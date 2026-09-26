# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Transport-agnostic watcher that follows a submitted simulation.

The MCP server no longer shares a filesystem with the simclient, so status is
*event push* (this watcher) plus request/response pulls (the ``status``/``logs``
handlers in :mod:`pic_agentic.simclient.client`).  The watcher uses an adaptive
backoff cadence -- a few ``scontrol`` calls per hour on a long run, not a fixed
high-frequency poll -- and tails the job's PIConGPU progress output to emit the
bounded ``simulation.step_finished`` cadence.

Deliberate simplification (documented in the plan): the backoff only widens
(initial -> max) and resets to the initial cadence when the job starts running;
it does not *tighten* as the expected walltime nears (the submit payload carries
no expected walltime) and a status/logs pull does not reset it (the pull is a
separate request/response path that answers live from ``scontrol``).  Both
refinements are unnecessary for correctness: the coarse terminal transition is
still observed within one ``max_interval_s`` tick.

The watcher is deliberately transport-agnostic: it takes an async ``emit``
callback and an async ``job_info`` callable, so it can be driven directly in
unit tests without a cluster, a transport or a room.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from pic_agentic.parsing.progress import parse_progress_line
from pic_agentic.protocol.simulation import (
    PROGRESS_EVENT_STEP_PERCENT,
    SimulationState,
)
from pic_agentic.simclient.simulation import find_stdout_path, link_run_results
from pic_agentic.slurm import SlurmJobState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pic_agentic.slurm import JobInfo

log = logging.getLogger(__name__)

#: A progress line at this percent is the terminal event.
_PERCENT_COMPLETE = 100


class TrackedSim(BaseModel):
    """Mutable follow-state for one submitted simulation.

    A pydantic model because this is protocol-adjacent data (it is read back by
    the ``status`` pull), not a runtime resource.  It is mutated in place by the
    watcher as it advances.
    """

    sim_id: str
    cmd_id: str
    job_id: int | None
    run_dir: str
    stdout_path: str | None
    submit_system: str
    #: Percent of the last emitted (or observed) progress line; -1 before any.
    last_percent: int = -1
    #: Fields of the last observed progress line, served by the status pull.
    last_step: int | None = None
    last_walltime: str | None = None
    last_avg_per_step: str | None = None
    last_eta_s: int | None = None
    #: Byte offset already consumed from the stdout file.
    stdout_offset: int = 0
    #: Whether ``simulation.job_running`` has been emitted.
    job_running_emitted: bool = False
    #: Whether a terminal event has been emitted.
    terminal_emitted: bool = False


def _eta_seconds(percent: int, step: int, avg_per_step_ms: int | None) -> int | None:
    """Derive a best-effort ETA from the progress line's average step time.

    Args:
        percent: Percent complete reported by the progress line.
        step: Current PIConGPU iteration.
        avg_per_step_ms: Average wall-clock milliseconds per step.

    Returns:
        The estimated remaining whole seconds, or None when not derivable.

    """
    if not avg_per_step_ms or percent <= 0 or step <= 0:
        return None
    total_steps = round(step * _PERCENT_COMPLETE / percent)
    remaining = max(0, total_steps - step)
    return int(avg_per_step_ms * remaining / 1000)


class JobFollower:
    """Follow one SLURM job to its terminal transition.

    The class is a plain class, not a pydantic model: it holds the async
    callbacks and the stop flag, i.e. live runtime state rather than data.
    """

    def __init__(
        self,
        *,
        sim: str,
        emit: Callable[..., Awaitable[None]],
        tracked: TrackedSim,
        job_info: Callable[[int], Awaitable[JobInfo]],
        initial_interval_s: float = 30.0,
        max_interval_s: float = 300.0,
    ) -> None:
        """Create a follower.

        Args:
            sim: The RCP simulation tag (for logging context).
            emit: Async callable ``emit(state, *, job_id, **fields)``.
            tracked: The mutable follow-state for the simulation.
            job_info: Async callable returning a :class:`~pic_agentic.slurm.JobInfo`.
            initial_interval_s: First (and reset) poll interval in seconds.
            max_interval_s: Cap for the backing-off poll interval in seconds.

        """
        self.sim = sim
        self.emit = emit
        self.tracked = tracked
        self.job_info = job_info
        self.initial_interval_s = initial_interval_s
        self.max_interval_s = max_interval_s
        self._stopped = False
        #: Last emitted ``percent // PROGRESS_EVENT_STEP_PERCENT`` bucket; 0 so
        #: the pre-first-boundary (0-24 %) range never emits on its own.
        self._last_emitted_bucket = 0

    def stop(self) -> None:
        """Request a prompt, idempotent shutdown of the loop."""
        self._stopped = True

    async def run(self) -> None:
        """Follow the job until it is terminal or :meth:`stop` is called.

        Unexpected exceptions are logged and end the loop: the watcher runs as a
        detached task, so letting one escape would surface as an unhandled task
        exception without stopping the serve loop.

        Raises:
            asyncio.CancelledError: If the detached task is cancelled.

        """
        try:
            await self._run()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("job follower for sim %s (job %s) failed", self.sim, self.tracked.job_id)
        finally:
            self._stopped = True

    async def _run(self) -> None:
        """Adaptive-backoff poll loop (see :meth:`run`)."""
        current = self.initial_interval_s
        while not self._stopped:
            # A submit system with no scheduler job id (local ``bash``, or an
            # unparseable scheduler id) has nothing to follow.
            if self.tracked.job_id is None:
                return
            info = await self._poll()
            if info is not None:
                if info.state is SlurmJobState.RUNNING and not self.tracked.job_running_emitted:
                    self.tracked.job_running_emitted = True
                    current = self.initial_interval_s
                    await self.emit(
                        SimulationState.JOB_RUNNING,
                        job_id=self.tracked.job_id,
                        slurm_state=info.state.value,
                    )
                await self._tail_progress()
                if info.state.terminal:
                    await self._emit_terminal(info)
                    return
            if self._stopped:
                return
            await asyncio.sleep(current)
            current = min(current * 2, self.max_interval_s)

    async def _poll(self) -> JobInfo | None:
        """Query the job state, tolerating a transient ``scontrol`` failure.

        The query is shielded so a cancellation (simclient shutdown) waits for
        the in-flight ``scontrol`` subprocess to be reaped instead of leaking it.

        Returns:
            The job snapshot, or None when the query failed.

        Raises:
            asyncio.CancelledError: If the follower is being cancelled.

        """
        job_id = self.tracked.job_id
        if job_id is None:  # pragma: no cover - guarded by the caller
            return None
        query = asyncio.ensure_future(self.job_info(job_id))
        try:
            return await asyncio.shield(query)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await query
            raise
        except Exception:  # ruff: ignore[blind-except] - transient scontrol failure
            log.warning("scontrol failed for sim %s job %s", self.sim, job_id)
            return None

    def _stdout_file(self) -> Path | None:
        """Return the progress stdout file, discovering it once if needed.

        Returns:
            The stdout path, or None when none is known or found.

        """
        if self.tracked.stdout_path:
            candidate = Path(self.tracked.stdout_path)
            if candidate.is_file():
                return candidate
        discovered = find_stdout_path(Path(self.tracked.run_dir))
        if discovered:
            # Cache the discovery so later ticks do not re-glob the cache.
            self.tracked.stdout_path = discovered
            return Path(discovered)
        return None

    def _read_new_lines(self) -> list[str]:
        """Read and consume complete new lines from the stdout file.

        Only whole (newline-terminated) lines are consumed, so a half-written
        progress line is picked up whole on the next tick.  Missing or
        unreadable files and a shrunk (rotated) file are tolerated.

        Returns:
            The freshly consumed lines (possibly empty).

        """
        path = self._stdout_file()
        if path is None:
            return []
        read = self._read_from(path, self.tracked.stdout_offset)
        if read is None:
            return []
        offset, data = read
        newline = data.rfind(b"\n")
        if newline < 0:
            # No complete line yet; leave the offset so the partial line is
            # completed (and consumed) on a later tick.
            self.tracked.stdout_offset = offset
            return []
        consumed = data[: newline + 1]
        self.tracked.stdout_offset = offset + len(consumed)
        return consumed.decode("utf-8", errors="replace").splitlines()

    @staticmethod
    def _read_from(path: Path, offset: int) -> tuple[int, bytes] | None:
        """Read a file from ``offset``, recovering from a shrunk file.

        Args:
            path: The file to read.
            offset: The byte offset to start from.

        Returns:
            The ``(effective_offset, bytes)`` read, or None when unreadable; a
            file smaller than ``offset`` (rotated) is re-read from the start.

        """
        try:
            with path.open("rb") as handle:
                size = handle.seek(0, 2)
                offset = 0 if offset > size else offset
                handle.seek(offset)
                data = handle.read()
        except OSError:
            return None
        return offset, data

    async def _tail_progress(self) -> None:
        """Parse newly written progress lines and emit their bounded cadence."""
        for line in self._read_new_lines():
            parsed = parse_progress_line(line)
            if parsed is None:
                continue
            eta_s = _eta_seconds(parsed.percent, parsed.step, parsed.avg_per_step_ms)
            self.tracked.last_percent = parsed.percent
            self.tracked.last_step = parsed.step
            self.tracked.last_walltime = parsed.elapsed
            self.tracked.last_avg_per_step = parsed.avg_per_step
            self.tracked.last_eta_s = eta_s
            crossed = parsed.percent // PROGRESS_EVENT_STEP_PERCENT > self._last_emitted_bucket
            terminal = parsed.percent >= _PERCENT_COMPLETE
            if crossed or terminal:
                await self.emit(
                    SimulationState.STEP_FINISHED,
                    job_id=self.tracked.job_id,
                    step=parsed.step,
                    percent=parsed.percent,
                    walltime=parsed.elapsed,
                    avg_per_step=parsed.avg_per_step,
                    eta_s=eta_s,
                )
                self._last_emitted_bucket = parsed.percent // PROGRESS_EVENT_STEP_PERCENT

    async def _emit_terminal(self, info: JobInfo) -> None:
        """Emit the terminal lifecycle events and stop following.

        Args:
            info: The terminal job snapshot.

        """
        if self.tracked.terminal_emitted:
            return
        self.tracked.terminal_emitted = True
        if info.state is SlurmJobState.COMPLETED and info.exit_code in {None, 0}:
            await self.emit(
                SimulationState.JOB_FINISHED,
                job_id=self.tracked.job_id,
                slurm_state=info.state.value,
                exit_code=info.exit_code,
            )
            linked = await asyncio.to_thread(link_run_results, Path(self.tracked.run_dir))
            # ``results.ready`` is only claimed once ``run_dir/simOutput`` really
            # exists (the plan's M2b contract).  A failed link emits a second,
            # explicit ``job_finished`` carrying ``results_linked=False`` (the
            # stored-terminal re-emission is idempotent and condenses to the
            # latest payload); a later status pull promotes it via
            # ``SimClient._state_for_info`` once the link appears.
            if linked:
                await self.emit(
                    SimulationState.RESULTS_READY,
                    job_id=self.tracked.job_id,
                    results_linked=True,
                )
            else:
                await self.emit(
                    SimulationState.JOB_FINISHED,
                    job_id=self.tracked.job_id,
                    slurm_state=info.state.value,
                    exit_code=info.exit_code,
                    results_linked=False,
                )
        else:
            await self.emit(
                SimulationState.JOB_FAILED,
                job_id=self.tracked.job_id,
                slurm_state=info.state.value,
                exit_code=info.exit_code,
            )


__all__ = ["JobFollower", "TrackedSim"]
