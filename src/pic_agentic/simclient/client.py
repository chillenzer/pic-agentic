# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""The simulation-side RCP client.

Runs on the submission node, holds the cluster session, verifies inbound
commands (HMAC + sender allow-list) and executes only the fixed M1 ``hello``
command.  It has no arbitrary-shell surface.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from pic_agentic.protocol.hello import HelloType, build_hello_ack
from pic_agentic.protocol.simulation import (
    PAYLOAD_KEY,
    SimulationStage,
    SimulationState,
    SimulationType,
    build_logs_ack,
    build_status_ack,
    build_submit_ack,
    build_submit_event,
)
from pic_agentic.rcp import DedupStore, Kind, RcpMessage, SenderRole, SequenceState
from pic_agentic.simclient.follow import JobFollower, TrackedSim
from pic_agentic.simclient.safety import safe_write_message
from pic_agentic.simclient.simulation import (
    PreparedSubmit,
    SimulationErrorCode,
    SimulationExecutionError,
    SubmitConfig,
    execute_submit,
    find_stdout_path,
    parse_payload,
    prepare_submit,
)
from pic_agentic.slurm import JobInfo, SlurmClient, SlurmError, SlurmJobState
from pic_agentic.version import local_provenance as _local_provenance

if TYPE_CHECKING:
    from pic_agentic.transport.base import Transport

log = logging.getLogger(__name__)

#: Upper bound on retained idempotency records, so the durable file cannot grow
#: without limit on a long-lived message directory.
_MAX_PROCESSED = 4096

#: M2 commands gated by the presence of a cluster-local ``submit_config``.
_M2_COMMANDS = frozenset(
    {SimulationType.COMMAND, SimulationType.STATUS_COMMAND, SimulationType.LOGS_COMMAND},
)

#: Default first (and reset) poll interval for the job-follow watcher, per the
#: M2b plan (30 s -> 5 min adaptive backoff); overridable per instance and via
#: ``PIC_AGENTIC_POLL_INTERVAL_S``.
DEFAULT_POLL_INTERVAL_S = 30.0

#: Default cap for the watcher's backing-off poll interval (the plan's 5 min).
DEFAULT_POLL_MAX_INTERVAL_S = 300.0

#: Assume a log line is at most this long when sizing the tail read window; a
#: longer line is still returned in full once the window is aligned to a
#: newline, but may cover fewer than ``tail`` lines.
_ASSUMED_MAX_LINE_BYTES = 4096

#: Hard cap on the bytes a single ``get_logs`` reads from the end of a stream,
#: so an unbounded PIConGPU ``stdout`` cannot OOM the simclient.
_MAX_LOG_READ_BYTES = 8 * 1024 * 1024


class HelloResult(BaseModel):
    """Outcome of one ``hello`` command execution."""

    job_id: int | None
    cluster_output: str | None
    error: str | None = None


class ProcessedCommand(BaseModel):
    """Durable idempotency record for one command.

    The record is written *before* execution (so a crash mid-job cannot cause a
    re-submission) and updated with the result afterwards.  Storing the result
    lets a replay re-send the same ack instead of leaving the MCP sender to time
    out on a command that already ran.
    """

    cmd_id: str
    #: False while the job is still running (or the process died mid-execution).
    completed: bool = False
    job_id: int | None = None
    cluster_output: str | None = None
    error: str | None = None
    #: For ``submit_simulation``: the payload hash this cmd_id was executed
    #: with, so an identical resend is re-acked while a *changed* payload under
    #: the same cmd_id is treated as a new simulation.
    payload_hash: str | None = None
    sim_id: str | None = None
    state: str | None = None
    #: Machine-readable code of a recorded failure (``hello`` leaves it unset).
    error_code: str | None = None

    def to_result(self) -> HelloResult:
        """Return the execution result to replay in an ack.

        Returns:
            The stored result, or a sentinel error when the outcome is unknown
            (the job started but no result was recorded before a restart).

        """
        if self.completed:
            return HelloResult(job_id=self.job_id, cluster_output=self.cluster_output, error=self.error)
        return HelloResult(job_id=self.job_id, cluster_output=None, error="already_submitted:outcome_unknown")


class SimClient:
    """Handle ``rcp.hello`` commands for one simulation on one transport."""

    def __init__(
        self,
        *,
        sim: str,
        secret: str,
        transport: Transport,
        slurm: SlurmClient,
        message_dir: Path,
        job_wait_timeout_s: float = 60.0,
        poll_interval_s: float = 0.2,
        poll_max_interval_s: float = 300.0,
        allowed_sender_user_id: str | None = None,
        submit_config: SubmitConfig | None = None,
    ) -> None:
        """Create a simulation-side client.

        Args:
            sim: Simulation id this client answers for.
            secret: Shared per-simulation RCP secret.
            transport: The RCP transport carrying commands and acks.
            slurm: The SLURM command layer.
            message_dir: Shared-filesystem base directory for payload files.
            job_wait_timeout_s: Maximum wait for a submitted job.
            poll_interval_s: Initial (and reset) interval between ``scontrol``
                polls in the job-follow watcher; also the ``hello`` wait loop.
            poll_max_interval_s: Cap for the watcher's backing-off interval.
            allowed_sender_user_id: Optional expected MCP-server identity.
            submit_config: Cluster-local policy for ``submit_simulation``;
                when omitted the M2 handler is disabled.

        """
        self.sim = sim
        self.secret = secret
        self.transport = transport
        self.slurm = slurm
        self.message_dir = message_dir
        self.job_wait_timeout_s = job_wait_timeout_s
        self.poll_interval_s = poll_interval_s
        self.poll_max_interval_s = poll_max_interval_s
        self.allowed_sender_user_id = allowed_sender_user_id
        self.submit_config = submit_config
        self.sequences = SequenceState()
        self.seen = DedupStore()
        #: Per-sim follow-state and detached watcher tasks, keyed by ``sim_id``.
        self._tracked: dict[str, TrackedSim] = {}
        #: Current watcher per ``sim_id`` (the latest run).
        self._follow_tasks: dict[str, asyncio.Task[None]] = {}
        #: Every live watcher task, including a superseded one still winding
        #: down, so shutdown reaps orphans the dict slot no longer points at.
        self._follow_tasks_all: set[asyncio.Task[None]] = set()
        #: Detached watchers are started only while :meth:`serve` owns the event
        #: loop, so a direct ``handle`` call in a test does not leave a task
        #: (and its subprocesses) running past the test.
        self._serving = False
        #: Command ids already executed, mapped to their result.  Persisted
        #: under ``message_dir`` so a restart that backfills the room does not
        #: re-submit cluster jobs for commands it already ran, and so a replay
        #: can re-send the original ack (the transport replays the whole
        #: timeline on reconnect).  The store assumes one simclient per
        #: ``message_dir`` (the supported topology); it is not cross-process
        #: locked.  ``cluster_output`` is small for the M1 ``hello`` job and the
        #: file is capped at :data:`_MAX_PROCESSED` records.
        self._processed: dict[str, ProcessedCommand] = {}
        self._processed_path = message_dir / "processed-cmds.jsonl"
        self._load_processed()

    def _accepts(self, message: RcpMessage) -> bool:
        if message.sim != self.sim or message.kind is not Kind.COMMAND:
            return False
        if not message.verify(self.secret):
            log.warning("rejecting RCP message with bad signature: %s", message.type)
            return False
        # Defence in depth (design section 6.4): the registered MCP server
        # identity must match, when configured.
        if (
            self.allowed_sender_user_id
            and message.transport_sender
            and message.transport_sender != self.allowed_sender_user_id
        ):
            log.warning("rejecting command from unexpected sender %s", message.transport_sender)
            return False
        return self.seen.seen(message)

    async def handle(self, message: RcpMessage) -> RcpMessage | None:
        """Validate and dispatch one inbound message.

        Args:
            message: The inbound message.

        Returns:
            The reply ack that was sent, or None if the message was ignored.

        """
        if not self._accepts(message):
            return None
        self.sequences.observe(message.sim, message.sender_role, message.seq)
        if message.type == HelloType.COMMAND:
            return await self._handle_hello(message)
        if message.type in _M2_COMMANDS:
            if self.submit_config is None:
                return await self._reject_m2(message, error="rejected_by_policy")
            return await self._dispatch_m2(message)
        return await self._ack(message, cmd_id=message.payload.get("cmd_id"), error="rejected_by_policy")

    async def _dispatch_m2(self, message: RcpMessage) -> RcpMessage:
        """Dispatch one of the M2 commands to its handler.

        Returns:
            The signed acknowledgement that was sent.

        """
        if message.type == SimulationType.COMMAND:
            return await self._handle_submit(message)
        if message.type == SimulationType.STATUS_COMMAND:
            return await self._handle_status(message)
        return await self._handle_logs(message)

    async def _reject_m2(self, message: RcpMessage, *, error: str) -> RcpMessage:
        """Reject an M2 command when the cluster-local handler is disabled.

        Returns:
            The signed acknowledgement that was sent.

        """
        if message.type == SimulationType.COMMAND:
            return await self._reject_submit(message, error=error)
        return await self._reject_pull(message, error=error)

    async def _reject_pull(self, message: RcpMessage, *, error: str) -> RcpMessage:
        """Send a status/logs-shaped rejection.

        Returns:
            The signed acknowledgement that was sent.

        """
        if message.type == SimulationType.LOGS_COMMAND:
            ack = self._build_logs_ack(
                message,
                cmd_id=str(message.payload.get("cmd_id", "")),
                sim_id=str(message.payload.get("sim_id", "")),
                stream=str(message.payload.get("stream", "stdout")),
                lines=[],
                total_lines=0,
                error=error,
                error_code=SimulationErrorCode.REJECTED,
            )
        else:
            ack = self._build_status_ack(
                message,
                cmd_id=str(message.payload.get("cmd_id", "")),
                sim_id=str(message.payload.get("sim_id", "")),
                state=SimulationState.FAILED.value,
                error=error,
                error_code=SimulationErrorCode.REJECTED,
            )
        await self.transport.send(ack)
        return ack

    async def _reject_submit(
        self,
        message: RcpMessage,
        *,
        error: str,
        sim_id: str = "",
        cmd_id: str = "",
    ) -> RcpMessage:
        """Send a submit-shaped rejection (never a ``hello_ack``).

        Returns:
            The signed acknowledgement that was sent.

        """
        header = message.payload.get("header")
        header = header if isinstance(header, dict) else {}
        ack = self._build_submit_ack(
            message,
            cmd_id=cmd_id or str(message.payload.get("cmd_id", "")),
            sim_id=sim_id or str(header.get("sim_id", "")),
            state=SimulationState.FAILED.value,
            job_id=None,
            error=error,
            error_code=SimulationErrorCode.REJECTED,
        )
        await self.transport.send(ack)
        return ack

    @staticmethod
    def _read_outcome(info: JobInfo, outfile: str) -> tuple[str | None, str | None]:
        """Map a terminal job state to a result pair.

        Returns:
            The ``(error, cluster_output)`` pair for the finished job.

        """
        if info.state.value == "COMPLETED":
            return None, Path(outfile).read_text(encoding="utf-8", errors="replace")
        if info.state.terminal:
            return f"job_failed:{info.state.value}", None
        return f"job_timeout:{info.state.value}", None

    async def _execute_hello(self, message: RcpMessage, cmd_id: str) -> HelloResult:
        message_path = str(message.payload.get("message_path", ""))
        content = str(message.payload.get("message", "Hello World"))
        outfile = str(self._outfile_path(cmd_id))
        result = HelloResult(job_id=None, cluster_output=None)
        try:
            safe_write_message(message_path, self.message_dir, default=content)
            result.job_id = await self.slurm.submit_wrap_cat(message_path, outfile)
            info = await self.slurm.wait_for_job(
                result.job_id,
                timeout_s=self.job_wait_timeout_s,
                interval_s=self.poll_interval_s,
            )
        except SlurmError as exc:
            result.error = f"signal_failed:{exc}" if result.job_id else f"submit_failed:{exc}"
            return result
        except Exception as exc:  # ruff: ignore[blind-except] - surfaced verbatim to the LLM
            result.error = f"unexpected:{exc}"
            return result
        result.error, result.cluster_output = self._read_outcome(info, outfile)
        return result

    def _load_processed(self) -> None:
        """Load persisted command records, ignoring a missing or unreadable file."""
        try:
            text = self._processed_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return
        except OSError as exc:
            log.warning("cannot read %s: %s", self._processed_path, exc)
            return
        records: dict[str, ProcessedCommand] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = ProcessedCommand.model_validate_json(line)
            except ValueError as exc:
                log.warning(
                    "ignoring malformed processed-command line in %s (%s)",
                    self._processed_path,
                    type(exc).__name__,
                )
                continue
            records[record.cmd_id] = record
        self._processed = records

    def _persist_processed(self, record: ProcessedCommand) -> None:
        """Append one record to the durable idempotency file.

        Records are append-only and last-write-wins on load, so updating an
        execution's outcome is just another append.  The file is compacted once
        it holds more than :data:`_MAX_PROCESSED` records.

        Args:
            record: The command record (pending or completed) to persist.

        """
        try:
            self._processed_path.parent.mkdir(parents=True, exist_ok=True)
            with self._processed_path.open("a", encoding="utf-8") as handle:
                handle.write(record.model_dump_json() + "\n")
        except OSError as exc:
            log.warning("cannot persist processed id %s: %s", record.cmd_id, exc)
            return
        self._processed[record.cmd_id] = record
        if len(self._processed) > _MAX_PROCESSED:
            self._rewrite_processed()

    def _rewrite_processed(self) -> None:
        """Rewrite the idempotency file with only the most recent records."""
        recent = list(self._processed.values())[-_MAX_PROCESSED:]
        self._processed = {record.cmd_id: record for record in recent}
        tmp = self._processed_path.with_suffix(self._processed_path.suffix + ".tmp")
        try:
            with os.fdopen(os.open(tmp, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600), "w", encoding="utf-8") as handle:
                for record in recent:
                    handle.write(record.model_dump_json() + "\n")
            tmp.replace(self._processed_path)
        except OSError as exc:
            log.warning("cannot compact %s: %s", self._processed_path, exc)

    async def _handle_hello(self, message: RcpMessage) -> RcpMessage | None:
        cmd_id = str(message.payload.get("cmd_id", ""))
        # Idempotency: a re-sent or backfilled command carries the same cmd_id;
        # do not submit a second job.  Re-ack the stored result so a sender
        # whose original ack was lost does not block until its timeout.
        if cmd_id and cmd_id in self._processed:
            log.info("re-acking already-processed hello command %s", cmd_id)
            ack = self._build_ack(message, cmd_id=cmd_id, result=self._processed[cmd_id].to_result())
            await self.transport.send(ack)
            return ack
        if cmd_id:
            # Persist *before* executing so a crash mid-job cannot resubmit.
            self._persist_processed(ProcessedCommand(cmd_id=cmd_id))
        result = await self._execute_hello(message, cmd_id)
        ack = self._build_ack(message, cmd_id=cmd_id, result=result)
        if cmd_id:
            self._persist_processed(
                ProcessedCommand(
                    cmd_id=cmd_id,
                    completed=True,
                    job_id=result.job_id,
                    cluster_output=result.cluster_output,
                    error=result.error,
                )
            )
        await self.transport.send(ack)
        return ack

    async def _submit_replay_ack(
        self,
        message: RcpMessage,
        existing: ProcessedCommand,
        *,
        cmd_id: str,
        sim_id: str,
        payload_hash: str,
    ) -> RcpMessage | None:
        """Handle an already-seen ``cmd_id``.

        Returns:
            The signed ack to re-send, or None when the command is genuinely new
            and should be executed.

        """
        if existing.payload_hash and payload_hash and existing.payload_hash != payload_hash:
            # A *different* payload under an already-executed cmd_id is a sender
            # bug or an attack: executing it would also overwrite the stored
            # record, so the original could later be re-run.  Reject and keep
            # the original record; a real resubmission always gets a new cmd_id.
            log.warning("rejecting submit command %s with a changed payload", cmd_id)
            ack = self._build_submit_ack(
                message,
                cmd_id=cmd_id,
                sim_id=sim_id,
                state=SimulationState.FAILED.value,
                job_id=None,
                error="cmd_id_conflict: a different payload was already submitted under this cmd_id",
                error_code=SimulationErrorCode.REJECTED,
            )
        elif existing.payload_hash == payload_hash and payload_hash:
            log.info("re-acking already-processed submit command %s", cmd_id)
            # A pending record means the process died mid-build: report that
            # honestly instead of claiming acceptance (mirrors the hello path's
            # already_submitted:outcome_unknown).  A finished record re-sends its
            # stored terminal state and error.
            if existing.completed:
                state = existing.state or SimulationState.FAILED.value
                error: str | None = existing.error
                code: str | None = existing.error_code
            else:
                state = SimulationState.FAILED.value
                error = "already_submitted:outcome_unknown"
                code = SimulationErrorCode.REJECTED
            ack = self._build_submit_ack(
                message,
                cmd_id=cmd_id,
                sim_id=existing.sim_id or sim_id,
                state=state,
                job_id=existing.job_id,
                error=error,
                error_code=code,
            )
        else:
            return None
        await self.transport.send(ack)
        return ack

    async def _prepare_or_reject(
        self,
        message: RcpMessage,
        *,
        header: dict,
        payload_hash: str,
        cmd_id: str,
        sim_id: str,
    ) -> PreparedSubmit | RcpMessage:
        """Validate a new submit command, or send and return a rejection ack.

        Returns:
            The prepared submission, or the rejection ack that was sent.

        """
        raw = message.payload.get(PAYLOAD_KEY)
        if not isinstance(raw, str):
            return await self._reject_submit(message, error="payload_missing", sim_id=sim_id, cmd_id=cmd_id)
        try:
            body = parse_payload(raw)
            return prepare_submit(
                body=body,
                header=header,
                params=message.payload.get("params"),
                config=self.submit_config,
                local_provenance=_local_provenance(),
                token=cmd_id or payload_hash,
            )
        except SimulationExecutionError as exc:
            ack = self._build_submit_ack(
                message,
                cmd_id=cmd_id,
                sim_id=sim_id,
                state=SimulationState.FAILED.value,
                job_id=None,
                error=str(exc),
                error_code=exc.code,
            )
            if cmd_id:
                self._persist_processed(
                    ProcessedCommand(
                        cmd_id=cmd_id,
                        completed=True,
                        payload_hash=payload_hash,
                        sim_id=sim_id,
                        state=SimulationState.FAILED.value,
                        error=str(exc),
                        error_code=exc.code,
                    )
                )
            await self.transport.send(ack)
            return ack

    async def _handle_submit(self, message: RcpMessage) -> RcpMessage | None:
        cmd_id = str(message.payload.get("cmd_id", ""))
        header = message.payload.get("header")
        header = header if isinstance(header, dict) else {}
        payload_hash = str(header.get("payload_hash", ""))
        sim_id = str(header.get("sim_id", ""))
        # Idempotency: same cmd_id + same payload hash is a replay (re-ack); a
        # different payload under the same cmd_id is rejected; a changed
        # simulation must use a fresh cmd_id.
        existing = self._processed.get(cmd_id) if cmd_id else None
        if existing is not None:
            replay_ack = await self._submit_replay_ack(
                message, existing, cmd_id=cmd_id, sim_id=sim_id, payload_hash=payload_hash
            )
            if replay_ack is not None:
                return replay_ack
        # A genuinely new command.  Persist *before* executing so a crash
        # mid-build cannot cause a re-run on restart.
        if cmd_id:
            self._persist_processed(ProcessedCommand(cmd_id=cmd_id, payload_hash=payload_hash, sim_id=sim_id))

        # Validate *before* accepting: a rejected command must report the reason
        # in its single ack, not as a lifecycle event for a sim that never
        # started (design section 2.2).
        prepared = await self._prepare_or_reject(
            message,
            header=header,
            payload_hash=payload_hash,
            cmd_id=cmd_id,
            sim_id=sim_id,
        )
        if isinstance(prepared, RcpMessage):
            return prepared

        sim_id = prepared.payload.sim_id
        # First ack: accepted (coarse; per-stage acks wait for upstream #55).
        accepted = self._build_submit_ack(
            message,
            cmd_id=cmd_id,
            sim_id=sim_id,
            state=SimulationState.ACCEPTED.value,
            job_id=None,
        )
        await self.transport.send(accepted)

        async def emit(state: SimulationState, *, job_id: int | None = None, **fields: object) -> None:
            event = self._build_submit_event(
                cmd_id=cmd_id,
                sim_id=sim_id,
                state=state,
                job_id=job_id,
                **fields,
            )
            await self.transport.send(event)

        try:
            result = await execute_submit(
                prepared=prepared,
                emit=emit,
                job_id_reader=self._read_submission_job_id,
            )
        except SimulationExecutionError as exc:
            await emit(
                SimulationState.FAILED,
                stage=exc.stage or SimulationStage.BUILD,
                error=str(exc),
                error_code=exc.code,
            )
            if cmd_id:
                self._persist_processed(
                    ProcessedCommand(
                        cmd_id=cmd_id,
                        completed=True,
                        payload_hash=payload_hash,
                        sim_id=sim_id,
                        state=SimulationState.FAILED.value,
                        error=str(exc),
                        error_code=exc.code,
                    )
                )
            return accepted
        if cmd_id:
            self._persist_processed(
                ProcessedCommand(
                    cmd_id=cmd_id,
                    completed=True,
                    job_id=result.get("job_id"),
                    payload_hash=payload_hash,
                    sim_id=str(result.get("sim_id", sim_id)),
                    state=str(result.get("state", SimulationState.WORKFLOW_FINISHED.value)),
                )
            )
        await self._start_follower(
            cmd_id=cmd_id,
            sim_id=str(result.get("sim_id", sim_id)),
            job_id=result.get("job_id"),
            run_dir=str(result.get("run_dir", "")),
            stdout_path=result.get("stdout_path"),
            submit_system=prepared.params.submit_system,
        )
        return accepted

    async def _start_follower(
        self,
        *,
        cmd_id: str,
        sim_id: str,
        job_id: int | None,
        run_dir: str,
        stdout_path: str | None,
        submit_system: str,
    ) -> None:
        """Store follow-state and start the detached watcher for one simulation.

        The watcher is skipped for a submit system with neither a scheduler job
        id nor an ``sbatch`` contract (e.g. local ``bash`` execution): there is
        no job to follow and the workflow already finished synchronously.
        """
        if not self._serving or (job_id is None and submit_system != "sbatch"):
            return
        tracked = TrackedSim(
            sim_id=sim_id,
            cmd_id=cmd_id,
            job_id=job_id,
            run_dir=run_dir,
            stdout_path=stdout_path,
            submit_system=submit_system,
        )
        # A resubmission of an identical simulation yields the same ``sim_id``
        # (payload-hash prefix).  Cancel the previous watcher *before* replacing
        # it, so it cannot keep polling and emitting under its stale ``cmd_id``;
        # keep it in ``_follow_tasks_all`` until it has actually finished so
        # shutdown reaps it even after the dict slot is reused.
        previous = self._follow_tasks.pop(sim_id, None)
        if previous is not None:
            previous.cancel()
            self._follow_tasks_all.add(previous)
            previous.add_done_callback(self._follow_tasks_all.discard)
            # Await it so the cancelled watcher's in-flight ``scontrol``
            # subprocess is reaped before the replacement starts (the follower
            # shields its poll for exactly this reason).
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await previous
        self._tracked[sim_id] = tracked

        async def emit(state: SimulationState, *, job_id: int | None = None, **fields: object) -> None:
            event = self._build_submit_event(cmd_id=cmd_id, sim_id=sim_id, state=state, job_id=job_id, **fields)
            await self.transport.send(event)

        follower = JobFollower(
            sim=self.sim,
            emit=emit,
            tracked=tracked,
            job_info=self.slurm.job_info,
            initial_interval_s=self.poll_interval_s,
            max_interval_s=self.poll_max_interval_s,
        )
        task = asyncio.create_task(follower.run())
        self._follow_tasks[sim_id] = task
        self._follow_tasks_all.add(task)
        task.add_done_callback(self._follow_tasks_all.discard)

    async def _cancel_followers(self) -> None:
        """Stop and await every detached watcher task (idempotent).

        Cancels both the current per-``sim_id`` watchers and any superseded
        (orphaned) watcher still winding down, then awaits them all.
        """
        tasks = set(self._follow_tasks.values()) | self._follow_tasks_all
        for follower_task in tasks:
            follower_task.cancel()
        for follower_task in tasks:
            # A cancelled follower raises CancelledError; a follower that crashed
            # before cancellation may raise its stored exception.  Best-effort.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await follower_task
        self._follow_tasks.clear()
        self._follow_tasks_all.clear()

    @staticmethod
    def _state_for_info(info: JobInfo, run_dir: str) -> str:
        """Map a live SLURM snapshot to a coarse lifecycle state.

        Returns:
            The :class:`SimulationState` value to report in a status ack.

        """
        if info.state is SlurmJobState.COMPLETED and info.exit_code in {None, 0}:
            if run_dir and (Path(run_dir) / "simOutput").exists():
                return SimulationState.RESULTS_READY.value
            return SimulationState.JOB_FINISHED.value
        mapping = {
            SlurmJobState.PENDING: SimulationState.SUBMITTED,
            SlurmJobState.RUNNING: SimulationState.JOB_RUNNING,
            SlurmJobState.COMPLETING: SimulationState.JOB_RUNNING,
            SlurmJobState.COMPLETED: SimulationState.JOB_FINISHED,
            SlurmJobState.FAILED: SimulationState.JOB_FAILED,
            SlurmJobState.CANCELLED: SimulationState.JOB_FAILED,
            SlurmJobState.TIMEOUT: SimulationState.JOB_FAILED,
            SlurmJobState.UNKNOWN: SimulationState.WORKFLOW_FINISHED,
        }
        return mapping[info.state].value

    async def _handle_status(self, message: RcpMessage) -> RcpMessage:
        """Answer a ``status_request`` with a live or last-known snapshot.

        Errors are reported as data in the ack, never raised: a status pull
        must not tear down the serve loop.

        Returns:
            The signed ``status_ack`` that was sent.

        """
        cmd_id = str(message.payload.get("cmd_id", ""))
        sim_id = str(message.payload.get("sim_id", ""))
        tracked = self._tracked.get(sim_id)
        if tracked is None:
            ack = self._build_status_ack(
                message,
                cmd_id=cmd_id,
                sim_id=sim_id,
                state=SimulationState.FAILED.value,
                error="unknown_sim",
                error_code="unknown_sim",
            )
            await self.transport.send(ack)
            return ack
        state = SimulationState.WORKFLOW_FINISHED.value
        slurm_state: str | None = None
        exit_code: int | None = None
        error: str | None = None
        error_code: str | None = None
        if tracked.job_id is not None:
            try:
                info = await self.slurm.job_info(tracked.job_id)
                state = self._state_for_info(info, tracked.run_dir)
                slurm_state = info.state.value
                exit_code = info.exit_code
            except Exception as exc:  # ruff: ignore[blind-except] - a transient scontrol failure is ack data
                log.warning("status query failed for sim %s: %s", sim_id, exc)
                error = f"job_info_failed:{exc}"
                error_code = "job_info_failed"
        ack = self._build_status_ack(
            message,
            cmd_id=cmd_id,
            sim_id=sim_id,
            state=state,
            slurm_state=slurm_state,
            job_id=tracked.job_id,
            step=tracked.last_step,
            percent=tracked.last_percent if tracked.last_percent >= 0 else None,
            walltime=tracked.last_walltime,
            avg_per_step=tracked.last_avg_per_step,
            eta_s=tracked.last_eta_s,
            exit_code=exit_code,
            error=error,
            error_code=error_code,
        )
        await self.transport.send(ack)
        return ack

    @staticmethod
    def _log_path(tracked: TrackedSim, stream: str) -> Path | None:
        """Resolve the on-disk file backing a log stream.

        Returns:
            The file path to read, or None when the stream has no known file.

        """
        run_dir = Path(tracked.run_dir)
        if stream == "stdout":
            if tracked.stdout_path and Path(tracked.stdout_path).is_file():
                return Path(tracked.stdout_path)
            return run_dir / "stdout"
        if stream == "stderr":
            return run_dir / "stderr"
        discovered = find_stdout_path(run_dir)
        return Path(discovered) if discovered else None

    @staticmethod
    def _read_tail(path: Path, tail: int) -> tuple[list[str], int]:
        """Read up to ``tail`` trailing lines without slurping the whole file.

        The read window is bounded by :data:`_MAX_LOG_READ_BYTES`, so a
        multi-hundred-MB PIConGPU ``stdout`` cannot OOM the simclient.  When
        the window does not reach the start of the file the first (partial)
        line is dropped, and ``total_lines`` then counts only the lines in the
        window (the true total is unknowable without reading everything).

        Args:
            path: The log file to read.
            tail: Maximum number of trailing lines to return.

        Returns:
            The ``(lines, total_lines)`` pair; ``([], 0)`` on an unreadable file.

        """
        try:
            size = path.stat().st_size
        except OSError:
            return [], 0
        window = min(_MAX_LOG_READ_BYTES, max(tail, 1) * _ASSUMED_MAX_LINE_BYTES + 1)
        start = max(0, size - window)
        try:
            with path.open("rb") as handle:
                handle.seek(start)
                data = handle.read()
        except OSError:
            return [], 0
        if start > 0:
            newline = data.find(b"\n")
            # No newline in the window means a single line longer than the
            # whole window; report nothing rather than a bogus partial line.
            if newline < 0:
                return [], 0
            data = data[newline + 1 :]
        all_lines = data.decode("utf-8", errors="replace").splitlines()
        return (all_lines[-tail:] if tail else []), len(all_lines)

    async def _handle_logs(self, message: RcpMessage) -> RcpMessage:
        """Answer a ``logs_request`` with up to ``tail`` lines of a stream.

        A missing file is reported as an empty log (not an error).  Errors are
        ack data, never raised.

        Returns:
            The signed ``logs_ack`` that was sent.

        """
        cmd_id = str(message.payload.get("cmd_id", ""))
        sim_id = str(message.payload.get("sim_id", ""))
        stream = str(message.payload.get("stream", "stdout"))
        try:
            tail = int(message.payload.get("tail", 100))
        except (TypeError, ValueError):
            tail = 100
        tail = max(0, min(tail, 10_000))
        tracked = self._tracked.get(sim_id)
        if tracked is None:
            ack = self._build_logs_ack(
                message,
                cmd_id=cmd_id,
                sim_id=sim_id,
                stream=stream,
                lines=[],
                total_lines=0,
                error="unknown_sim",
                error_code="unknown_sim",
            )
            await self.transport.send(ack)
            return ack
        lines: list[str] = []
        total = 0
        path = self._log_path(tracked, stream)
        if path is not None:
            lines, total = await asyncio.to_thread(self._read_tail, path, tail)
        ack = self._build_logs_ack(
            message,
            cmd_id=cmd_id,
            sim_id=sim_id,
            stream=stream,
            lines=lines,
            total_lines=total,
        )
        await self.transport.send(ack)
        return ack

    @staticmethod
    def _read_submission_job_id(run_dir: Path, _payload: object) -> int | None:
        """Parse the SLURM job id from ``run_dir/submission_information.txt``.

        Returns:
            The job id, or None when the file is absent or carries no id (the
            local ``bash`` submit system writes a PID instead of a job id).

        """
        try:
            text = (Path(run_dir) / "submission_information.txt").read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        return SlurmClient.parse_job_id(text)

    def _build_submit_ack(
        self,
        message: RcpMessage,
        *,
        cmd_id: str,
        sim_id: str,
        state: str,
        job_id: int | None,
        error: str | None = None,
        error_code: str | None = None,
    ) -> RcpMessage:
        return build_submit_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            sim_id=sim_id,
            state=SimulationState(state),
            in_reply_to=message.transport_event_id,
            job_id=job_id,
            error=error,
            error_code=error_code,
        ).sign(self.secret)

    def _build_status_ack(
        self,
        message: RcpMessage,
        *,
        cmd_id: str,
        sim_id: str,
        state: str,
        slurm_state: str | None = None,
        job_id: int | None = None,
        step: int | None = None,
        percent: int | None = None,
        walltime: str | None = None,
        avg_per_step: str | None = None,
        eta_s: int | None = None,
        exit_code: int | None = None,
        error: str | None = None,
        error_code: str | None = None,
    ) -> RcpMessage:
        return build_status_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            sim_id=sim_id,
            in_reply_to=message.transport_event_id,
            state=state,
            slurm_state=slurm_state,
            job_id=job_id,
            step=step,
            percent=percent,
            walltime=walltime,
            avg_per_step=avg_per_step,
            eta_s=eta_s,
            exit_code=exit_code,
            error=error,
            error_code=error_code,
        ).sign(self.secret)

    def _build_logs_ack(
        self,
        message: RcpMessage,
        *,
        cmd_id: str,
        sim_id: str,
        stream: str,
        lines: list[str],
        total_lines: int,
        error: str | None = None,
        error_code: str | None = None,
    ) -> RcpMessage:
        return build_logs_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            sim_id=sim_id,
            in_reply_to=message.transport_event_id,
            stream=stream,
            lines=lines,
            total_lines=total_lines,
            error=error,
            error_code=error_code,
        ).sign(self.secret)

    def _build_submit_event(
        self,
        *,
        cmd_id: str,
        sim_id: str,
        state: SimulationState,
        job_id: int | None,
        stage: SimulationStage | None = None,
        error: str | None = None,
        error_code: str | None = None,
        submit_system: str | None = None,
        results_linked: bool | None = None,
        step: int | None = None,
        percent: int | None = None,
        walltime: str | None = None,
        avg_per_step: str | None = None,
        eta_s: int | None = None,
        slurm_state: str | None = None,
        exit_code: int | None = None,
    ) -> RcpMessage:
        return build_submit_event(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            sim_id=sim_id,
            state=state,
            job_id=job_id,
            stage=stage,
            error=error,
            error_code=error_code,
            submit_system=submit_system,
            results_linked=results_linked,
            step=step,
            percent=percent,
            walltime=walltime,
            avg_per_step=avg_per_step,
            eta_s=eta_s,
            slurm_state=slurm_state,
            exit_code=exit_code,
        ).sign(self.secret)

    async def _ack(self, message: RcpMessage, *, cmd_id: object, error: str) -> RcpMessage:
        ack = self._build_ack(
            message,
            cmd_id=str(cmd_id or ""),
            result=HelloResult(job_id=None, cluster_output=None, error=error),
        )
        await self.transport.send(ack)
        return ack

    def _build_ack(self, message: RcpMessage, *, cmd_id: str, result: HelloResult) -> RcpMessage:
        return build_hello_ack(
            sim=self.sim,
            seq=self.sequences.next_seq(self.sim, SenderRole.SIMCLIENT),
            cmd_id=cmd_id,
            in_reply_to=message.transport_event_id,
            job_id=result.job_id,
            cluster_output=result.cluster_output,
            error=result.error,
        ).sign(self.secret)

    def _outfile_path(self, cmd_id: str) -> Path:
        outdir = self.message_dir / "out"
        outdir.mkdir(parents=True, exist_ok=True)
        return outdir / f"hello-{cmd_id}.out"

    async def serve(self) -> None:
        """Consume inbound messages until the transport closes.

        Raises:
            asyncio.CancelledError: If the serving task is cancelled.

        """
        self._serving = True
        try:
            async for message in self.transport.receive():
                try:
                    await self.handle(message)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    log.exception("error handling inbound RCP message")
        finally:
            self._serving = False
            await self._cancel_followers()
