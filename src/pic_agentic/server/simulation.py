# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""MCP-side ``submit_simulation`` orchestration (design sections 4.1, 8.2).

The service turns an LLM-supplied PICMI script into a ``Runner`` dump in a
disposable subprocess (the server never imports the script), wraps it in a
:class:`~pic_agentic.protocol.simulation.SimulationPayload`, embeds it in the
signed command and sends that.  It waits for the simclient's immediate
``accepted`` ack, so the LLM learns the ``sim_id`` right away; the later
lifecycle events (``simulation.submitted``/``workflow.finished``/
``simulation.failed``) are recorded as they arrive for the M2 reporting tools.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field

from pic_agentic.protocol.simulation import (
    LOG_STREAMS,
    SimulationPayload,
    SimulationState,
    SimulationType,
    SubmitParams,
    build_logs_command,
    build_status_command,
    build_submit_command,
)
from pic_agentic.rcp import Kind, RcpMessage, SenderRole, SequenceState, new_cmd_id
from pic_agentic.server.hello import AckTimeoutError, SendFn
from pic_agentic.simulation_build import BuiltSimulation, SimulationBuildError, build_runner_dump

log = logging.getLogger(__name__)

#: Signature of the injectable runner-dump builder (test seam).
RunnerDumpBuilder = Callable[..., Awaitable[BuiltSimulation]]

#: Registry states after which no further lifecycle event is expected.
TERMINAL_STATES = frozenset(
    {
        SimulationState.RESULTS_READY.value,
        SimulationState.FAILED.value,
        SimulationState.JOB_FAILED.value,
    },
)

#: Cap on the retained event log (the registry is projected from it and, on a
#: fresh start, from the signed-room backfill).
DEFAULT_EVENT_LOG_MAX = 1000

#: Cap on rows returned by ``get_events``.
MAX_EVENT_PAGE = 200

#: The ack type each request/response pull expects, so ``on_message`` can match
#: a pending pull by kind and not only by the (possibly colliding) cmd_id.
_PULL_ACK_FOR_REQUEST: dict[SimulationType, SimulationType] = {
    SimulationType.STATUS_COMMAND: SimulationType.STATUS_ACK,
    SimulationType.LOGS_COMMAND: SimulationType.LOGS_ACK,
}

#: Fields projected from an event/ack payload into a :class:`SimRecord` when the
#: payload actually carries a non-None value (a later event omitting the field
#: must not erase an earlier known one, e.g. the job id).
_RECORD_FIELDS = (
    "job_id",
    "slurm_state",
    "step",
    "percent",
    "walltime",
    "avg_per_step",
    "eta_s",
    "exit_code",
    "run_dir",
)


class SimRecord(BaseModel):
    """The server's projection of one simulation's lifecycle.

    The state lives in the signed room: this record is rebuilt by replaying the
    ``rcp.simulation_event`` log (and the submit acks) on server start, so it
    survives MCP-server restarts without a separate database.
    """

    model_config = ConfigDict(extra="forbid")

    sim_id: str
    cmd_id: str
    job_id: int | None = None
    state: str = ""
    slurm_state: str | None = None
    step: int | None = None
    percent: int | None = None
    walltime: str | None = None
    avg_per_step: str | None = None
    eta_s: int | None = None
    exit_code: int | None = None
    run_dir: str | None = None
    last_event_type: str | None = None
    last_event_ts: str | None = None
    active: bool = True


def condense_events(
    event_log: Iterable[RcpMessage],
    *,
    sim_id: str,
    since: str | None = None,
    types: Iterable[str] | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Condense the event log into a bounded, deduplicated event list.

    Consecutive events carrying the same ``state`` collapse to a single entry
    holding the most recent payload (the bounded progress cadence therefore
    yields one ``simulation.step_finished`` row, while the full stream stays
    available via ``get_logs``).

    Args:
        event_log: The retained event log (oldest first).
        sim_id: Only events for this simulation are considered.
        since: Optional ISO-8601 lower bound; events with ``ts < since`` are
            dropped (the canonical ``Z`` timestamps sort lexicographically).
        types: Optional set of ``state`` values to keep.
        limit: Maximum number of condensed rows (capped at
            :data:`MAX_EVENT_PAGE`).

    Returns:
        The condensed payload dicts, each augmented with its envelope ``ts``,
        oldest first.

    """
    wanted = set(types) if types else None
    capped = max(0, min(limit, MAX_EVENT_PAGE))
    if capped == 0:
        return []
    condensed: list[dict[str, Any]] = []
    for message in event_log:
        if message.kind is not Kind.EVENT or message.type != SimulationType.EVENT:
            continue
        payload = message.payload
        if str(payload.get("sim_id", "")) != sim_id:
            continue
        ts = message.ts
        if since is not None and ts < since:
            continue
        state = str(payload.get("state", ""))
        if wanted is not None and state not in wanted:
            continue
        entry: dict[str, Any] = {"ts": ts, **payload}
        if condensed and condensed[-1].get("state") == state:
            condensed[-1] = entry
        else:
            condensed.append(entry)
    return condensed[-capped:]


class SubmitOutcome(BaseModel):
    """The MCP-side result of one ``submit_simulation`` exchange."""

    model_config = ConfigDict(extra="forbid")

    sim: str
    cmd_id: str
    sim_id: str
    state: str
    job_id: int | None = None
    acked: bool = False
    error: str | None = None
    error_code: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def ok(self) -> bool:
        """Whether the command was accepted without an error."""
        return not self.error


class SubmitService:
    """Turn PICMI scripts into commands and await their acks."""

    def __init__(
        self,
        sim: str,
        secret: str,
        *,
        picongpu_python: str = "",
        picongpu_revision: str = "",
        ack_timeout_s: float = 90.0,
        runner_dump_builder: RunnerDumpBuilder = build_runner_dump,
        event_log_max: int = DEFAULT_EVENT_LOG_MAX,
    ) -> None:
        """Create a service for one simulation.

        Args:
            sim: Simulation id.
            secret: Shared per-simulation RCP secret.
            picongpu_python: Interpreter with the pinned PIConGPU install.
            picongpu_revision: Pinned revision carried in the payload header.
            ack_timeout_s: Maximum wait for an ack (submit and pull).
            runner_dump_builder: Subprocess runner-dump builder (test seam).
            event_log_max: Maximum retained lifecycle events.

        """
        self.sim = sim
        self.secret = secret
        self.picongpu_python = picongpu_python
        self.picongpu_revision = picongpu_revision
        self.ack_timeout_s = ack_timeout_s
        self.runner_dump_builder = runner_dump_builder
        self.event_log_max = event_log_max
        self.sequences = SequenceState()
        self._pending: dict[str, asyncio.Future[RcpMessage]] = {}
        #: Pending status/logs pulls, keyed by cmd_id, and the ack type each
        #: one expects (so a mis-routed ack cannot resolve the wrong future).
        self._pending_pull: dict[str, asyncio.Future[RcpMessage]] = {}
        self._pending_pull_kind: dict[str, SimulationType] = {}
        #: Ordered retained event log (all sims), capped *per sim_id* so one
        #: busy simulation cannot evict another's history; its sim_id-keyed
        #: projection is :attr:`registry`.  This is the single store
        #: ``get_events``/``condense_events`` read from.
        self.event_log: list[RcpMessage] = []
        self.registry: dict[str, SimRecord] = {}

    async def build_payload(
        self,
        script_path: Path,
        *,
        params: SubmitParams | None = None,
        cmd_id: str | None = None,
    ) -> tuple[str, SimulationPayload, RcpMessage]:
        """Build the payload and embed it in the signed command.

        Args:
            script_path: Path to the PICMI script (already resolved).
            params: Optional build/run flags.
            cmd_id: Optional command id (generated when omitted).

        Returns:
            The ``(cmd_id, payload, command)`` triple, the command signed.

        """
        command_id = cmd_id or new_cmd_id()
        built = await self.runner_dump_builder(script_path=script_path, interpreter=self.picongpu_python)
        # Provenance comes from the *child* that produced the dump, not from the
        # server process: the server may run a different interpreter (and, with
        # PIC_AGENTIC_PICONGPU_PYTHON, may not have PIConGPU at all).
        payload = SimulationPayload.build(
            picongpu_version=built.picongpu_version,
            picongpu_revision=self.picongpu_revision or built.picongpu_revision,
            schema_hash=built.schema_hash,
            runner_dump=built.runner,
        )
        payload.check_allowlist()
        seq = self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER)
        command = build_submit_command(
            sim=self.sim,
            seq=seq,
            payload=payload,
            params=params,
            cmd_id=command_id,
        ).sign(self.secret)
        return command_id, payload, command

    def on_message(self, message: RcpMessage) -> None:
        """Feed an inbound message; resolve pending futures and project events.

        Args:
            message: An inbound RCP message.

        """
        if not message.verify(self.secret) or message.sim != self.sim:
            return
        if message.sender_role is not SenderRole.SIMCLIENT:
            return
        cmd_id = str(message.payload.get("cmd_id", ""))
        if message.kind is Kind.EVENT and message.type == SimulationType.EVENT:
            self._append_event_log(message)
            self._project_event(message)
            return
        if message.kind is Kind.ACK and message.type in {SimulationType.STATUS_ACK, SimulationType.LOGS_ACK}:
            # Match the ack's *kind* to the pending request, not just its
            # cmd_id: a mis-routed ``logs_ack`` carrying a status request's
            # cmd_id must not resolve the status future.
            expected = _PULL_ACK_FOR_REQUEST.get(self._pending_pull_kind.get(cmd_id))
            if expected is not None and message.type == expected:
                future = self._pending_pull.get(cmd_id)
                if future is not None and not future.done():
                    future.set_result(message)
            else:
                log.warning(
                    "ignoring %s for pending %s request %s",
                    message.type,
                    self._pending_pull_kind.get(cmd_id),
                    cmd_id,
                )
            return
        if message.kind is Kind.ACK and message.type == SimulationType.ACK:
            self._project_ack(message)
            future = self._pending.get(cmd_id)
            if future is not None and not future.done():
                future.set_result(message)

    def ingest_backfill(self, messages: Iterable[RcpMessage]) -> None:
        """Rebuild the registry from a replayed (signed-room) message stream.

        Replaying the same events must converge to the same registry:
        :meth:`_project_event` and :meth:`_project_ack` overwrite record fields
        only when the payload carries them, and the event log is append-only, so
        a replay is idempotent in effect.

        Args:
            messages: The messages to replay, oldest first.

        """
        for message in messages:
            self.on_message(message)

    def _append_event_log(self, message: RcpMessage) -> None:
        """Append one event to the bounded, ordered event log.

        The cap is applied *per sim_id*: at most :attr:`event_log_max` events
        per simulation are retained, so a busy simulation cannot evict another
        simulation's early lifecycle history.

        Args:
            message: The event to retain.

        """
        self.event_log.append(message)
        sim_id = str(message.payload.get("sim_id", ""))
        matching = [
            index for index, entry in enumerate(self.event_log) if str(entry.payload.get("sim_id", "")) == sim_id
        ]
        overflow = len(matching) - self.event_log_max
        if overflow > 0:
            for index in reversed(matching[:overflow]):
                del self.event_log[index]

    def _project_event(self, message: RcpMessage) -> None:
        """Project one lifecycle event into the sim_id-keyed registry.

        Args:
            message: A verified ``rcp.simulation_event`` from the simclient.

        """
        payload = message.payload
        sim_id = str(payload.get("sim_id", ""))
        if not sim_id:
            return
        state = str(payload.get("state", ""))
        cmd_id = str(payload.get("cmd_id", ""))
        record = self._record_for(sim_id, cmd_id=cmd_id, ts=message.ts)
        # A replayed/old-run event (its cmd_id predates the latest run) must not
        # touch the current record.
        if cmd_id and cmd_id != record.cmd_id:
            return
        # Terminal is monotonic within a run: a replayed or out-of-order
        # non-terminal event (e.g. a late ``step_finished``) must never flip a
        # finished record back to active.  A genuinely new run under the same
        # sim_id gets a fresh (non-terminal) record from :meth:`_record_for`.
        if record.state in TERMINAL_STATES:
            return
        for field in _RECORD_FIELDS:
            value = payload.get(field)
            if value is not None:
                setattr(record, field, value)
        record.state = state
        record.last_event_type = state or record.last_event_type
        record.last_event_ts = message.ts
        record.active = state not in TERMINAL_STATES
        self.registry[sim_id] = record

    def _project_ack(self, message: RcpMessage) -> None:
        """Register the simulation named by a submit ack, before its first event.

        Args:
            message: A verified submit ack from the simclient.

        """
        payload = message.payload
        sim_id = str(payload.get("sim_id", ""))
        if not sim_id:
            return
        cmd_id = str(payload.get("cmd_id", ""))
        record = self._record_for(sim_id, cmd_id=cmd_id, ts=message.ts)
        # A replayed ack from an older run must not touch the latest record.
        if cmd_id and cmd_id != record.cmd_id:
            return
        # The ack seeds a fresh record only: a late or re-delivered ack must
        # never regress a state already projected from a later event.
        state = str(payload.get("state", ""))
        if state and not record.state:
            record.state = state
            record.last_event_type = state
            record.active = state not in TERMINAL_STATES
        if payload.get("job_id") is not None:
            record.job_id = payload["job_id"]
        if record.last_event_ts is None:
            record.last_event_ts = message.ts
        self.registry[sim_id] = record

    def _record_for(self, sim_id: str, *, cmd_id: str, ts: str | None = None) -> SimRecord:
        """Return the record for the simulation's latest run.

        A resubmission of an identical simulation yields the same ``sim_id``
        but a fresh ``cmd_id``.  Such a message starts a *new run*: the record
        is reset to the new run, rather than reporting the first run's
        ``cmd_id`` alongside the second run's ``state``/``job_id``.  The
        registry therefore keeps one (latest-run) record per ``sim_id`` and the
        event log separates runs by ``cmd_id``.

        A message from an *older* run (a backfill replay of run 1 after run 2
        has started) must not create or switch records: a new ``cmd_id`` only
        starts a run when its timestamp is not older than the record's.

        Args:
            sim_id: The simulation id.
            cmd_id: The command id naming this run.
            ts: The message timestamp (for the old-run guard), if known.

        Returns:
            The mutable record (also stored in :attr:`registry`).

        """
        record = self.registry.get(sim_id)
        if record is None:
            record = SimRecord(sim_id=sim_id, cmd_id=cmd_id)
            self.registry[sim_id] = record
        elif (
            cmd_id
            and cmd_id != record.cmd_id
            and (ts is None or record.last_event_ts is None or ts >= record.last_event_ts)
        ):
            # New run under the same sim_id: reset the run-scoped projection
            # (keep the sim_id) so it reflects the latest run only.
            record = SimRecord(sim_id=sim_id, cmd_id=cmd_id)
            self.registry[sim_id] = record
        return record

    def get(self, sim_id: str) -> SimRecord | None:
        """Return the registry record for ``sim_id``, if known.

        Returns:
            The record, or None.

        """
        return self.registry.get(sim_id)

    def list(self, *, active_only: bool = False) -> list[SimRecord]:
        """Return the registry records in registration order.

        Args:
            active_only: When True, only simulations still in a non-terminal
                state are returned.

        Returns:
            The selected records.

        """
        records = list(self.registry.values())
        if active_only:
            return [record for record in records if record.active]
        return records

    def _outcome_from_ack(self, cmd_id: str, ack: RcpMessage) -> SubmitOutcome:
        return SubmitOutcome(
            sim=self.sim,
            cmd_id=cmd_id,
            sim_id=str(ack.payload.get("sim_id", "")),
            state=str(ack.payload.get("state", "")),
            job_id=ack.payload.get("job_id"),
            acked=True,
            error=ack.payload.get("error"),
            error_code=ack.payload.get("error_code"),
        )

    async def submit(
        self,
        send: SendFn,
        script_path: Path,
        *,
        params: SubmitParams | None = None,
    ) -> SubmitOutcome:
        """Build, send and await one ``submit_simulation`` command.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            script_path: Path to the PICMI script.
            params: Optional build/run flags.

        Returns:
            The outcome; ``state`` is the simclient's first ack state (normally
            ``accepted``).

        Raises:
            AckTimeoutError: If no ack arrives within the configured wait.

        """
        cmd_id, _payload, command = await self.build_payload(script_path, params=params)
        future: asyncio.Future[RcpMessage] = asyncio.get_running_loop().create_future()
        self._pending[cmd_id] = future
        try:
            await send(command)
            try:
                ack = await asyncio.wait_for(future, timeout=self.ack_timeout_s)
            except TimeoutError:
                msg = f"no ack for submit command {cmd_id} within {self.ack_timeout_s}s"
                raise AckTimeoutError(msg) from None
        finally:
            self._pending.pop(cmd_id, None)
        return self._outcome_from_ack(cmd_id, ack)

    async def fetch_status(self, send: SendFn, sim_id: str) -> dict[str, Any]:
        """Send a live-status request and await its ack.

        A timeout is returned as data (``{"error": "timeout"}``), never raised:
        the caller can fall back to its registry projection.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            sim_id: The simulation to query.

        Returns:
            The ``status_ack`` payload, or ``{"sim_id", "error"}`` on timeout.

        """
        return await self._fetch(
            send,
            sim_id,
            lambda cmd_id: build_status_command(
                sim=self.sim,
                seq=self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER),
                sim_id=sim_id,
                cmd_id=cmd_id,
            ),
        )

    async def fetch_logs(
        self,
        send: SendFn,
        sim_id: str,
        *,
        stream: str = "stdout",
        tail: int = 100,
    ) -> dict[str, Any]:
        """Send a log request and await its ack.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            sim_id: The simulation to query.
            stream: One of :data:`~pic_agentic.protocol.simulation.LOG_STREAMS`.
            tail: Maximum number of trailing lines.

        Returns:
            The ``logs_ack`` payload, or ``{"sim_id", "stream", "error"}`` on
            timeout.

        Raises:
            ValueError: If ``stream`` is not a known log stream.

        """
        if stream not in LOG_STREAMS:
            msg = f"unknown log stream {stream!r}; expected one of {LOG_STREAMS}"
            raise ValueError(msg)
        return await self._fetch(
            send,
            sim_id,
            lambda cmd_id: build_logs_command(
                sim=self.sim,
                seq=self.sequences.next_seq(self.sim, SenderRole.MCP_SERVER),
                sim_id=sim_id,
                stream=stream,
                tail=tail,
                cmd_id=cmd_id,
            ),
        )

    async def _fetch(self, send: SendFn, sim_id: str, build: Callable[[str], RcpMessage]) -> dict[str, Any]:
        """Sign, send and await one request/response pull.

        Args:
            send: Async callable ``send(RcpMessage) -> event_id``.
            sim_id: The simulation the request names (for the timeout payload).
            build: Builder taking a fresh ``cmd_id`` and returning the command.

        Returns:
            The ack payload, or an error dict on timeout.

        """
        cmd_id = new_cmd_id()
        command = build(cmd_id).sign(self.secret)
        future: asyncio.Future[RcpMessage] = asyncio.get_running_loop().create_future()
        self._pending_pull[cmd_id] = future
        self._pending_pull_kind[cmd_id] = command.type
        try:
            await send(command)
            try:
                ack = await asyncio.wait_for(future, timeout=self.ack_timeout_s)
            except TimeoutError:
                return {"sim_id": sim_id, "error": "timeout"}
        finally:
            self._pending_pull.pop(cmd_id, None)
            self._pending_pull_kind.pop(cmd_id, None)
        return dict(ack.payload)


def resolve_script(picmi_script: str, *, workdir: Path) -> Path:
    """Resolve the tool's ``picmi_script`` argument to a file path.

    Args:
        picmi_script: An existing file path, or inline PICMI code.
        workdir: Directory for inline code (a temp dir on the server).

    Returns:
        The path to the PICMI script.

    """
    candidate = Path(picmi_script).expanduser()
    if "\n" not in picmi_script and candidate.is_file():
        return candidate
    workdir.mkdir(parents=True, exist_ok=True)
    # A unique name so concurrent submissions never overwrite each other.
    fd, name = tempfile.mkstemp(prefix="picmi_script-", suffix=".py", dir=str(workdir))
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(picmi_script)
    return Path(name)


__all__ = [
    "DEFAULT_EVENT_LOG_MAX",
    "MAX_EVENT_PAGE",
    "TERMINAL_STATES",
    "AckTimeoutError",
    "SimRecord",
    "SimulationBuildError",
    "SubmitOutcome",
    "SubmitService",
    "condense_events",
    "resolve_script",
]
