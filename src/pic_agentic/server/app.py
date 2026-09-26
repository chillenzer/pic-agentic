# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""MCP stdio server exposing the M1 ``hello`` tool (design sections 4, 8.1)."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server import MCPServer
from mcp.types import ToolAnnotations

from pic_agentic.auth import MasTokenStore
from pic_agentic.protocol.simulation import LOG_STREAMS, PayloadTooLargeError, SubmitParams, UnsupportedPayloadError
from pic_agentic.server.hello import AckTimeoutError, HelloOutcome, HelloService
from pic_agentic.server.simulation import (
    SimRecord,
    SubmitOutcome,
    SubmitService,
    condense_events,
    resolve_script,
)
from pic_agentic.simclient.safety import UnsafePathError
from pic_agentic.simulation_build import SimulationBuildError
from pic_agentic.transport.matrix import MatrixTransport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pic_agentic.config import Config
    from pic_agentic.rcp.envelope import RcpMessage

log = logging.getLogger(__name__)

#: The canonical RCP timestamp format.
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: Errors the ``submit_simulation`` tool turns into a soft ``{"ok": false}``
#: result rather than letting them escape as an unhandled tool exception.
#: Includes the protocol ``ValueError``s raised before/while sending
#: (``PayloadTooLargeError``, ``UnsupportedPayloadError``) and pydantic's
#: ``ValidationError`` for bad params; no ``ack`` reaches the LLM otherwise.
_SUBMIT_TOOL_ERRORS: tuple[type[BaseException], ...] = (
    AckTimeoutError,
    SimulationBuildError,
    UnsafePathError,
    PayloadTooLargeError,
    UnsupportedPayloadError,
    OSError,
    ValueError,
)

#: Read-tier annotations shared by the M2b reporting tools.
_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True)

#: Cap on the number of log lines a single ``get_logs`` call may return.
_MAX_LOG_TAIL = 10_000

#: Recursion guard for :func:`_redact_dict`.
_REDACT_MAX_DEPTH = 8


class HelloRuntime:
    """Owns the Matrix transport and the async RCP service."""

    def __init__(self, config: Config, sim: str) -> None:
        """Create the runtime (the transport starts in :meth:`start`).

        Args:
            config: Resolved configuration.
            sim: Simulation id the server operates under.

        """
        self.config = config
        self.sim = sim
        message_dir = Path(config.message_dir)
        self.service = HelloService(
            sim=sim,
            secret=config.rcp_secret,
            message_dir=message_dir,
            ack_timeout_s=config.ack_timeout_s,
        )
        self.submit_service = SubmitService(
            sim=sim,
            secret=config.rcp_secret,
            picongpu_python=config.picongpu_python,
            picongpu_revision=config.picongpu_revision,
            ack_timeout_s=config.ack_timeout_s,
        )
        self._transport: MatrixTransport | None = None
        self._pump: asyncio.Task | None = None

    async def start(self) -> None:
        """Open the Matrix transport and start pumping inbound messages."""
        config = self.config
        config.require("homeserver", "user_id", "access_token", "room_id", "rcp_secret", "message_dir")
        token_provider = None
        if config.has_refresh_chain():
            store = MasTokenStore.from_config(config)
            token_provider = store.access_token
        self._transport = MatrixTransport(
            config.homeserver,
            config.user_id,
            config.access_token,
            config.room_id,
            store_path=config.nio_store_dir or None,
            token_provider=token_provider,
        )
        # Drain anything already in the room before we start awaiting.  The
        # hello service only needs acks matching a live exchange; the submit
        # service rebuilds its registry from the signed-room replay.
        backfilled = await self._transport.backfill()
        for message in backfilled:
            self.service.on_message(message)
        self.submit_service.ingest_backfill(backfilled)
        self._pump = asyncio.create_task(self._pump_forever())

    def _route(self, message: RcpMessage) -> None:
        # Both services filter by envelope kind/type/sim/signature, so feeding
        # every message to both is safe and keeps the routing trivial.
        self.service.on_message(message)
        self.submit_service.on_message(message)

    async def _pump_forever(self) -> None:
        if self._transport is None:
            msg = "runtime is not started"
            raise RuntimeError(msg)
        async for message in self._transport.receive():
            self._route(message)

    async def hello(self, message: str) -> HelloOutcome:
        """Run one ``hello`` exchange.

        Args:
            message: The LLM-supplied message text.

        Returns:
            The outcome of the exchange.

        Raises:
            RuntimeError: If the runtime has not been started.

        """
        if self._transport is None:
            msg = "runtime is not started"
            raise RuntimeError(msg)
        return await self.service.hello(self._transport.send, message)

    async def submit(
        self,
        picmi_script: str,
        *,
        params: SubmitParams | None = None,
    ) -> SubmitOutcome:
        """Run one ``submit_simulation`` exchange.

        Args:
            picmi_script: A path to a PICMI script or inline PICMI code.
            params: Optional build/run flags.

        Returns:
            The outcome; ``state`` is the simclient's first ack state.

        Raises:
            RuntimeError: If the runtime has not been started.

        """
        if self._transport is None:
            msg = "runtime is not started"
            raise RuntimeError(msg)
        script_path = resolve_script(picmi_script, workdir=Path(tempfile.gettempdir()) / "pic-agentic")
        return await self.submit_service.submit(self._transport.send, script_path, params=params)

    def registry(self) -> dict[str, SimRecord]:
        """Return the submit service's sim_id-keyed registry.

        Returns:
            The registry mapping (mutated in place by the service).

        """
        return self.submit_service.registry

    def get_sim(self, sim_id: str) -> SimRecord | None:
        """Return the registry record for ``sim_id``, if known.

        Returns:
            The record, or None.

        """
        return self.submit_service.get(sim_id)

    def list_sims(self, *, active_only: bool = False) -> list[SimRecord]:
        """Return the registry records.

        Returns:
            The selected records.

        """
        return self.submit_service.list(active_only=active_only)

    async def fetch_status(self, sim_id: str) -> dict[str, Any]:
        """Run a live-status pull, if the transport is started.

        Returns:
            The ``status_ack`` payload, or an empty dict when not started.

        """
        if self._transport is None:
            return {}
        return await self.submit_service.fetch_status(self._transport.send, sim_id)

    async def fetch_logs(self, sim_id: str, *, stream: str = "stdout", tail: int = 100) -> dict[str, Any]:
        """Run a log pull, if the transport is started.

        Returns:
            The ``logs_ack`` payload, or an empty dict when not started.

        """
        if self._transport is None:
            return {}
        return await self.submit_service.fetch_logs(self._transport.send, sim_id, stream=stream, tail=tail)

    def condensed_events(
        self,
        sim_id: str,
        *,
        since: str | None = None,
        types: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Return the condensed event list for ``sim_id``.

        Returns:
            The condensed payload dicts, oldest first.

        """
        return condense_events(self.submit_service.event_log, sim_id=sim_id, since=since, types=types, limit=limit)

    async def stop(self) -> None:
        """Cancel the pump and close the transport."""
        if self._pump:
            self._pump.cancel()
        if self._transport:
            await self._transport.close()


def build_server(config: Config, sim: str) -> tuple[MCPServer, HelloRuntime]:
    """Create the MCP server and its runtime, wired together via lifespan.

    Returns:
        The ``(server, runtime)`` pair.

    """
    runtime = HelloRuntime(config, sim)

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[None]:
        await runtime.start()
        try:
            yield
        finally:
            await runtime.stop()

    server = MCPServer(
        "pic-agentic",
        instructions=(
            "Submit and follow PIConGPU simulations on a remote SLURM cluster. "
            "The 'hello' tool performs an end-to-end connectivity check."
        ),
        lifespan=lifespan,
    )

    @server.tool(
        title="Hello World connectivity check",
        description=(
            "Send a short message through the Matrix control channel to the "
            "simulation-side client, which runs a trivial SLURM job that "
            "prints it back. Returns the SLURM job id and captured output."
        ),
        # write/resource tier: consumes cluster resources, not destructive
        # (design section 6.2).  Not read-only, not idempotent (each call
        # submits a fresh job).
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
    )
    async def hello(message: str = "Hello World") -> dict[str, Any]:
        outcome = await runtime.hello(message)
        return _outcome_dict(runtime, outcome)

    @server.tool(
        title="Submit a PIConGPU simulation",
        description=(
            "Build a PICMI simulation script into a PyPIConGPU runner, send it "
            "through the Matrix control channel to the simulation-side client "
            "and submit it to the remote SLURM cluster. Returns the simulation "
            "id and the coarse accepted/submitted state."
        ),
        # write/resource tier: consumes cluster resources, not destructive
        # (design section 6.2).  The server-side MCP client prompts for human
        # confirmation; each call is a fresh submission.
        annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False),
    )
    async def submit_simulation(
        picmi_script: str,
        *,
        build_jobs: int | None = None,
        build_cmake: str | None = None,
        build_preset: int | None = None,
        build_force: bool = False,
        cfg_file: str | None = None,
        overwrite_vars: list[str] | None = None,
    ) -> dict[str, Any]:
        try:
            params = SubmitParams(
                build_jobs=build_jobs,
                build_cmake=build_cmake,
                build_preset=build_preset,
                build_force=build_force,
                cfg_file=cfg_file,
                overwrite_vars=overwrite_vars,
            )
            outcome = await runtime.submit(picmi_script, params=params)
        except _SUBMIT_TOOL_ERRORS as exc:
            return {"ok": False, "state": "error", "error": runtime.config.redact(str(exc))}
        return _submit_outcome_dict(runtime, outcome)

    _register_reporting_tools(server, runtime)
    return server, runtime


def _register_reporting_tools(server: MCPServer, runtime: HelloRuntime) -> None:
    """Register the M2b read-tier reporting tools on ``server``.

    Args:
        server: The MCP server to add the tools to.
        runtime: The runtime the tools delegate to.

    """

    @server.tool(
        title="Get simulation status",
        description=(
            "Report the lifecycle state of one simulation. For a known, "
            "non-terminal simulation a live scontrol view is fetched from the "
            "cluster and merged over the last-event projection; otherwise the "
            "signed-room projection is returned."
        ),
        annotations=_READ_ONLY,
    )
    async def get_status(sim_id: str) -> dict[str, Any]:
        record = runtime.get_sim(sim_id)
        if record is None:
            return {"sim_id": sim_id, "known": False, "error": "unknown_sim"}
        projection = _status_dict(record)
        if record.active:
            live = await runtime.fetch_status(sim_id)
            if live and not live.get("error"):
                _merge_status(projection, live)
        return _redact_dict(runtime, projection)

    @server.tool(
        title="List simulations",
        description="List the simulations the server knows about, optionally only the still-active ones.",
        annotations=_READ_ONLY,
    )
    def list_simulations(*, active_only: bool = False) -> dict[str, Any]:
        rows = [
            {
                "sim_id": record.sim_id,
                "cmd_id": record.cmd_id,
                "state": record.state,
                "job_id": record.job_id,
                "last_event_type": record.last_event_type,
                "last_event_ts": record.last_event_ts,
                "active": record.active,
            }
            for record in runtime.list_sims(active_only=active_only)
        ]
        return _redact_dict(runtime, {"simulations": rows})

    @server.tool(
        title="Get simulation events",
        description=(
            "Return the condensed lifecycle-event history of one simulation "
            "(consecutive duplicate states collapse). Optionally filter by an "
            "ISO timestamp lower bound and by state type."
        ),
        annotations=_READ_ONLY,
    )
    def get_events(
        sim_id: str,
        *,
        since: str | None = None,
        types: list[str] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        events = runtime.condensed_events(sim_id, since=since, types=types, limit=limit)
        return _redact_dict(runtime, {"sim_id": sim_id, "events": events, "count": len(events)})

    @server.tool(
        title="Get simulation logs",
        description="Return up to `tail` lines of a simulation's stdout, stderr or workflow log stream.",
        annotations=_READ_ONLY,
    )
    async def get_logs(sim_id: str, *, stream: str = "stdout", tail: int = 100) -> dict[str, Any]:
        if stream not in LOG_STREAMS:
            return {"sim_id": sim_id, "stream": stream, "error": "unknown_stream"}
        tail = max(0, min(tail, _MAX_LOG_TAIL))
        payload = await runtime.fetch_logs(sim_id, stream=stream, tail=tail)
        if not payload:
            return {"sim_id": sim_id, "stream": stream, "lines": [], "total_lines": 0, "error": "unavailable"}
        return _redact_dict(runtime, payload)


def _status_dict(record: SimRecord) -> dict[str, Any]:
    """Build the registry projection of one simulation's status.

    Returns:
        The fields shared by the live and projected status responses.

    """
    return {
        "sim_id": record.sim_id,
        "state": record.state,
        "slurm_state": record.slurm_state,
        "job_id": record.job_id,
        "step": record.step,
        "percent": record.percent,
        "walltime": record.walltime,
        "eta_s": record.eta_s,
        "since_last_event_s": _since_last_event_s(record.last_event_ts),
    }


def _merge_status(projection: dict[str, Any], live: dict[str, Any]) -> None:
    """Overlay the live ``status_ack`` fields on the registry projection.

    Only fields the ack actually carries override the projection, so a live view
    missing a field does not erase a value known from the event log.

    Args:
        projection: The registry projection, updated in place.
        live: The live ``status_ack`` payload (without ids).

    """
    for field in (
        "state",
        "slurm_state",
        "job_id",
        "step",
        "percent",
        "walltime",
        "avg_per_step",
        "eta_s",
        "exit_code",
    ):
        if live.get(field) is not None:
            projection[field] = live[field]


def _since_last_event_s(ts: str | None) -> int | None:
    """Seconds elapsed since ``ts``, or None when unparsable.

    Returns:
        The whole seconds since the timestamp, or None.

    """
    if not ts:
        return None
    try:
        when = datetime.strptime(ts, _TS_FORMAT).replace(tzinfo=UTC)
    except ValueError:
        return None
    return max(0, int((datetime.now(UTC) - when).total_seconds()))


def _redact_dict(runtime: HelloRuntime, payload: Any, _depth: int = 0) -> Any:
    """Redact every string in an outbound payload.

    Args:
        runtime: The runtime whose config holds the secrets.
        payload: The value about to leave toward the LLM.
        _depth: Recursion guard for nested containers.

    Returns:
        The payload with all strings redacted.

    """
    if _depth > _REDACT_MAX_DEPTH:
        return payload
    redact = runtime.config.redact
    if isinstance(payload, str):
        return redact(payload)
    if isinstance(payload, dict):
        return {key: _redact_dict(runtime, value, _depth + 1) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_redact_dict(runtime, value, _depth + 1) for value in payload]
    return payload


def _submit_outcome_dict(runtime: HelloRuntime, outcome: SubmitOutcome) -> dict[str, Any]:
    redact = runtime.config.redact
    payload: dict[str, Any] = {
        "ok": outcome.ok,
        "sim": outcome.sim,
        "sim_id": outcome.sim_id,
        "state": outcome.state,
        "job_id": outcome.job_id,
        "acked": outcome.acked,
    }
    if outcome.error:
        payload["error"] = redact(outcome.error)
    if outcome.error_code:
        payload["error_code"] = outcome.error_code
    return payload


def _outcome_dict(runtime: HelloRuntime, outcome: HelloOutcome) -> dict[str, Any]:
    redact = runtime.config.redact
    payload: dict[str, Any] = {
        "ok": outcome.ok,
        "sim": outcome.sim,
        "job_id": outcome.job_id,
        "acked": outcome.acked,
        "cluster_output": redact(outcome.cluster_output) if outcome.cluster_output else None,
    }
    if outcome.error:
        payload["error"] = redact(outcome.error)
    return payload
