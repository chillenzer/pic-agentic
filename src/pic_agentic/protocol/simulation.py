# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""M2 ``submit_simulation`` RCP messages (design sections 4.1, 8.2).

Wire format: the signed command carries a ``pypicongpu.Runner`` *spec* inline
-- only the ``sim`` field -- and never the runner's cluster-local directories.
The simclient validates the provenance tuple and the payload hash, then rebuilds
a fresh ``Runner`` with cluster-local ``setup_dir``/``run_dir``/``template_dir``
(design section 6.4, gap 3).

The payload travels *inside* the Matrix ``m.room.message`` (not as a shared-FS
file), so the MCP server and the simclient need no common file system: the
container/cluster split of the deployment is preserved.  The cost is that the
command is bounded by the homeserver's event-size limit, hence
:data:`MAX_INLINE_PAYLOAD_BYTES` (a realistic simulation is a few KiB; the
largest stress case measured ~53 KiB).

The payload itself contains no ``rc_params``: those are cluster-local (the
``picongpurc.toml``) and some of their fields are shell code (design section
4.1, milestone note).

The module stays import-safe without PIConGPU; importing/validating the runner
is the simclient's job (the ``sim`` extra).
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, computed_field, field_validator

from pic_agentic.rcp import Kind, RcpMessage, SenderRole, canonical_bytes, new_cmd_id
from pic_agentic.version import WIRE_FORMAT_VERSION

#: The only top-level key a transmitted runner spec may carry.
ALLOWED_SIMULATION_KEYS = frozenset({"sim"})

#: Default submit command; a payload asking for anything else is rejected.
DEFAULT_SUBMIT_SYSTEM = "sbatch"

#: Cap on the *encoded* event content carried in one Matrix command.  Synapse's
#: default limit is 64 KiB for the whole event content, so 48 KiB leaves headroom
#: for the envelope, the human-readable body and the params.  The size check
#: counts the **escaped** payload (it is embedded as a JSON string, so its
#: quotes/backslashes are doubled) plus :data:`_ENVELOPE_ALLOWANCE_BYTES`, i.e.
#: what actually goes on the wire -- not the inner simulation object.
MAX_INLINE_PAYLOAD_BYTES = 48 * 1024

#: Reserved budget for the signed envelope, the room body line, the copy of the
#: provenance header and the params that travel alongside the payload in the
#: same event content.
_ENVELOPE_ALLOWANCE_BYTES = 4 * 1024

#: ``cfg_file``: a relative path to a ``.cfg`` inside the generated setup.
#: Absolute paths, ``..`` and shell metacharacters are rejected outright so the
#: value can never be interpreted as shell code by the cluster's ``tbg`` (which
#: ``eval``\\s the configuration file name).
_CFG_FILE_RE = re.compile(r"^[A-Za-z0-9._/-]+\.cfg$")

#: One ``NAME=value`` overwrite entry.  The strict charset excludes every shell
#: metacharacter (spaces, ``$``, backticks, ``;``, quotes, ``(``/``)``, ``<``,
#: ``>``, ``|``, ``&``, ``\\``, ``*``, ``?``, ``~``, ``%``); ``tbg`` applies
#: these with ``eval``/``for word in $extra_op``, so only inert ``name=value``
#: data may pass.
_OVERWRITE_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9._:/+-]*$")


#: Key of the embedded :class:`SimulationPayload` inside the command.
#:
#: The value is the payload's canonical JSON as a *string*, not a nested
#: object: Matrix's canonical JSON (Synapse) rejects any floating-point value
#: in an event content object, and the simulation is full of floats (every
#: length, density and timestep).  Floats inside a JSON string are fine, so the
#: payload is serialised once and parsed on receipt.
PAYLOAD_KEY = "payload"


class PayloadTooLargeError(ValueError):
    """Raised when a simulation is too large to send inline in one command."""


class UnsupportedPayloadError(ValueError):
    """Raised when a payload carries fields outside the accepted wire schema."""


class SimulationType(StrEnum):
    """RCP ``type`` values of the M2 ``submit_simulation`` exchange."""

    COMMAND = "rcp.simulation_submit"
    ACK = "rcp.simulation_submit_ack"
    EVENT = "rcp.simulation_event"
    #: M2b request/response: the MCP server asks the simclient (which sees the
    #: cluster) for live status / logs; the simclient answers with an ``*_ACK``.
    STATUS_COMMAND = "rcp.status_request"
    STATUS_ACK = "rcp.status_ack"
    LOGS_COMMAND = "rcp.logs_request"
    LOGS_ACK = "rcp.logs_ack"


class SimulationState(StrEnum):
    """Coarse lifecycle state reported in acks and events (gap 9).

    ``workflow.finished`` marks the end of the CWL workflow the simclient
    drives (build -> prepare -> submit -> organize).  It deliberately does not
    claim the SLURM job finished, nor that results exist: the job may still be
    queued or running.

    The ``simulation.job_*`` events (M2b) follow the SLURM job itself:
    ``job_running`` once it starts, ``job_finished`` on a clean terminal state,
    ``job_failed`` otherwise.  ``results.ready`` is emitted only after
    ``job_finished`` and a present ``run_dir/simOutput``.
    """

    ACCEPTED = "accepted"
    SUBMITTED = "simulation.submitted"
    WORKFLOW_FINISHED = "workflow.finished"
    JOB_RUNNING = "simulation.job_running"
    JOB_FINISHED = "simulation.job_finished"
    JOB_FAILED = "simulation.job_failed"
    STEP_FINISHED = "simulation.step_finished"
    RESULTS_READY = "results.ready"
    FAILED = "simulation.failed"


class SimulationStage(StrEnum):
    """Which pipeline stage a failure occurred in."""

    BUILD = "build"
    PREPARE = "prepare"
    SUBMIT = "submit"
    RUN = "run"


#: Progress events at least this far apart (percent) are emitted; the terminal
#: event is always emitted.  Matches the design's condensation (section 4.2).
PROGRESS_EVENT_STEP_PERCENT = 25

#: Log streams ``get_logs`` can request.
LOG_STREAMS = ("stdout", "stderr", "workflow")


class SubmitParams(BaseModel):
    """Build/run flags carried alongside the payload (design section 4.1).

    Only flags that are not cluster-local policy are accepted; unlike the
    ``rc_params`` they are validated JSON scalars, never shell code.

    The field names mirror the design's tool signature (``build_*``/``cfg_*``);
    :meth:`picongpu_flags` maps them to the aliases the pinned
    ``PicBuildFlags``/``TBGFlags`` models actually accept (``jobs``, ``cmake``,
    ``preset``, ``force``, ``cfg``, ``submit``).  Passing the field names
    straight through is silently ignored by pydantic (their validation aliases
    do not include the ``build_`` prefix; ``populate_by_name`` is off), which
    would drop ``submit_system`` and run the job locally via ``bash``.
    """

    model_config = ConfigDict(extra="forbid")

    build_jobs: int | None = None
    build_cmake: str | None = None
    build_preset: int | None = None
    build_force: bool = False
    cfg_file: str | None = None
    #: The submit command; the simclient enforces its local ``tbg_submit``
    #: matches this.  A NON-sbatch value cannot be requested over the wire:
    #: ``prepare_submit`` rejects anything but ``sbatch`` outright.
    submit_system: str = DEFAULT_SUBMIT_SYSTEM
    overwrite_vars: list[str] | None = None

    @field_validator("cfg_file")
    @classmethod
    def _validate_cfg_file(cls, value: str | None) -> str | None:
        r"""Reject a ``cfg_file`` that is not a relative, inert ``.cfg`` path.

        The cluster's ``tbg`` ``eval``\s the configuration file name, so an
        arbitrary path or any shell metacharacter would be wire-supplied shell
        code -- forbidden by the design (sections 5/9).

        Returns:
            The validated path, or ``None`` when unset.

        Raises:
            ValueError: If the path is absolute, escapes upward, or contains
                characters outside the safe set.

        """
        if value is None:
            return None
        if not _CFG_FILE_RE.fullmatch(value) or ".." in value.split("/") or value.startswith("/"):
            msg = f"cfg_file must be a relative path matching {_CFG_FILE_RE.pattern!r}, got {value!r}"
            raise ValueError(msg)
        return value

    @field_validator("overwrite_vars")
    @classmethod
    def _validate_overwrite_vars(cls, value: list[str] | None) -> list[str] | None:
        """Reject any ``overwrite_vars`` entry that is not inert ``NAME=value``.

        ``tbg`` expands each entry with ``eval``/``for word in $extra_op``, so a
        value such as ``PARAM=$(cmd)`` or one containing whitespace/backticks
        would be remote code execution on the submission node.

        Returns:
            The validated list, or ``None`` when unset.

        Raises:
            ValueError: If any entry contains shell metacharacters or does not
                match ``NAME=value``.

        """
        if value is None:
            return None
        for entry in value:
            if not _OVERWRITE_VAR_RE.fullmatch(entry):
                msg = f"overwrite_vars entries must match {_OVERWRITE_VAR_RE.pattern!r}, got {entry!r}"
                raise ValueError(msg)
        return value

    def picongpu_flags(self) -> dict[str, Any]:
        """Map to the aliases ``Runner.generate(**flags)`` forwards to picongpu.

        Returns:
            The flags with ``build_``/``cfg_`` names translated and unset
            options dropped (so picongpu keeps its own defaults).

        """
        mapping = {
            "build_jobs": "jobs",
            "build_cmake": "cmake",
            "build_preset": "preset",
            "cfg_file": "cfg",
            "submit_system": "submit",
            # The pinned TBGFlags accepts overwrite_vars only under the short
            # ``o`` alias (it has no populate_by_name), so the long name alone
            # would be silently ignored.
            "overwrite_vars": "o",
        }
        flags = {alias: getattr(self, field) for field, alias in mapping.items() if getattr(self, field) is not None}
        if self.build_force:
            flags["force"] = True
        return flags


def simulation_spec_from_runner_dump(runner_dump: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full ``Runner.model_dump(mode="json")`` to the wire spec.

    The cluster-local directories are deliberately dropped: the simclient always
    sets its own, and a payload trying to set them fails
    :func:`SimulationPayload.check_allowlist`.

    Args:
        runner_dump: A full runner dump as produced by the pinned PIConGPU.

    Returns:
        ``{"sim": <pypicongpu Simulation dump>}``.

    Raises:
        UnsupportedPayloadError: If the dump has no ``sim`` field.

    """
    if "sim" not in runner_dump:
        msg = "runner dump has no 'sim' field"
        raise UnsupportedPayloadError(msg)
    return {"sim": runner_dump["sim"]}


class SimulationPayload(BaseModel):
    """The serialised simulation plus its provenance tuple (design section 2.2).

    ``extra="forbid"`` rejects unknown top-level fields; the nested
    ``simulation`` mapping is separately allow-listed by
    :meth:`check_allowlist`.
    """

    model_config = ConfigDict(extra="forbid")

    #: Required: a payload that omits the version is rejected rather than
    #: silently treated as the current version (design section 2.2).  The
    #: sender always supplies it via :meth:`build`.
    wire_format_version: int
    picongpu_version: str
    picongpu_revision: str = ""
    schema_hash: str
    simulation: dict[str, Any]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def payload_hash(self) -> str:
        """SHA-256 of the canonical simulation bytes."""
        return hashlib.sha256(canonical_bytes(self.simulation)).hexdigest()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def sim_id(self) -> str:
        """8-hex simulation id derived from :attr:`payload_hash`."""
        return self.payload_hash[:8]

    def check_allowlist(self) -> None:
        """Reject any simulation key outside :data:`ALLOWED_SIMULATION_KEYS`.

        Raises:
            UnsupportedPayloadError: If the simulation mapping carries an
                unsupported (or absent) top-level key.

        """
        keys = set(self.simulation)
        unsupported = keys - ALLOWED_SIMULATION_KEYS
        missing = ALLOWED_SIMULATION_KEYS - keys
        if unsupported or missing:
            parts = []
            if unsupported:
                parts.append(f"unsupported field(s): {', '.join(sorted(unsupported))}")
            if missing:
                parts.append(f"missing field(s): {', '.join(sorted(missing))}")
            raise UnsupportedPayloadError("; ".join(parts))

    def counts(self) -> dict[str, Any]:
        """Return a small redaction-safe summary for logs and acks.

        Returns:
            The provenance header fields plus the simulation id.

        """
        return {
            "wire_format_version": self.wire_format_version,
            "picongpu_version": self.picongpu_version,
            "picongpu_revision": self.picongpu_revision,
            "schema_hash": self.schema_hash,
            "sim_id": self.sim_id,
            "payload_hash": self.payload_hash,
        }

    @classmethod
    def build(
        cls,
        *,
        picongpu_version: str,
        picongpu_revision: str,
        schema_hash: str,
        runner_dump: dict[str, Any],
        wire_format_version: int = WIRE_FORMAT_VERSION,
    ) -> SimulationPayload:
        """Build a payload from a full runner dump and provenance values.

        Args:
            picongpu_version: The sender's PIConGPU version string.
            picongpu_revision: The sender's pinned revision.
            schema_hash: The sender's ``Runner`` schema hash.
            runner_dump: A full ``Runner.model_dump(mode="json")``.
            wire_format_version: The payload contract version.

        Returns:
            The validated payload.

        """
        return cls(
            wire_format_version=wire_format_version,
            picongpu_version=picongpu_version,
            picongpu_revision=picongpu_revision,
            schema_hash=schema_hash,
            simulation=simulation_spec_from_runner_dump(runner_dump),
        )


def provenance_mismatches(payload: SimulationPayload, local: dict[str, str]) -> list[str]:
    """Compare the payload provenance tuple against the local install.

    A blank revision on either side is treated as "unknown" and skipped, so an
    editable/local install without ``vcs_info`` does not spuriously reject a
    payload; the version and schema hash are always compared.

    Args:
        payload: The received payload.
        local: The local provenance tuple from ``version.local_provenance()``.

    Returns:
        Human-readable mismatch descriptions (empty when compatible).

    """
    mismatches: list[str] = []
    if payload.wire_format_version != WIRE_FORMAT_VERSION:
        mismatches.append(f"wire_format_version {payload.wire_format_version} != {WIRE_FORMAT_VERSION}")
    if local["picongpu_version"] and payload.picongpu_version != local["picongpu_version"]:
        mismatches.append(f"picongpu_version {payload.picongpu_version!r} != {local['picongpu_version']!r}")
    if local["schema_hash"] and payload.schema_hash != local["schema_hash"]:
        mismatches.append("schema_hash differs")
    payload_rev = payload.picongpu_revision
    local_rev = local["picongpu_revision"]
    if payload_rev and local_rev and payload_rev != local_rev:
        mismatches.append(f"picongpu_revision {payload_rev[:12]} != {local_rev[:12]}")
    return mismatches


def payload_wire_bytes(payload: SimulationPayload) -> bytes:
    """Serialise a payload for inline transport, enforcing the size cap.

    The cap is checked against the size the payload actually occupies on the
    wire: the payload body is a JSON *string* embedded in the event content, so
    its quotes/backslashes are escaped once more.  Measuring the inner
    simulation object (as an earlier version did) undercounted by up to 2x and
    let an "under-cap" payload produce an over-64-KiB Matrix event.

    The computed fields (``payload_hash``/``sim_id``) are excluded: they are
    recomputed on read and would otherwise be rejected by
    ``extra="forbid"``.

    Args:
        payload: The payload to serialise.

    Returns:
        The canonical JSON bytes to embed in the command.

    Raises:
        PayloadTooLargeError: If the encoded payload plus
            :data:`_ENVELOPE_ALLOWANCE_BYTES` exceeds
            :data:`MAX_INLINE_PAYLOAD_BYTES`.

    """
    body = payload.model_dump_json(exclude_computed_fields=True).encode("utf-8")
    # The body is carried as a JSON string inside the event content, so measure
    # the escaped form (json.dumps doubles every quote/backslash) plus the
    # envelope budget -- that is what the homeserver's 64 KiB event limit sees.
    encoded = len(json.dumps(body.decode("utf-8"), ensure_ascii=True).encode("ascii"))
    size = encoded + _ENVELOPE_ALLOWANCE_BYTES
    if size > MAX_INLINE_PAYLOAD_BYTES:
        msg = (
            f"encoded simulation payload is ~{size} bytes; the inline limit is "
            f"{MAX_INLINE_PAYLOAD_BYTES} (out-of-band payload transport is not implemented yet)"
        )
        raise PayloadTooLargeError(msg)
    return body


def build_submit_command(
    *,
    sim: str,
    seq: int,
    payload: SimulationPayload,
    params: SubmitParams | None = None,
    cmd_id: str | None = None,
    in_reply_to: str | None = None,
) -> RcpMessage:
    """Build the MCP-server-to-simclient ``submit_simulation`` command.

    The payload travels inline in the signed envelope, so the command is
    self-contained: the simclient needs no shared file system to read it.

    Args:
        sim: Simulation id.
        seq: Per-sender sequence number.
        payload: The simulation payload to embed.
        params: Optional build/run flags.
        cmd_id: Optional command id (generated when omitted).
        in_reply_to: Optional transport event id being replied to.

    Returns:
        The unsigned ``rcp.simulation_submit`` command.  A simulation larger
        than :data:`MAX_INLINE_PAYLOAD_BYTES` is rejected by
        :func:`payload_wire_bytes` with :class:`PayloadTooLargeError`.

    """
    # The payload is carried as a JSON string (see PAYLOAD_KEY): Synapse
    # rejects floats in event-content objects, and the simulation has many.
    body = payload_wire_bytes(payload).decode("utf-8")
    return RcpMessage(
        sim=sim,
        kind=Kind.COMMAND,
        type=SimulationType.COMMAND,
        seq=seq,
        sender_role=SenderRole.MCP_SERVER,
        in_reply_to=in_reply_to,
        payload={
            "cmd_id": cmd_id or new_cmd_id(),
            "header": payload.counts(),
            PAYLOAD_KEY: body,
            "params": (params or SubmitParams()).model_dump(mode="json"),
        },
    )


def build_submit_ack(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    sim_id: str,
    state: SimulationState,
    in_reply_to: str | None,
    job_id: int | None = None,
    error: str | None = None,
    error_code: str | None = None,
) -> RcpMessage:
    """Build the simclient's acknowledgement of a submit command.

    Returns:
        The unsigned ``rcp.simulation_submit_ack`` message.

    """
    payload: dict[str, Any] = {"cmd_id": cmd_id, "sim_id": sim_id, "state": state.value, "job_id": job_id}
    if error:
        payload["error"] = error
    if error_code:
        payload["error_code"] = error_code
    return RcpMessage(
        sim=sim,
        kind=Kind.ACK,
        type=SimulationType.ACK,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        in_reply_to=in_reply_to,
        payload=payload,
    )


def build_submit_event(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    sim_id: str,
    state: SimulationState,
    job_id: int | None = None,
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
    """Build one M2 lifecycle event.

    Only the fields relevant to the event's ``state`` are carried; the rest stay
    absent so the payloads remain small and the room body readable.

    Returns:
        The unsigned ``rcp.simulation_event`` event.

    """
    payload: dict[str, Any] = {"cmd_id": cmd_id, "sim_id": sim_id, "state": state.value, "job_id": job_id}
    if stage is not None:
        payload["stage"] = stage.value
    if submit_system is not None:
        payload["submit_system"] = submit_system
    if error:
        payload["error"] = error
    if error_code:
        payload["error_code"] = error_code
    if results_linked is not None:
        payload["results_linked"] = results_linked
    payload.update(
        {
            key: value
            for key, value in (
                ("step", step),
                ("percent", percent),
                ("walltime", walltime),
                ("avg_per_step", avg_per_step),
                ("eta_s", eta_s),
                ("slurm_state", slurm_state),
                ("exit_code", exit_code),
            )
            if value is not None
        },
    )
    return RcpMessage(
        sim=sim,
        kind=Kind.EVENT,
        type=SimulationType.EVENT,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        payload=payload,
    )


def build_status_command(
    *,
    sim: str,
    seq: int,
    sim_id: str,
    cmd_id: str | None = None,
    in_reply_to: str | None = None,
) -> RcpMessage:
    """Build the MCP-server-to-simclient live-status request (M2b).

    Returns:
        The unsigned ``rcp.status_request`` command.

    """
    return RcpMessage(
        sim=sim,
        kind=Kind.COMMAND,
        type=SimulationType.STATUS_COMMAND,
        seq=seq,
        sender_role=SenderRole.MCP_SERVER,
        in_reply_to=in_reply_to,
        payload={"cmd_id": cmd_id or new_cmd_id(), "sim_id": sim_id},
    )


def build_logs_command(
    *,
    sim: str,
    seq: int,
    sim_id: str,
    stream: str = "stdout",
    tail: int = 100,
    cmd_id: str | None = None,
    in_reply_to: str | None = None,
) -> RcpMessage:
    """Build the MCP-server-to-simclient log request (M2b).

    Returns:
        The unsigned ``rcp.logs_request`` command.

    Raises:
        ValueError: If ``stream`` is not one of :data:`LOG_STREAMS`.

    """
    if stream not in LOG_STREAMS:
        msg = f"unknown log stream {stream!r}; expected one of {LOG_STREAMS}"
        raise ValueError(msg)
    return RcpMessage(
        sim=sim,
        kind=Kind.COMMAND,
        type=SimulationType.LOGS_COMMAND,
        seq=seq,
        sender_role=SenderRole.MCP_SERVER,
        in_reply_to=in_reply_to,
        payload={"cmd_id": cmd_id or new_cmd_id(), "sim_id": sim_id, "stream": stream, "tail": tail},
    )


def build_status_ack(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    sim_id: str,
    in_reply_to: str | None,
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
    """Build the simclient's live-status response (M2b).

    Returns:
        The unsigned ``rcp.status_ack`` message.

    """
    payload: dict[str, Any] = {"cmd_id": cmd_id, "sim_id": sim_id, "state": state}
    payload.update(
        {
            key: value
            for key, value in (
                ("slurm_state", slurm_state),
                ("job_id", job_id),
                ("step", step),
                ("percent", percent),
                ("walltime", walltime),
                ("avg_per_step", avg_per_step),
                ("eta_s", eta_s),
                ("exit_code", exit_code),
                ("error", error),
                ("error_code", error_code),
            )
            if value is not None
        },
    )
    return RcpMessage(
        sim=sim,
        kind=Kind.ACK,
        type=SimulationType.STATUS_ACK,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        in_reply_to=in_reply_to,
        payload=payload,
    )


def build_logs_ack(
    *,
    sim: str,
    seq: int,
    cmd_id: str,
    sim_id: str,
    in_reply_to: str | None,
    stream: str,
    lines: list[str],
    total_lines: int,
    error: str | None = None,
    error_code: str | None = None,
) -> RcpMessage:
    """Build the simclient's log response (M2b).

    Returns:
        The unsigned ``rcp.logs_ack`` message.

    """
    payload: dict[str, Any] = {
        "cmd_id": cmd_id,
        "sim_id": sim_id,
        "stream": stream,
        "lines": lines,
        "total_lines": total_lines,
    }
    if error:
        payload["error"] = error
    if error_code:
        payload["error_code"] = error_code
    return RcpMessage(
        sim=sim,
        kind=Kind.ACK,
        type=SimulationType.LOGS_ACK,
        seq=seq,
        sender_role=SenderRole.SIMCLIENT,
        in_reply_to=in_reply_to,
        payload=payload,
    )
