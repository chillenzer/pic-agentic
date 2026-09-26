# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Simulation-side execution of an M2 ``submit_simulation`` command.

The handler is deliberately paranoid (design sections 6.4, 8.2): it re-validates
the inline payload, checks its byte hash and the provenance tuple against the
local install, and only then imports PIConGPU.  All cluster locations come from
local configuration, never from the payload.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pic_agentic.protocol.simulation import (
    DEFAULT_SUBMIT_SYSTEM,
    SimulationPayload,
    SimulationStage,
    SimulationState,
    SubmitParams,
    UnsupportedPayloadError,
    provenance_mismatches,
)
from pic_agentic.rcp import canonical_bytes

log = logging.getLogger(__name__)

#: Per-command directory token: the command id (or, defensively, the payload
#: hash) -- lowercase hex only, so it can never traverse or be absolute.
_TOKEN_RE = re.compile(r"^[0-9a-f]{8,64}$")


class SimulationErrorCode(StrEnum):
    """Stable machine-readable error codes reported in acks and events."""

    PATH_UNSAFE = "path_unsafe"
    PAYLOAD_INVALID = "payload_invalid"
    UNSUPPORTED = "unsupported"
    HASH_MISMATCH = "hash_mismatch"
    VERSION_MISMATCH = "version_mismatch"
    SUBMIT_SYSTEM_MISMATCH = "submit_system_mismatch"
    REJECTED = "rejected_by_policy"
    PICONGPU_UNAVAILABLE = "picongpu_unavailable"
    GENERATE_FAILED = "generate_failed"
    RUN_FAILED = "run_failed"


class SimulationExecutionError(RuntimeError):
    """A stage failure carrying a machine-readable error code."""

    def __init__(self, code: SimulationErrorCode, message: str, stage: SimulationStage | None = None) -> None:
        """Create the error.

        Args:
            code: Stable error code reported in the ack/event.
            message: Human-readable detail.
            stage: Pipeline stage, when the failure happened after acceptance.

        """
        super().__init__(message)
        self.code = code
        self.stage = stage


@dataclass
class SubmitConfig:
    """Cluster-local policy for executing a submitted simulation."""

    setup_root: Path
    template_dir: str = ""
    preset: int | None = None

    def __post_init__(self) -> None:
        """Normalise the path to absolute form."""
        self.setup_root = Path(self.setup_root)


def parse_payload(raw: str) -> dict[str, Any]:
    """Parse the inline payload JSON string into a mapping.

    The payload is transported as a JSON *string* (see
    :data:`~pic_agentic.protocol.simulation.PAYLOAD_KEY`): Matrix's canonical
    JSON rejects floats in event-content objects, and the simulation has many.

    Args:
        raw: The JSON string from the command's payload.

    Returns:
        The decoded payload mapping.

    Raises:
        SimulationExecutionError: If the string is not a JSON object.

    """
    try:
        body = json.loads(raw)
    except ValueError as exc:
        msg = f"inline payload is not JSON: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg) from exc
    if not isinstance(body, dict):
        msg = "inline payload is not a JSON object"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg)
    return body


def check_payload_hash(body: dict[str, Any], header: dict[str, Any]) -> None:
    """Verify the transmitted hash against the embedded payload.

    Args:
        body: The embedded ``SimulationPayload`` mapping from the command.
        header: The command's ``header`` mapping.

    Raises:
        SimulationExecutionError: On a malformed payload or hash mismatch.

    """
    try:
        simulation = body["simulation"]
    except (KeyError, TypeError) as exc:
        msg = f"cannot read simulation from payload: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg) from exc
    actual = hashlib.sha256(canonical_bytes(simulation)).hexdigest()
    expected = str(header.get("payload_hash", ""))
    if not expected or actual != expected:
        msg = f"payload hash {actual[:12]} does not match header {expected[:12] or '<missing>'}"
        raise SimulationExecutionError(SimulationErrorCode.HASH_MISMATCH, msg)


def _detect_submit_system() -> str | None:
    """Return the local ``tbg_submit`` from the cluster rc params.

    Returns:
        The configured submit command, ``None`` when PIConGPU or the rc
        parameter is unavailable.

    """
    try:
        from picongpu import rc_params  # ruff: ignore[import-outside-top-level] - optional dependency
    except ImportError:
        return None
    value = rc_params.get("tbg_submit", "")
    return str(value) if value else None


def _check_submit_system(requested: str) -> None:
    """Reject a request that would not submit via SLURM ``sbatch``.

    The picongpu workflow default is ``"bash"`` (local execution on the
    submission node, no SLURM job), so the request must itself be ``sbatch``
    (the wire contract only supports SLURM), and a *configured* cluster-local
    ``tbg_submit`` must not contradict it.  An unset ``tbg_submit`` is not an
    error: the explicit ``submit="sbatch"`` flag still overrides the workflow
    default, so the job lands on SLURM either way.

    Args:
        requested: The submit system the command asked for.

    Raises:
        SimulationExecutionError: If the request is not ``sbatch`` or a
            configured cluster-local setting differs.

    """
    if requested != DEFAULT_SUBMIT_SYSTEM:
        msg = f"only {DEFAULT_SUBMIT_SYSTEM!r} submissions are supported, got {requested!r}"
        raise SimulationExecutionError(SimulationErrorCode.SUBMIT_SYSTEM_MISMATCH, msg)
    local = _detect_submit_system()
    if local is not None and local != requested:
        msg = f"cluster tbg_submit={local!r} but the command requests {requested!r}"
        raise SimulationExecutionError(SimulationErrorCode.SUBMIT_SYSTEM_MISMATCH, msg)


def runner_from_payload(payload: SimulationPayload, config: SubmitConfig, token: str) -> Any:
    """Rebuild a fresh ``Runner`` with cluster-local, per-command directories.

    The payload's own directories are never read; only ``sim`` is taken.  The
    ``token`` makes the directories unique per command: ``Runner.generate()``
    asserts the setup directory does not exist, so a legitimate *resubmission*
    of an identical simulation (a new ``cmd_id``) would otherwise collide with
    the previous run's directory.  ``token`` is a validated hex command id.

    Args:
        payload: The validated payload.
        config: The cluster-local submit policy.
        token: Per-command unique token (the ``cmd_id``).

    Returns:
        A ``pypicongpu.Runner`` instance.

    Raises:
        SimulationExecutionError: If PIConGPU is unavailable or the simulation
            does not validate against the local schema.

    """
    if not _TOKEN_RE.match(token):
        msg = f"unsafe per-command token: {token!r}"
        raise SimulationExecutionError(SimulationErrorCode.PATH_UNSAFE, msg)
    base = (config.setup_root / payload.sim_id / token).resolve()
    # Defence in depth: even with a validated token, never build outside the
    # configured root (pathlib discards earlier components on an absolute path).
    root = config.setup_root.resolve()
    if root != base and root not in base.parents:
        msg = f"generated setup dir escapes {config.setup_root}: {base}"
        raise SimulationExecutionError(SimulationErrorCode.PATH_UNSAFE, msg)
    try:
        from picongpu.pypicongpu.runner import Runner  # ruff: ignore[import-outside-top-level] - optional dependency
    except ImportError as exc:
        msg = "PIConGPU is not installed on the cluster"
        raise SimulationExecutionError(SimulationErrorCode.PICONGPU_UNAVAILABLE, msg) from exc
    setup_dir = (base / "input").absolute()
    run_dir = (base / "run").absolute()
    sim_dump = payload.simulation["sim"]
    dump: dict[str, Any] = {"sim": sim_dump, "setup_dir": str(setup_dir), "run_dir": str(run_dir)}
    if config.template_dir:
        dump["template_dir"] = [config.template_dir]
    try:
        runner = Runner.model_validate(dump)
    except Exception as exc:
        msg = f"simulation does not validate: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg) from exc
    # The nested pypicongpu models do not set ``extra="forbid"``, so an unknown
    # field inside ``sim`` would be silently dropped instead of rejected.  The
    # pin guarantees a lossless ``Runner`` round-trip, so a dump that does not
    # reproduce itself carried something outside the schema: reject it as
    # ``unsupported`` per the plan rather than running a silently altered sim.
    if runner.sim.model_dump(mode="json") != sim_dump:
        msg = "simulation carries fields outside the pinned pypicongpu schema"
        raise SimulationExecutionError(SimulationErrorCode.UNSUPPORTED, msg)
    return runner


@dataclass
class PreparedSubmit:
    """A validated, ready-to-execute submit command."""

    payload: SimulationPayload
    params: SubmitParams
    runner: Any
    config: SubmitConfig


def prepare_submit(
    *,
    body: dict[str, Any],
    header: dict[str, Any],
    params: dict[str, Any] | None,
    config: SubmitConfig,
    local_provenance: dict[str, str],
    token: str,
) -> PreparedSubmit:
    """Validate a submit command and rebuild its runner.

    Every check here happens *before* an ``accepted`` ack is sent: a failure means
    the command was never accepted, so the error belongs in the ack (design
    section 2.2).  Stage failures during execution are reported as events
    instead (see :func:`execute_submit`).

    Args:
        body: The embedded payload mapping from the command.
        header: The command's provenance header.
        params: The command's build/run flags.
        config: Cluster-local submit policy.
        local_provenance: This install's provenance tuple.
        token: Per-command unique token for the generated directories.

    Returns:
        The validated payload, flags and fresh runner.

    Raises:
        SimulationExecutionError: On any validation failure.

    """
    check_payload_hash(body, header)

    try:
        payload = SimulationPayload.model_validate(body)
    except ValueError as exc:
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, str(exc)) from exc
    if payload.payload_hash != str(header.get("payload_hash", "")):
        msg = "payload/header hash mismatch"
        raise SimulationExecutionError(SimulationErrorCode.HASH_MISMATCH, msg)
    if payload.sim_id != str(header.get("sim_id", "")):
        msg = f"payload sim_id {payload.sim_id} does not match header {header.get('sim_id')!r}"
        raise SimulationExecutionError(SimulationErrorCode.HASH_MISMATCH, msg)
    try:
        payload.check_allowlist()
    except UnsupportedPayloadError as exc:
        raise SimulationExecutionError(SimulationErrorCode.UNSUPPORTED, str(exc)) from exc
    mismatches = provenance_mismatches(payload, local_provenance)
    if mismatches:
        raise SimulationExecutionError(SimulationErrorCode.VERSION_MISMATCH, "; ".join(mismatches))

    try:
        submit_params = SubmitParams.model_validate(params or {})
    except ValueError as exc:
        msg = f"invalid submit params: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.PAYLOAD_INVALID, msg) from exc
    _check_submit_system(submit_params.submit_system)

    return PreparedSubmit(
        payload=payload,
        params=submit_params,
        runner=runner_from_payload(payload, config, token),
        config=config,
    )


#: Captured stderr of the last workflow execution.  Single process-global slot:
#: safe only because the simclient executes one submission at a time (the
#: single-threaded event loop serialises ``execute_submit``, and the only
#: concurrent caller is that one submission's ``asyncio.to_thread``).  It is
#: reset at the start of every ``_run_workflow`` and read immediately after via
#: ``_workflow_failure_detail``; do not introduce concurrent workflow runs
#: without replacing it with a per-run return value.
_LAST_WORKFLOW_STDERR = ""

#: Cap on the captured error text sent in a failure event.
_MAX_ERROR_DETAIL = 4000
#: Files above this size are skipped by the cache scan (compiled binaries).
_MAX_SCAN_FILE_BYTES = 2_000_000
#: Stop the cache scan after this many matching lines.
_MAX_SCAN_HITS = 50

#: Lines worth keeping from the captured stderr for the failure event.
_ERROR_LINE_RE = re.compile(
    r"error|fatal|not found|No such file|command not found|exited with status|permanentFail|missing expected",
    re.IGNORECASE,
)


def _run_workflow(runner: Any) -> None:
    """Run the CWL workflow, capturing its stderr for failure reporting.

    cwltool raises only ``Completed permanentFail``; the actual step command
    error (e.g. ``cmake: command not found``) is written to file descriptor 2
    by cwltool and the step subprocess, bypassing both the ``cwltool`` logger
    and Python-level ``sys.stderr`` redirection.  Redirect fd 2 to a temporary
    file for the duration of the run and retain the matching lines.

    Args:
        runner: The ``pypicongpu.Runner`` to run.

    """
    global _LAST_WORKFLOW_STDERR  # ruff: ignore[global-statement] - single-slot capture, one run at a time
    # Reset the single-slot capture at the start of every run (see the module
    # note on _LAST_WORKFLOW_STDERR); _workflow_failure_detail consumes it.
    _LAST_WORKFLOW_STDERR = ""
    saved = os.dup(2)
    read_fd, write_fd = os.pipe()
    chunks: list[bytes] = []

    def reader() -> None:
        # Tee: forward everything to the real stderr (so operator logs stay
        # visible) and keep a copy for the failure event.
        with os.fdopen(read_fd, "rb", closefd=True) as stream:
            for chunk in iter(lambda: stream.read(4096), b""):
                os.write(saved, chunk)
                chunks.append(chunk)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        runner.run()
    finally:
        # Restore fd 2 first so the pipe's write end closes and the reader sees
        # EOF; only then join (and do NOT close ``saved`` before the join, the
        # reader still writes to it).  Parse here, not after the try/finally:
        # ``runner.run()`` raises ``Completed permanentFail``, which would
        # otherwise skip the capture assignment.
        os.dup2(saved, 2)
        thread.join(timeout=10)
        os.close(saved)
        text = b"".join(chunks).decode("utf-8", errors="replace")
        lines = [line.rstrip() for line in text.splitlines() if _ERROR_LINE_RE.search(line)]
        _LAST_WORKFLOW_STDERR = "\n".join(lines)


def _workflow_failure_detail(run_dir: Path) -> str:
    """Return the captured workflow stderr plus any retained step log.

    Args:
        run_dir: The run directory (for the .cwl_cache fallback scan).

    Returns:
        A truncated, redaction-ready error string (possibly empty).

    """
    detail = _LAST_WORKFLOW_STDERR.strip()
    if not detail:
        detail = _scan_retained_step_logs(run_dir)
    return detail[-_MAX_ERROR_DETAIL:]


def _scan_retained_step_logs(run_dir: Path) -> str:
    """Best-effort scan of the retained cwltool cache for an error line.

    Returns:
        The most relevant-looking error line, or ``""``.

    """
    cache = Path(run_dir) / ".cwl_cache"
    if not cache.is_dir():
        return ""
    pattern = re.compile(r"error|fatal|not found|No such file|command not found|exited with status", re.IGNORECASE)
    hits: list[str] = []
    for path in cache.rglob("*"):
        if not path.is_file() or path.stat().st_size > _MAX_SCAN_FILE_BYTES:
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                hits.extend(line.rstrip() for line in handle if pattern.search(line))
        except OSError:
            continue
        if len(hits) > _MAX_SCAN_HITS:
            break
    return "\n".join(hits[-20:])


def _normalise_workflow_vars(setup_dir: Path) -> None:
    """Patch the generated ``input.yaml`` so CWL accepts ``run_overwrite_vars``.

    The pinned ``Runner.generate()`` serialises its ``TBGFlags.overwrite_vars``
    list straight into the workflow input as a YAML/JSON list, but the pinned
    ``workflow.cwl`` declares ``run_overwrite_vars`` as ``type: string?`` (tbg
    takes a single ``-o`` argument it word-splits itself).  Left as a list, CWL
    validation fails with "value is a CommentedSeq, expected null or string",
    so every submission using the flag would die as ``RUN_FAILED`` after
    ``accepted``.  Join the (already validated, single-token) entries into the
    one space-separated string the tool expects.

    Args:
        setup_dir: The runner's setup directory (holds ``workflow/input.yaml``).

    """
    input_path = Path(setup_dir) / "workflow" / "input.yaml"
    if not input_path.is_file():
        return
    data = json.loads(input_path.read_text(encoding="utf-8"))
    value = data.get("run_overwrite_vars")
    if isinstance(value, list):
        data["run_overwrite_vars"] = " ".join(str(entry) for entry in value)
        input_path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def link_run_results(run_dir: Path) -> bool:
    """Link the simulation output into ``run_dir`` via the generated script.

    The CWL workflow writes PIConGPU output inside its per-step cache directory
    and generates ``link_results.sh`` to expose it as ``run_dir/simOutput``, but
    never runs that script in the run directory.  Running it here makes
    ``run_dir/simOutput`` present before the workflow-finished event, matching
    the design's expectation that results are organised when the run finishes.

    Args:
        run_dir: The runner's run directory.

    Returns:
        True if the script ran successfully (or the link already exists), False
        otherwise; a missing link is not fatal, the job may still be running.

    """
    run_dir = Path(run_dir)
    if (run_dir / "simOutput").exists():
        return True
    script = run_dir / "link_results.sh"
    if not script.is_file():
        return False
    try:
        result = subprocess.run(
            ["/bin/bash", str(script), str(run_dir)],
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and (run_dir / "simOutput").exists()


def find_stdout_path(run_dir: Path) -> str | None:
    """Locate the job's ``stdout`` inside the cwltool step cache.

    The CWL step is submitted with ``#SBATCH -o stdout`` and a ``--chdir`` into
    its per-step cache directory, so the SLURM output lands at
    ``run_dir/.cwl_cache/*/stdout``.  The cache layout is internal to cwltool
    (the plan's known risk), so discovery is a best-effort glob: when several
    step caches match, the newest file (by mtime) wins.

    Args:
        run_dir: The runner's run directory.

    Returns:
        The absolute path to the newest matching ``stdout``, or None when the
        cache carries none.

    """
    run_dir = Path(run_dir)
    candidates = [path for path in run_dir.glob(".cwl_cache/*/stdout") if path.is_file()]
    if not candidates:
        return None
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    return str(newest)


async def execute_submit(
    *,
    prepared: PreparedSubmit,
    emit: Any,
    job_id_reader: Any,
) -> dict[str, Any]:
    """Run a prepared submission (accepted already sent).

    Args:
        prepared: The validated submission.
        emit: Async callable ``emit(state, **fields)`` posting a lifecycle event.
        job_id_reader: Callable ``job_id_reader(run_dir, payload) -> int | None``.

    Returns:
        ``{"sim_id", "state", "job_id", "run_dir", "stdout_path"}``.

    Raises:
        SimulationExecutionError: On a build/run stage failure.

    """
    runner = prepared.runner
    flags = prepared.params.picongpu_flags()
    # The cluster-local preset is a default: an explicit command flag wins.
    if prepared.config.preset is not None and flags.get("preset") is None:
        flags["preset"] = prepared.config.preset
    # build stage: generate the setup (must not pre-exist).  generate() is
    # synchronous and can run for minutes, so keep it off the event loop.
    try:
        await asyncio.to_thread(lambda: runner.generate(**flags))
    except Exception as exc:
        msg = f"generate failed: {exc}"
        raise SimulationExecutionError(SimulationErrorCode.GENERATE_FAILED, msg, SimulationStage.BUILD) from exc
    # The pinned workflow.cwl types run_overwrite_vars as a single string while
    # Runner.generate() writes the list through; patch the input so a
    # submission using -o does not fail CWL validation (see the helper).
    await asyncio.to_thread(_normalise_workflow_vars, runner.setup_dir)

    # Submit stage: run the workflow; job id from submission_information.txt.
    # cwltool raises a generic ``Completed permanentFail`` and logs the actual
    # step error at ERROR level, so capture its log for the failure event;
    # otherwise the event is undiagnosable from the room.
    try:
        await asyncio.to_thread(_run_workflow, runner)
    except Exception as exc:
        detail = _workflow_failure_detail(runner.run_dir)
        msg = f"workflow failed: {exc}"
        if detail:
            msg = f"{msg}\n{detail}"
        raise SimulationExecutionError(SimulationErrorCode.RUN_FAILED, msg, SimulationStage.RUN) from exc

    job_id = job_id_reader(runner.run_dir, prepared.payload)
    if job_id is not None:
        await emit(SimulationState.SUBMITTED, job_id=job_id, submit_system=prepared.params.submit_system)
    # A submit system without a scheduler job id (e.g. local ``bash`` execution,
    # or a scheduler whose output has no parseable id) has nothing to report in
    # ``simulation.submitted``; the ``workflow.finished`` event below still fires
    # with ``job_id=None``, so the lifecycle is not silently truncated.
    link_ready = await asyncio.to_thread(link_run_results, runner.run_dir)
    await emit(SimulationState.WORKFLOW_FINISHED, job_id=job_id, results_linked=link_ready)
    stdout_path = await asyncio.to_thread(find_stdout_path, runner.run_dir)
    return {
        "sim_id": prepared.payload.sim_id,
        "state": SimulationState.WORKFLOW_FINISHED.value,
        "job_id": job_id,
        "run_dir": str(runner.run_dir),
        "stdout_path": stdout_path,
    }


__all__ = [
    "PreparedSubmit",
    "SimulationErrorCode",
    "SimulationExecutionError",
    "SubmitConfig",
    "check_payload_hash",
    "execute_submit",
    "find_stdout_path",
    "prepare_submit",
]
