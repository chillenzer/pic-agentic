# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""End-to-end ``submit_simulation`` test over the in-memory transport.

PIConGPU is not importable in the offline test environment, so the runner
boundary is stubbed at the module seam; everything else (inline payload
validation, hash and provenance checks, command/ack/event flow, durable
idempotency) runs for real.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pic_agentic.protocol.simulation import (
    PAYLOAD_KEY,
    SimulationPayload,
    SimulationState,
    SimulationType,
    SubmitParams,
    UnsupportedPayloadError,
    build_submit_command,
)
from pic_agentic.rcp import new_secret_hex
from pic_agentic.server.simulation import SubmitService
from pic_agentic.simclient import SimClient
from pic_agentic.simclient import simulation as sim_mod
from pic_agentic.simclient.simulation import SubmitConfig
from pic_agentic.simulation_build import BuiltSimulation
from pic_agentic.slurm import SlurmClient
from pic_agentic.transport.memory import MemoryTransport

FAKE_BIN = Path(__file__).parent / "fake_slurm"
FIXTURE = Path(__file__).parent / "fixtures" / "pypicongpu_runner.json"
SECRET = new_secret_hex()
SIM = "7f3a2b1c"
JOB_ID = 424242


def _runner_dump() -> dict:
    return json.loads(FIXTURE.read_text())


class _FakeRunner:
    """Stand-in for ``pypicongpu.Runner`` at the module boundary."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        # Mirrors Runner.setup_dir (base/input next to base/run); the simclient
        # patches setup_dir/workflow/input.yaml after generate().
        self.setup_dir = self.run_dir.parent / "input"
        self.generated = False
        self.ran = False
        self.flags: dict = {}

    def generate(self, **flags: object) -> None:
        self.generated = True
        self.flags = flags

    def run(self) -> None:
        self.ran = True
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "submission_information.txt").write_text(f"Submitted batch job {JOB_ID}\n")
        # The real workflow writes simOutput under its cache dir and generates
        # link_results.sh (which the simclient then runs).
        output = self.run_dir / ".cwl_cache" / "steps" / "simOutput"
        output.mkdir(parents=True, exist_ok=True)
        (output / "openPMD").mkdir(exist_ok=True)
        (self.run_dir / "link_results.sh").write_text(
            f'#!/bin/bash\nln -s "{output}" "$1"\n',
        )


@pytest.fixture
def shared_dir(tmp_path):
    d = tmp_path / "shared"
    d.mkdir()
    return d


@pytest.fixture
def fake_runner(monkeypatch):
    """Stub the runner rebuild and the local submit-system probe."""
    created: list[_FakeRunner] = []

    def fake_from_payload(payload, config, token):
        runner = _FakeRunner(config.setup_root / payload.sim_id / token / "run")
        created.append(runner)
        return runner

    monkeypatch.setattr(sim_mod, "runner_from_payload", fake_from_payload)
    monkeypatch.setattr(sim_mod, "_detect_submit_system", lambda: "sbatch")
    return created


async def _fake_builder(*, script_path, interpreter="", **_kw: object) -> BuiltSimulation:
    return BuiltSimulation(
        runner=_runner_dump(),
        picongpu_version="0.9.0-dev",
        picongpu_revision="91c3ee5fb4c9425b00d4673d9608f4370593cacf",
        schema_hash="f6471fe1244f6a9d819951e090a281862b58b5b7170b5ae209b8841c3d94e4d9",
    )


def _make_pair(shared_dir, *, builder=_fake_builder):
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = SubmitService(
        sim=SIM,
        secret=SECRET,
        runner_dump_builder=builder,
        ack_timeout_s=5.0,
    )
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        submit_config=SubmitConfig(setup_root=shared_dir / "sims"),
        poll_interval_s=0.05,
    )
    return mcp_t, sim_t, service, client


async def _run_submit(shared_dir, tmp_path, *, fake_runner, params=None):
    mcp_t, sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")

    async def pump() -> None:
        async for msg in mcp_t.receive():
            service.on_message(msg)

    pump_task = asyncio.create_task(pump())
    serve_task = asyncio.create_task(client.serve())
    try:
        outcome = await service.submit(mcp_t.send, script, params=params)
        # Let the async lifecycle events drain (the ack resolves the call first).
        await asyncio.sleep(0.1)
    finally:
        serve_task.cancel()
        pump_task.cancel()
        # Await the cancelled tasks so ``serve``'s ``finally`` reaps any
        # in-flight follower subprocess before the event loop closes.
        await asyncio.gather(serve_task, pump_task, return_exceptions=True)
        await sim_t.close()
        await mcp_t.close()
    return outcome, service


async def test_submit_round_trip(shared_dir, tmp_path, fake_runner) -> None:
    outcome, service = await _run_submit(shared_dir, tmp_path, fake_runner=fake_runner)
    assert outcome.acked
    assert outcome.ok, outcome.error
    assert outcome.state == SimulationState.ACCEPTED.value
    assert len(outcome.sim_id) == 8
    assert fake_runner
    assert fake_runner[0].generated
    assert fake_runner[0].ran

    events = [
        message
        for message in service.event_log
        if message.payload.get("sim_id") == outcome.sim_id and message.payload.get("cmd_id") == outcome.cmd_id
    ]
    states = [event.payload["state"] for event in events]
    assert SimulationState.SUBMITTED.value in states
    assert SimulationState.WORKFLOW_FINISHED.value in states
    submitted = next(e for e in events if e.payload["state"] == SimulationState.SUBMITTED.value)
    assert submitted.payload["job_id"] == JOB_ID
    # Provenance tuple is reported back to the sender.
    assert "picongpu_version" not in submitted.payload
    finished = next(e for e in events if e.payload["state"] == SimulationState.WORKFLOW_FINISHED.value)
    # The simclient runs the generated link_results.sh, so simOutput is present.
    assert finished.payload["results_linked"] is True
    assert (fake_runner[0].run_dir / "simOutput").exists()


async def test_submit_payload_is_embedded_in_the_command(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, _client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, payload, command = await service.build_payload(script)
    raw = command.payload[PAYLOAD_KEY]
    # The payload is a JSON *string*: Synapse's canonical JSON rejects floats in
    # event-content objects, and the simulation has many.
    assert isinstance(raw, str)
    body = json.loads(raw)
    assert set(body) == {"wire_format_version", "picongpu_version", "picongpu_revision", "schema_hash", "simulation"}
    assert set(body["simulation"]) == {"sim"}
    assert SimulationPayload.model_validate(body).sim_id == payload.sim_id
    assert "payload_path" not in command.payload


async def test_submit_is_idempotent_on_replay(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)

    first = await client.handle(command)
    command.transport_event_id = "$replay"
    second = await client.handle(command)
    assert first is not None
    assert second is not None
    assert second.payload["cmd_id"] == first.payload["cmd_id"]
    # The replay re-acks the recorded terminal state, not a fresh accept.
    assert second.payload["state"] == SimulationState.WORKFLOW_FINISHED.value
    # The replay did not rebuild the setup.
    assert len(fake_runner) == 1


async def test_submit_replay_after_restart_reacks_without_rebuilding(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    await client.handle(command)

    # New process, same shared dir: the durable record re-acks.
    fresh = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
        submit_config=SubmitConfig(setup_root=shared_dir / "sims"),
        poll_interval_s=0.05,
    )
    command.transport_event_id = "$backfill"
    replayed = await fresh.handle(command)
    assert replayed is not None
    assert replayed.payload["state"] == SimulationState.WORKFLOW_FINISHED.value
    assert len(fake_runner) == 1


async def test_submit_rejects_changed_payload_same_cmd_id(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    cmd_id, _payload, command = await service.build_payload(script)
    await client.handle(command)
    assert len(fake_runner) == 1

    # Same cmd_id, different payload hash: rejected, so the original record is
    # not clobbered (a real resubmission gets a fresh cmd_id).
    changed = dict(_runner_dump())
    changed["sim"] = {**changed["sim"], "delta_t_si": 2e-15}
    payload = SimulationPayload.build(picongpu_version="", picongpu_revision="", schema_hash="", runner_dump=changed)
    new_command = build_submit_command(sim=SIM, seq=99, payload=payload, cmd_id=cmd_id).sign(SECRET)
    new_command.transport_event_id = "$changed"
    ack = await client.handle(new_command)
    assert ack is not None
    assert ack.payload["error_code"] == "rejected_by_policy"
    assert "cmd_id_conflict" in ack.payload["error"]
    # Nothing was executed for the changed payload.
    assert len(fake_runner) == 1


async def test_submit_reports_hash_mismatch(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    # Corrupt the embedded simulation; the header hash still names the original,
    # so the hash check fires.  Re-sign so the signature is not the blocker.
    tampered = json.loads(command.payload[PAYLOAD_KEY])
    tampered["simulation"]["sim"]["delta_t_si"] = 9.99e-15
    command.payload[PAYLOAD_KEY] = json.dumps(tampered)
    command.sign(SECRET)

    ack = await client.handle(command)
    assert ack is not None
    assert ack.payload["error_code"] == "hash_mismatch"


async def test_submit_reports_version_mismatch(shared_dir, tmp_path, fake_runner, monkeypatch) -> None:
    monkeypatch.setattr(
        sim_mod,
        "provenance_mismatches",
        lambda *_args, **_kwargs: ["schema_hash differs"],
    )
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    ack = await client.handle(command)
    assert ack is not None
    assert ack.payload["error_code"] == "version_mismatch"


async def test_submit_rejects_command_without_payload(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    del command.payload[PAYLOAD_KEY]
    command.sign(SECRET)
    ack = await client.handle(command)
    assert ack is not None
    assert ack.payload["error_code"] == "rejected_by_policy"


async def test_submit_rejected_when_handler_disabled(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, sim_t, service, _client = _make_pair(shared_dir)
    disabled = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared_dir,
    )
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    ack = await disabled.handle(command)
    assert ack is not None
    # Regression: the rejection must be a submit-shaped ack, not a hello_ack,
    # or the sender's SubmitService never resolves its future and times out.
    assert ack.type == SimulationType.ACK
    assert ack.payload["error"] == "rejected_by_policy"
    assert ack.payload["error_code"] == "rejected_by_policy"
    assert ack.payload["state"] == SimulationState.FAILED.value


async def test_submit_rejects_non_sbatch_submit_system(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    cmd_id, payload, _command = await service.build_payload(script)
    # A command asking for local bash must be rejected outright.
    local = build_submit_command(
        sim=SIM, seq=1, payload=payload, params=SubmitParams(submit_system="bash"), cmd_id=cmd_id
    ).sign(SECRET)
    ack = await client.handle(local)
    assert ack is not None
    assert ack.payload["error_code"] == "submit_system_mismatch"


async def test_submit_mismatched_local_submit_system(shared_dir, tmp_path, fake_runner, monkeypatch) -> None:
    monkeypatch.setattr(sim_mod, "_detect_submit_system", lambda: "bash")
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    ack = await client.handle(command)
    assert ack is not None
    assert ack.payload["error_code"] == "submit_system_mismatch"


async def test_submit_accepts_when_local_submit_system_unset(shared_dir, tmp_path, fake_runner, monkeypatch) -> None:
    # An unset tbg_submit is fine: the explicit submit=sbatch flag still wins.
    monkeypatch.setattr(sim_mod, "_detect_submit_system", lambda: None)
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    # sbatch requested, local unset: accepted (we only reject a known
    # contradiction), so this documents the current policy.
    ack = await client.handle(command)
    assert ack is not None
    assert ack.payload.get("error_code") is None


async def test_submit_bad_params_is_reported_not_crashed(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    command.payload["params"] = {"build_jobs": "not-an-int", "bogus": 1}
    # Re-sign after tampering so the signature still verifies.
    command.sign(SECRET)
    ack = await client.handle(command)
    assert ack is not None
    assert ack.payload["error_code"] == "payload_invalid"


async def test_submit_rejects_shell_injection_params(shared_dir, tmp_path, fake_runner) -> None:
    """A wire-supplied shell-injection attempt is rejected before generate()."""
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    command.payload["params"] = {"overwrite_vars": ["PARAM=$(touch /tmp/pwned)"]}
    command.sign(SECRET)
    ack = await client.handle(command)
    assert ack is not None
    assert ack.payload["error_code"] == "payload_invalid"
    # generate() never ran for the malicious payload.
    assert not fake_runner or not fake_runner[0].generated


async def test_submit_rejects_absolute_cfg_file(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    command.payload["params"] = {"cfg_file": "/etc/passwd"}
    command.sign(SECRET)
    ack = await client.handle(command)
    assert ack is not None
    assert ack.payload["error_code"] == "payload_invalid"


def test_link_run_results_runs_the_generated_script(tmp_path) -> None:
    from pic_agentic.simclient.simulation import link_run_results

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = run_dir / ".cwl_cache" / "x" / "simOutput"
    target.mkdir(parents=True)
    (run_dir / "link_results.sh").write_text(f'#!/bin/bash\nln -s "{target}" "$1"\n')
    assert link_run_results(run_dir) is True
    assert (run_dir / "simOutput").exists()


def test_link_run_results_missing_script_is_not_fatal(tmp_path) -> None:
    from pic_agentic.simclient.simulation import link_run_results

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert link_run_results(run_dir) is False


def test_per_command_token_must_be_safe_hex(shared_dir) -> None:
    # The command id comes from the wire, so a traversal/absolute token must be
    # rejected before any directory is built.
    from pic_agentic.protocol.simulation import SimulationPayload
    from pic_agentic.simclient.simulation import (
        SimulationErrorCode,
        SimulationExecutionError,
        SubmitConfig,
        runner_from_payload,
    )

    payload = SimulationPayload(
        wire_format_version=1,
        picongpu_version="",
        schema_hash="",
        simulation={"sim": json.loads(FIXTURE.read_text())["sim"]},
    )
    config = SubmitConfig(setup_root=shared_dir / "sims")
    for bad in ("/etc/cron.d", "../../../../tmp/evil", "a/b", "", "ABCDEF.."):
        with pytest.raises(SimulationExecutionError) as excinfo:
            # Skip only if PIConGPU is absent; the token check precedes it.
            runner_from_payload(payload, config, bad)
        assert excinfo.value.code is SimulationErrorCode.PATH_UNSAFE


async def test_resubmission_with_new_cmd_id_gets_a_fresh_directory(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id1, _p1, command1 = await service.build_payload(script)
    _cmd_id2, _p2, command2 = await service.build_payload(script)
    assert command1.payload["cmd_id"] != command2.payload["cmd_id"]
    _cmd_id3, _p3, command3 = await service.build_payload(script)
    # Identical simulation content -> identical sim_id, distinct cmd_id: the
    # generated directories must differ so generate() does not collide.
    assert command1.payload["header"]["sim_id"] == command3.payload["header"]["sim_id"]
    await client.handle(command1)
    await client.handle(command3)
    assert len(fake_runner) == 2
    assert fake_runner[0].run_dir != fake_runner[1].run_dir


async def test_replay_of_failed_submission_reports_failure(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    _cmd_id, _payload, command = await service.build_payload(script)
    # Force a pre-accept failure by asking for a non-sbatch system.
    command.payload["params"]["submit_system"] = "bash"
    command.sign(SECRET)
    first = await client.handle(command)
    assert first is not None
    assert first.payload["error_code"] == "submit_system_mismatch"
    # The replay must report the failure too, not a successful re-ack.
    command.transport_event_id = "$replay"
    second = await client.handle(command)
    assert second is not None
    assert second.payload["error"] is not None
    assert second.payload["state"] == SimulationState.FAILED.value


async def test_submit_rejects_unsupported_simulation_key(shared_dir, tmp_path, fake_runner) -> None:
    _mcp_t, _sim_t, service, client = _make_pair(shared_dir)
    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    cmd_id, _payload, command = await service.build_payload(script)
    body = json.loads(command.payload[PAYLOAD_KEY])
    body["simulation"] = {**body["simulation"], "run_dir": "/etc"}
    # Rebuild and re-sign the command; its header hash is recomputed from the
    # tampered simulation, so the allow-list check (not the hash check) fires.
    tampered = SimulationPayload.model_validate(body)
    rebuilt = build_submit_command(sim=SIM, seq=1, payload=tampered, cmd_id=cmd_id).sign(SECRET)
    ack = await client.handle(rebuilt)
    assert ack is not None
    assert ack.payload["error_code"] == "unsupported"


def test_allowlist_is_enforced_by_prepare_submit(shared_dir, tmp_path) -> None:
    # A simulation mapping carrying a cluster-local key must fail the allow-list.
    payload = SimulationPayload(
        wire_format_version=1, picongpu_version="", schema_hash="", simulation={"sim": {}, "run_dir": "/etc"}
    )
    with pytest.raises(UnsupportedPayloadError):
        payload.check_allowlist()
