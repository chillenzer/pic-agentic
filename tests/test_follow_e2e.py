# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""End-to-end M2b follow test over the in-memory transport and fake SLURM.

A submit drives the detached :class:`~pic_agentic.simclient.follow.JobFollower`
through ``PENDING`` -> ``RUNNING`` -> ``COMPLETED`` using the sequence-state
``scontrol`` double; the same transport pair then answers a ``status_request``
and a ``logs_request``.  No cluster or homeserver is involved.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from pic_agentic.protocol.simulation import (
    SimulationState,
    SimulationType,
    build_logs_command,
    build_status_command,
)
from pic_agentic.rcp import RcpMessage, new_secret_hex
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
#: State sequence the fake scontrol walks on successive invocations.
STATES = "PENDING\nRUNNING\nCOMPLETED 0\n"


def _setw(width: int, value: object) -> str:
    text = str(value)
    return " " * max(0, width - len(text)) + text


def _print_time(h: int, m: int, s: int, ms: int) -> str:
    if h > 0:
        return f"{h:2d}h {m:2d}min {s:2d}sec {ms:3d}msec"
    if m > 0:
        return f"{m:2d}min {s:2d}sec {ms:3d}msec"
    if s > 0:
        return f"{s:2d}sec {ms:3d}msec"
    return f"{ms:3d}msec"


def _progress_line(percent: int, step: int) -> str:
    return (
        _setw(3, percent)
        + " % = "
        + _setw(8, step)
        + " | time elapsed:"
        + _setw(25, _print_time(0, 0, 1, 0))
        + " | avg time per step: "
        + _print_time(0, 0, 1, 0)
    )


#: Small max_steps-style stdout: startup line plus the 25 % cadence.
STDOUT_TEXT = (
    "\n".join(
        [
            "PIConGPU: simulation startup",
            _progress_line(25, 250),
            _progress_line(50, 500),
            _progress_line(75, 750),
            _progress_line(100, 1000),
        ]
    )
    + "\n"
)
TOTAL_STDOUT_LINES = len(STDOUT_TEXT.splitlines())


def _runner_dump() -> dict:
    return json.loads(FIXTURE.read_text())


class _FakeRunner:
    """Stand-in for ``pypicongpu.Runner`` that mimics the follow-time artifacts."""

    def __init__(self, run_dir: Path, state_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.setup_dir = self.run_dir.parent / "input"
        self.state_dir = Path(state_dir)
        self.generated = False
        self.ran = False

    def generate(self, **_flags: object) -> None:
        self.generated = True

    def run(self) -> None:
        self.ran = True
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "submission_information.txt").write_text(f"Submitted batch job {JOB_ID}\n")
        # The sequence the watcher's scontrol calls will walk.
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / f"{JOB_ID}.state").write_text(STATES, encoding="utf-8")
        # cwltool's per-step cache stdout, where --chdir + -o stdout lands.
        cache = self.run_dir / ".cwl_cache" / "steps"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / "stdout").write_text(STDOUT_TEXT, encoding="utf-8")
        output = cache / "simOutput"
        output.mkdir(exist_ok=True)
        (self.run_dir / "link_results.sh").write_text(f'#!/bin/bash\nln -s "{output}" "$1"\n')


@pytest.fixture
def shared_dir(tmp_path, monkeypatch):
    state_dir = tmp_path / "fake-slurm"
    monkeypatch.setenv("FAKE_SLURM_STATE", str(state_dir))
    d = tmp_path / "shared"
    d.mkdir()
    return d, state_dir


@pytest.fixture
def fake_runner(monkeypatch):
    created: list[_FakeRunner] = []

    def fake_from_payload(payload, config, token):
        runner = _FakeRunner(
            config.setup_root / payload.sim_id / token / "run",
            state_dir=Path(os.environ["FAKE_SLURM_STATE"]),
        )
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


def _make_pair(shared: Path, *, poll_interval_s: float = 30.0):
    mcp_t, sim_t = MemoryTransport.create_pair()
    service = SubmitService(sim=SIM, secret=SECRET, runner_dump_builder=_fake_builder, ack_timeout_s=5.0)
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared,
        submit_config=SubmitConfig(setup_root=shared / "sims"),
        poll_interval_s=poll_interval_s,
        poll_max_interval_s=max(poll_interval_s, 0.02),
    )
    return mcp_t, sim_t, service, client


async def _wait_for(predicate, *, limit_s: float = 5.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + limit_s
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    pytest.fail("condition not met before timeout")


async def _collect_ack(mcp_t, received: list[RcpMessage], message, ack_type: SimulationType) -> RcpMessage:
    await mcp_t.send(message)
    await _wait_for(lambda: any(entry.type == ack_type for entry in received))
    return next(entry for entry in received if entry.type == ack_type)


async def test_follow_events_and_status_logs(shared_dir, tmp_path, fake_runner) -> None:
    mcp_t, sim_t, service, client = _make_pair(shared_dir[0], poll_interval_s=0.01)
    received: list[RcpMessage] = []

    async def pump() -> None:
        async for msg in mcp_t.receive():
            received.append(msg)
            service.on_message(msg)

    script = tmp_path / "picmi_script.py"
    script.write_text("# picmi\n")
    pump_task = asyncio.create_task(pump())
    serve_task = asyncio.create_task(client.serve())
    try:
        outcome = await service.submit(mcp_t.send, script)
        assert outcome.acked, outcome.error
        assert outcome.ok, outcome.error

        await _wait_for(
            lambda: (
                service.get(outcome.sim_id) is not None
                and service.get(outcome.sim_id).state == SimulationState.RESULTS_READY.value
            ),
        )
        events = [
            message
            for message in service.event_log
            if message.payload.get("sim_id") == outcome.sim_id and message.payload.get("cmd_id") == outcome.cmd_id
        ]
        states = [event.payload["state"] for event in events]

        # Coarse transitions, each once, in order.
        assert states.count(SimulationState.JOB_RUNNING.value) == 1
        assert states.count(SimulationState.JOB_FINISHED.value) == 1
        assert states.count(SimulationState.RESULTS_READY.value) == 1
        assert states.index(SimulationState.JOB_RUNNING.value) < states.index(SimulationState.JOB_FINISHED.value)
        # Bounded progress cadence.
        step_percents = [
            event.payload["percent"]
            for event in events
            if event.payload["state"] == SimulationState.STEP_FINISHED.value
        ]
        assert step_percents == [25, 50, 75, 100]
        assert (fake_runner[0].run_dir / "simOutput").exists()

        # The same pair now answers a live status pull and a logs pull.
        await _assert_pulls(mcp_t, received, outcome.sim_id)
    finally:
        serve_task.cancel()
        pump_task.cancel()
        # Await the cancelled tasks so ``serve``'s ``finally`` reaps the
        # follower's in-flight ``scontrol`` subprocess before the loop closes.
        await asyncio.gather(serve_task, pump_task, return_exceptions=True)
        await sim_t.close()
        await mcp_t.close()


async def _assert_pulls(mcp_t, received: list[RcpMessage], sim_id: str) -> None:
    status_cmd = build_status_command(sim=SIM, seq=100, sim_id=sim_id, cmd_id="status-1").sign(SECRET)
    status_ack = await _collect_ack(mcp_t, received, status_cmd, SimulationType.STATUS_ACK)
    assert status_ack.sender_role.value == "simclient"
    assert status_ack.payload["sim_id"] == sim_id
    assert status_ack.payload["job_id"] == JOB_ID
    assert status_ack.payload["slurm_state"] == "COMPLETED"
    assert status_ack.payload["state"] == SimulationState.RESULTS_READY.value
    assert status_ack.payload["percent"] == 100
    assert status_ack.payload["step"] == 1000
    assert status_ack.payload["exit_code"] == 0

    logs_cmd = build_logs_command(sim=SIM, seq=101, sim_id=sim_id, stream="stdout", tail=100, cmd_id="logs-1").sign(
        SECRET
    )
    logs_ack = await _collect_ack(mcp_t, received, logs_cmd, SimulationType.LOGS_ACK)
    assert logs_ack.payload["stream"] == "stdout"
    assert logs_ack.payload["total_lines"] == TOTAL_STDOUT_LINES
    assert logs_ack.payload["lines"][-1] == _progress_line(100, 1000)


async def test_same_sim_id_resubmission_cancels_previous_follower(shared_dir, tmp_path, fake_runner) -> None:
    """A second follower for one sim_id must cancel (and reap) the first (#1)."""
    shared, state_dir = shared_dir
    _mcp_t, sim_t, _service, client = _make_pair(shared)
    # A job that never terminates: the follower parks after its first poll.
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "4711.state").write_text("RUNNING\n", encoding="utf-8")
    client._serving = True
    try:
        await client._start_follower(
            cmd_id="cmd-a",
            sim_id="same1234",
            job_id=4711,
            run_dir=str(tmp_path),
            stdout_path=None,
            submit_system="sbatch",
        )
        first = client._follow_tasks["same1234"]
        await client._start_follower(
            cmd_id="cmd-b",
            sim_id="same1234",
            job_id=4711,
            run_dir=str(tmp_path),
            stdout_path=None,
            submit_system="sbatch",
        )
        second = client._follow_tasks["same1234"]
        assert second is not first
        # The superseded watcher was cancelled and awaited, not merely dropped.
        assert first.done()
        assert not second.done()
        assert [task for task in client._follow_tasks_all if not task.done()] == [second]
    finally:
        client._serving = False
        await client._cancel_followers()
        await sim_t.close()
    assert not client._follow_tasks
    assert not client._follow_tasks_all


async def test_get_logs_reads_only_a_bounded_suffix(shared_dir, tmp_path, monkeypatch) -> None:
    """``get_logs`` must not slurp the whole file (#4)."""
    from pic_agentic.simclient import client as client_mod
    from pic_agentic.simclient.follow import TrackedSim

    monkeypatch.setattr(client_mod, "_MAX_LOG_READ_BYTES", 64)
    monkeypatch.setattr(client_mod, "_ASSUMED_MAX_LINE_BYTES", 1024)
    shared, _state_dir = shared_dir
    _mcp_t, sim_t = MemoryTransport.create_pair()
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared,
        submit_config=SubmitConfig(setup_root=shared / "sims"),
    )
    run = tmp_path / "run"
    run.mkdir()
    stdout = run / "stdout"
    lines = [f"log-line-{index:03d}" for index in range(100)]
    stdout.write_text("\n".join(lines) + "\n", encoding="utf-8")
    client._tracked["sim12345"] = TrackedSim(
        sim_id="sim12345",
        cmd_id="c",
        job_id=1,
        run_dir=str(run),
        stdout_path=str(stdout),
        submit_system="sbatch",
    )
    ack = await client.handle(
        build_logs_command(sim=SIM, seq=1, sim_id="sim12345", stream="stdout", tail=3, cmd_id="l").sign(SECRET)
    )
    assert ack is not None
    assert ack.payload["lines"] == lines[-3:]
    # Only the bounded suffix was counted: a whole-file read would report 100.
    assert ack.payload["total_lines"] < len(lines)


async def test_status_unknown_sim_is_ack_error(shared_dir, tmp_path, fake_runner) -> None:
    shared, _state_dir = shared_dir
    _mcp_t, sim_t = MemoryTransport.create_pair()
    client = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared,
        submit_config=SubmitConfig(setup_root=shared / "sims"),
    )
    status_cmd = build_status_command(sim=SIM, seq=1, sim_id="deadbeef", cmd_id="s").sign(SECRET)
    ack = await client.handle(status_cmd)
    assert ack is not None
    assert ack.payload["error_code"] == "unknown_sim"
    assert ack.payload["error"] == "unknown_sim"

    logs_cmd = build_logs_command(sim=SIM, seq=2, sim_id="deadbeef", cmd_id="l").sign(SECRET)
    logs_ack = await client.handle(logs_cmd)
    assert logs_ack is not None
    assert logs_ack.payload["error_code"] == "unknown_sim"
    assert logs_ack.payload["lines"] == []
    assert logs_ack.payload["total_lines"] == 0


async def test_pull_rejected_when_submit_disabled(shared_dir, tmp_path) -> None:
    shared, _state_dir = shared_dir
    _mcp_t, sim_t = MemoryTransport.create_pair()
    disabled = SimClient(
        sim=SIM,
        secret=SECRET,
        transport=sim_t,
        slurm=SlurmClient(bin_dir=str(FAKE_BIN)),
        message_dir=shared,
    )
    ack = await disabled.handle(build_status_command(sim=SIM, seq=1, sim_id="deadbeef").sign(SECRET))
    assert ack is not None
    assert ack.type == SimulationType.STATUS_ACK
    assert ack.payload["error_code"] == "rejected_by_policy"

    logs_ack = await disabled.handle(build_logs_command(sim=SIM, seq=2, sim_id="deadbeef").sign(SECRET))
    assert logs_ack is not None
    assert logs_ack.type == SimulationType.LOGS_ACK
    assert logs_ack.payload["error_code"] == "rejected_by_policy"
