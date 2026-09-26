# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

"""Unit tests for the transport-agnostic :class:`JobFollower`.

The follower is driven directly with a fake async ``job_info`` and a real
progress file, so no cluster, transport or room is involved.  Intervals are
tiny (10/20 ms) to keep the suite fast.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pic_agentic.protocol.simulation import SimulationState
from pic_agentic.simclient.follow import JobFollower, TrackedSim
from pic_agentic.slurm import JobInfo, SlurmJobState


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


def _progress_line(percent: int, step: int, avg_ms: int) -> str:
    avg = _print_time(0, 0, avg_ms // 1000, avg_ms % 1000)
    elapsed = _print_time(0, 0, 1, 0)
    return (
        _setw(3, percent)
        + " % = "
        + _setw(8, step)
        + " | time elapsed:"
        + _setw(25, elapsed)
        + " | avg time per step: "
        + avg
    )


def _running_sequence(*states: SlurmJobState, exit_code: int = 0):
    """Return a ``job_info`` callable repeating the last state forever."""
    calls = {"n": 0}

    async def job_info(_job_id: int) -> JobInfo:
        index = min(calls["n"], len(states) - 1)
        calls["n"] += 1
        state = states[index]
        code = exit_code if state.terminal else None
        return JobInfo(job_id=4711, state=state, exit_code=code)

    return job_info


def _tracked(run_dir: Path, *, stdout_path: Path | None, job_id: int | None = 4711) -> TrackedSim:
    return TrackedSim(
        sim_id="abc12345",
        cmd_id="cmd-1",
        job_id=job_id,
        run_dir=str(run_dir),
        stdout_path=str(stdout_path) if stdout_path else None,
        submit_system="sbatch",
    )


def _collector():
    events: list[tuple[SimulationState, dict]] = []

    async def emit(state: SimulationState, *, job_id: int | None = None, **fields: object) -> None:
        events.append((state, {"job_id": job_id, **fields}))

    return events, emit


async def _run_to_terminal(follower: JobFollower) -> list[tuple[SimulationState, dict]]:
    events, emit = _collector()
    follower.emit = emit
    await asyncio.wait_for(follower.run(), timeout=3.0)
    return events


async def test_running_progress_terminal_and_results(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = run_dir / ".cwl_cache" / "x" / "simOutput"
    output.mkdir(parents=True)
    (run_dir / "link_results.sh").write_text(f'#!/bin/bash\nln -s "{output}" "$1"\n')
    stdout = tmp_path / "stdout"
    stdout.write_text(
        "\n".join(
            [
                "some startup log",
                _progress_line(10, 100, 1000),
                _progress_line(25, 250, 1000),
                _progress_line(50, 500, 1000),
                _progress_line(75, 750, 1000),
                _progress_line(100, 1000, 1000),
            ]
        )
        + "\n"
    )

    follower = JobFollower(
        sim="abc12345",
        emit=None,  # type: ignore[arg-type]
        tracked=_tracked(run_dir, stdout_path=stdout),
        job_info=_running_sequence(SlurmJobState.PENDING, SlurmJobState.RUNNING, SlurmJobState.COMPLETED),
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    events = await _run_to_terminal(follower)
    states = [state for state, _ in events]

    assert states.count(SimulationState.JOB_RUNNING) == 1
    step_percents = [fields["percent"] for state, fields in events if state is SimulationState.STEP_FINISHED]
    assert step_percents == [25, 50, 75, 100]
    assert states[-2:] == [SimulationState.JOB_FINISHED, SimulationState.RESULTS_READY]
    assert (run_dir / "simOutput").exists()
    # The ETA is derived from the average step time (1000 ms/step, 1000 steps).
    first_step = next(fields for state, fields in events if state is SimulationState.STEP_FINISHED)
    assert first_step["step"] == 250
    assert first_step["eta_s"] == 750
    assert first_step["walltime"] == _print_time(0, 0, 1, 0).strip()
    assert first_step["avg_per_step"] == _print_time(0, 0, 1, 0).strip()


async def test_running_is_emitted_only_once(tmp_path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text("")
    follower = JobFollower(
        sim="abc12345",
        emit=None,  # type: ignore[arg-type]
        tracked=_tracked(tmp_path, stdout_path=stdout),
        job_info=_running_sequence(SlurmJobState.RUNNING, SlurmJobState.RUNNING, SlurmJobState.COMPLETED),
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    events = await _run_to_terminal(follower)
    assert [state for state, _ in events].count(SimulationState.JOB_RUNNING) == 1


async def test_failed_job_reports_exit_code(tmp_path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text("")
    follower = JobFollower(
        sim="abc12345",
        emit=None,  # type: ignore[arg-type]
        tracked=_tracked(tmp_path, stdout_path=stdout),
        job_info=_running_sequence(SlurmJobState.RUNNING, SlurmJobState.FAILED, exit_code=3),
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    events = await _run_to_terminal(follower)
    states = [state for state, _ in events]
    assert SimulationState.JOB_FINISHED not in states
    assert SimulationState.RESULTS_READY not in states
    assert states[-1] is SimulationState.JOB_FAILED
    failed = events[-1][1]
    assert failed["slurm_state"] == "FAILED"
    assert failed["exit_code"] == 3


async def test_nonzero_exit_on_completed_is_failed(tmp_path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_text("")
    follower = JobFollower(
        sim="abc12345",
        emit=None,  # type: ignore[arg-type]
        tracked=_tracked(tmp_path, stdout_path=stdout),
        job_info=_running_sequence(SlurmJobState.RUNNING, SlurmJobState.COMPLETED, exit_code=1),
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    events = await _run_to_terminal(follower)
    assert [state for state, _ in events][-1] is SimulationState.JOB_FAILED


async def test_malformed_stdout_is_tolerated(tmp_path) -> None:
    stdout = tmp_path / "stdout"
    stdout.write_bytes(
        b"\xff\xfe not a progress line\npartial line without newline" + _progress_line(50, 5, 0).encode(),
    )
    follower = JobFollower(
        sim="abc12345",
        emit=None,  # type: ignore[arg-type]
        tracked=_tracked(tmp_path, stdout_path=stdout),
        job_info=_running_sequence(SlurmJobState.RUNNING, SlurmJobState.COMPLETED),
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    events = await _run_to_terminal(follower)
    # The malformed bytes do not crash the loop and the terminal transition
    # still fires; a partial final line without a newline is not consumed.
    # The link cannot run (no ``link_results.sh``), so no RESULTS_READY is
    # claimed: the run stops at job_finished with results_linked=False.
    assert [state for state, _ in events][-1] is SimulationState.JOB_FINISHED
    assert SimulationState.RESULTS_READY not in [state for state, _ in events]
    assert events[-1][1]["results_linked"] is False
    assert SimulationState.STEP_FINISHED not in [state for state, _ in events]


async def test_missing_stdout_file_is_tolerated(tmp_path) -> None:
    follower = JobFollower(
        sim="abc12345",
        emit=None,  # type: ignore[arg-type]
        tracked=_tracked(tmp_path, stdout_path=tmp_path / "does-not-exist"),
        job_info=_running_sequence(SlurmJobState.RUNNING, SlurmJobState.COMPLETED),
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    events = await _run_to_terminal(follower)
    # No ``link_results.sh`` in ``tmp_path``: the link fails, so results are not
    # claimed.
    assert [state for state, _ in events][-1] is SimulationState.JOB_FINISHED
    assert events[-1][1]["results_linked"] is False


async def test_job_info_failure_is_tolerated(tmp_path) -> None:
    calls = {"n": 0}

    async def flaky(_job_id: int) -> JobInfo:
        calls["n"] += 1
        if calls["n"] == 1:
            message = "transient scontrol failure"
            raise RuntimeError(message)
        return JobInfo(job_id=4711, state=SlurmJobState.COMPLETED, exit_code=0)

    follower = JobFollower(
        sim="abc12345",
        emit=None,  # type: ignore[arg-type]
        tracked=_tracked(tmp_path, stdout_path=tmp_path / "stdout"),
        job_info=flaky,
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    events = await _run_to_terminal(follower)
    assert calls["n"] >= 2
    # No link script here, so the terminal state is job_finished.
    assert [state for state, _ in events][-1] is SimulationState.JOB_FINISHED


async def test_no_job_id_stops_immediately(tmp_path) -> None:
    events, emit = _collector()
    follower = JobFollower(
        sim="abc12345",
        emit=emit,
        tracked=_tracked(tmp_path, stdout_path=None, job_id=None),
        job_info=_running_sequence(SlurmJobState.RUNNING),
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    await asyncio.wait_for(follower.run(), timeout=1.0)
    assert events == []


async def test_failed_link_emits_job_finished_not_results_ready(tmp_path) -> None:
    """A terminal job whose link is missing stays at job_finished (#2)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stdout = tmp_path / "stdout"
    stdout.write_text("")
    follower = JobFollower(
        sim="abc12345",
        emit=None,  # type: ignore[arg-type]
        tracked=_tracked(run_dir, stdout_path=stdout),
        job_info=_running_sequence(SlurmJobState.RUNNING, SlurmJobState.COMPLETED),
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    events = await _run_to_terminal(follower)
    states = [state for state, _ in events]
    assert SimulationState.RESULTS_READY not in states
    assert states[-1] is SimulationState.JOB_FINISHED
    # A second job_finished carries the explicit ``results_linked=False``.
    finished = [fields for state, fields in events if state is SimulationState.JOB_FINISHED]
    assert finished[-1]["results_linked"] is False


async def test_missing_link_emits_results_ready_once_link_exists(tmp_path) -> None:
    """A present link still yields results_ready (#2, positive branch)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    output = run_dir / ".cwl_cache" / "x" / "simOutput"
    output.mkdir(parents=True)
    (run_dir / "link_results.sh").write_text(f'#!/bin/bash\nln -s "{output}" "$1"\n')
    stdout = tmp_path / "stdout"
    stdout.write_text("")
    follower = JobFollower(
        sim="abc12345",
        emit=None,  # type: ignore[arg-type]
        tracked=_tracked(run_dir, stdout_path=stdout),
        job_info=_running_sequence(SlurmJobState.RUNNING, SlurmJobState.COMPLETED),
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    events = await _run_to_terminal(follower)
    states = [state for state, _ in events]
    assert states[-2:] == [SimulationState.JOB_FINISHED, SimulationState.RESULTS_READY]
    assert events[-1][1]["results_linked"] is True


async def test_stop_ends_the_loop(tmp_path) -> None:
    async def never_terminal(_job_id: int) -> JobInfo:
        return JobInfo(job_id=4711, state=SlurmJobState.RUNNING, exit_code=None)

    follower = JobFollower(
        sim="abc12345",
        emit=None,  # type: ignore[arg-type]
        tracked=_tracked(tmp_path, stdout_path=tmp_path / "stdout"),
        job_info=never_terminal,
        initial_interval_s=0.01,
        max_interval_s=0.02,
    )
    task = asyncio.create_task(follower.run())
    await asyncio.sleep(0.05)
    follower.stop()
    await asyncio.wait_for(task, timeout=1.0)
    assert task.done()
