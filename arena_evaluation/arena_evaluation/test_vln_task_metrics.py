import csv
import json

import pytest
import yaml
from PIL import Image

from arena_evaluation.vln_task_metrics import _read_odom, generate_vln_task_metrics


def _write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_world(tmp_path, map_pixels=None):
    repo = tmp_path / "repo"
    world = repo / "arena_simulation_setup" / "worlds" / "grscenes_test"
    scenario_dir = world / "scenarios" / "default"
    map_dir = world / "map"
    scenario_dir.mkdir(parents=True)
    map_dir.mkdir(parents=True)
    (scenario_dir / "scenario.yaml").write_text(
        yaml.safe_dump({"robots": [{"start": [0.0, 0.0, 0.0], "goal": [1.0, 0.0, 0.0]}], "dynamic": []}),
        encoding="utf-8",
    )
    (map_dir / "map.yaml").write_text(
        yaml.safe_dump(
            {
                "image": "map.png",
                "resolution": 0.1,
                "origin": [0.0, 0.0, 0.0],
                "occupied_thresh": 0.65,
                "free_thresh": 0.196,
                "negate": 0,
            }
        ),
        encoding="utf-8",
    )
    image = Image.new("L", (20, 20), color=255)
    if map_pixels:
        for pixel in map_pixels:
            image.putpixel(pixel, 0)
    image.save(map_dir / "map.png")
    return repo


def _make_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump({"parameters": {"world": "grscenes_test", "scenario_file": "default", "timeout": 120.0}}),
        encoding="utf-8",
    )
    _write_rows(
        run_dir / "start_goal.csv",
        ["episode", "start", "goal"],
        [{"episode": "0", "start": "[0.0, 0.0, 0.0]", "goal": "[1.0, 0.0, 0.0]"}],
    )
    _write_rows(run_dir / "metrics.csv", ["result"], [{"result": "GOAL_REACHED"}])
    return run_dir


def test_vln_task_metrics_goal_success(monkeypatch, tmp_path):
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _write_rows(
        run_dir / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": "10000000000", "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [0.1, 0.0, 0.0]}"},
        ],
    )
    _write_rows(run_dir / "cmd_vel.csv", ["time", "data"], [{"time": "0", "data": "[0.0, 0.0, 0.0]"}])

    result = generate_vln_task_metrics(run_dir, thresholds={"robot_radius_m": 0.0})

    assert result["goal"]["goal_reached"] is True
    assert result["strict_task_success"] is True
    assert result["vln"]["spl"] == pytest.approx(1.0)
    assert result["language_task_contract"]["contract_type"] == "native_scenario_goal"
    assert "goal_reached(robot, native_scenario_goal)" in result["language_task_contract"]["evaluated_predicates"]
    assert "bddl_semantic_predicates" in result["language_task_contract"]["unsupported_predicates"]
    assert result["language_task_contract"]["bddl_evaluator"]["used_for_this_score"] is False
    assert (run_dir / "vln_task_metrics.json").exists()


def test_vln_task_metrics_prefers_instruction_file(monkeypatch, tmp_path):
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    instruction_file = run_dir / "instruction.txt"
    instruction_file.write_text("Turn left and stop at the desk.", encoding="utf-8")
    (run_dir / "run_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "parameters": {
                    "world": "grscenes_test",
                    "scenario_file": "default",
                    "timeout": 120.0,
                    "vln_instruction": "navigate",
                    "vln_instruction_file": str(instruction_file),
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _write_rows(
        run_dir / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": "10000000000", "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [0.1, 0.0, 0.0]}"},
        ],
    )
    _write_rows(run_dir / "cmd_vel.csv", ["time", "data"], [])

    result = generate_vln_task_metrics(run_dir, thresholds={"robot_radius_m": 0.0})

    assert result["instruction"] == "Turn left and stop at the desk."
    assert result["scenario_contract"]["manifest_instruction"] == "navigate"
    assert result["language_task_contract"]["instruction_source"] == "run_manifest.vln_instruction_file"
    assert result["language_task_contract"]["instruction_file"] == str(instruction_file)


def test_vln_task_metrics_fails_stale_start_goal(monkeypatch, tmp_path):
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _write_rows(
        run_dir / "start_goal.csv",
        ["episode", "start", "goal"],
        [{"episode": "0", "start": "[0.0, 0.0, 0.0]", "goal": "[0.0, 0.0, 0.0]"}],
    )
    _write_rows(
        run_dir / "odom.csv",
        ["time", "data"],
        [{"time": "0", "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"}],
    )
    _write_rows(run_dir / "cmd_vel.csv", ["time", "data"], [])

    result = generate_vln_task_metrics(run_dir, thresholds={"robot_radius_m": 0.0})

    assert result["start_goal_consistency"]["pass"] is False
    assert "start_goal_inconsistent" in result["strict_task_failure_reasons"]


def test_vln_task_metrics_detects_commanded_stuck(monkeypatch, tmp_path):
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _write_rows(
        run_dir / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": "100000000000", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
        ],
    )
    _write_rows(run_dir / "cmd_vel.csv", ["time", "data"], [{"time": "0", "data": "[0.2, 0.0, 0.0]"}])

    result = generate_vln_task_metrics(run_dir, thresholds={"robot_radius_m": 0.0, "stuck_min_duration_sec": 5.0})

    assert result["commanded_stuck"]["commanded_stuck_time_sec"] == pytest.approx(10.0)
    assert "commanded_stuck" in result["strict_task_failure_reasons"]


def test_vln_task_metrics_detects_map_occupancy(monkeypatch, tmp_path):
    repo = _make_world(tmp_path, map_pixels=[(0, 19)])
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _write_rows(
        run_dir / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.05, 0.05, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": "10000000000", "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
        ],
    )
    _write_rows(run_dir / "cmd_vel.csv", ["time", "data"], [])

    result = generate_vln_task_metrics(run_dir, thresholds={"robot_radius_m": 0.0})

    assert result["static_occupancy"]["collision_sample_count"] == 1
    assert "static_occupancy_collision" in result["strict_task_failure_reasons"]


def test_vln_task_metrics_marks_timeout_when_duration_reaches_manifest_timeout(monkeypatch, tmp_path):
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _write_rows(
        run_dir / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.1, 0.0, 0.0]}"},
            {"time": "1200000000000", "data": "{'position': [0.2, 0.0, 0.0], 'velocity': [0.1, 0.0, 0.0]}"},
        ],
    )
    _write_rows(run_dir / "cmd_vel.csv", ["time", "data"], [])

    result = generate_vln_task_metrics(run_dir, thresholds={"robot_radius_m": 0.0})

    assert result["episode_timing"]["duration_sec"] == pytest.approx(120.0)
    assert result["episode_timing"]["timed_out"] is True
    assert "episode_timeout" in result["strict_task_failure_reasons"]
    assert "goal_not_reached" in result["strict_task_failure_reasons"]


# --------------------------------------------------------------------------
# Time segments.
#
# `odom.csv` and `cmd_vel.csv` are each several concatenated time segments: while
# /clock stalls during scene load the recorder fabricates timestamps, then adopts
# the lower resumed value as a new baseline and keeps appending to the same file.
# `time` is therefore an ordering key only WITHIN a segment, and the previous
# `sorted(samples, key=time)` was a shuffle that inflated
# `vln.trajectory_length_m` -- by up to x41 in the delivered runs.
#
# The segment count is not fixed; it tracks how many times /clock stalls.  None
# of these fixtures assumes one.
# --------------------------------------------------------------------------

_SEC = 10_000_000_000  # RECORDER_TIME_UNITS_PER_SECOND


def _odom_row(t_sec, x):
    return {"time": str(int(t_sec * _SEC)), "data": f"{{'position': [{x}, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}}"}


def _spliced_odom(tmp_path):
    """Two time segments whose `time` ranges interleave.

    Rows 0-1 are a pre-episode segment that ran ahead to 2.0-3.0 s and never
    moved.  Rows 2-4 are the episode, restarting at 0.5 s and travelling 0.2 m.
    A global `time` sort puts the stale pair AFTER the episode rows and charges
    the polyline for travelling back to x=0.0.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            _odom_row(2.0, 0.0),
            _odom_row(3.0, 0.0),
            _odom_row(0.5, 0.0),
            _odom_row(1.0, 0.1),
            _odom_row(1.5, 0.2),
        ],
    )
    _write_rows(tmp_path / "cmd_vel.csv", ["time", "data"], [])


def test_read_odom_preserves_file_order_and_tags_recording_segments(monkeypatch, tmp_path):
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _spliced_odom(run_dir)

    samples = _read_odom(run_dir)

    # file order, not `time` order.  Pre-fix this was [0.5, 1.0, 1.5, 2.0, 3.0].
    assert [round(s["time"] / _SEC, 3) for s in samples] == [2.0, 3.0, 0.5, 1.0, 1.5]
    assert [s["order"] for s in samples] == [0, 1, 2, 3, 4]
    assert [s["segment"] for s in samples] == [0, 0, 1, 1, 1]


def test_trajectory_length_excludes_backwards_clock_segment_splice(monkeypatch, tmp_path):
    """The defect itself, on hand-derived numbers.

    Segment 0 never moves, so it contributes 0.0 m.  Segment 1 travels
    0.0 -> 0.1 -> 0.2, so it contributes 0.2 m.  The correct total is 0.2 m.

    Pre-fix the `time` sort produced the order
    [x=0.0, 0.1, 0.2, 0.0, 0.0] and charged 0.1 + 0.1 + 0.2 + 0.0 = 0.4 m, i.e.
    exactly double, because the polyline had to travel back from the episode's
    end to the stale pre-episode position.
    """
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _spliced_odom(run_dir)

    result = generate_vln_task_metrics(run_dir, thresholds={"robot_radius_m": 0.0})

    # numeric first: pre-fix 0.4
    assert result["vln"]["trajectory_length_m"] == pytest.approx(0.2)
    # `spl` short-circuits on goal_reached=False, so it cannot expose the defect
    assert result["goal"]["goal_reached"] is False
    assert result["vln"]["spl"] == pytest.approx(0.0)
    assert result["sample_counts"]["odom_recording_segments"] == 2
    assert result["sample_counts"]["odom_segment_boundaries_skipped"] == 1


def test_large_teleport_not_reported_across_segment_boundary(monkeypatch, tmp_path):
    """A backwards /clock step is not a teleport and must not fail the run.

    The only step above the 1.0 m threshold used here is the segment boundary
    itself, which isolates teleport semantics from the path arithmetic.
    Pre-fix: trajectory 2.0 m, one teleport, and `large_teleport` among the
    strict-task failure reasons.
    """
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _write_rows(
        tmp_path / "run" / "odom.csv",
        ["time", "data"],
        [
            # segment 0 -- pre-episode, already at 2.0-3.0 s
            _odom_row(2.0, 0.0),
            _odom_row(3.0, 0.1),
            # segment 1 -- episode restarts at 0.5 s, 1.8 m from the stale position
            _odom_row(0.5, 1.9),
            _odom_row(1.0, 1.8),
        ],
    )
    _write_rows(run_dir / "cmd_vel.csv", ["time", "data"], [])

    result = generate_vln_task_metrics(
        run_dir, thresholds={"robot_radius_m": 0.0, "large_teleport_threshold_m": 1.0}
    )

    # numeric first: pre-fix 2.0
    assert result["vln"]["trajectory_length_m"] == pytest.approx(0.2)
    assert result["large_teleports"] == []
    assert "large_teleport" not in result["strict_task_failure_reasons"]


def test_commanded_stuck_interval_does_not_span_a_segment_boundary(monkeypatch, tmp_path):
    """A stuck interval must not start in one time segment and end in another.

    Hand-derived.  Segment 0 is stuck from 100 s to 120 s; segment 1 is stuck from
    5 s to 25 s.  The recording therefore contains 20 + 20 = 40 s of samples, and
    nothing at all between 25 s and 100 s.

    Pre-fix, the global `time` sort made the order 5, 25, 100, 120 look like one
    continuous stationary run, so a single interval was reported from 5 s to 120 s
    and charged as 115 s -- crediting 75 s of "stuck" time for which no sample
    exists.  It also produced an interval whose endpoints came from two different
    segments, making its length meaningless.

    After the fix each segment yields its own interval and the total is 40 s.  The
    two segments do not nest here, deliberately: when a pre-episode segment lies
    entirely inside the episode segment both implementations happen to agree on
    the total, so a nesting fixture could not distinguish them numerically.
    """
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _write_rows(
        run_dir / "odom.csv",
        ["time", "data"],
        [
            # segment 0 -- fabricated stamps ran ahead to 100-120 s
            _odom_row(100.0, 0.0),
            _odom_row(120.0, 0.0),
            # segment 1 -- /clock resumed at 5 s and reached 25 s
            _odom_row(5.0, 0.0),
            _odom_row(25.0, 0.0),
        ],
    )
    _write_rows(run_dir / "cmd_vel.csv", ["time", "data"], [{"time": "0", "data": "[0.2, 0.0, 0.0]"}])

    result = generate_vln_task_metrics(run_dir, thresholds={"robot_radius_m": 0.0})
    stuck = result["commanded_stuck"]

    # numeric first: pre-fix 115.0 from a single boundary-spanning interval
    assert stuck["commanded_stuck_time_sec"] == pytest.approx(40.0)
    assert len(stuck["commanded_stuck_intervals"]) == 2
    first, second = stuck["commanded_stuck_intervals"]
    assert (first["start_sec"], first["end_sec"]) == pytest.approx((100.0, 120.0))
    assert (second["start_sec"], second["end_sec"]) == pytest.approx((5.0, 25.0))
    # every interval lies within one segment, so no duration is fabricated across
    # a stretch the recorder never sampled
    assert stuck["commanded_stuck_time_sec"] == pytest.approx(
        sum(i["end_sec"] - i["start_sec"] for i in stuck["commanded_stuck_intervals"])
    )


def test_commanded_stuck_time_superseded_when_segments_overlap(monkeypatch, tmp_path):
    """A stuck instant two overlapping segments both report is credited once.

    Hand-derived.  Segment 0 is stuck 2-3 s, entirely inside segment 1's 0.5-20 s
    range, so supersession credits it 0 s and the total is segment 1's 19.5 s.

    This is a GUARD on the fix's arithmetic, not a demonstration of the pre-fix
    defect: because the segments nest, the pre-fix global sort also produced 19.5
    s, and only the interval structure differs (1 interval vs 2).  Without
    supersession the two per-segment intervals would sum to 20.5 s from a 19.5 s
    recording, so `credited_duration_sec` is asserted explicitly.
    """
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _write_rows(
        run_dir / "odom.csv",
        ["time", "data"],
        [
            _odom_row(2.0, 0.0),
            _odom_row(3.0, 0.0),
            _odom_row(0.5, 0.0),
            _odom_row(20.0, 0.0),
        ],
    )
    _write_rows(run_dir / "cmd_vel.csv", ["time", "data"], [{"time": "0", "data": "[0.2, 0.0, 0.0]"}])

    result = generate_vln_task_metrics(run_dir, thresholds={"robot_radius_m": 0.0, "stuck_min_duration_sec": 0.5})
    stuck = result["commanded_stuck"]

    assert stuck["commanded_stuck_time_sec"] == pytest.approx(19.5)
    intervals = stuck["commanded_stuck_intervals"]
    assert len(intervals) == 2
    assert intervals[0]["duration_sec"] == pytest.approx(1.0)
    assert intervals[0]["credited_duration_sec"] == pytest.approx(0.0)
    assert intervals[1]["credited_duration_sec"] == pytest.approx(19.5)
    assert stuck["commanded_stuck_time_sec"] <= (20.0 - 0.5) + 1e-9


def test_trajectory_length_unchanged_when_odom_time_order_is_already_clean(monkeypatch, tmp_path):
    """No-op guard: a single-segment, monotonic odom.csv keeps its previous values.

    Every assertion here holds on the pre-fix code as well, which is the point --
    the change must be inert when the recorder produced one clean time segment.
    It deliberately asserts no new diagnostic key so it stays runnable on both.
    """
    repo = _make_world(tmp_path)
    run_dir = _make_run(tmp_path)
    monkeypatch.setenv("ARENA_SOURCE_DIR", str(repo))
    _write_rows(
        run_dir / "odom.csv",
        ["time", "data"],
        [_odom_row(0.0, 0.0), _odom_row(1.0, 0.3), _odom_row(2.0, 0.7)],
    )
    _write_rows(run_dir / "cmd_vel.csv", ["time", "data"], [{"time": "0", "data": "[0.0, 0.0, 0.0]"}])

    result = generate_vln_task_metrics(run_dir, thresholds={"robot_radius_m": 0.0})

    assert result["vln"]["trajectory_length_m"] == pytest.approx(0.7)
    assert result["sample_counts"]["odom"] == 3
    assert result["episode_timing"]["duration_sec"] == pytest.approx(2.0)
    assert result["goal"]["final_xy"] == pytest.approx([0.7, 0.0])
    assert result["large_teleports"] == []
    assert result["commanded_stuck"]["commanded_stuck_time_sec"] == pytest.approx(0.0)
