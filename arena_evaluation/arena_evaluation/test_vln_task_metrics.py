import csv
import json

import pytest
import yaml
from PIL import Image

from arena_evaluation.vln_task_metrics import generate_vln_task_metrics


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
