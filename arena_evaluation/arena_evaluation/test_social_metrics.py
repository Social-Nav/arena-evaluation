import csv
import json

import pytest

from arena_evaluation.social_metrics import generate_social_metrics


def _write_rows(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_social_metrics_reads_pedsim_agents_data(tmp_path):
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": "10000000000", "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [1.0, 0.0, 0.0]}"},
        ],
    )
    _write_rows(
        tmp_path / "pedsim_agents_data.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "[{'id': '1', 'position': [4.0, 0.0]}]"},
            {"time": "10000000000", "data": "[{'id': '1', 'position': [4.0, 0.0]}]"},
        ],
    )

    result = generate_social_metrics(tmp_path)

    assert result["human_source_csv"] == "pedsim_agents_data.csv"
    assert result["humans_present"] is True
    assert result["human_sample_count"] == 2
    assert result["odom_sample_count"] == 2
    assert result["max_humans_observed"] == 1
    assert result["path_length_m"] == 1.0
    assert result["near_miss_count"] == 0
    assert result["human_collision_count"] == 0
    assert result["social_success"] is True
    assert json.loads((tmp_path / "social_metrics.json").read_text(encoding="utf-8"))["humans_present"] is True


def test_social_metrics_missing_humans_still_writes_json(tmp_path):
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": "10000000000", "data": "{'position': [0.5, 0.0, 0.0], 'velocity': [0.5, 0.0, 0.0]}"},
        ],
    )

    result = generate_social_metrics(tmp_path)

    assert result["humans_present"] is False
    assert result["human_sample_count"] == 0
    assert result["max_humans_observed"] == 0
    assert result["social_success"] is False
    assert (tmp_path / "social_metrics.json").exists()


def test_social_metrics_reports_dynamic_scene_overlap(tmp_path):
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
            {"time": "10000000000", "data": "{'position': [0.2, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
            {"time": "20000000000", "data": "{'position': [0.4, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
            {"time": "30000000000", "data": "{'position': [0.6, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
        ],
    )
    _write_rows(
        tmp_path / "human_states.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "[{'id': '1', 'position': [1.0, 0.0]}]"},
            {"time": "10000000000", "data": "[{'id': '1', 'position': [1.2, 0.0]}]"},
            {"time": "20000000000", "data": "[{'id': '1', 'position': [1.4, 0.0]}]"},
            {"time": "30000000000", "data": "[{'id': '1', 'position': [1.6, 0.0]}]"},
        ],
    )

    result = generate_social_metrics(
        tmp_path,
        thresholds={
            "min_human_motion_time_sec": 2.0,
            "min_human_robot_motion_overlap_time_sec": 2.0,
            "min_human_robot_interaction_time_sec": 2.0,
        },
    )

    assert result["human_source_csv"] == "human_states.csv"
    assert result["moving_human_count"] == 1
    assert result["human_motion_total_m"] == pytest.approx(0.6)
    assert result["human_motion_time_sec"] == 3.0
    assert result["robot_motion_time_sec"] == 3.0
    assert result["human_robot_motion_overlap_time_sec"] == 3.0
    assert result["human_robot_interaction_time_sec"] == 3.0
    assert result["dynamic_scene_success"] is True
    assert result["time_scale"]["raw_units_per_second"] == 10000000000.0


def test_social_metrics_fails_dynamic_scene_when_humans_are_static(tmp_path):
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
            {"time": "10000000000", "data": "{'position': [0.2, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
        ],
    )
    _write_rows(
        tmp_path / "human_states.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "[{'id': '1', 'position': [1.0, 0.0]}]"},
            {"time": "10000000000", "data": "[{'id': '1', 'position': [1.0, 0.0]}]"},
        ],
    )

    result = generate_social_metrics(tmp_path)

    assert result["humans_present"] is True
    assert result["moving_human_count"] == 0
    assert result["human_motion_time_sec"] == 0.0
    assert result["human_robot_motion_overlap_time_sec"] == 0.0
    assert result["dynamic_scene_success"] is False


def test_social_metrics_reports_strict_footprint_collision(tmp_path):
    (tmp_path / "vln_task_metrics.json").write_text(
        json.dumps(
            {
                "strict_task_success": True,
                "strict_task_failure_reasons": [],
                "static_occupancy": {"collision_sample_count": 0, "intervals": []},
                "commanded_stuck": {"commanded_stuck_time_sec": 0.0, "commanded_stuck_intervals": []},
            }
        ),
        encoding="utf-8",
    )
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
            {"time": "10000000000", "data": "{'position': [0.2, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
            {"time": "20000000000", "data": "{'position': [0.4, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
        ],
    )
    _write_rows(
        tmp_path / "human_states.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "[{'id': '1', 'position': [0.4, 0.0]}]"},
            {"time": "10000000000", "data": "[{'id': '1', 'position': [0.6, 0.0]}]"},
            {"time": "20000000000", "data": "[{'id': '1', 'position': [0.8, 0.0]}]"},
        ],
    )

    result = generate_social_metrics(
        tmp_path,
        thresholds={
            "robot_radius_m": 0.3,
            "human_radius_m": 0.25,
            "min_human_motion_time_sec": 1.0,
            "min_human_robot_motion_overlap_time_sec": 1.0,
            "min_human_robot_interaction_time_sec": 1.0,
        },
    )

    assert result["min_footprint_clearance_m"] < 0.0
    assert result["footprint_human_collision_count"] >= 1
    assert result["strict_social_success"] is False
    assert "footprint_human_collision" in result["strict_social_failure_reasons"]


def test_social_metrics_strict_failure_includes_task_failures(tmp_path):
    (tmp_path / "vln_task_metrics.json").write_text(
        json.dumps(
            {
                "strict_task_success": False,
                "strict_task_failure_reasons": ["static_occupancy_collision"],
                "static_occupancy": {"collision_sample_count": 3, "intervals": [{"start_sec": 1.0, "end_sec": 2.0}]},
                "commanded_stuck": {"commanded_stuck_time_sec": 0.0, "commanded_stuck_intervals": []},
            }
        ),
        encoding="utf-8",
    )
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
            {"time": "10000000000", "data": "{'position': [0.2, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
        ],
    )
    _write_rows(
        tmp_path / "human_states.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "[{'id': '1', 'position': [2.0, 0.0]}]"},
            {"time": "10000000000", "data": "[{'id': '1', 'position': [2.2, 0.0]}]"},
        ],
    )

    result = generate_social_metrics(
        tmp_path,
        thresholds={
            "min_human_motion_time_sec": 1.0,
            "min_human_robot_motion_overlap_time_sec": 1.0,
            "min_human_robot_interaction_time_sec": 1.0,
        },
    )

    assert result["social_success"] is True
    assert result["strict_social_success"] is False
    assert "static_occupancy_collision" in result["strict_social_failure_reasons"]
    assert result["review_intervals"]["static_occupancy"] == [{"start_sec": 1.0, "end_sec": 2.0}]
