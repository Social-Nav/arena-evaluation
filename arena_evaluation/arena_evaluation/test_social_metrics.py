import csv
import json

import pytest

from arena_evaluation.social_metrics import (
    DEFAULT_THRESHOLDS,
    _consolidate_human_samples,
    _read_humans,
    _read_odom,
    generate_social_metrics,
)


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
    assert result["social_success"] is False
    assert result["strict_social_success"] is False
    assert "dynamic_scene_failed" in result["strict_social_failure_reasons"]
    assert "missing_vln_task_metrics" in result["strict_social_failure_reasons"]
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
    assert result["min_footprint_clearance_sample"]["human_id"] == "1"
    assert result["min_footprint_clearance_sample"]["footprint_clearance_m"] == pytest.approx(
        result["min_footprint_clearance_m"]
    )
    assert result["footprint_human_collision_count"] >= 1
    assert result["footprint_human_collision_events"]
    assert result["footprint_human_collision_events"][0]["human_id"] == "1"
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

    assert result["social_success"] is False
    assert result["strict_social_success"] is False
    assert "static_occupancy_collision" in result["strict_social_failure_reasons"]
    assert result["review_intervals"]["static_occupancy"] == [{"start_sec": 1.0, "end_sec": 2.0}]


# ---------------------------------------------------------------------------
# Recording-segment handling in the pedestrian-motion metric.
#
# The Arena recorder appends several sim-time segments to one human_states.csv:
# whenever /clock jumps backwards it adopts the lower value as the new baseline and
# keeps appending (data_recorder_node.py:767-778), without bumping the `episode`
# column.  `time` is therefore only monotonic within a time segment, and the old global
# `sorted(key=time)` interleaved rows from different time segments, splicing unrelated
# positions into the pedestrian polyline.
# ---------------------------------------------------------------------------

# One time unit is 1 / RECORDER_TIME_UNITS_PER_SECOND second, so 1e9 units = 0.1 s
# and the 0.2 s consolidation window is 2e9 units wide.
_UNIT = 1_000_000_000


def _segment_splice_fixture(tmp_path):
    """Two recording time segments whose `time` ranges interleave.

    Rows 0-1 are a pre-episode segment that ran ahead to 0.7-0.9 s; rows 2-4 are the
    episode, restarting at 0.1 s.  A global `time` sort places the pre-episode pair
    between episode rows 3 and 4 and charges the polyline for jumping out to
    x=2.0/2.4 and back.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": str(1 * _UNIT), "data": "{'position': [20.0, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
            {"time": str(13 * _UNIT), "data": "{'position': [20.2, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
        ],
    )
    _write_rows(
        tmp_path / "human_states.csv",
        ["time", "data"],
        [
            # segment 0 -- pre-episode, timestamps already advanced to 0.7/0.9 s
            {"time": str(7 * _UNIT), "data": "[{'id': '1', 'position': [2.0, 0.0]}]"},
            {"time": str(9 * _UNIT), "data": "[{'id': '1', 'position': [2.4, 0.0]}]"},
            # segment 1 -- episode, /clock restarts at 0.1 s (backwards step)
            {"time": str(1 * _UNIT), "data": "[{'id': '1', 'position': [0.0, 0.0]}]"},
            {"time": str(3 * _UNIT), "data": "[{'id': '1', 'position': [0.3, 0.0]}]"},
            {"time": str(13 * _UNIT), "data": "[{'id': '1', 'position': [0.6, 0.0]}]"},
        ],
    )
    return generate_social_metrics(tmp_path)


def test_read_humans_preserves_file_order_and_tags_recording_segments(tmp_path):
    _segment_splice_fixture(tmp_path)

    path, samples = _read_humans(tmp_path)

    assert path.name == "human_states.csv"
    # file order, not `time` order
    assert [s["order"] for s in samples] == [0, 1, 2, 3, 4]
    assert [s["time"] for s in samples] == [
        7 * _UNIT, 9 * _UNIT, 1 * _UNIT, 3 * _UNIT, 13 * _UNIT
    ]
    # a new segment starts at the single backwards step (row 2)
    assert [s["segment"] for s in samples] == [0, 0, 1, 1, 1]

    consolidated = _consolidate_human_samples(
        samples, DEFAULT_THRESHOLDS["human_sample_consolidation_window_sec"]
    )
    # every row falls in its own 0.2 s window here, so nothing is dropped, and the
    # segment/acquisition tags survive consolidation for the motion integral to use
    assert [s["order"] for s in consolidated] == [0, 1, 2, 3, 4]
    assert [s["segment"] for s in consolidated] == [0, 0, 1, 1, 1]


def test_human_motion_excludes_backwards_clock_segment_splice(tmp_path):
    result = _segment_splice_fixture(tmp_path)

    # Assert the corrected DISTANCE first, so this test fails on the pre-fix code for
    # the numeric defect itself and not merely because a new diagnostic key is absent.
    #
    # Real travel: 0.4 m inside segment 0 plus 0.3 + 0.3 m inside segment 1.
    # The old global `time` sort reported 4.2 m for exactly these rows, because it
    # charged 1.7 m going out to the pre-episode position and 1.8 m coming back.
    assert result["human_motion_total_m"] == pytest.approx(1.0)
    assert result["human_motion_by_id_m"] == {"1": pytest.approx(1.0)}
    assert result["moving_human_count"] == 1

    # Motion time is NOT simply reduced: the corrected intervals are
    # 0.2 s (segment 0) + 0.2 s + 1.0 s (segment 1) = 1.4 s, whereas the spliced ordering
    # reported 1.2 s.  The fix removes an artefact; it is not a scale factor.
    assert result["human_motion_time_sec"] == pytest.approx(1.4)

    # Consolidation is untouched, so the surviving sample set is the same as before.
    assert result["human_raw_sample_count"] == 5
    assert result["human_sample_count"] == 5
    assert result["human_motion_recording_segments"] == 2
    assert result["human_motion_segment_boundaries_skipped"] == 1


def test_human_motion_unchanged_when_time_order_is_already_clean(tmp_path):
    """No-op guard: a single-segment, monotonic CSV keeps exactly its previous values.

    Every assertion here holds on the pre-fix code as well.  That is the point: the
    change must be inert when the recorder produced one clean time segment, so this test is
    only meaningful if it passes both before and after.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [20.0, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
            {"time": str(6 * _UNIT), "data": "{'position': [20.2, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
        ],
    )
    _write_rows(
        tmp_path / "human_states.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "[{'id': '1', 'position': [0.0, 0.0]}]"},
            {"time": str(2 * _UNIT), "data": "[{'id': '1', 'position': [0.3, 0.0]}]"},
            {"time": str(4 * _UNIT), "data": "[{'id': '1', 'position': [0.6, 0.0]}]"},
            {"time": str(6 * _UNIT), "data": "[{'id': '1', 'position': [0.9, 0.0]}]"},
        ],
    )

    result = generate_social_metrics(tmp_path)

    assert result["human_raw_sample_count"] == 4
    assert result["human_sample_count"] == 4
    assert result["human_motion_total_m"] == pytest.approx(0.9)
    assert result["human_motion_time_sec"] == pytest.approx(0.6)
    assert result["moving_human_count"] == 1
    assert result["human_motion_by_id_m"] == {"1": pytest.approx(0.9)}


def test_consolidation_still_suppresses_within_window_jitter(tmp_path):
    """Invariant: the 0.2 s consolidation keeps suppressing jitter, before and after.

    This guards against "fixing" the inflation by weakening the smoothing step.  Like
    the clean-order test, it must hold on the pre-fix code too.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [20.0, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
            {"time": str(4 * _UNIT), "data": "{'position': [20.2, 0.0, 0.0], 'velocity': [0.2, 0.0, 0.0]}"},
        ],
    )
    _write_rows(
        tmp_path / "human_states.csv",
        ["time", "data"],
        [
            # three samples inside the first 0.2 s window, jittering by 0.05 m
            {"time": "0", "data": "[{'id': '1', 'position': [0.0, 0.0]}]"},
            {"time": str(_UNIT // 2), "data": "[{'id': '1', 'position': [0.05, 0.0]}]"},
            {"time": str(_UNIT), "data": "[{'id': '1', 'position': [0.0, 0.0]}]"},
            {"time": str(2 * _UNIT), "data": "[{'id': '1', 'position': [0.3, 0.0]}]"},
            {"time": str(4 * _UNIT), "data": "[{'id': '1', 'position': [0.6, 0.0]}]"},
        ],
    )

    result = generate_social_metrics(tmp_path)

    assert result["human_raw_sample_count"] == 5
    # first window collapses to its last-written sample
    assert result["human_sample_count"] == 3
    # 0.6 m of real travel; the 0.1 m of within-window jitter stays suppressed
    assert result["human_motion_total_m"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# odom.csv carries the SAME multi-segment recorder defect as the pedestrian CSV,
# so `path_length_m` and every dt-weighted robot metric were inflated too.
# ---------------------------------------------------------------------------

_FAR_HUMAN = "[{'id': '1', 'position': [200.0, 0.0]}]"


def _static_far_humans(tmp_path, times):
    """A single, distant, motionless pedestrian.

    Keeps `agents` non-empty (the odom loop skips samples with no agents) while
    putting every proximity threshold far out of range, so these tests isolate the
    robot track instead of measuring pedestrian behaviour.
    """
    _write_rows(
        tmp_path / "human_states.csv",
        ["time", "data"],
        [{"time": str(t), "data": _FAR_HUMAN} for t in times],
    )


def _odom_segment_splice_fixture(tmp_path):
    """Two odom recording time segments whose `time` ranges interleave.

    Rows 0-1 are a pre-episode segment that already ran ahead to 1.0-2.0 s at x=10;
    rows 2-4 are the episode, restarting at 0.5 s at x=0..2.  A global `time` sort
    interleaves them as x = 0, 10, 1, 10, 2 and charges the robot polyline
    10 + 9 + 9 + 8 = 36 m for jumping out to the stale position and back twice.
    In acquisition order the real travel is 0 m (segment 0) + 1 + 1 m (segment 1) = 2 m.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            # segment 0 -- pre-episode, /clock already advanced to 1.0/2.0 s
            {"time": str(10 * _UNIT), "data": "{'position': [10.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(20 * _UNIT), "data": "{'position': [10.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            # segment 1 -- episode, /clock restarts at 0.5 s (the backwards step)
            {"time": str(5 * _UNIT), "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(15 * _UNIT), "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(25 * _UNIT), "data": "{'position': [2.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
        ],
    )
    _static_far_humans(tmp_path, [5 * _UNIT, 15 * _UNIT, 25 * _UNIT])
    return generate_social_metrics(tmp_path)


def test_read_odom_preserves_file_order_and_tags_recording_segments(tmp_path):
    _odom_segment_splice_fixture(tmp_path)

    samples = _read_odom(tmp_path)

    # Assert the POSITION SEQUENCE first, so this fails on the pre-fix code for the
    # wrong ordering itself rather than for a missing `segment`/`order` key.
    # Pre-fix (global `time` sort) yields [0.0, 10.0, 1.0, 10.0, 2.0].
    assert [s["x"] for s in samples] == [10.0, 10.0, 0.0, 1.0, 2.0]
    assert [s["time"] for s in samples] == [
        10 * _UNIT, 20 * _UNIT, 5 * _UNIT, 15 * _UNIT, 25 * _UNIT
    ]
    assert [s["order"] for s in samples] == [0, 1, 2, 3, 4]
    # a new segment starts at the single backwards step (row 2)
    assert [s["segment"] for s in samples] == [0, 0, 1, 1, 1]


def test_path_length_excludes_backwards_clock_segment_splice(tmp_path):
    result = _odom_segment_splice_fixture(tmp_path)

    # The numeric defect: 36.0 m before, 2.0 m after, for exactly these five rows.
    assert result["path_length_m"] == pytest.approx(2.0)
    assert result["odom_sample_count"] == 5
    assert result["odom_recording_segments"] == 2
    assert result["odom_segment_boundaries_skipped"] == 1


def test_large_teleport_not_reported_across_segment_boundary(tmp_path):
    """A backwards /clock step is not a teleport, so it must not fail the run.

    Here the ONLY step above the 5 m teleport threshold is the time-segment boundary
    itself, which isolates teleport semantics from the path-length arithmetic.
    Pre-fix: path 8.2 m, one teleport, and `large_teleport` in the failure reasons.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            # segment 0 -- pre-episode, already at 10.0 s
            {"time": str(100 * _UNIT), "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            # segment 1 -- episode restarts at 5.0 s, 8 m away from the stale position
            {"time": str(50 * _UNIT), "data": "{'position': [8.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(60 * _UNIT), "data": "{'position': [8.1, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
        ],
    )
    _static_far_humans(tmp_path, [50 * _UNIT, 60 * _UNIT])

    result = generate_social_metrics(tmp_path)

    assert result["path_length_m"] == pytest.approx(0.1)
    assert result["large_teleports"] == []
    assert "large_teleport" not in result["strict_social_failure_reasons"]


def test_robot_motion_time_not_credited_across_segment_boundary(tmp_path):
    """dt is not taken from a cross-segment successor, which carries a LOWER `time`.

    The corrected total is not simply smaller than the spliced one: the spliced
    ordering FRAGMENTED each 0.2 s step into 0.1 s pieces and reported 0.3 s.  The
    fix removes an artefact; it is not a scale factor.

    Expectation revised from 0.4 s to 0.3 s.  The two segments here cover
    0.0-0.2 s and 0.1-0.3 s, so the recording spans only 0.3 s of sim time and the
    two 0.2 s steps overlap on [0.1, 0.2].  Summing them to 0.4 s credited that
    0.1 s twice and claimed more elapsed time than the recording contains, which
    is impossible.  Under supersession the later segment owns the overlap, segment
    0 credits 0.1 s, segment 1 credits 0.2 s, and the total is exactly the 0.3 s
    span.  The invariant is asserted below so this cannot regress silently.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            # segment 0 -- one real 0.2 s step of 1.0 m
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(2 * _UNIT), "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            # segment 1 -- /clock restarts at 0.1 s; another real 0.2 s step of 1.0 m
            {"time": str(1 * _UNIT), "data": "{'position': [50.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(3 * _UNIT), "data": "{'position': [51.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
        ],
    )
    _static_far_humans(tmp_path, [0, 2 * _UNIT])

    result = generate_social_metrics(tmp_path)

    assert result["robot_motion_time_sec"] == pytest.approx(0.3)
    assert result["odom_recording_span_sec"] == pytest.approx(0.3)
    assert result["robot_motion_time_sec"] <= result["odom_recording_span_sec"] + 1e-9
    # the distant pedestrian keeps every proximity-gated duration at zero, so the
    # figure above is attributable to the odom track alone
    assert result["personal_space_violation_time_sec"] == pytest.approx(0.0)
    assert result["footprint_personal_space_violation_time_sec"] == pytest.approx(0.0)
    assert result["odom_segment_boundaries_skipped"] == 1


def test_path_length_unchanged_when_odom_time_order_is_already_clean(tmp_path):
    """No-op guard: a single-segment, monotonic odom.csv keeps its previous values.

    Every assertion here holds on the pre-fix code as well, which is the point --
    the change must be inert when the recorder produced one clean time segment.  It
    deliberately asserts no new diagnostic key, so it stays runnable on both.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(2 * _UNIT), "data": "{'position': [0.5, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(4 * _UNIT), "data": "{'position': [1.2, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
        ],
    )
    _static_far_humans(tmp_path, [0, 2 * _UNIT, 4 * _UNIT])

    result = generate_social_metrics(tmp_path)

    assert result["odom_sample_count"] == 3
    assert result["path_length_m"] == pytest.approx(1.2)
    assert result["large_teleports"] == []
    assert result["robot_motion_time_sec"] == pytest.approx(0.4)
    assert result["personal_space_violation_time_sec"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# Overlapping time segments must not double-count elapsed time.
#
# The recorder's segments overlap in sim time: every pre-episode segment restarts
# near t=0 while the final segment spans the whole episode.  Summing each
# segment's step durations therefore charges the first few seconds several times
# and reports an elapsed time LONGER than the episode, which is impossible.  The
# supersession rule credits an instant to the LAST segment covering it.
# --------------------------------------------------------------------------

_NEAR_HUMAN = "[{'id': '1', 'position': [0.5, 0.0]}]"


def _near_static_humans(tmp_path, times):
    """One motionless pedestrian 0.5 m from the robot's track.

    Close enough that every proximity-gated duration accrues, and motionless so
    `human_robot_motion_overlap_time_sec` stays out of the picture.
    """
    _write_rows(
        tmp_path / "human_states.csv",
        ["time", "data"],
        [{"time": str(t), "data": _NEAR_HUMAN} for t in times],
    )


def test_elapsed_time_not_double_counted_when_a_segment_is_fully_superseded(tmp_path):
    """Segment 0 lies entirely inside segment 1's range, so it must credit no time.

    Hand-derived.  Segment 0 spans 0.2-0.6 s (0.4 s) and segment 1 spans
    0.1-1.1 s (1.0 s), so the recording covers 1.0 s of sim time in total.
    Pre-fix each segment's duration was summed independently and every
    `*_time_sec` came out at 0.4 + 1.0 = 1.4 s -- longer than the recording it was
    measured from.  With supersession segment 0 contributes 0.0 s and the totals
    are exactly 1.0 s.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            # segment 0 -- pre-episode, /clock already advanced to 0.2-0.6 s
            {"time": str(2 * _UNIT), "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(6 * _UNIT), "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            # segment 1 -- /clock resumes at 0.1 s and runs to 1.1 s
            {"time": str(1 * _UNIT), "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(11 * _UNIT), "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
        ],
    )
    _near_static_humans(tmp_path, [1 * _UNIT, 2 * _UNIT, 6 * _UNIT, 11 * _UNIT])

    result = generate_social_metrics(tmp_path)

    # Numeric assertions FIRST, so this test fails on the pre-fix VALUES (every one
    # of these was 1.4) rather than on a missing diagnostic key.
    assert result["robot_motion_time_sec"] == pytest.approx(1.0)
    assert result["personal_space_violation_time_sec"] == pytest.approx(1.0)
    assert result["footprint_personal_space_violation_time_sec"] == pytest.approx(1.0)
    assert result["crowd_freezing_time_sec"] == pytest.approx(1.0)
    # supersession is about time attribution only: path geometry is untouched and
    # still integrates both segments (1.0 m + 1.0 m).  Holds pre-fix too.
    assert result["path_length_m"] == pytest.approx(2.0)
    # the impossible-duration invariant the fix exists to restore
    assert result["odom_recording_span_sec"] == pytest.approx(1.0)
    assert result["odom_superseded_time_sec"] == pytest.approx(0.4)
    assert result["robot_motion_time_sec"] <= result["odom_recording_span_sec"] + 1e-9


def test_elapsed_time_clipped_where_segments_partially_overlap(tmp_path):
    """Only the re-covered PART of a step is dropped, not the whole step.

    Hand-derived.  Segment 0 spans 0.0-1.0 s, segment 1 spans 0.5-1.5 s, so the
    recording covers 1.5 s.  Segment 0's single 1.0 s step overlaps segment 1's
    range on [0.5, 1.0], i.e. 0.5 s, so it credits 0.5 s and segment 1 credits its
    full 1.0 s.  Pre-fix the total was 1.0 + 1.0 = 2.0 s against a 1.5 s recording.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            # segment 0 -- 0.0 s to 1.0 s
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(10 * _UNIT), "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            # segment 1 -- /clock resumes at 0.5 s and runs to 1.5 s
            {"time": str(5 * _UNIT), "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(15 * _UNIT), "data": "{'position': [1.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
        ],
    )
    _near_static_humans(tmp_path, [0, 5 * _UNIT, 10 * _UNIT, 15 * _UNIT])

    result = generate_social_metrics(tmp_path)

    # Numeric assertions FIRST: pre-fix both of these were 2.0, measured from a
    # recording only 1.5 s long.
    assert result["robot_motion_time_sec"] == pytest.approx(1.5)
    assert result["footprint_personal_space_violation_time_sec"] == pytest.approx(1.5)
    assert result["odom_recording_span_sec"] == pytest.approx(1.5)
    assert result["odom_superseded_time_sec"] == pytest.approx(0.5)
    assert result["robot_motion_time_sec"] <= result["odom_recording_span_sec"] + 1e-9


def test_supersession_does_not_rescale_measured_step_speed(tmp_path):
    """A clipped step keeps its MEASURED speed, so motion classification is unmoved.

    Regression guard for a wrong way to implement supersession: dividing the step
    distance by the CREDITED duration instead of the measured one would inflate the
    speed.  Segment 0 moves 0.01 m over a measured 1.0 s, i.e. 0.01 m/s, which is
    below the 0.02 m/s motion threshold.  Its credited duration is only 0.4 s, and
    0.01 / 0.4 = 0.025 m/s WOULD clear the threshold and wrongly add 0.4 s.

    This passes on the pre-fix code as well -- that is what makes it a guard on the
    fix rather than a test of it.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            # segment 0 -- 0.0 s to 1.0 s, 0.01 m travelled
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(10 * _UNIT), "data": "{'position': [0.01, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            # segment 1 -- 0.4 s to 1.4 s, stationary
            {"time": str(4 * _UNIT), "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(14 * _UNIT), "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
        ],
    )
    _near_static_humans(tmp_path, [0, 4 * _UNIT, 10 * _UNIT, 14 * _UNIT])

    result = generate_social_metrics(tmp_path)

    assert result["robot_motion_time_sec"] == pytest.approx(0.0)


def test_elapsed_time_unchanged_when_recording_has_one_segment(tmp_path):
    """No-op guard: a single-segment recording keeps its previous durations.

    Every assertion holds on the pre-fix code as well, which is the point --
    supersession must be inert when the recorder produced one clean time segment.
    It deliberately asserts no new diagnostic key so it stays runnable on both.
    """
    _write_rows(
        tmp_path / "odom.csv",
        ["time", "data"],
        [
            {"time": "0", "data": "{'position': [0.0, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(2 * _UNIT), "data": "{'position': [0.5, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
            {"time": str(4 * _UNIT), "data": "{'position': [1.2, 0.0, 0.0], 'velocity': [0.0, 0.0, 0.0]}"},
        ],
    )
    _write_rows(
        tmp_path / "human_states.csv",
        ["time", "data"],
        [{"time": str(t), "data": "[{'id': '1', 'position': [0.6, 0.0]}]"} for t in (0, 2 * _UNIT, 4 * _UNIT)],
    )

    result = generate_social_metrics(tmp_path)

    assert result["path_length_m"] == pytest.approx(1.2)
    assert result["robot_motion_time_sec"] == pytest.approx(0.4)
    assert result["personal_space_violation_time_sec"] == pytest.approx(0.4)
    assert result["footprint_personal_space_violation_time_sec"] == pytest.approx(0.4)
    assert result["crowd_freezing_time_sec"] == pytest.approx(0.4)
