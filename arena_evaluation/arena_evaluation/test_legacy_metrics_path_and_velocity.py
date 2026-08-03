"""Legacy `metrics.csv` (scripts/metrics.py) episode-analysis regression tests.

`Metrics.__init__` loads CSVs and resolves robot model params from an installed
share directory, none of which `_analyze_episode` needs, so these tests follow
the pattern already used by `test_legacy_metrics_timeout.py`: build the instance
with `object.__new__` and drive `_analyze_episode` on a hand-built frame.
"""

import numpy as np
import pandas as pd
import pytest

from arena_evaluation.scripts.metrics import Math, Metrics


_SEC = 10_000_000_000  # `_analyze_episode` divides `time` by 10**10


def _episode(positions, velocities, times=None):
    """A minimal episode frame in the shape `_analyze_episode` expects."""
    n = len(positions)
    times = times or [i * _SEC for i in range(n)]
    return pd.DataFrame(
        {
            "time": [float(t) for t in times],
            "odom": [{"position": list(p), "velocity": list(v)} for p, v in zip(positions, velocities)],
            "start": [list(positions[0])] * n,
            "goal": [list(positions[-1])] * n,
            "episode": [0] * n,
        }
    )


def test_velocity_is_read_from_the_velocity_field_not_the_position_field():
    """`velocity`, `acceleration` and `jerk` must not be functions of position.

    The fixture keeps every position on a circle of radius 5 about the origin, so
    reading `position` yields the constant 5.0 regardless of how the robot is
    actually moving.  That is exactly the signature seen in every delivered
    metrics.csv: case01's first three velocity samples are 5.257, 5.257, 5.257 for
    a robot standing at (3.926, -2.012, -2.86), whose norm is 5.257.

    Hand-derived expectations.  Pre-fix: velocity [5.0]*5, and because it is
    constant, acceleration [0.0]*4 and jerk [0.0]*3 -- a robot that appears to
    have no acceleration at all while its commanded speed octuples.
    """
    positions = [(3.0, 4.0, 0.0), (4.0, 3.0, 0.0), (5.0, 0.0, 0.0), (0.0, 5.0, 0.0), (-3.0, 4.0, 0.0)]
    speeds = [0.1, 0.2, 0.4, 0.8, 0.1]
    velocities = [(s, 0.0, 0.0) for s in speeds]
    # the premise of the fixture, asserted rather than assumed
    assert [round(float(np.linalg.norm(p)), 3) for p in positions] == [5.0] * 5

    metric = object.__new__(Metrics)._analyze_episode(_episode(positions, velocities), 0)

    assert metric["velocity"] == [0.1, 0.2, 0.4, 0.8, 0.1]
    assert metric["acceleration"] == [0.1, 0.2, 0.4, -0.7]
    assert metric["jerk"] == [0.1, 0.2, -1.1]


def test_velocity_length_matches_the_sample_count():
    """One velocity per odom sample; no window trimming applies to this series."""
    positions = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0), (0.3, 0.0, 0.0)]
    velocities = [(0.05, 0.0, 0.0)] * 4

    metric = object.__new__(Metrics)._analyze_episode(_episode(positions, velocities), 0)

    assert len(metric["velocity"]) == 4


# --------------------------------------------------------------------------
# Consecutive-window grouping must not wrap around the episode.
#
# `Math.grouping` was built with `np.roll`, so window 0 paired the FIRST sample
# with the LAST, and a trailing `[:-size]` slice discarded the final windows.
# --------------------------------------------------------------------------


def test_grouping_windows_are_consecutive_and_do_not_wrap():
    """Every consecutive window, newest-first, with none dropped and none wrapped.

    Pre-fix, `grouping(base, 2)` on six samples produced the index windows
    [[0,5],[1,0],[2,1],[3,2]]: window 0 pairs sample 0 with sample 5, and the
    windows [4,3] and [5,4] are missing.  For size 3 it produced
    [[0,5,4],[1,0,5],[2,1,0]], i.e. two wrapped windows and two dropped.
    """
    base = np.arange(6.0).reshape(6, 1)

    def windows(size):
        return [[int(v[0]) for v in window] for window in Math.grouping(base, size)]

    assert windows(2) == [[1, 0], [2, 1], [3, 2], [4, 3], [5, 4]]
    assert windows(3) == [[2, 1, 0], [3, 2, 1], [4, 3, 2], [5, 4, 3]]


def test_path_length_has_no_wrap_term_and_drops_no_step():
    """A path length may not include the distance from the last pose to the first.

    Hand-derived on steps of 1, 2, 3 and 4 m along x, i.e. a 10 m path whose
    endpoints are 10 m apart.  Pre-fix `Math.path_length` returned three terms,
    [10.0, 1.0, 2.0], summing to 13.0: the wrap distance plus the first two steps,
    with the 3 m and 4 m steps dropped.  That is the arithmetic that made the
    delivered case01 `path_length_values` start at 5.910, exactly the distance
    from its final pose back to its first.
    """
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    terms = Math.path_length(positions)

    assert list(terms) == [1.0, 2.0, 3.0, 4.0]
    assert terms.sum() == 10.0
    # one term per step, not len - 2
    assert len(terms) == len(positions) - 1


def test_turn_has_no_wrap_term():
    """`turn` feeds `angle_over_length`, so its wrap term is equally spurious.

    Hand-derived on yaws 0, 0.1, 0.3, 0.6, 1.0 rad: the real turns are 0.1, 0.2,
    0.3, 0.4 summing to 1.0 rad.  Pre-fix the terms were [1.0, 0.1, 0.2] -- the
    full 1.0 rad sweep from the last yaw back to the first, plus the first two
    turns -- summing to 1.3 rad.
    """
    yaw = np.array([0.0, 0.1, 0.3, 0.6, 1.0])

    turns = Math.turn(yaw)

    # sum first, so the failure on the pre-fix module is numeric (1.3 vs 1.0)
    # rather than a shape mismatch
    assert turns.sum() == pytest.approx(1.0)
    assert np.allclose(turns, [0.1, 0.2, 0.3, 0.4])


def test_shape_descriptors_cover_every_consecutive_triple():
    """`curvature` and `roughness` gain the windows the wrap slice discarded."""
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    curvature, normalized = Math.curvature(positions)
    roughness = Math.roughness(positions)

    # len - 2 consecutive triples, not len - 3
    assert len(curvature) == len(positions) - 2
    assert len(normalized) == len(positions) - 2
    assert len(roughness) == len(positions) - 2


def test_analyze_episode_path_length_excludes_the_wrap_term():
    """The same defect through the production entry point.

    Yaw is held at 0 so this exercises only the wrap/truncation change and is
    independent of whether `path_length` norms two or three columns.
    Pre-fix: 13.0.
    """
    positions = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0), (6.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    velocities = [(0.1, 0.0, 0.0)] * 5

    metric = object.__new__(Metrics)._analyze_episode(_episode(positions, velocities), 0)

    assert metric["path_length"] == pytest.approx(10.0)
    assert metric["path_length_values"] == [1.0, 2.0, 3.0, 4.0]


# --------------------------------------------------------------------------
# Planar quantities must not norm radians together with metres.
#
# Odom `position` is [x, y, yaw] and `velocity` is [linear.x, linear.y,
# angular.z].  A Euclidean norm over all three columns is dimensionally invalid.
# --------------------------------------------------------------------------


def test_path_length_ignores_yaw():
    """A pure rotation travels no distance.

    Hand-derived.  The robot stands still and turns 1 rad per sample.  The planar
    path length is 0.0 m.  Pre-fix, yaw entered the norm and the same fixture
    reported 4.0, i.e. the yaw sweep in radians reported as metres.
    """
    positions = np.array([[2.0, 3.0, 0.0], [2.0, 3.0, 1.0], [2.0, 3.0, 2.0], [2.0, 3.0, 3.0], [2.0, 3.0, 4.0]])

    terms = Math.path_length(positions)

    assert terms.sum() == pytest.approx(0.0)
    assert list(terms) == [0.0, 0.0, 0.0, 0.0]


def test_path_length_of_a_diagonal_move_is_the_planar_distance():
    """Translation is unaffected; only the yaw contribution is removed.

    A 3-4-5 step with a simultaneous 12 rad yaw change is 5.0 m.  Pre-fix the norm
    over [3, 4, 12] gave 13.0.
    """
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 12.0], [6.0, 8.0, 24.0]])

    terms = Math.path_length(positions)

    assert list(terms) == [5.0, 5.0]


def test_velocity_is_linear_speed_only():
    """`velocity` is m/s, so angular.z must not enter the norm.

    Hand-derived: linear (0.3, 0.4) and angular 12.0 gives 0.5 m/s.  Pre-fix the
    norm over all three components gave 12.0104.
    """
    positions = [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0), (0.2, 0.0, 0.0)]
    velocities = [(0.3, 0.4, 12.0)] * 3

    metric = object.__new__(Metrics)._analyze_episode(_episode(positions, velocities), 0)

    assert metric["velocity"] == [0.5, 0.5, 0.5]


def test_analyze_episode_reports_planar_path_length_and_angle_over_length():
    """End to end: 1 m of travel while turning 1 rad.

    Hand-derived.  Planar path length 1.0 m over two 0.5 m steps.  `turn` sums the
    two 0.5 rad steps to 1.0 rad, so `angle_over_length` is 1.0 rad/m.  Pre-fix
    the path length was the norm over [0.5, 0, 0.5] twice, i.e. 1.4142, and
    `angle_over_length` was 0.7071.
    """
    positions = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5), (1.0, 0.0, 1.0)]
    velocities = [(0.1, 0.0, 0.0)] * 3

    metric = object.__new__(Metrics)._analyze_episode(_episode(positions, velocities), 0)

    assert metric["path_length"] == pytest.approx(1.0)
    assert metric["angle_over_length"] == pytest.approx(1.0)


def test_planar_helper_leaves_two_column_input_alone():
    """Guard: already-planar data passes through unchanged, so the helper is inert
    wherever no yaw column exists.

    This one cannot run on the pre-fix module at all -- `Math.planar` is new -- so
    unlike the other guards in this file it is not a pre/post no-op proof.  The
    no-op proof for the dimensional change is that `path_length` on a fixture with
    yaw held at 0 is unaffected, which the wrap-term tests above already assert.
    """
    two_col = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert np.array_equal(Math.planar(two_col), two_col)
    assert np.array_equal(Math.planar(np.array([[1.0, 2.0, 9.0]])), np.array([[1.0, 2.0]]))


# --------------------------------------------------------------------------
# Recording time segments.
#
# The recorder concatenates several time segments into one CSV, marked only by a
# backwards step in `time`.  A pair straddling a boundary is a clock rebase, not
# travel and not a turn.
# --------------------------------------------------------------------------


def test_segment_boundaries_are_detected_from_backwards_time_steps():
    """One boundary per backwards step, with no assumed segment count."""
    assert list(Math.segment_boundaries(np.array([0.0, 1.0, 2.0]))) == [False, False]
    assert list(Math.segment_boundaries(np.array([2.0, 3.0, 0.5, 1.0]))) == [False, True, False]
    # three segments, to show nothing is hard-coded to two
    assert list(Math.segment_boundaries(np.array([5.0, 1.0, 9.0, 2.0]))) == [True, False, True]
    assert list(Math.segment_boundaries(np.array([1.0]))) == []


def test_path_length_and_turn_skip_the_segment_boundary_step():
    """Hand-derived: the boundary jump is excluded from both numerator and denominator.

    Segment 0 sits at x=0 and turns 0 -> 0.1 rad.  Segment 1 restarts the clock at
    0.5 s, 9 m away, and travels 9.0 -> 9.2 with yaw 1.0 -> 1.3.  The real travel
    is 0.2 m and the real turning is 0.1 + 0.3 = 0.4 rad.

    Without the times argument the 9 m boundary jump and its 0.9 rad yaw step are
    both charged, giving 9.2 m and 1.3 rad.
    """
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [9.0, 0.0, 1.0], [9.2, 0.0, 1.3]])
    times = np.array([2.0, 3.0, 0.5, 1.0])

    assert Math.path_length(positions, times).sum() == pytest.approx(0.2)
    assert Math.turn(positions[:, 2], times).sum() == pytest.approx(0.4)
    # the same data without segment awareness, to show the argument is load-bearing
    assert Math.path_length(positions).sum() == pytest.approx(9.2)
    assert Math.turn(positions[:, 2]).sum() == pytest.approx(1.3)


def test_analyze_episode_excludes_the_segment_boundary(monkeypatch):
    """End to end.  Pre-fix path_length 9.2, angle_over_length 1.3/9.2 = 0.1413."""
    positions = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (9.0, 0.0, 1.0), (9.2, 0.0, 1.3)]
    velocities = [(0.1, 0.0, 0.0)] * 4
    times = [2.0 * _SEC, 3.0 * _SEC, 0.5 * _SEC, 1.0 * _SEC]

    metric = object.__new__(Metrics)._analyze_episode(_episode(positions, velocities, times), 0)

    assert metric["path_length"] == pytest.approx(0.2)
    assert metric["angle_over_length"] == pytest.approx(0.4 / 0.2)


def test_segment_awareness_is_inert_on_a_single_segment():
    """No-op guard: with monotonic timestamps the times argument changes nothing.

    Holds on both modules for the `times=None` call, and pins that supplying
    monotonic times is identical to omitting them.
    """
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    times = np.array([0.0, 1.0, 2.0, 3.0])

    assert list(Math.path_length(positions, times)) == list(Math.path_length(positions))
    assert Math.path_length(positions, times).sum() == pytest.approx(6.0)
