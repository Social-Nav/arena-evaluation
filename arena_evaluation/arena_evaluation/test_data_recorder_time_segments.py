"""Recorder time-segment detection tests.

`BagRecorder` is an rclpy Node whose constructor opens a rosbag, subscribes to
topics and resolves package share directories, none of which `_record_tick` needs.
These tests therefore follow the pattern of `test_legacy_metrics_timeout.py`:
build the instance with `object.__new__` and drive the tick directly, with
`write_data` and the logger captured.

SCOPE.  These prove the DETECTION logic.  They cannot prove the behaviour under a
real /clock stall -- that needs a full eval run -- so the recorder change is
reported as implemented, NOT runtime-validated.
"""

import pytest

from arena_evaluation.data_recorder_node import BagRecorder, Recorder


class _Logger:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.infos = []

    def warn(self, message):
        self.warnings.append(str(message))

    def error(self, message):
        self.errors.append(str(message))

    def info(self, message):
        self.infos.append(str(message))


# Literal, not BagRecorder.SEGMENT_MARKER_FILE, so the harness also works against
# the pre-fix module, which has no such attribute.  Pinned by
# `test_marker_file_name_matches_the_module_constant`.
_MARKER_FILE = "recording_segments"


class _Parameter:
    value = None


class _ClockMsg:
    """Minimal stand-in for builtin_interfaces/Time inside rosgraph_msgs/Clock."""

    def __init__(self, units):
        seconds, remainder = divmod(int(units), 10_000_000_000)
        self.clock = type("_Time", (), {"sec": seconds, "nanosec": remainder})()


class _Harness:
    """A recorder with every ROS dependency replaced by a capture."""

    def __init__(self, cls, record_frequency=400):
        self.recorder = object.__new__(cls)
        self.rows = []
        self.logger = _Logger()
        rec = self.recorder
        rec.config = {"record_frequency": record_frequency}
        rec.current_time = None
        rec.current_episode = 0
        rec.current_start = None
        rec.current_goal = None
        rec.data_collectors = []
        rec._csv_topic_to_file = {}
        rec._record_period_sec = record_frequency / 1000.0
        rec._last_clock_sim_time = None
        rec._last_clock_progress_wall_time = 0.0
        rec._last_record_wall_time = 0.0
        # Works on the pre-fix module too, so a failure below is a BEHAVIOUR
        # difference (no marker, no warning) rather than a harness AttributeError.
        if hasattr(rec, "_init_time_segment_state"):
            rec._init_time_segment_state()
        rec.write_data = self._write_data
        rec.get_logger = lambda: self.logger
        # the legacy Recorder reads `start`/`goal` node parameters inside its tick
        rec.get_parameter = lambda _name: _Parameter()

    @property
    def recording_segment(self):
        return getattr(self.recorder, "_recording_segment", 0)

    @property
    def synthetic_row_count(self):
        return getattr(self.recorder, "_synthetic_row_count", 0)

    @property
    def synthetic_active(self):
        return getattr(self.recorder, "_synthetic_time_active", False)

    def _write_data(self, file_name, data, mode="a"):
        self.rows.append((file_name, list(data), mode))

    def clock(self, units):
        """Drive the real `clock_callback`, so /clock bookkeeping is exercised."""
        self.recorder.clock_callback(_ClockMsg(units))

    def stall(self, ticks):
        """Drive the real fallback timer, as if /clock had stopped publishing."""
        for _ in range(ticks):
            self.recorder._last_record_wall_time = 0.0  # bypass the wall-clock rate limit
            self.recorder.wall_clock_fallback_callback()

    def marker_rows(self):
        return [r for r in self.rows if r[0] == _MARKER_FILE]

    def boundaries(self):
        return [r[1] for r in self.marker_rows() if r[1][0] != "segment"]


_UNIT_PER_MS = 10_000_000  # 10**10 units per second / 1000 ms


def _ms(milliseconds):
    return int(milliseconds * _UNIT_PER_MS)


@pytest.mark.parametrize("cls", [BagRecorder, Recorder])
def test_backwards_clock_step_is_logged_and_marked(cls):
    """The core defect: a backwards /clock step must not be absorbed silently.

    Pre-fix `_record_tick` set `self.current_time` to the lower value and returned,
    writing nothing, logging nothing and leaving no marker, so the CSV silently
    became two concatenated time segments.

    Here /clock reaches 2000 ms, then resumes at 500 ms.  Exactly one boundary is
    reported, with both stamps and the size of the step.
    """
    h = _Harness(cls)

    h.clock(_ms(0))
    h.clock(_ms(1000))
    h.clock(_ms(2000))
    assert h.marker_rows() == []  # nothing marked while /clock moves forward

    h.clock(_ms(500))

    # Observable output first.  Pre-fix both of these are 0: the rebase produced no
    # marker row and no log line at all, which IS the defect.
    markers = h.marker_rows()
    assert len(markers) == 2  # header + one boundary
    assert len(h.logger.warnings) == 1

    assert markers[0][1] == BagRecorder.SEGMENT_MARKER_HEADER
    assert markers[0][2] == "w"
    segment, previous, resumed, backwards_by, kind, synthetic_before = markers[1][1]
    assert segment == 1
    assert previous == _ms(2000)
    assert resumed == _ms(500)
    assert backwards_by == _ms(1500)
    assert synthetic_before == 0
    assert h.recording_segment == 1
    assert "BACKWARDS" in h.logger.warnings[0]
    assert "2 concatenated time segments" in h.logger.warnings[0]


@pytest.mark.parametrize("cls", [BagRecorder, Recorder])
def test_no_marker_and_no_warning_while_the_clock_moves_forward(cls):
    """No-op guard: a clean run produces no marker file and no warning at all.

    The behaviour asserted here is identical on the pre-fix code, which is what
    makes it a no-op proof: the change must be inert unless /clock steps back.
    """
    h = _Harness(cls)

    for ms in (0, 400, 800, 1200, 1600):
        h.clock(_ms(ms))

    assert h.marker_rows() == []
    assert h.logger.warnings == []
    assert h.recording_segment == 0


@pytest.mark.parametrize("cls", [BagRecorder, Recorder])
def test_every_boundary_is_counted_without_assuming_a_segment_count(cls):
    """Three backwards steps give three boundaries and four segments.

    The delivered runs have four or five segments depending on how long scene load
    stalls /clock, so nothing may be pinned to a particular count.
    """
    h = _Harness(cls)

    for ms in (0, 1000, 2000, 400, 1400, 300, 1300, 200):
        h.clock(_ms(ms))

    assert len(h.logger.warnings) == 3
    assert [row[0] for row in h.boundaries()] == [1, 2, 3]
    assert h.recording_segment == 3


def test_boundary_from_fabricated_stamps_is_classified_as_an_overshoot():
    """The cause of every boundary in the delivered runs, named rather than guessed.

    /clock publishes 40 ms and then stalls.  Each fallback tick adds
    `record_frequency * 1e6` = 4e8 units, which in the 1e10-units-per-second scale
    is 40 ms, so two ticks take the stamp to 120 ms.  /clock then resumes at 45 ms:
    above the last value it actually published, so sim time never went backwards --
    our own fabricated stamps overshot.  This is the shape measured in the
    delivered artifacts, whose pre-episode segments consist almost entirely of
    consecutive fixed 4e8 steps.
    """
    h = _Harness(BagRecorder)

    h.clock(_ms(40))
    h.stall(2)

    assert h.recorder.current_time == _ms(120)
    assert h.synthetic_active is True
    assert h.synthetic_row_count == 2
    assert any("FABRICATED" in w for w in h.logger.warnings)

    h.clock(_ms(45))

    boundary = h.boundaries()[0]
    assert boundary[1] == _ms(120)
    assert boundary[2] == _ms(45)
    assert boundary[4] == "synthetic_overshoot"
    assert boundary[5] == 2  # the fabricated rows are counted, not hidden


def test_boundary_below_the_last_real_clock_reading_is_a_sim_clock_reset():
    """The other cause, which needs a different response, is labelled differently."""
    h = _Harness(BagRecorder)

    h.clock(_ms(0))
    h.clock(_ms(2000))
    h.clock(_ms(500))

    assert h.boundaries()[0][4] == "sim_clock_reset"


def test_fabrication_warning_is_emitted_once_per_stall_not_per_tick():
    """A stall lasting many ticks must not flood the log."""
    h = _Harness(BagRecorder)

    h.clock(_ms(40))
    h.stall(5)

    assert sum("FABRICATED" in w for w in h.logger.warnings) == 1

    # /clock progressing again re-arms the warning for the next stall
    h.clock(_ms(4000))
    h.stall(1)
    assert sum("FABRICATED" in w for w in h.logger.warnings) == 2


def test_a_failing_marker_write_does_not_stop_recording():
    """Recording must survive a marker-write failure; the failure is logged."""
    h = _Harness(BagRecorder)

    def explode(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    h.clock(_ms(0))
    h.clock(_ms(2000))
    h.recorder.write_data = explode
    h.clock(_ms(500))

    assert h.recorder.current_time == _ms(500)
    assert len(h.logger.warnings) == 1
    assert h.recording_segment == 1
    assert any("time-segment marker" in e for e in h.logger.errors)


def test_marker_file_name_matches_the_module_constant():
    """Pins the literal the harness uses against the production constant."""
    assert BagRecorder.SEGMENT_MARKER_FILE == _MARKER_FILE
    assert Recorder.SEGMENT_MARKER_FILE == _MARKER_FILE
