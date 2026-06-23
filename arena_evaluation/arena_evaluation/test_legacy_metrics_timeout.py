from arena_evaluation.scripts.metrics import Config, DoneReason, Metrics


def test_legacy_metrics_timeout_threshold_is_seconds():
    metrics = object.__new__(Metrics)

    assert Config.TIMEOUT_TRESHOLD == 180.0
    assert metrics._get_success(179.9, None) == DoneReason.GOAL_REACHED
    assert metrics._get_success(180.0, None) == DoneReason.TIMEOUT
