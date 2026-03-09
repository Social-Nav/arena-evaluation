from __future__ import annotations

import math

from .common import l2, safe_div


def path_length_xy(path_xy: list[tuple[float, float]]) -> float:
    if len(path_xy) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(path_xy[:-1], path_xy[1:]):
        total += l2(x1 - x0, y1 - y0)
    return total


def navigation_error(end_xy: tuple[float, float], goal_xy: tuple[float, float]) -> float:
    return l2(end_xy[0] - goal_xy[0], end_xy[1] - goal_xy[1])


def oracle_error(path_xy: list[tuple[float, float]], goal_xy: tuple[float, float]) -> float:
    if not path_xy:
        return math.inf
    gx, gy = goal_xy
    return min(l2(x - gx, y - gy) for x, y in path_xy)


def spl(success: bool, shortest_path_len: float, executed_path_len: float) -> float:
    if not success:
        return 0.0
    denom = max(shortest_path_len, executed_path_len)
    return safe_div(shortest_path_len, denom, default=0.0)


def _dtw_distance(a_xy: list[tuple[float, float]], b_xy: list[tuple[float, float]]) -> float:
    """Classic DTW on 2D points with L2 cost; O(n*m)."""

    if not a_xy or not b_xy:
        return math.inf

    n, m = len(a_xy), len(b_xy)
    inf = float("inf")
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0

    for i in range(1, n + 1):
        ax, ay = a_xy[i - 1]
        for j in range(1, m + 1):
            bx, by = b_xy[j - 1]
            cost = l2(ax - bx, ay - by)
            dp[i][j] = cost + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def ndtw(
    executed_xy: list[tuple[float, float]],
    reference_xy: list[tuple[float, float]],
    success_distance_m: float,
) -> float:
    """Normalized DTW as used in VLN benchmarks.

    nDTW = exp(-DTW / (|R| * d_th))
    """

    if not executed_xy or not reference_xy:
        return 0.0
    dtw = _dtw_distance(executed_xy, reference_xy)
    denom = max(1.0, float(len(reference_xy)) * float(success_distance_m))
    return math.exp(-dtw / denom)


def sdtw(
    success: bool,
    executed_xy: list[tuple[float, float]],
    reference_xy: list[tuple[float, float]],
    success_distance_m: float,
) -> float:
    return ndtw(executed_xy, reference_xy, success_distance_m) if success else 0.0