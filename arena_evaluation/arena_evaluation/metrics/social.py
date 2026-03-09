from __future__ import annotations

import math
from dataclasses import dataclass

from .common import clamp, l2


@dataclass(frozen=True)
class TTCResult:
    ttc_s: float | None
    distance_at_ttc_m: float | None
    counterpart_id: str | None


def min_distance_to_any_ped(
    robot_xy: tuple[float, float],
    pedestrians_xy: list[tuple[str, float, float]],
) -> tuple[float | None, str | None]:
    if not pedestrians_xy:
        return None, None
    rx, ry = robot_xy
    best_d = float("inf")
    best_id: str | None = None
    for pid, px, py in pedestrians_xy:
        d = l2(px - rx, py - ry)
        if d < best_d:
            best_d = d
            best_id = pid
    return best_d, best_id


def ttc_to_circle(
    *,
    rel_pos_xy: tuple[float, float],
    rel_vel_xy: tuple[float, float],
    radius_m: float,
    horizon_s: float,
) -> float | None:
    """Time-to-collision (circle) using closest-approach check.

    - rel_pos = ped_pos - robot_pos
    - rel_vel = ped_vel - robot_vel

    Returns earliest positive time within horizon where distance <= radius.
    """

    px, py = rel_pos_xy
    vx, vy = rel_vel_xy
    v2 = vx * vx + vy * vy
    if v2 < 1e-12:
        return None

    # time of closest approach
    t_star = -((px * vx + py * vy) / v2)
    if t_star <= 0.0:
        return None

    t_star = clamp(t_star, 0.0, horizon_s)
    cx = px + vx * t_star
    cy = py + vy * t_star
    d_star = l2(cx, cy)
    if d_star > radius_m:
        return None

    # Conservative: return t_star as TTC proxy.
    return t_star


def min_ttc(
    *,
    robot_xy: tuple[float, float],
    robot_vxy: tuple[float, float],
    peds: list[tuple[str, float, float, float, float]],
    collision_radius_m: float,
    horizon_s: float,
) -> TTCResult:
    """Compute minimum TTC across pedestrians.

    peds: list of (pid, px, py, vx, vy) in world frame.
    """

    rx, ry = robot_xy
    rvx, rvy = robot_vxy

    best_t: float | None = None
    best_id: str | None = None
    for pid, px, py, pvx, pvy in peds:
        rel_pos = (px - rx, py - ry)
        rel_vel = (pvx - rvx, pvy - rvy)
        t = ttc_to_circle(
            rel_pos_xy=rel_pos,
            rel_vel_xy=rel_vel,
            radius_m=collision_radius_m,
            horizon_s=horizon_s,
        )
        if t is None:
            continue
        if best_t is None or t < best_t:
            best_t = t
            best_id = pid

    return TTCResult(ttc_s=best_t, distance_at_ttc_m=None, counterpart_id=best_id)