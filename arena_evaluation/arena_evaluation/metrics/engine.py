from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from ..schema import EpisodeLog
from .common import diff, l2, safe_div
from .social import min_distance_to_any_ped, min_ttc
from .vln import ndtw, oracle_error, path_length_xy, navigation_error, spl as spl_metric, sdtw


@dataclass(frozen=True)
class MetricConfig:
    # Termination / success
    goal_tolerance_m: float = 0.25
    time_limit_s: float | None = None

    # Geometry
    robot_radius_m: float = 0.30
    human_radius_m: float = 0.30

    # Social thresholds
    personal_space_radius_m: float = 0.80
    ttc_horizon_s: float = 5.0

    # Motion thresholds
    stall_speed_mps: float = 0.05


@dataclass(frozen=True)
class ScoringProfile:
    """Optional scalar score based on computed metrics.

    Keep this separate from raw metrics so you can publish raw numbers
    while iterating on scoring weights.
    """

    collision_penalty: float = 1000.0
    timeout_penalty: float = 200.0
    discomfort_penalty_per_s: float = 2.0
    time_penalty_per_s: float = 1.0
    success_bonus: float = 100.0


class MetricEngine:
    def __init__(self, config: MetricConfig | None = None):
        self._cfg = config or MetricConfig()

    def compute(self, episode: EpisodeLog, scoring: ScoringProfile | None = None) -> dict:
        """Compute a metric dictionary for one episode."""

        if not episode.frames:
            raise ValueError("episode has no frames")

        cfg = self._merge_config_with_meta(episode)

        goal_xy = (episode.goal_x, episode.goal_y)
        robot_xy = [(f.robot.x, f.robot.y) for f in episode.frames]
        times = [f.robot.t for f in episode.frames]
        dt = [max(0.0, b - a) for a, b in zip(times[:-1], times[1:])]
        duration_s = times[-1] - times[0]

        # Basic VLN metrics
        start_xy = robot_xy[0]
        end_xy = robot_xy[-1]
        nav_err = navigation_error(end_xy=end_xy, goal_xy=goal_xy)
        orc_err = oracle_error(path_xy=robot_xy, goal_xy=goal_xy)
        executed_len = path_length_xy(robot_xy)
        shortest_len = l2(goal_xy[0] - start_xy[0], goal_xy[1] - start_xy[1])

        # Success / timeout detection
        success = nav_err <= cfg.goal_tolerance_m
        timed_out = False
        if episode.meta.truncated is True:
            timed_out = True
        elif cfg.time_limit_s is not None and duration_s >= cfg.time_limit_s - 1e-6:
            timed_out = True

        # Ped distance + comfort violations
        dh_min: float | None = None
        dh_min_id: str | None = None
        comfort_viol_time_s = 0.0
        collision = False
        collision_with: str | None = None

        collision_dist_m = cfg.robot_radius_m + cfg.human_radius_m

        for i, frame in enumerate(episode.frames):
            peds_xy = [(p.pid, p.x, p.y) for p in frame.pedestrians]
            d, pid = min_distance_to_any_ped((frame.robot.x, frame.robot.y), peds_xy)
            if d is not None:
                if dh_min is None or d < dh_min:
                    dh_min = d
                    dh_min_id = pid
                if d < cfg.personal_space_radius_m:
                    if i < len(dt):
                        comfort_viol_time_s += dt[i]
                if d < collision_dist_m:
                    collision = True
                    collision_with = "human"

            # If the logger provides collision events, honor them.
            ev = frame.events or {}
            if isinstance(ev, dict) and ev.get("collision") is True:
                collision = True
                collision_with = str(ev.get("collision_with") or collision_with or "unknown")

        # TTC
        ttc_min_s: float | None = None
        ttc_min_id: str | None = None

        for frame in episode.frames:
            # robot world velocity estimate
            if frame.robot.v is None:
                continue
            rvx = frame.robot.v * math.cos(frame.robot.yaw)
            rvy = frame.robot.v * math.sin(frame.robot.yaw)
            peds = []
            for p in frame.pedestrians:
                if p.vx is None or p.vy is None:
                    continue
                peds.append((p.pid, p.x, p.y, p.vx, p.vy))
            if not peds:
                continue
            res = min_ttc(
                robot_xy=(frame.robot.x, frame.robot.y),
                robot_vxy=(rvx, rvy),
                peds=peds,
                collision_radius_m=collision_dist_m,
                horizon_s=cfg.ttc_horizon_s,
            )
            if res.ttc_s is None:
                continue
            if ttc_min_s is None or res.ttc_s < ttc_min_s:
                ttc_min_s = res.ttc_s
                ttc_min_id = res.counterpart_id

        # Smoothness: velocity/acceleration/jerk (linear speed only)
        v_series = [f.robot.v for f in episode.frames]
        v_series_f = [float(v) for v in v_series if v is not None]
        jerk_rms: float | None = None
        accel_rms: float | None = None
        if len(v_series_f) >= 4 and all(dti > 0.0 for dti in dt[: len(v_series_f) - 1]):
            dv = diff(v_series_f)
            acc = [dv_i / dt_i for dv_i, dt_i in zip(dv, dt[: len(dv)])]
            da = diff(acc)
            jerk = [da_i / dt_i for da_i, dt_i in zip(da, dt[: len(da)])]
            accel_rms = math.sqrt(sum(a * a for a in acc) / max(1, len(acc)))
            jerk_rms = math.sqrt(sum(j * j for j in jerk) / max(1, len(jerk)))

        # Stalled time
        stalled_time_s = 0.0
        for i, frame in enumerate(episode.frames[:-1]):
            v = frame.robot.v
            if v is None:
                continue
            if abs(v) < cfg.stall_speed_mps and navigation_error((frame.robot.x, frame.robot.y), goal_xy) > cfg.goal_tolerance_m:
                stalled_time_s += dt[i]

        # VLN DTW metrics (optional)
        ref_xy = list(episode.reference_path_xy)
        ndtw_val: float | None = None
        sdtw_val: float | None = None
        if ref_xy:
            ndtw_val = ndtw(robot_xy, ref_xy, success_distance_m=cfg.goal_tolerance_m)
            sdtw_val = sdtw(success, robot_xy, ref_xy, success_distance_m=cfg.goal_tolerance_m)

        # SPL
        spl_val = spl_metric(success, shortest_len, executed_len)

        # Compose
        result = {
            "meta": dataclasses.asdict(episode.meta),
            "success": bool(success and not collision and not timed_out),
            "success_nav_only": bool(success),
            "timeout": bool(timed_out),
            "collision": bool(collision),
            "collision_with": collision_with,
            "duration_s": float(duration_s),
            "path_length_m": float(executed_len),
            "shortest_path_m": float(shortest_len),
            "spl": float(spl_val),
            "nav_error_m": float(nav_err),
            "oracle_error_m": float(orc_err),
            "ndtw": None if ndtw_val is None else float(ndtw_val),
            "sdtw": None if sdtw_val is None else float(sdtw_val),
            "dh_min_m": None if dh_min is None else float(dh_min),
            "dh_min_id": dh_min_id,
            "personal_space_viol_time_s": float(comfort_viol_time_s),
            "personal_space_viol_ratio": safe_div(comfort_viol_time_s, max(1e-9, duration_s), default=0.0),
            "ttc_min_s": None if ttc_min_s is None else float(ttc_min_s),
            "ttc_min_id": ttc_min_id,
            "stalled_time_s": float(stalled_time_s),
            "accel_rms": None if accel_rms is None else float(accel_rms),
            "jerk_rms": None if jerk_rms is None else float(jerk_rms),
        }

        if scoring is not None:
            result["score"] = float(self._score(result, scoring=scoring))

        return result

    def _merge_config_with_meta(self, episode: EpisodeLog) -> MetricConfig:
        cfg = self._cfg
        meta = episode.meta

        return MetricConfig(
            goal_tolerance_m=float(meta.goal_tolerance_m or cfg.goal_tolerance_m),
            time_limit_s=cfg.time_limit_s if meta.time_limit_s is None else float(meta.time_limit_s),
            robot_radius_m=float(meta.robot_radius_m or cfg.robot_radius_m),
            human_radius_m=float(meta.human_radius_m or cfg.human_radius_m),
            personal_space_radius_m=cfg.personal_space_radius_m,
            ttc_horizon_s=cfg.ttc_horizon_s,
            stall_speed_mps=cfg.stall_speed_mps,
        )

    @staticmethod
    def _score(metrics: dict, scoring: ScoringProfile) -> float:
        score = 0.0
        if metrics.get("success"):
            score += scoring.success_bonus
        score -= scoring.time_penalty_per_s * float(metrics.get("duration_s") or 0.0)
        score -= scoring.discomfort_penalty_per_s * float(metrics.get("personal_space_viol_time_s") or 0.0)
        if metrics.get("timeout"):
            score -= scoring.timeout_penalty
        if metrics.get("collision"):
            score -= scoring.collision_penalty
        return score