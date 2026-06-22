"""Generate social-navigation metrics from Arena recorder CSV artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from bisect import bisect_left
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "personal_space_radius_m": 1.0,
    "near_miss_radius_m": 0.5,
    "human_collision_radius_m": 0.25,
    "robot_radius_m": 0.30,
    "human_radius_m": 0.25,
    "crowd_radius_m": 1.5,
    "crowd_freezing_speed_mps": 0.05,
    "large_teleport_threshold_m": 5.0,
    "human_moving_speed_mps": 0.05,
    "human_motion_min_distance_m": 0.25,
    "robot_moving_speed_mps": 0.02,
    "human_robot_interaction_radius_m": 3.0,
    "min_moving_human_count": 1,
    "min_human_motion_time_sec": 5.0,
    "min_human_robot_motion_overlap_time_sec": 3.0,
    "min_human_robot_interaction_time_sec": 1.0,
    "human_sample_consolidation_window_sec": 0.2,
}

# The current Arena recorder stores /clock-derived integer timestamps with a
# legacy sec*10_000_000_000 + nanosec scale. Keep metric time integration
# compatible with existing artifacts and report the scale explicitly.
RECORDER_TIME_UNITS_PER_SECOND = 10_000_000_000.0


def _parse_value(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value:
        return None
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_odom(run_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in _read_csv(run_dir / "odom.csv"):
        data = _parse_value(row.get("data"))
        if not isinstance(data, dict):
            continue
        position = data.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        velocity = data.get("velocity")
        vx = vy = 0.0
        if isinstance(velocity, (list, tuple)):
            if velocity:
                vx = _as_float(velocity[0])
            if len(velocity) > 1:
                vy = _as_float(velocity[1])
        samples.append(
            {
                "time": int(_as_float(row.get("time"))),
                "x": _as_float(position[0]),
                "y": _as_float(position[1]),
                "speed": math.hypot(vx, vy),
            }
        )
    return sorted(samples, key=lambda sample: sample["time"])


def _human_csv_path(run_dir: Path) -> Path:
    human_states = run_dir / "human_states.csv"
    if human_states.exists():
        return human_states
    return run_dir / "pedsim_agents_data.csv"


def _read_humans(run_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    path = _human_csv_path(run_dir)
    samples: list[dict[str, Any]] = []
    for order, row in enumerate(_read_csv(path)):
        agents = _parse_value(row.get("data"))
        if isinstance(agents, str):
            agents = _parse_value(agents)
        if not isinstance(agents, list):
            agents = []
        parsed_agents: list[dict[str, Any]] = []
        for agent in agents:
            if not isinstance(agent, dict):
                continue
            position = agent.get("position")
            if not isinstance(position, (list, tuple)) or len(position) < 2:
                continue
            agent_id = str(agent.get("id") or agent.get("name") or len(parsed_agents))
            velocity = agent.get("velocity")
            vx = vy = 0.0
            if isinstance(velocity, (list, tuple)):
                if velocity:
                    vx = _as_float(velocity[0])
                if len(velocity) > 1:
                    vy = _as_float(velocity[1])
            parsed_agents.append(
                {
                    "id": agent_id,
                    "x": _as_float(position[0]),
                    "y": _as_float(position[1]),
                    "vx": vx,
                    "vy": vy,
                }
            )
        samples.append(
            {
                "time": int(_as_float(row.get("time"))),
                "agents": parsed_agents,
                "order": order,
            }
        )
    return path, sorted(samples, key=lambda sample: sample["time"])


def _consolidate_human_samples(
    human_samples: list[dict[str, Any]],
    window_sec: float,
) -> list[dict[str, Any]]:
    if not human_samples:
        return []
    window_units = int(max(window_sec, 0.0) * RECORDER_TIME_UNITS_PER_SECOND)
    if window_units <= 0:
        return human_samples

    buckets: dict[int, dict[str, Any]] = {}
    for sample in human_samples:
        bucket = int(sample["time"]) // window_units
        current = buckets.get(bucket)
        if current is None or int(sample.get("order", -1)) >= int(current.get("order", -1)):
            buckets[bucket] = sample

    consolidated = sorted(buckets.values(), key=lambda sample: sample["time"])
    for sample in consolidated:
        sample.pop("order", None)
    return consolidated


def _nearest_human_sample(human_samples: list[dict[str, Any]], time_ns: int) -> dict[str, Any] | None:
    if not human_samples:
        return None
    times = [sample["time"] for sample in human_samples]
    idx = bisect_left(times, time_ns)
    if idx <= 0:
        return human_samples[0]
    if idx >= len(human_samples):
        return human_samples[-1]
    before = human_samples[idx - 1]
    after = human_samples[idx]
    if abs(time_ns - before["time"]) <= abs(after["time"] - time_ns):
        return before
    return after


def _time_seconds(sample: dict[str, Any]) -> float:
    return int(sample["time"]) / RECORDER_TIME_UNITS_PER_SECOND


def _human_motion_summary(
    human_samples: list[dict[str, Any]],
    cfg: dict[str, float],
) -> dict[str, Any]:
    totals: dict[str, float] = {}
    active_intervals: list[tuple[int, int]] = []
    moving_speed = float(cfg["human_moving_speed_mps"])
    min_distance = float(cfg["human_motion_min_distance_m"])
    teleport_threshold = float(cfg["large_teleport_threshold_m"])

    for idx in range(len(human_samples) - 1):
        current = human_samples[idx]
        next_sample = human_samples[idx + 1]
        dt = _dt_seconds(current, next_sample)
        if dt <= 0.0:
            continue

        current_agents = {
            str(agent["id"]): agent
            for agent in current.get("agents", [])
            if isinstance(agent, dict) and "id" in agent
        }
        next_agents = {
            str(agent["id"]): agent
            for agent in next_sample.get("agents", [])
            if isinstance(agent, dict) and "id" in agent
        }

        any_human_moving = False
        for agent_id, agent in current_agents.items():
            next_agent = next_agents.get(agent_id)
            if not next_agent:
                continue
            distance = math.hypot(next_agent["x"] - agent["x"], next_agent["y"] - agent["y"])
            if distance > teleport_threshold:
                continue
            totals[agent_id] = totals.get(agent_id, 0.0) + distance
            if (distance / dt) >= moving_speed:
                any_human_moving = True

        if any_human_moving:
            active_intervals.append((int(current["time"]), int(next_sample["time"])))

    moving_ids = sorted(agent_id for agent_id, total in totals.items() if total >= min_distance)
    return {
        "human_motion_total_m": float(sum(totals.values())),
        "human_motion_by_id_m": {agent_id: float(total) for agent_id, total in sorted(totals.items())},
        "moving_human_ids": moving_ids,
        "moving_human_count": len(moving_ids),
        "human_motion_time_sec": float(sum((end - start) / RECORDER_TIME_UNITS_PER_SECOND for start, end in active_intervals)),
        "human_motion_intervals": active_intervals,
    }


def _intervals_overlap(intervals: list[tuple[int, int]], start: int, end: int) -> bool:
    for interval_start, interval_end in intervals:
        if interval_end <= start:
            continue
        if interval_start >= end:
            break
        return True
    return False


def _path_length_and_teleports(
    odom_samples: list[dict[str, Any]],
    threshold_m: float,
) -> tuple[float, list[dict[str, Any]]]:
    path_length = 0.0
    teleports: list[dict[str, Any]] = []
    for idx in range(1, len(odom_samples)):
        prev = odom_samples[idx - 1]
        curr = odom_samples[idx]
        distance = math.hypot(curr["x"] - prev["x"], curr["y"] - prev["y"])
        path_length += distance
        if distance > threshold_m:
            teleports.append({"index": idx, "distance_m": distance})
    return path_length, teleports


def _dt_seconds(current: dict[str, Any], next_sample: dict[str, Any] | None) -> float:
    if next_sample is None:
        return 0.0
    return max(0.0, (int(next_sample["time"]) - int(current["time"])) / RECORDER_TIME_UNITS_PER_SECOND)


def _read_base_metrics(run_dir: Path) -> dict[str, Any]:
    rows = _read_csv(run_dir / "metrics.csv")
    if not rows:
        return {"rows": 0, "first": {}}
    first = dict(rows[0])
    for key in ("path_length", "time", "angle_over_length"):
        if key in first:
            parsed = _parse_value(first[key])
            if parsed is not None:
                first[key] = parsed
            else:
                first[key] = _as_float(first[key], first[key])
    for key in ("goal", "start", "collisions"):
        if key in first:
            parsed = _parse_value(first[key])
            if parsed is not None:
                first[key] = parsed
    return {"rows": len(rows), "first": first}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def generate_social_metrics(
    run_dir: str | os.PathLike[str],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    cfg = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    odom_samples = _read_odom(run_path)
    human_path, human_samples = _read_humans(run_path)
    raw_human_sample_count = len(human_samples)
    human_samples = _consolidate_human_samples(
        human_samples,
        float(cfg.get("human_sample_consolidation_window_sec", 0.0) or 0.0),
    )

    path_length_m, large_teleports = _path_length_and_teleports(
        odom_samples,
        cfg["large_teleport_threshold_m"],
    )

    observed_ids: set[str] = set()
    max_humans_observed = 0
    human_nonempty_samples = 0
    min_human_distance_m: float | None = None
    personal_space_violation_time_sec = 0.0
    crowd_freezing_time_sec = 0.0
    robot_motion_time_sec = 0.0
    human_robot_motion_overlap_time_sec = 0.0
    human_robot_interaction_time_sec = 0.0
    near_miss_count = 0
    human_collision_count = 0
    footprint_personal_space_violation_time_sec = 0.0
    footprint_near_miss_count = 0
    footprint_human_collision_count = 0
    min_footprint_clearance_m: float | None = None
    near_miss_active = False
    human_collision_active = False
    footprint_near_miss_active = False
    footprint_human_collision_active = False
    human_motion = _human_motion_summary(human_samples, cfg)
    human_motion_intervals = human_motion.pop("human_motion_intervals")
    robot_radius = float(cfg.get("robot_radius_m", 0.0) or 0.0)
    human_radius = float(cfg.get("human_radius_m", 0.0) or 0.0)
    footprint_collision_threshold = robot_radius + human_radius

    for idx, odom in enumerate(odom_samples):
        human_sample = _nearest_human_sample(human_samples, odom["time"])
        agents = human_sample.get("agents", []) if isinstance(human_sample, dict) else []
        if not agents:
            near_miss_active = False
            human_collision_active = False
            continue

        human_nonempty_samples += 1
        max_humans_observed = max(max_humans_observed, len(agents))
        nearest = None
        for agent in agents:
            observed_ids.add(str(agent["id"]))
            distance = math.hypot(agent["x"] - odom["x"], agent["y"] - odom["y"])
            if nearest is None or distance < nearest:
                nearest = distance
        if nearest is None:
            continue

        if min_human_distance_m is None or nearest < min_human_distance_m:
            min_human_distance_m = nearest
        footprint_clearance = nearest - footprint_collision_threshold
        if min_footprint_clearance_m is None or footprint_clearance < min_footprint_clearance_m:
            min_footprint_clearance_m = footprint_clearance

        next_odom = odom_samples[idx + 1] if idx + 1 < len(odom_samples) else None
        dt = _dt_seconds(odom, next_odom)
        robot_moving = False
        human_moving = False
        if next_odom is not None and dt > 0.0:
            robot_step_distance = math.hypot(next_odom["x"] - odom["x"], next_odom["y"] - odom["y"])
            robot_step_speed = robot_step_distance / dt
            robot_moving = max(robot_step_speed, float(odom.get("speed", 0.0) or 0.0)) >= cfg["robot_moving_speed_mps"]
            human_moving = _intervals_overlap(
                human_motion_intervals,
                int(odom["time"]),
                int(next_odom["time"]),
            )
            if robot_moving:
                robot_motion_time_sec += dt
            if robot_moving and human_moving:
                human_robot_motion_overlap_time_sec += dt
                if nearest <= cfg["human_robot_interaction_radius_m"]:
                    human_robot_interaction_time_sec += dt
        if nearest < cfg["personal_space_radius_m"]:
            personal_space_violation_time_sec += dt
            if odom["speed"] < cfg["crowd_freezing_speed_mps"]:
                crowd_freezing_time_sec += dt
        if footprint_clearance < cfg["personal_space_radius_m"]:
            footprint_personal_space_violation_time_sec += dt

        in_near_miss = nearest < cfg["near_miss_radius_m"]
        if in_near_miss and not near_miss_active:
            near_miss_count += 1
        near_miss_active = in_near_miss

        in_human_collision = nearest < cfg["human_collision_radius_m"]
        if in_human_collision and not human_collision_active:
            human_collision_count += 1
        human_collision_active = in_human_collision

        in_footprint_near_miss = footprint_clearance < cfg["near_miss_radius_m"]
        if in_footprint_near_miss and not footprint_near_miss_active:
            footprint_near_miss_count += 1
        footprint_near_miss_active = in_footprint_near_miss

        in_footprint_human_collision = footprint_clearance < 0.0
        if in_footprint_human_collision and not footprint_human_collision_active:
            footprint_human_collision_count += 1
        footprint_human_collision_active = in_footprint_human_collision

    humans_present = human_nonempty_samples > 0
    legacy_social_success = (
        humans_present
        and human_collision_count == 0
        and near_miss_count == 0
        and not large_teleports
    )
    dynamic_scene_success = (
        humans_present
        and human_motion["moving_human_count"] >= int(cfg["min_moving_human_count"])
        and human_motion["human_motion_time_sec"] >= cfg["min_human_motion_time_sec"]
        and human_robot_motion_overlap_time_sec >= cfg["min_human_robot_motion_overlap_time_sec"]
        and human_robot_interaction_time_sec >= cfg["min_human_robot_interaction_time_sec"]
    )
    strict_task_metrics = _read_json(run_path / "vln_task_metrics.json")
    strict_social_failure_reasons: list[str] = []
    if not dynamic_scene_success:
        strict_social_failure_reasons.append("dynamic_scene_failed")
    if footprint_human_collision_count > 0:
        strict_social_failure_reasons.append("footprint_human_collision")
    if footprint_near_miss_count > 0:
        strict_social_failure_reasons.append("footprint_near_miss")
    if near_miss_count > 0:
        strict_social_failure_reasons.append("point_near_miss")
    if human_collision_count > 0:
        strict_social_failure_reasons.append("point_human_collision")
    if large_teleports:
        strict_social_failure_reasons.append("large_teleport")
    if strict_task_metrics:
        static_occupancy = strict_task_metrics.get("static_occupancy") or {}
        commanded_stuck = strict_task_metrics.get("commanded_stuck") or {}
        if int(static_occupancy.get("collision_sample_count") or 0) > 0:
            strict_social_failure_reasons.append("static_occupancy_collision")
        if float(commanded_stuck.get("commanded_stuck_time_sec") or 0.0) > 0.0:
            strict_social_failure_reasons.append("commanded_stuck")
    else:
        strict_social_failure_reasons.append("missing_vln_task_metrics")
    strict_social_success = not strict_social_failure_reasons

    result = {
        "schema_version": 1,
        "run_dir": str(run_path),
        "time_scale": {
            "source": "arena_recorder_legacy_clock",
            "raw_units_per_second": RECORDER_TIME_UNITS_PER_SECOND,
            "raw_time_field": "time",
        },
        "human_source_csv": human_path.name,
        "humans_present": humans_present,
        "human_sample_count": len(human_samples),
        "human_raw_sample_count": raw_human_sample_count,
        "human_nonempty_sample_count": human_nonempty_samples,
        "odom_sample_count": len(odom_samples),
        "max_humans_observed": max_humans_observed,
        "observed_human_ids": sorted(observed_ids),
        "path_length_m": path_length_m,
        "min_human_distance_m": min_human_distance_m,
        "min_footprint_clearance_m": min_footprint_clearance_m,
        "personal_space_violation_time_sec": personal_space_violation_time_sec,
        "footprint_personal_space_violation_time_sec": footprint_personal_space_violation_time_sec,
        "near_miss_count": near_miss_count,
        "human_collision_count": human_collision_count,
        "footprint_near_miss_count": footprint_near_miss_count,
        "footprint_human_collision_count": footprint_human_collision_count,
        "crowd_freezing_time_sec": crowd_freezing_time_sec,
        "robot_motion_time_sec": robot_motion_time_sec,
        "human_robot_motion_overlap_time_sec": human_robot_motion_overlap_time_sec,
        "human_robot_interaction_time_sec": human_robot_interaction_time_sec,
        "dynamic_scene_success": dynamic_scene_success,
        **human_motion,
        "large_teleports": large_teleports,
        "social_success": strict_social_success,
        "legacy_social_success": legacy_social_success,
        "strict_social_success": strict_social_success,
        "strict_social_failure_reasons": strict_social_failure_reasons,
        "strict_task_metrics_path": str(run_path / "vln_task_metrics.json") if strict_task_metrics else None,
        "strict_task_success": strict_task_metrics.get("strict_task_success") if strict_task_metrics else None,
        "strict_task_failure_reasons": strict_task_metrics.get("strict_task_failure_reasons") if strict_task_metrics else [],
        "review_intervals": {
            "commanded_stuck": (strict_task_metrics.get("commanded_stuck") or {}).get("commanded_stuck_intervals", [])
            if strict_task_metrics
            else [],
            "static_occupancy": (strict_task_metrics.get("static_occupancy") or {}).get("intervals", [])
            if strict_task_metrics
            else [],
        },
        "video_paths": {
            "sim_top_down": str(run_path / "videos" / "episode_0000" / "sim_top_down.mp4"),
            "ego_debug_overlay": str(run_path / "videos" / "episode_0000" / "ego_debug_overlay.mp4"),
        },
        "thresholds": cfg,
        "base_metrics": _read_base_metrics(run_path),
    }
    (run_path / "social_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Arena social-navigation metrics.")
    parser.add_argument("--dir", required=True, help="Eval run directory containing odom and human CSV artifacts")
    args = parser.parse_args()
    generate_social_metrics(args.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
