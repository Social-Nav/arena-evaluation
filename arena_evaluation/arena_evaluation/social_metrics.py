#!/usr/bin/env python3
"""Social-navigation metrics for HuNav-backed Arena eval outputs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = {
    "personal_space_radius_m": 1.0,
    "near_miss_radius_m": 0.5,
    "human_collision_radius_m": 0.25,
    "crowd_radius_m": 1.5,
    "crowd_freezing_speed_mps": 0.05,
}


def _safe_load_value(value: str) -> Any:
    value = str(value or '').strip()
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
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def _time_seconds(raw: str) -> float:
    try:
        value = float(raw)
    except Exception:
        return 0.0
    # Arena recorder stores ``sec * 1e10 + nanosec`` for historical reasons;
    # existing metrics normalizes by 1e10, so keep the same convention here.
    return value / 1e10 if abs(value) > 1e6 else value


def _odom_position(row: dict[str, str]) -> tuple[float, float, float] | None:
    data = _safe_load_value(row.get('data', ''))
    if not isinstance(data, dict):
        return None
    position = data.get('position')
    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    return (
        float(position[0]),
        float(position[1]),
        float(position[2]) if len(position) > 2 else 0.0,
    )


def _odom_speed(row: dict[str, str]) -> float:
    data = _safe_load_value(row.get('data', ''))
    velocity = data.get('velocity') if isinstance(data, dict) else None
    if not isinstance(velocity, (list, tuple)) or len(velocity) < 2:
        return 0.0
    return math.hypot(float(velocity[0]), float(velocity[1]))


def _human_agents(row: dict[str, str]) -> list[dict[str, Any]]:
    data = _safe_load_value(row.get('data', ''))
    if isinstance(data, str):
        data = _safe_load_value(data)
    if not isinstance(data, list):
        return []
    agents = []
    for agent in data:
        if not isinstance(agent, dict):
            continue
        position = agent.get('position')
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            continue
        agents.append(agent)
    return agents


def _nearest_human_row(time_sec: float, human_rows: list[dict[str, Any]], start_index: int) -> tuple[int, dict[str, Any] | None]:
    if not human_rows:
        return start_index, None
    best_index = min(max(start_index, 0), len(human_rows) - 1)
    best_dt = abs(float(human_rows[best_index]['time_sec']) - time_sec)
    i = best_index + 1
    while i < len(human_rows):
        dt = abs(float(human_rows[i]['time_sec']) - time_sec)
        if dt > best_dt and human_rows[i]['time_sec'] > time_sec:
            break
        if dt < best_dt:
            best_dt = dt
            best_index = i
        i += 1
    return best_index, human_rows[best_index]


def _path_length(positions: list[tuple[float, float, float]]) -> float:
    return sum(
        math.hypot(positions[i][0] - positions[i - 1][0], positions[i][1] - positions[i - 1][1])
        for i in range(1, len(positions))
    )


def _large_teleports(positions: list[tuple[float, float, float]], threshold_m: float = 5.0) -> list[dict[str, Any]]:
    jumps = []
    for i in range(1, len(positions)):
        distance = math.hypot(positions[i][0] - positions[i - 1][0], positions[i][1] - positions[i - 1][1])
        if distance > threshold_m:
            jumps.append({"index": i, "distance_m": distance})
    return jumps


def _read_base_metrics(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    if not rows:
        return {"present": False, "rows": 0}
    return {"present": True, "rows": len(rows), "first": rows[0]}


def generate_social_metrics(run_dir: str | os.PathLike[str], *, thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    run_path = Path(run_dir)
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    odom_rows = _read_csv(run_path / 'odom.csv')
    human_csv_rows = _read_csv(run_path / 'human_states.csv')

    human_rows = []
    observed_human_ids = set()
    max_humans = 0
    for row in human_csv_rows:
        agents = _human_agents(row)
        for agent in agents:
            observed_human_ids.add(agent.get('id', agent.get('name')))
        max_humans = max(max_humans, len(agents))
        human_rows.append({
            "time_sec": _time_seconds(row.get('time', '0')),
            "agents": agents,
        })

    odom_samples = []
    for row in odom_rows:
        position = _odom_position(row)
        if position is None:
            continue
        odom_samples.append({
            "time_sec": _time_seconds(row.get('time', '0')),
            "position": position,
            "speed": _odom_speed(row),
        })

    min_distance = math.inf
    distance_samples = 0
    personal_space_time = 0.0
    near_miss_count = 0
    human_collision_count = 0
    crowd_freezing_time = 0.0
    personal_active = False
    near_active = False
    collision_active = False
    nearest_index = 0
    positions = [sample['position'] for sample in odom_samples]
    teleports = _large_teleports(positions)

    for idx, sample in enumerate(odom_samples):
        dt = 0.0
        if idx > 0:
            dt = max(0.0, sample['time_sec'] - odom_samples[idx - 1]['time_sec'])
        nearest_index, human_row = _nearest_human_row(sample['time_sec'], human_rows, nearest_index)
        agents = human_row['agents'] if human_row else []
        if not agents:
            personal_active = near_active = collision_active = False
            continue
        rx, ry, _ = sample['position']
        distances = []
        for agent in agents:
            ax, ay = float(agent['position'][0]), float(agent['position'][1])
            distances.append(math.hypot(rx - ax, ry - ay))
        if not distances:
            continue
        distance_samples += 1
        frame_min = min(distances)
        min_distance = min(min_distance, frame_min)

        in_personal = frame_min <= thresholds['personal_space_radius_m']
        in_near_miss = frame_min <= thresholds['near_miss_radius_m']
        in_collision = frame_min <= thresholds['human_collision_radius_m']

        if in_personal:
            personal_space_time += dt
        if in_personal and sample['speed'] <= thresholds['crowd_freezing_speed_mps']:
            crowd_freezing_time += dt
        if in_near_miss and not near_active:
            near_miss_count += 1
        if in_collision and not collision_active:
            human_collision_count += 1

        personal_active = in_personal
        near_active = in_near_miss
        collision_active = in_collision

    humans_present = max_humans > 0 and distance_samples > 0
    base_metrics = _read_base_metrics(run_path / 'metrics.csv')
    social_success = bool(
        humans_present
        and human_collision_count == 0
        and near_miss_count == 0
        and len(teleports) == 0
    )

    duration = 0.0
    if len(odom_samples) >= 2:
        duration = max(0.0, odom_samples[-1]['time_sec'] - odom_samples[0]['time_sec'])

    result = {
        "schema_version": 1,
        "run_dir": str(run_path),
        "thresholds": thresholds,
        "humans_present": humans_present,
        "human_sample_count": sum(1 for row in human_rows if row['agents']),
        "max_humans_observed": max_humans,
        "observed_human_ids": sorted(str(v) for v in observed_human_ids if v is not None),
        "odom_sample_count": len(odom_samples),
        "duration_sec": duration,
        "path_length_m": _path_length(positions),
        "min_human_distance_m": None if math.isinf(min_distance) else min_distance,
        "personal_space_violation_time_sec": personal_space_time,
        "near_miss_count": near_miss_count,
        "human_collision_count": human_collision_count,
        "crowd_freezing_time_sec": crowd_freezing_time,
        "large_teleports": teleports,
        "base_metrics": base_metrics,
        "social_success": social_success,
    }
    output_path = run_path / 'social_metrics.json'
    output_path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate HuNav social navigation metrics for an Arena eval output directory.')
    parser.add_argument('--dir', required=True, help='Eval run directory')
    args = parser.parse_args()
    generate_social_metrics(args.dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
