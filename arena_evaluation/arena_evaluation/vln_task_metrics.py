"""Strict VLN task metrics for Arena GRScenes/InternNav eval artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from arena_evaluation.metrics.vln import navigation_error, ndtw, oracle_error, path_length_xy, sdtw, spl


RECORDER_TIME_UNITS_PER_SECOND = 10_000_000_000.0

DEFAULT_THRESHOLDS = {
    "goal_tolerance_m": 0.75,
    "start_goal_consistency_tolerance_m": 0.25,
    "robot_radius_m": 0.30,
    "commanded_speed_threshold_mps": 0.05,
    "stuck_motion_threshold_mps": 0.02,
    "stuck_min_duration_sec": 5.0,
    "large_teleport_threshold_m": 5.0,
    "max_static_collision_samples": 0,
    "max_commanded_stuck_time_sec": 0.0,
    "timeout_margin_sec": 1.0,
}


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


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _time_seconds(raw_time: int | float) -> float:
    return float(raw_time) / RECORDER_TIME_UNITS_PER_SECOND


def _xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return (_as_float(value[0]), _as_float(value[1]))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _read_odom(run_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in _read_csv(run_dir / "odom.csv"):
        data = _parse_value(row.get("data"))
        if not isinstance(data, dict):
            continue
        position = _xy(data.get("position"))
        if position is None:
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
                "x": position[0],
                "y": position[1],
                "speed": math.hypot(vx, vy),
            }
        )
    return sorted(samples, key=lambda sample: sample["time"])


def _read_cmd_vel(run_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for row in _read_csv(run_dir / "cmd_vel.csv"):
        data = _parse_value(row.get("data"))
        if not isinstance(data, (list, tuple)):
            continue
        vx = _as_float(data[0]) if data else 0.0
        vy = _as_float(data[1]) if len(data) > 1 else 0.0
        wz = _as_float(data[2]) if len(data) > 2 else 0.0
        samples.append(
            {
                "time": int(_as_float(row.get("time"))),
                "vx": vx,
                "vy": vy,
                "wz": wz,
                "linear_speed": math.hypot(vx, vy),
            }
        )
    return sorted(samples, key=lambda sample: sample["time"])


def _latest_before_or_equal(samples: list[dict[str, Any]], time_raw: int) -> dict[str, Any] | None:
    latest: dict[str, Any] | None = None
    for sample in samples:
        if int(sample["time"]) > time_raw:
            break
        latest = sample
    return latest


def _read_start_goal_csv(run_dir: Path) -> dict[str, Any]:
    rows = _read_csv(run_dir / "start_goal.csv")
    if not rows:
        return {"present": False}
    first = rows[0]
    start = _xy(_parse_value(first.get("start")))
    goal = _xy(_parse_value(first.get("goal")))
    return {
        "present": True,
        "rows": len(rows),
        "start_xy": list(start) if start else None,
        "goal_xy": list(goal) if goal else None,
    }


def _find_repo_root(run_dir: Path) -> Path:
    for candidate in (
        Path(os.environ.get("ARENA_SOURCE_DIR", "")),
        Path("/opt/arena_ws/src/Arena"),
        Path("/home/ubuntu/arena_jazzy_ws/src/Arena"),
        run_dir.parents[1] / "src" / "Arena" if len(run_dir.parents) > 1 else Path(),
    ):
        if candidate and (candidate / "arena_simulation_setup" / "worlds").exists():
            return candidate
    return Path("/home/ubuntu/arena_jazzy_ws/src/Arena")


def _manifest_params(run_dir: Path) -> dict[str, Any]:
    manifest = _read_yaml(run_dir / "run_manifest.yaml")
    params = manifest.get("parameters")
    if not isinstance(params, dict):
        params = {}
    return {"manifest": manifest, "parameters": params}


def _scenario_paths(run_dir: Path) -> dict[str, Any]:
    manifest_data = _manifest_params(run_dir)
    params = manifest_data["parameters"]
    world = str(params.get("world") or "")
    scenario = str(params.get("scenario_file") or params.get("scenario_config_id") or "default")
    repo_root = _find_repo_root(run_dir)
    world_dir = repo_root / "arena_simulation_setup" / "worlds" / world if world else None
    scenario_dir = world_dir / "scenarios" / scenario if world_dir else None
    scenario_path = scenario_dir / "scenario.yaml" if scenario_dir else None
    map_yaml = world_dir / "map" / "map.yaml" if world_dir else None
    return {
        "world": world,
        "scenario": scenario,
        "repo_root": str(repo_root),
        "scenario_path": scenario_path,
        "map_yaml": map_yaml,
        "instruction": params.get("vln_instruction") or "",
    }


def _read_scenario_contract(run_dir: Path) -> dict[str, Any]:
    paths = _scenario_paths(run_dir)
    scenario_path = paths["scenario_path"]
    scenario = _read_yaml(scenario_path) if isinstance(scenario_path, Path) else {}
    robots = scenario.get("robots") if isinstance(scenario, dict) else None
    robot = robots[0] if isinstance(robots, list) and robots and isinstance(robots[0], dict) else {}
    start = _xy(robot.get("start"))
    goal = _xy(robot.get("goal"))
    dynamic = scenario.get("dynamic") if isinstance(scenario, dict) else []
    return {
        "world": paths["world"],
        "scenario": paths["scenario"],
        "scenario_path": str(scenario_path) if isinstance(scenario_path, Path) else None,
        "map_yaml": str(paths["map_yaml"]) if isinstance(paths["map_yaml"], Path) else None,
        "instruction": paths["instruction"],
        "start_xy": list(start) if start else None,
        "goal_xy": list(goal) if goal else None,
        "dynamic_actor_count": len(dynamic) if isinstance(dynamic, list) else 0,
    }


def _task_contract_summary(contract: dict[str, Any]) -> dict[str, Any]:
    """Document which language-task predicates this strict metric scores.

    GRScenes instructions can contain rich landmark and social clauses.  The
    minimal strict task metric currently grounds them through the recorded
    native scenario goal rather than through a full BDDL semantic evaluator.
    """
    evaluated = [
        "goal_reached(robot, native_scenario_goal)",
        "robot_moved(robot, min_path_length_m=0.1)",
        "not_timeout",
        "not_commanded_stuck",
        "not_static_occupancy_collision",
        "not_large_teleport",
        "start_goal_matches_native_scenario",
    ]
    unsupported = [
        "landmark_sequence_followed",
        "object_or_region_facing",
        "orientation_at_goal",
        "natural_language_spatial_relations",
        "bddl_semantic_predicates",
    ]
    return {
        "contract_type": "native_scenario_goal",
        "instruction_source": "run_manifest.vln_instruction",
        "goal_source": "arena_simulation_setup.worlds.<world>.scenarios.<scenario>.scenario.yaml robots[0].goal",
        "goal_tolerance_source": "vln_task_metrics.DEFAULT_THRESHOLDS.goal_tolerance_m",
        "evaluated_predicates": evaluated,
        "unsupported_predicates": unsupported,
        "unsupported_predicates_present": bool(str(contract.get("instruction") or "").strip()),
        "bddl_evaluator": {
            "available_in_package": True,
            "used_for_this_score": False,
            "reason": "GRScenes recorded episodes currently provide native scenario goals and free-form instructions, not grounded per-episode BDDL task files.",
        },
    }


def _reference_path(contract: dict[str, Any]) -> list[tuple[float, float]]:
    start = _xy(contract.get("start_xy"))
    goal = _xy(contract.get("goal_xy"))
    return [point for point in (start, goal) if point is not None]


def _read_base_metrics(run_dir: Path) -> dict[str, Any]:
    rows = _read_csv(run_dir / "metrics.csv")
    if not rows:
        return {"present": False, "rows": 0, "first": {}}
    first = dict(rows[0])
    return {"present": True, "rows": len(rows), "first": first}


def _episode_timing(run_dir: Path, odom_samples: list[dict[str, Any]], cfg: dict[str, float]) -> dict[str, Any]:
    manifest_data = _manifest_params(run_dir)
    params = manifest_data["parameters"]
    result = manifest_data["manifest"].get("result", {}) if isinstance(manifest_data["manifest"], dict) else {}
    timeout_sec = _as_float(params.get("timeout"), math.inf)
    if not math.isfinite(timeout_sec) or timeout_sec <= 0.0:
        timeout_sec = math.inf
    duration_sec = 0.0
    if len(odom_samples) >= 2:
        duration_sec = _time_seconds(int(odom_samples[-1]["time"]) - int(odom_samples[0]["time"]))
    timed_out_by_manifest = bool(result.get("timed_out")) or str(result.get("end_reason") or "").lower() == "timeout"
    timed_out_by_duration = math.isfinite(timeout_sec) and duration_sec >= max(0.0, timeout_sec - float(cfg["timeout_margin_sec"]))
    return {
        "timeout_sec": timeout_sec if math.isfinite(timeout_sec) else None,
        "duration_sec": duration_sec,
        "timed_out_by_manifest": timed_out_by_manifest,
        "timed_out_by_duration": timed_out_by_duration,
        "timed_out": timed_out_by_manifest or timed_out_by_duration,
        "end_reason": result.get("end_reason"),
        "finished_observed": bool(result.get("finished_observed")),
    }


class OccupancyMap:
    def __init__(self, map_yaml: Path, robot_radius_m: float) -> None:
        self.map_yaml = map_yaml
        self.metadata = _read_yaml(map_yaml)
        image_name = self.metadata.get("image")
        image_path = map_yaml.parent / str(image_name) if image_name else None
        self.image_path = image_path
        self.available = bool(image_path and image_path.exists())
        self.resolution = _as_float(self.metadata.get("resolution"), 0.0)
        origin = self.metadata.get("origin") or [0.0, 0.0, 0.0]
        self.origin_x = _as_float(origin[0]) if isinstance(origin, list) and origin else 0.0
        self.origin_y = _as_float(origin[1]) if isinstance(origin, list) and len(origin) > 1 else 0.0
        self.negate = int(_as_float(self.metadata.get("negate"), 0.0))
        self.occupied_thresh = _as_float(self.metadata.get("occupied_thresh"), 0.65)
        self.radius_px = max(0, int(math.ceil(robot_radius_m / self.resolution))) if self.resolution > 0 else 0
        self.width = 0
        self.height = 0
        self._occupied: set[tuple[int, int]] = set()
        if self.available and image_path is not None:
            self._load(image_path)

    def _load(self, image_path: Path) -> None:
        image = Image.open(image_path).convert("L")
        self.width, self.height = image.size
        pix = image.load()
        occupied: set[tuple[int, int]] = set()
        for y in range(self.height):
            for x in range(self.width):
                value = pix[x, y] / 255.0
                occ = value if self.negate else 1.0 - value
                if occ >= self.occupied_thresh:
                    occupied.add((x, y))
        self._occupied = occupied

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        px = int(math.floor((x - self.origin_x) / self.resolution))
        py_from_bottom = int(math.floor((y - self.origin_y) / self.resolution))
        return px, self.height - 1 - py_from_bottom

    def is_occupied_footprint(self, x: float, y: float) -> tuple[bool, tuple[int, int] | None]:
        if not self.available or self.resolution <= 0:
            return False, None
        px, py = self.world_to_pixel(x, y)
        if px < 0 or py < 0 or px >= self.width or py >= self.height:
            return True, (px, py)
        radius = self.radius_px
        for oy in range(-radius, radius + 1):
            for ox in range(-radius, radius + 1):
                if ox * ox + oy * oy > radius * radius:
                    continue
                candidate = (px + ox, py + oy)
                if candidate in self._occupied:
                    return True, candidate
        return False, None

    def summary(self) -> dict[str, Any]:
        return {
            "map_yaml": str(self.map_yaml),
            "image": str(self.image_path) if self.image_path else None,
            "available": self.available,
            "resolution": self.resolution,
            "origin": [self.origin_x, self.origin_y],
            "width": self.width,
            "height": self.height,
            "robot_radius_px": self.radius_px,
        }


def _static_occupancy_collisions(
    odom_samples: list[dict[str, Any]],
    map_yaml: str | None,
    robot_radius_m: float,
) -> dict[str, Any]:
    if not map_yaml:
        return {"map_available": False, "collision_sample_count": 0, "collision_samples": [], "intervals": []}
    occ_map = OccupancyMap(Path(map_yaml), robot_radius_m)
    collision_samples: list[dict[str, Any]] = []
    intervals: list[dict[str, float]] = []
    active_start: int | None = None
    active_end: int | None = None
    for sample in odom_samples:
        occupied, pixel = occ_map.is_occupied_footprint(sample["x"], sample["y"])
        if occupied:
            if len(collision_samples) < 50:
                collision_samples.append(
                    {
                        "time_sec": _time_seconds(sample["time"]),
                        "position": [sample["x"], sample["y"]],
                        "pixel": list(pixel) if pixel else None,
                    }
                )
            if active_start is None:
                active_start = sample["time"]
            active_end = sample["time"]
        elif active_start is not None and active_end is not None:
            intervals.append({"start_sec": _time_seconds(active_start), "end_sec": _time_seconds(active_end)})
            active_start = None
            active_end = None
    if active_start is not None and active_end is not None:
        intervals.append({"start_sec": _time_seconds(active_start), "end_sec": _time_seconds(active_end)})
    return {
        **occ_map.summary(),
        "map_available": occ_map.available,
        "collision_sample_count": sum(1 for s in odom_samples if occ_map.is_occupied_footprint(s["x"], s["y"])[0]),
        "collision_samples": collision_samples,
        "intervals": intervals,
    }


def _commanded_stuck_intervals(
    odom_samples: list[dict[str, Any]],
    cmd_samples: list[dict[str, Any]],
    cfg: dict[str, float],
) -> dict[str, Any]:
    intervals: list[dict[str, float]] = []
    active_start: int | None = None
    active_end: int | None = None
    total = 0.0
    commanded_threshold = float(cfg["commanded_speed_threshold_mps"])
    motion_threshold = float(cfg["stuck_motion_threshold_mps"])
    min_duration = float(cfg["stuck_min_duration_sec"])

    for idx in range(len(odom_samples) - 1):
        current = odom_samples[idx]
        nxt = odom_samples[idx + 1]
        dt = max(0.0, _time_seconds(nxt["time"] - current["time"]))
        if dt <= 0.0:
            continue
        cmd = _latest_before_or_equal(cmd_samples, int(current["time"]))
        commanded = bool(cmd and float(cmd["linear_speed"]) >= commanded_threshold)
        step_speed = _distance((current["x"], current["y"]), (nxt["x"], nxt["y"])) / dt
        stuck = commanded and step_speed < motion_threshold
        if stuck:
            if active_start is None:
                active_start = int(current["time"])
            active_end = int(nxt["time"])
        elif active_start is not None and active_end is not None:
            duration = _time_seconds(active_end - active_start)
            if duration >= min_duration:
                intervals.append({"start_sec": _time_seconds(active_start), "end_sec": _time_seconds(active_end), "duration_sec": duration})
                total += duration
            active_start = None
            active_end = None
    if active_start is not None and active_end is not None:
        duration = _time_seconds(active_end - active_start)
        if duration >= min_duration:
            intervals.append({"start_sec": _time_seconds(active_start), "end_sec": _time_seconds(active_end), "duration_sec": duration})
            total += duration
    return {"commanded_stuck_time_sec": total, "commanded_stuck_intervals": intervals}


def _large_teleports(odom_samples: list[dict[str, Any]], threshold_m: float) -> list[dict[str, Any]]:
    teleports: list[dict[str, Any]] = []
    for idx in range(1, len(odom_samples)):
        prev = odom_samples[idx - 1]
        curr = odom_samples[idx]
        dist = _distance((prev["x"], prev["y"]), (curr["x"], curr["y"]))
        if dist > threshold_m:
            teleports.append({"index": idx, "time_sec": _time_seconds(curr["time"]), "distance_m": dist})
    return teleports


def generate_vln_task_metrics(
    run_dir: str | os.PathLike[str],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    cfg = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    odom_samples = _read_odom(run_path)
    cmd_samples = _read_cmd_vel(run_path)
    contract = _read_scenario_contract(run_path)
    start_goal_csv = _read_start_goal_csv(run_path)
    reference_xy = _reference_path(contract)
    executed_xy = [(sample["x"], sample["y"]) for sample in odom_samples]

    goal_xy = _xy(contract.get("goal_xy"))
    start_xy = _xy(contract.get("start_xy"))
    final_xy = executed_xy[-1] if executed_xy else None
    nav_error = navigation_error(final_xy, goal_xy) if final_xy and goal_xy else math.inf
    oracle = oracle_error(executed_xy, goal_xy) if goal_xy else math.inf
    goal_reached = nav_error <= float(cfg["goal_tolerance_m"])
    executed_len = path_length_xy(executed_xy)
    shortest_len = path_length_xy(reference_xy)
    metric_spl = spl(goal_reached, shortest_len, executed_len)
    metric_ndtw = ndtw(executed_xy, reference_xy, float(cfg["goal_tolerance_m"])) if reference_xy else 0.0
    metric_sdtw = sdtw(goal_reached, executed_xy, reference_xy, float(cfg["goal_tolerance_m"])) if reference_xy else 0.0

    start_goal_consistency: dict[str, Any] = {
        "start_goal_csv_present": bool(start_goal_csv.get("present")),
        "scenario_start_xy": contract.get("start_xy"),
        "scenario_goal_xy": contract.get("goal_xy"),
        "csv_start_xy": start_goal_csv.get("start_xy"),
        "csv_goal_xy": start_goal_csv.get("goal_xy"),
        "start_delta_m": None,
        "goal_delta_m": None,
        "pass": False,
    }
    if start_xy and goal_xy and start_goal_csv.get("start_xy") and start_goal_csv.get("goal_xy"):
        csv_start = _xy(start_goal_csv.get("start_xy"))
        csv_goal = _xy(start_goal_csv.get("goal_xy"))
        start_delta = _distance(start_xy, csv_start) if csv_start else math.inf
        goal_delta = _distance(goal_xy, csv_goal) if csv_goal else math.inf
        start_goal_consistency.update(
            {
                "start_delta_m": start_delta,
                "goal_delta_m": goal_delta,
                "pass": start_delta <= cfg["start_goal_consistency_tolerance_m"]
                and goal_delta <= cfg["start_goal_consistency_tolerance_m"],
            }
        )

    stuck = _commanded_stuck_intervals(odom_samples, cmd_samples, cfg)
    occupancy = _static_occupancy_collisions(odom_samples, contract.get("map_yaml"), float(cfg["robot_radius_m"]))
    teleports = _large_teleports(odom_samples, float(cfg["large_teleport_threshold_m"]))
    timing = _episode_timing(run_path, odom_samples, cfg)

    failure_reasons: list[str] = []
    if not odom_samples:
        failure_reasons.append("missing_odom")
    if not goal_xy:
        failure_reasons.append("missing_scenario_goal")
    if timing["timed_out"] and not goal_reached:
        failure_reasons.append("episode_timeout")
    if not goal_reached:
        failure_reasons.append("goal_not_reached")
    if not start_goal_consistency["pass"]:
        failure_reasons.append("start_goal_inconsistent")
    if stuck["commanded_stuck_time_sec"] > float(cfg["max_commanded_stuck_time_sec"]):
        failure_reasons.append("commanded_stuck")
    if occupancy["collision_sample_count"] > int(cfg["max_static_collision_samples"]):
        failure_reasons.append("static_occupancy_collision")
    if teleports:
        failure_reasons.append("large_teleport")

    legacy_result = _read_base_metrics(run_path)
    strict_task_success = not failure_reasons
    false_positive = (
        str(legacy_result.get("first", {}).get("result", "")).upper() == "GOAL_REACHED"
        and not strict_task_success
    )

    result = {
        "schema_version": 1,
        "run_dir": str(run_path),
        "time_scale": {
            "source": "arena_recorder_legacy_clock",
            "raw_units_per_second": RECORDER_TIME_UNITS_PER_SECOND,
            "raw_time_field": "time",
        },
        "world": contract.get("world"),
        "scenario": contract.get("scenario"),
        "instruction": contract.get("instruction"),
        "sources": {
            "scenario_path": contract.get("scenario_path"),
            "map_yaml": contract.get("map_yaml"),
            "odom_csv": str(run_path / "odom.csv"),
            "cmd_vel_csv": str(run_path / "cmd_vel.csv"),
            "start_goal_csv": str(run_path / "start_goal.csv"),
        },
        "scenario_contract": contract,
        "language_task_contract": _task_contract_summary(contract),
        "sample_counts": {
            "odom": len(odom_samples),
            "cmd_vel": len(cmd_samples),
        },
        "start_goal_consistency": start_goal_consistency,
        "goal": {
            "goal_tolerance_m": cfg["goal_tolerance_m"],
            "start_xy": contract.get("start_xy"),
            "goal_xy": contract.get("goal_xy"),
            "final_xy": list(final_xy) if final_xy else None,
            "navigation_error_m": nav_error,
            "oracle_error_m": oracle,
            "goal_reached": goal_reached,
        },
        "vln": {
            "trajectory_length_m": executed_len,
            "shortest_path_length_m": shortest_len,
            "spl": metric_spl,
            "ndtw": metric_ndtw,
            "sdtw": metric_sdtw,
        },
        "episode_timing": timing,
        "commanded_stuck": stuck,
        "static_occupancy": occupancy,
        "large_teleports": teleports,
        "legacy_metrics": legacy_result,
        "strict_task_success": strict_task_success,
        "strict_task_failure_reasons": failure_reasons,
        "legacy_goal_reached_false_positive": false_positive,
        "thresholds": cfg,
    }
    (run_path / "vln_task_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate strict Arena VLN task metrics.")
    parser.add_argument("--dir", required=True, help="Eval run directory containing Arena recorder artifacts")
    args = parser.parse_args()
    generate_vln_task_metrics(args.dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
