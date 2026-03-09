#!/usr/bin/env python3
"""Minimal demo: evaluate a synthetic multi-room patrol episode with BDDL 3.x.

Run:
    python -m arena_evaluation.bddl.demo

This generates a synthetic robot trajectory that visits three rooms in sequence,
then evaluates it against real .bddl subtask files using the bddl 3.x package
(ArenaBackend + ArenaEntity + bddl.condition_evaluation).

It demonstrates:
  - Real BDDL problem file parsing (subtask0.bddl, subtask1.bddl, subtask2.bddl)
  - ArenaBackend predicate grounding (nextto -> Euclidean distance)
  - Long-horizon sequential subtask evaluation via bddl.condition_evaluation
  - Per-subtask success/failure/timeout tracking
  - Subtask success rate metric
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

# Ensure the package is importable when run directly
_pkg_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from arena_evaluation.schema import EpisodeLog, EpisodeMeta, Frame, RobotState
from arena_evaluation.bddl.evaluator import BDDLEvaluator


def _lerp_path(
    waypoints: list[tuple[float, float]],
    speed: float = 1.0,
    dt: float = 0.1,
) -> list[tuple[float, float, float, float]]:
    """Generate (t, x, y, yaw) samples moving at constant *speed* through *waypoints*."""
    samples: list[tuple[float, float, float, float]] = []
    t = 0.0
    for i in range(len(waypoints) - 1):
        sx, sy = waypoints[i]
        ex, ey = waypoints[i + 1]
        dx, dy = ex - sx, ey - sy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-9:
            continue
        yaw = math.atan2(dy, dx)
        n_steps = max(1, int(dist / (speed * dt)))
        for k in range(n_steps):
            frac = k / n_steps
            x = sx + frac * dx
            y = sy + frac * dy
            samples.append((t, x, y, yaw))
            t += dt
    # Final point
    if waypoints:
        lx, ly = waypoints[-1]
        samples.append((t, lx, ly, samples[-1][2] if samples else 0.0))
    return samples


def build_synthetic_episode(
    success_subtasks: int = 3,
) -> EpisodeLog:
    """Build an episode where the robot visits 0..success_subtasks of the 3 rooms.

    Waypoints:
      start(0,0) -> kitchen(5,3) -> living_room(10,7) -> bedroom(15,2)
    If success_subtasks < 3, the trajectory stops early.
    """
    all_waypoints = [
        (0.0, 0.0),   # start
        (5.0, 3.0),   # kitchen_table
        (10.0, 7.0),  # living_room_sofa
        (15.0, 2.0),  # bedroom_desk
    ]
    waypoints = all_waypoints[: success_subtasks + 1]

    samples = _lerp_path(waypoints, speed=1.5, dt=0.1)

    frames = tuple(
        Frame(robot=RobotState(t=t, x=x, y=y, yaw=yaw, v=1.5))
        for t, x, y, yaw in samples
    )

    return EpisodeLog(
        meta=EpisodeMeta(
            run_id="bddl_demo",
            episode_id="patrol_001",
            scenario_id="multi_room_patrol",
            instruction="Navigate through kitchen, living room, and bedroom.",
        ),
        goal_x=15.0,
        goal_y=2.0,
        frames=frames,
    )


def main() -> None:
    task_dir = pathlib.Path(__file__).resolve().parent / "tasks" / "multi_room_patrol"
    evaluator = BDDLEvaluator.from_task_dir(task_dir)

    separator = "=" * 60
    print(separator)
    print("BDDL 3.x Evaluation Demo — Multi-Room Patrol")
    print(f"  Backend: ArenaBackend (bddl {_bddl_version()})")
    print(f"  Task dir: {task_dir}")
    print(separator)

    # --- Scenario A: Full success (visits all 3 rooms) ---
    print(f"\n{separator}")
    print("Scenario A: Robot visits ALL 3 rooms")
    print(separator)
    episode_a = build_synthetic_episode(success_subtasks=3)
    result_a = evaluator.evaluate(episode_a)
    _print_result(result_a)

    # --- Scenario B: Partial success (visits only 2 rooms) ---
    print(f"\n{separator}")
    print("Scenario B: Robot visits only 2 rooms (stops before bedroom)")
    print(separator)
    episode_b = build_synthetic_episode(success_subtasks=2)
    result_b = evaluator.evaluate(episode_b)
    _print_result(result_b)

    # --- Scenario C: Minimal progress (visits 1 room) ---
    print(f"\n{separator}")
    print("Scenario C: Robot visits only 1 room (kitchen only)")
    print(separator)
    episode_c = build_synthetic_episode(success_subtasks=1)
    result_c = evaluator.evaluate(episode_c)
    _print_result(result_c)

    # --- Write full JSON output ---
    out_path = task_dir / "demo_results.json"
    results_json = {
        "scenario_a_full_success": result_a.to_dict(),
        "scenario_b_partial": result_b.to_dict(),
        "scenario_c_minimal": result_c.to_dict(),
    }
    out_path.write_text(json.dumps(results_json, indent=2) + "\n", encoding="utf-8")
    print(f"\nFull results written to: {out_path}")


def _print_result(result) -> None:
    print(f"  Overall success: {result.success}")
    print(f"  Subtasks completed: {result.subtasks_completed}/{result.subtasks_total}")
    print(f"  Subtask success rate: {result.subtask_success_rate:.1%}")
    print(f"  Total duration: {result.total_duration_s:.1f}s")
    print()
    for sr in result.subtask_results:
        status = "PASS" if sr.success else ("TIMEOUT" if sr.timed_out else "FAIL")
        dur = f"{sr.duration_s:.1f}s" if sr.duration_s is not None else "N/A"
        print(f"    [{status}] {sr.subtask_id}: {sr.description} ({dur})")


def _bddl_version() -> str:
    try:
        import importlib.metadata
        return importlib.metadata.version("bddl")
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
