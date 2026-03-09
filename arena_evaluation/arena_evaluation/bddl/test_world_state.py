#!/usr/bin/env python3
"""Minimal integration test: launch Arena + print BDDL world state.

This script:
  1. Initialises ROS 2 and creates a node
  2. Loads a BDDL task definition (task.yaml + .bddl files)
  3. Auto-discovers the odom topic (or uses --odom-topic)
  4. Optionally calls /isaac/GetPrims for GT entity poses (if --isaac flag)
  5. Runs bddl 3.x condition evaluation every tick
  6. Prints a live world-state table: entity poses + predicate truth values

Usage (with Isaac Sim):
    ros2 launch arena_bringup arena.launch.py sim:=isaac &
    python3 -m arena_evaluation.bddl.test_world_state --isaac

Usage (standalone — no sim, just verify BDDL pipeline):
    python3 -m arena_evaluation.bddl.test_world_state --standalone
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
import time

# Ensure package is importable when run directly
_pkg_root = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from arena_evaluation.bddl.arena_entity import ArenaEntity
from arena_evaluation.bddl.evaluator import BDDLEvaluator

from bddl.condition_evaluation import evaluate_state


TASK_DIR = pathlib.Path(__file__).resolve().parent / "tasks" / "multi_room_patrol"


def _quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


# ──────────────────────────────────────────────
# Standalone mode: pure Python, no ROS 2 needed
# ──────────────────────────────────────────────

def run_standalone() -> None:
    """Verify the full BDDL pipeline without ROS 2 or any simulator."""
    print("=" * 60)
    print("BDDL World State Test — STANDALONE MODE (no ROS 2)")
    print("=" * 60)

    evaluator = BDDLEvaluator.from_task_dir(TASK_DIR)
    entities = evaluator._entities
    subtask_defs = evaluator._subtask_defs

    print(f"\nTask: {evaluator._task_id}")
    print(f"Description: {evaluator._description}")
    print(f"Subtasks: {len(subtask_defs)}")
    print(f"Entities: {list(entities.keys())}")

    # Print initial world state
    print("\n--- Initial World State ---")
    _print_entity_table(entities)
    _print_predicate_table(entities)

    # Simulate the robot moving to each target
    targets = [
        ("kitchen_table_1", 5.0, 3.0),
        ("living_room_sofa_1", 10.0, 7.0),
        ("bedroom_desk_1", 15.0, 2.0),
    ]

    robot = entities["robot_1"]
    for target_name, tx, ty in targets:
        print(f"\n>>> Moving robot to {target_name} ({tx}, {ty}) ...")
        robot.update_pose(tx, ty)

        print(f"\n--- World State after moving to {target_name} ---")
        _print_entity_table(entities)
        _print_predicate_table(entities)

        # Evaluate all subtask goals
        print("\n--- Subtask Goal Evaluation ---")
        for i, stdef in enumerate(subtask_defs):
            success, details = evaluate_state(stdef.compiled_goals)
            status = "SATISFIED" if success else "NOT SATISFIED"
            print(f"  subtask[{i}] {stdef.id}: {status}  (satisfied={details['satisfied']}, unsatisfied={details['unsatisfied']})")

    print("\n" + "=" * 60)
    print("STANDALONE TEST PASSED — BDDL pipeline is functional")
    print("=" * 60)


# ──────────────────────────────────────────────
# ROS 2 mode: connect to running simulation
# ──────────────────────────────────────────────

def _discover_odom_topic(node, timeout_sec: float = 10.0) -> str | None:
    """Auto-discover the odom topic by scanning available topics."""
    from rclpy.node import Node
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        topics = node.get_topic_names_and_types()
        for name, types in topics:
            if name.endswith("/odom") and "nav_msgs/msg/Odometry" in types:
                return name
        time.sleep(0.5)
    return None


def run_ros2(use_isaac_gt: bool = False, odom_topic: str | None = None, max_ticks: int = 0) -> None:
    """Run as a ROS 2 node, subscribing to live simulation state."""
    import rclpy
    from nav_msgs.msg import Odometry

    print("=" * 60)
    mode = "ISAAC GT" if use_isaac_gt else "ODOM"
    print(f"BDDL World State Test — ROS 2 MODE ({mode})")
    print("=" * 60)

    rclpy.init()

    evaluator = BDDLEvaluator.from_task_dir(TASK_DIR)
    entities = evaluator._entities
    subtask_defs = evaluator._subtask_defs

    node = rclpy.create_node("bddl_world_state_test")
    logger = node.get_logger()

    logger.info(f"Task: {evaluator._task_id}, entities: {list(entities.keys())}")

    # Auto-discover odom topic if not specified
    if odom_topic is None:
        logger.info("Auto-discovering odom topic...")
        odom_topic = _discover_odom_topic(node, timeout_sec=15.0)
        if odom_topic is None:
            logger.error("Could not find any odom topic. Available topics:")
            for name, types in node.get_topic_names_and_types():
                logger.error(f"  {name} [{', '.join(types)}]")
            node.destroy_node()
            rclpy.shutdown()
            return

    logger.info(f"Subscribing to odom topic: {odom_topic}")

    # Track odom updates
    odom_received = [False]

    def on_odom(msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = _quat_to_yaw(msg.pose.pose.orientation)

        robot = entities.get("robot_1")
        if robot is not None:
            robot.update_pose(x, y, yaw)
        odom_received[0] = True

    node.create_subscription(Odometry, odom_topic, on_odom, 50)

    # Isaac Sim GT client (optional)
    get_prims_client = None
    object_names = [n for n in entities if not n.startswith("robot")]
    if use_isaac_gt:
        try:
            from isaacsim_msgs.srv import GetPrims
            get_prims_client = node.create_client(GetPrims, "/isaac/GetPrims")
            logger.info("Waiting for /isaac/GetPrims service...")
            if not get_prims_client.wait_for_service(timeout_sec=10.0):
                logger.warn("/isaac/GetPrims not available — using static positions")
                get_prims_client = None
            else:
                logger.info("/isaac/GetPrims service connected")
        except ImportError:
            logger.warn("isaacsim_msgs not available — using static positions")

    print(f"\nWaiting for odom on {odom_topic}... (Ctrl+C to exit)\n")

    tick_count = 0
    print_count = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)

            if not odom_received[0]:
                continue

            tick_count += 1
            if tick_count % 10 != 1:  # Print every ~5s
                continue

            print_count += 1

            # Update entities from Isaac Sim GT if available
            if get_prims_client is not None and object_names:
                from isaacsim_msgs.srv import GetPrims
                req = GetPrims.Request()
                req.names = object_names
                future = get_prims_client.call_async(req)
                rclpy.spin_until_future_complete(node, future, timeout_sec=0.5)
                if future.done() and future.result() is not None:
                    for prim in future.result().prims:
                        ent = entities.get(prim.name)
                        if ent is not None:
                            ent.update_pose(
                                float(prim.pose.position.x),
                                float(prim.pose.position.y),
                                _quat_to_yaw(prim.pose.orientation),
                            )

            # Print world state
            print(f"\n{'=' * 60}")
            print(f"BDDL World State — tick {tick_count}  (mode: {mode})")
            print(f"{'=' * 60}")
            _print_entity_table(entities)
            _print_predicate_table(entities)

            print("\n--- Subtask Goal Evaluation ---")
            for i, stdef in enumerate(subtask_defs):
                success, details = evaluate_state(stdef.compiled_goals)
                status = "SATISFIED" if success else "NOT SATISFIED"
                print(f"  [{i}] {stdef.id}: {status}")

            sys.stdout.flush()

            # Auto-exit after N prints if requested (for CI/testing)
            if max_ticks > 0 and print_count >= max_ticks:
                print(f"\n>>> Reached {max_ticks} print cycles, exiting.")
                break

    except (KeyboardInterrupt, Exception) as e:
        if not isinstance(e, KeyboardInterrupt):
            # Handle ExternalShutdownException etc.
            print(f"\n{type(e).__name__}: {e}")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

    print("\nTest completed.")


# ──────────────────────────────────────────────
# Pretty-print helpers
# ──────────────────────────────────────────────

def _print_entity_table(entities: dict[str, ArenaEntity]) -> None:
    print(f"\n  {'Entity':<25} {'X':>8} {'Y':>8} {'Yaw':>8} {'Radius':>8} {'Room?':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for name, ent in entities.items():
        room = "yes" if ent.is_room else ""
        print(f"  {name:<25} {ent.x:8.3f} {ent.y:8.3f} {ent.yaw:8.3f} {ent.radius:8.2f} {room:>6}")


def _print_predicate_table(entities: dict[str, ArenaEntity]) -> None:
    names = list(entities.keys())
    print(f"\n  {'Pair':<45} {'nextto':>8} {'touching':>10} {'dist':>8}")
    print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*8}")
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            ea, eb = entities[na], entities[nb]
            dx, dy = ea.x - eb.x, ea.y - eb.y
            dist = math.sqrt(dx * dx + dy * dy)
            nt = ea.get_nextto(eb)
            tc = ea.get_touching(eb)
            print(f"  ({na}, {nb}){'':<{max(0, 42 - len(na) - len(nb))}} {str(nt):>8} {str(tc):>10} {dist:8.3f}")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test BDDL world state evaluation against Arena simulation"
    )
    parser.add_argument(
        "--standalone", action="store_true",
        help="Run without ROS 2 — pure BDDL pipeline verification"
    )
    parser.add_argument(
        "--isaac", action="store_true",
        help="Use /isaac/GetPrims for ground-truth entity poses"
    )
    parser.add_argument(
        "--odom-topic", type=str, default=None,
        help="Odom topic (auto-discovered if not set)"
    )
    parser.add_argument(
        "--max-ticks", type=int, default=0,
        help="Exit after N print cycles (0 = run forever, for CI use e.g. 3)"
    )
    args = parser.parse_args()

    if args.standalone:
        run_standalone()
    else:
        run_ros2(
            use_isaac_gt=args.isaac,
            odom_topic=args.odom_topic,
            max_ticks=args.max_ticks,
        )


if __name__ == "__main__":
    main()
