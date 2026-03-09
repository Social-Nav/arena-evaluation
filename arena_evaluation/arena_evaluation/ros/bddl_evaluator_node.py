"""ROS 2 node for online BDDL task evaluation.

Subscribes to odom to track the robot pose. When ``use_isaac_gt`` is True,
also calls the ``/isaac/GetPrims`` service to fetch ground-truth USD poses
for all non-robot entities (obstacles, landmarks, pedestrians) every tick.

This connects the BDDL evaluation pipeline to Isaac Sim's physical state:
  Isaac Sim USD scene --> /isaac/GetPrims service --> ArenaEntity.update_pose()
  /odom topic --> ArenaEntity.update_pose() (robot)
  ArenaEntity --> ArenaBackend predicate classes --> bddl.condition_evaluation

Parameters:
  - ``task_dir`` (str): path to a task directory containing ``task.yaml`` and
    subtask ``.bddl`` files.
  - ``namespace`` (str): robot namespace prefix (default: "").
  - ``odom_topic`` (str): default "odom".
  - ``eval_rate_hz`` (float): evaluation frequency (default: 10.0).
  - ``use_isaac_gt`` (bool): if True, poll /isaac/GetPrims for entity poses
    (default: False — use static positions from task.yaml).
  - ``verbose`` (bool): if True, print PDDL world state to terminal every
    3 seconds (default: False).

Published topics:
  - ``~/bddl_status`` (std_msgs/String): JSON with per-subtask status and
    full world state predicate dump, published every tick.

Services:
  - ``~/bddl_result`` (std_srvs/Trigger): returns the final TaskResult JSON.
"""

from __future__ import annotations

import json
import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String as StdString
from std_srvs.srv import Trigger

from ..bddl.arena_entity import ArenaEntity
from ..bddl.evaluator import BDDLEvaluator, SubtaskResult

from bddl.condition_evaluation import evaluate_state


def _quat_to_yaw(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class BDDLEvaluatorNode(Node):
    """Online BDDL evaluator: subscribes to live state, evaluates subtask goals."""

    def __init__(self):
        super().__init__("bddl_evaluator")

        self.declare_parameter("task_dir", "")
        self.declare_parameter("namespace", "")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("eval_rate_hz", 10.0)
        self.declare_parameter("use_isaac_gt", False)
        self.declare_parameter("verbose", False)

        task_dir = str(self.get_parameter("task_dir").value)
        if not task_dir:
            self.get_logger().fatal("Parameter 'task_dir' is required")
            raise SystemExit(1)

        ns = self.get_parameter("namespace").value
        ns = ns.strip("/") if ns else ""
        prefix = f"/{ns}" if ns else ""
        odom_topic = prefix + "/" + str(self.get_parameter("odom_topic").value).strip("/")
        rate = float(self.get_parameter("eval_rate_hz").value)
        self._use_isaac_gt = bool(self.get_parameter("use_isaac_gt").value)
        self._verbose = bool(self.get_parameter("verbose").value)

        # Load the BDDL task
        self._evaluator = BDDLEvaluator.from_task_dir(task_dir)
        self._entities = self._evaluator._entities
        self._subtask_defs = self._evaluator._subtask_defs

        # Identify robot vs non-robot entities for Isaac GT queries
        self._robot_entity_names = [
            name for name, ent in self._entities.items()
            if name.startswith("robot")
        ]
        self._object_entity_names = [
            name for name in self._entities
            if name not in self._robot_entity_names
        ]

        # Evaluation state
        self._current_idx = 0
        self._subtask_results: list[SubtaskResult] = []
        self._subtask_start_wall: float | None = None
        self._t0_wall: float | None = None
        self._finished = False

        # ROS interface — odom subscription
        self.create_subscription(Odometry, odom_topic, self._on_odom, 50)
        self._status_pub = self.create_publisher(StdString, "~/bddl_status", 10)
        self.create_service(Trigger, "~/bddl_result", self._on_result_request)

        # Isaac Sim GT state client (optional)
        self._get_prims_client = None
        if self._use_isaac_gt:
            try:
                from isaacsim_msgs.srv import GetPrims
                self._get_prims_client = self.create_client(GetPrims, "/isaac/GetPrims")
                self.get_logger().info("Isaac GT mode: will poll /isaac/GetPrims for entity poses")
            except ImportError:
                self.get_logger().warn(
                    "isaacsim_msgs not available — falling back to static positions"
                )
                self._use_isaac_gt = False

        self.create_timer(1.0 / max(0.1, rate), self._tick)

        # Verbose mode: print PDDL world state to terminal every 3 seconds
        if self._verbose:
            self.create_timer(3.0, self._verbose_print)

        self.get_logger().info(
            f"BDDL evaluator started: task_dir={task_dir}, "
            f"subtasks={len(self._subtask_defs)}, odom={odom_topic}, "
            f"isaac_gt={self._use_isaac_gt}, verbose={self._verbose}"
        )

    def _on_odom(self, msg: Odometry) -> None:
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        yaw = _quat_to_yaw(msg.pose.pose.orientation)

        for rname in self._robot_entity_names:
            robot = self._entities.get(rname)
            if robot is not None:
                robot.update_pose(x, y, yaw)

    def _verbose_print(self) -> None:
        """Print PDDL world state table to terminal (called every 3s when verbose)."""
        import sys

        entities = self._entities
        names = list(entities.keys())

        print(f"\n{'=' * 60}")
        print(f"BDDL World State (verbose)")
        print(f"{'=' * 60}")

        # Entity table
        print(f"\n  {'Entity':<25} {'X':>8} {'Y':>8} {'Yaw':>8} {'Radius':>8}")
        print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for name, ent in entities.items():
            print(f"  {name:<25} {ent.x:8.3f} {ent.y:8.3f} {ent.yaw:8.3f} {ent.radius:8.2f}")

        # Predicate table
        print(f"\n  {'Pair':<45} {'nextto':>8} {'touching':>10} {'dist':>8}")
        print(f"  {'-'*45} {'-'*8} {'-'*10} {'-'*8}")
        for i, na in enumerate(names):
            for nb in names[i + 1:]:
                ea, eb = entities[na], entities[nb]
                dx, dy = ea.x - eb.x, ea.y - eb.y
                dist = math.sqrt(dx * dx + dy * dy)
                nt = ea.get_nextto(eb)
                tc = ea.get_touching(eb)
                pad = max(0, 42 - len(na) - len(nb))
                print(f"  ({na}, {nb}){'':<{pad}} {str(nt):>8} {str(tc):>10} {dist:8.3f}")

        # Subtask evaluation
        print(f"\n--- Subtask Goal Evaluation ---")
        for i, stdef in enumerate(self._subtask_defs):
            if i < self._current_idx:
                status = "COMPLETED"
            else:
                success, _ = evaluate_state(stdef.compiled_goals)
                status = "SATISFIED" if success else "NOT SATISFIED"
            print(f"  [{i}] {stdef.id}: {status}")

        sys.stdout.flush()

    def _update_entities_from_isaac(self) -> None:
        """Call /isaac/GetPrims to fetch GT poses for non-robot entities."""
        if not self._get_prims_client or not self._object_entity_names:
            return

        if not self._get_prims_client.service_is_ready():
            return

        from isaacsim_msgs.srv import GetPrims
        request = GetPrims.Request()
        request.names = list(self._object_entity_names)

        future = self._get_prims_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=0.5)

        if future.done() and future.result() is not None:
            response = future.result()
            for prim in response.prims:
                entity = self._entities.get(prim.name)
                if entity is not None:
                    entity.update_pose(
                        float(prim.pose.position.x),
                        float(prim.pose.position.y),
                        _quat_to_yaw(prim.pose.orientation),
                    )

    def _get_world_state(self) -> dict:
        """Dump current world state: all entity poses + all predicate truth values."""
        entities_state = {}
        for name, ent in self._entities.items():
            entities_state[name] = {
                "x": round(ent.x, 3),
                "y": round(ent.y, 3),
                "yaw": round(ent.yaw, 3),
            }

        predicates_state = {}
        entity_names = list(self._entities.keys())
        for i, name_a in enumerate(entity_names):
            for name_b in entity_names[i + 1:]:
                ea = self._entities[name_a]
                eb = self._entities[name_b]
                pair_key = f"({name_a}, {name_b})"
                predicates_state[pair_key] = {
                    "nextto": ea.get_nextto(eb),
                    "touching": ea.get_touching(eb),
                }
                if eb.is_room:
                    predicates_state[pair_key]["inroom"] = ea.get_inroom(eb)
                if ea.is_room:
                    predicates_state[f"({name_b}, {name_a})"] = {
                        "inroom": eb.get_inroom(ea),
                    }

        return {"entities": entities_state, "predicates": predicates_state}

    def _tick(self) -> None:
        if self._finished:
            return

        now = time.time()
        if self._t0_wall is None:
            self._t0_wall = now

        # Fetch GT state from Isaac Sim if enabled
        if self._use_isaac_gt:
            self._update_entities_from_isaac()

        if self._current_idx >= len(self._subtask_defs):
            self._finished = True
            self._publish_status()
            return

        stdef = self._subtask_defs[self._current_idx]
        if self._subtask_start_wall is None:
            self._subtask_start_wall = now

        elapsed = now - self._subtask_start_wall

        # Timeout check
        if stdef.timeout_s is not None and elapsed > stdef.timeout_s:
            self._subtask_results.append(SubtaskResult(
                subtask_id=stdef.id,
                description=stdef.description,
                success=False,
                started_at_s=self._subtask_start_wall - self._t0_wall,
                completed_at_s=now - self._t0_wall,
                duration_s=elapsed,
                timed_out=True,
            ))
            self.get_logger().warn(f"Subtask '{stdef.id}' TIMED OUT ({elapsed:.1f}s)")
            self._current_idx += 1
            self._subtask_start_wall = None
            self._publish_status()
            return

        # Evaluate via bddl 3.x
        success, _ = evaluate_state(stdef.compiled_goals)
        if success:
            self._subtask_results.append(SubtaskResult(
                subtask_id=stdef.id,
                description=stdef.description,
                success=True,
                started_at_s=self._subtask_start_wall - self._t0_wall,
                completed_at_s=now - self._t0_wall,
                duration_s=elapsed,
            ))
            self.get_logger().info(f"Subtask '{stdef.id}' COMPLETED ({elapsed:.1f}s)")
            self._current_idx += 1
            self._subtask_start_wall = None

        self._publish_status()

    def _publish_status(self) -> None:
        completed = sum(1 for sr in self._subtask_results if sr.success)
        total = len(self._subtask_defs)
        status = {
            "task_id": self._evaluator._task_id,
            "finished": self._finished,
            "subtasks_completed": completed,
            "subtasks_total": total,
            "subtask_success_rate": completed / max(1, total),
            "current_subtask": (
                self._subtask_defs[self._current_idx].id
                if self._current_idx < total
                else None
            ),
            "subtask_results": [sr.to_dict() for sr in self._subtask_results],
            "world_state": self._get_world_state(),
        }
        self._status_pub.publish(StdString(data=json.dumps(status)))

    def _on_result_request(self, request, response):
        completed = sum(1 for sr in self._subtask_results if sr.success)
        total = len(self._subtask_defs)
        result = {
            "task_id": self._evaluator._task_id,
            "description": self._evaluator._description,
            "success": completed == total,
            "subtasks_completed": completed,
            "subtasks_total": total,
            "subtask_success_rate": completed / max(1, total),
            "total_duration_s": (
                (time.time() - self._t0_wall) if self._t0_wall else 0.0
            ),
            "subtask_results": [sr.to_dict() for sr in self._subtask_results],
            "world_state": self._get_world_state(),
        }
        response.success = True
        response.message = json.dumps(result, indent=2)
        return response


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv)
    node = BDDLEvaluatorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
