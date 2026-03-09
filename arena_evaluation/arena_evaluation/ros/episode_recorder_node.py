from __future__ import annotations

import json
import math
import pathlib
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import Trigger

try:
    from arena_people_msgs.msg import Pedestrians
except Exception:  # pragma: no cover
    Pedestrians = None  # type: ignore

from ..schema import EpisodeLog, EpisodeMeta, Frame, PedestrianState, RobotState


class EpisodeRecorderNode(Node):
    """Lightweight recorder that writes an EpisodeLog JSON.

    This intentionally does not attempt to auto-detect episode boundaries.
    You start/stop recording via services:
    - `~/start_episode` (Trigger)
    - `~/stop_episode` (Trigger)
    """

    def __init__(self):
        super().__init__("arena_episode_recorder")

        self.declare_parameter("namespace", "")
        self.declare_parameter("output_dir", str(pathlib.Path.home() / "arena_runs"))
        self.declare_parameter("goal_topic", "goal_pose")
        self.declare_parameter("odom_topic", "odom")
        self.declare_parameter("pedestrians_topic", "arena_peds")
        self.declare_parameter("goal_tolerance_m", 0.25)
        self.declare_parameter("robot_radius_m", 0.30)
        self.declare_parameter("human_radius_m", 0.30)

        ns = self.get_parameter("namespace").value
        ns = ns.strip("/")
        prefix = f"/{ns}" if ns else ""

        self._goal_topic = prefix + "/" + str(self.get_parameter("goal_topic").value).strip("/")
        self._odom_topic = prefix + "/" + str(self.get_parameter("odom_topic").value).strip("/")
        self._peds_topic = prefix + "/" + str(self.get_parameter("pedestrians_topic").value).strip("/")

        self._goal_xy: tuple[float, float] | None = None
        self._latest_odom: Odometry | None = None
        self._latest_peds: list[PedestrianState] = []

        self._recording = False
        self._frames: list[Frame] = []
        self._t0_wall: float | None = None
        self._run_id: str | None = None

        self.create_subscription(PoseStamped, self._goal_topic, self._on_goal, 10)
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 50)
        if Pedestrians is not None:
            self.create_subscription(Pedestrians, self._peds_topic, self._on_peds, 10)
        else:
            self.get_logger().warning(
                "arena_people_msgs not available; pedestrians will not be recorded"
            )

        self._start_srv = self.create_service(Trigger, "start_episode", self._start_episode)
        self._stop_srv = self.create_service(Trigger, "stop_episode", self._stop_episode)
        self._timer = self.create_timer(0.1, self._tick)

        self.get_logger().info(f"Recording sources: odom={self._odom_topic}, goal={self._goal_topic}, peds={self._peds_topic}")

    def _on_goal(self, msg: PoseStamped) -> None:
        self._goal_xy = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg

    def _on_peds(self, msg) -> None:
        peds: list[PedestrianState] = []
        for ped in getattr(msg, "pedestrians", []):
            peds.append(
                PedestrianState(
                    pid=str(getattr(ped, "name", "")),
                    x=float(ped.pose.position.x),
                    y=float(ped.pose.position.y),
                    vx=float(ped.twist.linear.x),
                    vy=float(ped.twist.linear.y),
                )
            )
        self._latest_peds = peds

    def _tick(self) -> None:
        if not self._recording:
            return
        if self._goal_xy is None or self._latest_odom is None:
            return

        t = float(time.time() - (self._t0_wall or time.time()))
        odom = self._latest_odom

        x = float(odom.pose.pose.position.x)
        y = float(odom.pose.pose.position.y)

        # Quaternion -> yaw (robust enough for planar base)
        q = odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        v = float(odom.twist.twist.linear.x)
        w = float(odom.twist.twist.angular.z)

        robot = RobotState(t=t, x=x, y=y, yaw=yaw, v=v, w=w)
        self._frames.append(Frame(robot=robot, pedestrians=tuple(self._latest_peds), events={}))

    def _start_episode(self, request, response):
        if self._recording:
            response.success = False
            response.message = "Already recording"
            return response

        if self._goal_xy is None:
            response.success = False
            response.message = "No goal received yet; wait for goal_pose"
            return response

        self._frames = []
        self._t0_wall = time.time()
        self._run_id = f"t{int(time.time())}"
        self._recording = True

        response.success = True
        response.message = f"Started recording run_id={self._run_id}"
        return response

    def _stop_episode(self, request, response):
        if not self._recording:
            response.success = False
            response.message = "Not recording"
            return response

        self._recording = False
        goal_xy = self._goal_xy
        if goal_xy is None:
            response.success = False
            response.message = "No goal available"
            return response

        meta = EpisodeMeta(
            run_id=self._run_id,
            goal_tolerance_m=float(self.get_parameter("goal_tolerance_m").value),
            robot_radius_m=float(self.get_parameter("robot_radius_m").value),
            human_radius_m=float(self.get_parameter("human_radius_m").value),
        )
        episode = EpisodeLog(
            meta=meta,
            goal_x=float(goal_xy[0]),
            goal_y=float(goal_xy[1]),
            frames=tuple(self._frames),
        )

        out_dir = pathlib.Path(str(self.get_parameter("output_dir").value))
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self._run_id or 'run'}_episode.json"
        out_path.write_text(json.dumps(episode.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

        response.success = True
        response.message = f"Wrote {out_path}"
        return response


def main(argv: list[str] | None = None) -> None:
    rclpy.init(args=argv)
    node = EpisodeRecorderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()