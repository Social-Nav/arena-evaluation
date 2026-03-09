from __future__ import annotations

import dataclasses
import math
import typing


@dataclasses.dataclass(frozen=True)
class EpisodeMeta:
    """Metadata for an episode.

    This is intentionally permissive: the metric engine will fall back to
    sensible defaults if fields are missing.
    """

    run_id: str | None = None
    episode_id: str | None = None
    scenario_id: str | None = None
    seed: int | None = None
    instruction: str | None = None
    context: str | None = None  # e.g. "normal", "urgent"

    time_limit_s: float | None = None
    goal_tolerance_m: float | None = None

    robot_radius_m: float | None = None
    human_radius_m: float | None = None

    terminated: bool | None = None
    truncated: bool | None = None
    done_reason: str | None = None


@dataclasses.dataclass(frozen=True)
class RobotState:
    t: float
    x: float
    y: float
    yaw: float
    v: float | None = None  # linear speed in robot frame (m/s)
    w: float | None = None  # yaw rate (rad/s)


@dataclasses.dataclass(frozen=True)
class PedestrianState:
    """A pedestrian/human state in world frame."""

    pid: str
    x: float
    y: float
    vx: float | None = None
    vy: float | None = None


@dataclasses.dataclass(frozen=True)
class Frame:
    """One timestep sample."""

    robot: RobotState
    pedestrians: tuple[PedestrianState, ...] = ()
    events: dict[str, typing.Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class EpisodeLog:
    meta: EpisodeMeta
    goal_x: float
    goal_y: float
    frames: tuple[Frame, ...]
    reference_path_xy: tuple[tuple[float, float], ...] = ()

    @property
    def start_xy(self) -> tuple[float, float]:
        if not self.frames:
            return (math.nan, math.nan)
        r0 = self.frames[0].robot
        return (r0.x, r0.y)

    @property
    def end_xy(self) -> tuple[float, float]:
        if not self.frames:
            return (math.nan, math.nan)
        rN = self.frames[-1].robot
        return (rN.x, rN.y)

    def to_dict(self) -> dict[str, typing.Any]:
        return {
            "meta": dataclasses.asdict(self.meta),
            "goal": {"x": self.goal_x, "y": self.goal_y},
            "reference_path": [
                {"x": float(x), "y": float(y)} for x, y in self.reference_path_xy
            ],
            "frames": [
                {
                    "t": f.robot.t,
                    "robot": {
                        "x": f.robot.x,
                        "y": f.robot.y,
                        "yaw": f.robot.yaw,
                        "v": f.robot.v,
                        "w": f.robot.w,
                    },
                    "pedestrians": [
                        {
                            "id": p.pid,
                            "x": p.x,
                            "y": p.y,
                            "vx": p.vx,
                            "vy": p.vy,
                        }
                        for p in f.pedestrians
                    ],
                    "events": f.events,
                }
                for f in self.frames
            ],
        }

    @staticmethod
    def from_dict(obj: dict[str, typing.Any]) -> "EpisodeLog":
        if not isinstance(obj, dict):
            raise TypeError(f"expected dict, got {type(obj)}")

        meta_obj = obj.get("meta", {}) or {}
        if not isinstance(meta_obj, dict):
            raise TypeError("episode.meta must be a dict")

        goal_obj = obj.get("goal", {}) or {}
        if not isinstance(goal_obj, dict):
            raise TypeError("episode.goal must be a dict")

        goal_x = float(goal_obj["x"])
        goal_y = float(goal_obj["y"])

        ref = obj.get("reference_path", ()) or ()
        reference_path_xy: list[tuple[float, float]] = []
        if isinstance(ref, list):
            for item in ref:
                if isinstance(item, dict):
                    reference_path_xy.append((float(item["x"]), float(item["y"])))
                else:
                    x, y = item
                    reference_path_xy.append((float(x), float(y)))
        elif isinstance(ref, tuple):
            for x, y in ref:
                reference_path_xy.append((float(x), float(y)))

        frames_obj = obj.get("frames", [])
        if not isinstance(frames_obj, list):
            raise TypeError("episode.frames must be a list")

        frames: list[Frame] = []
        for f in frames_obj:
            if not isinstance(f, dict):
                raise TypeError("frame must be a dict")
            t = float(f.get("t", f.get("time", 0.0)))
            robot_obj = f.get("robot", {})
            if not isinstance(robot_obj, dict):
                raise TypeError("frame.robot must be a dict")

            robot = RobotState(
                t=t,
                x=float(robot_obj["x"]),
                y=float(robot_obj["y"]),
                yaw=float(robot_obj.get("yaw", 0.0)),
                v=None if robot_obj.get("v") is None else float(robot_obj["v"]),
                w=None if robot_obj.get("w") is None else float(robot_obj["w"]),
            )

            peds: list[PedestrianState] = []
            for p in f.get("pedestrians", []) or []:
                if not isinstance(p, dict):
                    continue
                peds.append(
                    PedestrianState(
                        pid=str(p.get("id", p.get("name", ""))),
                        x=float(p["x"]),
                        y=float(p["y"]),
                        vx=None if p.get("vx") is None else float(p["vx"]),
                        vy=None if p.get("vy") is None else float(p["vy"]),
                    )
                )

            events = f.get("events", {}) or {}
            if not isinstance(events, dict):
                events = {}

            frames.append(Frame(robot=robot, pedestrians=tuple(peds), events=events))

        meta = EpisodeMeta(**meta_obj)
        return EpisodeLog(
            meta=meta,
            goal_x=goal_x,
            goal_y=goal_y,
            frames=tuple(frames),
            reference_path_xy=tuple(reference_path_xy),
        )