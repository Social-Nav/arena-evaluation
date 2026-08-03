"""
This file is used to calculate from the simulation data, various metrics, such as
- did a collision occur
- how long did the robot take form start to goal
the metrics / evaluation data will be saved to be preproccesed in the next step
"""

import os
import yaml
import enum
import json
import typing
import pandas as pd
import numpy as np

from typing import List
from pandas.core.api import DataFrame as DataFrame

from ament_index_python.packages import get_package_share_directory
from arena_evaluation.scripts.utils import Utils


# Optional dependency for polygon checks (zone metrics)
try:
    from shapely.geometry import Point, Polygon
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


class Action(str, enum.Enum):
    STOP = "STOP"
    ROTATE = "ROTATE"
    MOVE = "MOVE"


class DoneReason(str, enum.Enum):
    TIMEOUT = "TIMEOUT"
    GOAL_REACHED = "GOAL_REACHED"
    COLLISION = "COLLISION"


class Metric(typing.TypedDict):

    time: typing.List[int]
    time_diff: int
    episode: int
    goal: typing.List
    start: typing.List

    path: typing.List
    path_length_values: typing.List
    path_length: float
    angle_over_length: float
    curvature: typing.List
    normalized_curvature: typing.List
    roughness: typing.List

#    cmd_vel: typing.List
    velocity: typing.List
    acceleration: typing.List
    jerk: typing.List

    collision_amount: typing.Optional[int]
    collisions: typing.List

#    action_type: typing.List[Action]
    result: DoneReason


class SubjectMetric(typing.TypedDict):

    subject_id: str
    subject_type: str
    subject_position: typing.List[typing.List[float]]
    robot_subject_distances: typing.List[float]
    subject_scores: typing.List[float]
    total_subject_score: float
    scoring_function_type: str


class SubjectAwareMetric(Metric, typing.TypedDict):
    overall_subject_score: float
    average_subject_score: float
    subject_violation_count: int


class ZoneViolation(typing.TypedDict):

    timestamp: int
    position: typing.List[float]
    duration: int
    severity: float
    violation_type: str


class ZoneAwareMetric(Metric, typing.TypedDict):
    overall_zone_score: float
    zone_violation_count: int
    zone_violation_time: int


class Config:

    TIMEOUT_TRESHOLD = 180.0
    MAX_COLLISIONS = 3
    MIN_EPISODE_LENGTH = 5

    PERSONAL_SPACE_RADIUS = 1  # personal space is estimated at around 1'-4'
    ROBOT_GAZE_ANGLE = np.radians(5)  # size of (one half of) direct robot gaze cone
    PEDESTRIAN_GAZE_ANGLE = np.radians(5)  # size of (one half of) direct ped gaze cone


class Math:

    @classmethod
    def round_values(cls, values, digits=3):
        return [round(v, digits) for v in values]

    @classmethod
    def grouping(cls, base: np.ndarray, size: int) -> np.ndarray:
        """Every consecutive window of ``size`` samples, newest-first within a window.

        Window ``i`` is ``(base[i+size-1], ..., base[i])``, so ``window[:, 0]`` is
        the later sample and ``window[:, -1]`` the earlier one, matching how the
        callers index.  There are ``len(base) - size + 1`` windows.

        This used to be built with ``np.roll``, which WRAPS: window 0 was
        ``(base[0], base[-1], ...)``, pairing the first sample of the episode with
        the last.  A trailing ``[:-size]`` slice then discarded the final windows.
        For `path_length` that meant the first term was the distance from the last
        pose all the way back to the first -- 5.910 m of case01's reported
        17.565 m -- while the last two real steps were dropped.  Reproduced
        exactly: the delivered metrics.csv `path_length_values` has
        ``len(odom) - 2`` entries whose first element equals that wrap distance in
        all seven runs.
        """
        n = len(base)
        if size <= 0 or n < size:
            return np.empty((0, size) + base.shape[1:], dtype=base.dtype)
        return np.array([base[i + size - 1 :: -1][:size] for i in range(n - size + 1)])

    @classmethod
    def triangles(cls, position: np.ndarray) -> np.ndarray:
        return cls.grouping(position, 3)

    @classmethod
    def triangle_area(cls, vertices: np.ndarray) -> np.ndarray:
        return np.linalg.norm(
            np.cross(
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
                axis=1
            ),
            axis=1
        ) / 2

    # Odom `position` is [x, y, yaw] and `velocity` is [linear.x, linear.y,
    # angular.z], so the first two components are metres or metres per second and
    # the third is radians or radians per second.  Anything that takes a Euclidean
    # norm must therefore use only the planar columns; norming all three adds
    # radians in quadrature with metres and yields a number in no unit at all.
    PLANAR_COLUMNS = 2

    @classmethod
    def planar(cls, values: np.ndarray) -> np.ndarray:
        """Drop the yaw column of an [x, y, yaw]-style sample array.

        Applied to the SAMPLE array, before any grouping, so the trailing axis of
        the result is the coordinate axis.
        """
        values = np.asarray(values)
        if values.ndim < 2 or values.shape[-1] <= cls.PLANAR_COLUMNS:
            return values
        return values[..., : cls.PLANAR_COLUMNS]

    @classmethod
    def segment_boundaries(cls, times: np.ndarray) -> np.ndarray:
        """Mask of consecutive-pair indices that straddle a recording time segment.

        The recorder concatenates several time segments into one CSV: while /clock
        is stalled during scene load it fabricates timestamps, then adopts the
        lower resumed value as a new baseline and keeps appending, with no
        rotation, no marker and no log.  A backwards step in `time` is the only
        observable boundary.

        Rows reach this module in acquisition order -- pandas preserves file order
        and nothing here sorts -- so pair ``i`` spans ``times[i] -> times[i+1]`` and
        crosses a boundary exactly when that step is negative.  Such a pair is not
        a displacement and not a turn: the recorder simply resumed from a lower
        clock value.

        Returns a boolean array of length ``len(times) - 1``.  Nothing assumes how
        many segments there are; the count tracks how often /clock stalls.
        """
        times = np.asarray(times, dtype=float)
        if times.size < 2:
            return np.zeros(0, dtype=bool)
        return np.diff(times) < 0.0

    @classmethod
    def _drop_segment_boundaries(cls, steps: np.ndarray, times: np.ndarray | None) -> np.ndarray:
        if times is None:
            return steps
        boundaries = cls.segment_boundaries(times)
        if boundaries.size != steps.size:
            return steps
        return np.where(boundaries, 0.0, steps)

    @classmethod
    def path_length(cls, position: np.ndarray, times: np.ndarray | None = None) -> np.ndarray:
        """Distance travelled between consecutive samples, in metres.

        Only x and y participate.  Previously the norm was taken over the full
        [x, y, yaw] row, so a pure rotation registered as translation and the
        result was dimensionally invalid.  The effect is large, not cosmetic: the
        delivered case01 total was 11.6575 with yaw included against 2.1203 for the
        planar path, because that run turned through 359.3 degrees while advancing
        1.44 m.

        When ``times`` is supplied, a step that straddles a recording time-segment
        boundary contributes 0.0.  That is what makes this agree with
        ``social_metrics.path_length_m``, which skips the same pairs.
        """
        pairs = cls.grouping(cls.planar(position), 2)
        steps = np.linalg.norm(pairs[:, 0, :] - pairs[:, 1, :], axis=1)
        return cls._drop_segment_boundaries(steps, times)

    @classmethod
    def curvature(cls, position: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:

        triangles = cls.triangles(position)

        d01 = np.linalg.norm(triangles[:, 0, :] - triangles[:, 1, :], axis=1)
        d12 = np.linalg.norm(triangles[:, 1, :] - triangles[:, 2, :], axis=1)
        d20 = np.linalg.norm(triangles[:, 2, :] - triangles[:, 0, :], axis=1)

        triangle_area = cls.triangle_area(triangles)
        divisor = np.prod([d01, d12, d20], axis=0)
        divisor[divisor == 0] = np.nan

        curvature = 4 * triangle_area / divisor
        curvature[np.isnan(divisor)] = 0

        normalized = np.multiply(
            curvature,
            d01 + d12
        )

        return curvature, normalized

    @classmethod
    def roughness(cls, position: np.ndarray) -> np.ndarray:

        triangles = cls.triangles(position)

        triangle_area = cls.triangle_area(triangles)
        length = np.linalg.norm(triangles[:, :, 0] - triangles[:, :, 2], axis=1)
        length[length == 0] = np.nan

        roughness = 2 * triangle_area / np.square(length)
        roughness[np.isnan(length)] = 0

        return roughness

    @classmethod
    def acceleration(cls, speed: np.ndarray) -> np.ndarray:
        return np.diff(speed)

    @classmethod
    def jerk(cls, speed: np.ndarray) -> np.ndarray:
        return np.diff(np.diff(speed))

    @classmethod
    def turn(cls, yaw: np.ndarray, times: np.ndarray | None = None) -> np.ndarray:
        """Absolute yaw change between consecutive samples, in radians.

        When ``times`` is supplied, a step across a recording time-segment boundary
        contributes 0.0: the yaw difference between the last pose of one segment
        and the first of the next is a clock rebase, not a turn.  `turn` feeds
        `angle_over_length`, whose denominator already excludes those pairs.
        """
        pairs = cls.grouping(yaw, 2)
        steps = cls.angle_difference(pairs[:, 0], pairs[:, 1])
        return cls._drop_segment_boundaries(steps, times)

    @classmethod
    def angle_difference(cls, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        return np.pi - np.abs(np.abs(x1 - x2) - np.pi)

    @classmethod
    def point_in_polygon(cls, point, polygon):
        """Ray casting point-in-polygon test (fallback when shapely is unavailable)."""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @classmethod
    def u_curve_score(cls, distance: np.ndarray, optimal_distance: float, curve_width: float) -> np.ndarray:
        deviation = np.abs(distance - optimal_distance)
        return 1.0 / (1.0 + (deviation / curve_width) ** 2)

    @classmethod
    def distance_penalty_score(
        cls,
        distance: np.ndarray,
        safe_distance: float,
        penalty_weight: float = 1.0
    ) -> np.ndarray:
        violation = np.maximum(0, safe_distance - distance)
        return np.exp(-penalty_weight * violation / safe_distance)


class Metrics:

    dir: str
    _episode_data: typing.Dict[int, Metric]

    def _load_data(self) -> typing.List[pd.DataFrame]:

        odom = pd.read_csv(os.path.join(self.dir, "odom.csv"), converters={
            "data": lambda col: json.loads(col.replace("'", "\""))
        }).rename(columns={"data": "odom"})

        laserscan = pd.read_csv(os.path.join(self.dir, "scan.csv"), converters={
            "data": Utils.string_to_float_list
        }).rename(columns={"data": "laserscan"})

        episode = pd.read_csv(os.path.join(self.dir, "episode.csv"), converters={
            "data": lambda val: 0 if len(val) <= 0 else int(val)
        })

        start_goal = pd.read_csv(os.path.join(self.dir, "start_goal.csv"), converters={
            "start": Utils.string_to_float_list,
            "goal": Utils.string_to_float_list
        })

#        cmd_vel = pd.read_csv(os.path.join(self.dir, "cmd_vel.csv"), converters={
#            "data": Utils.string_to_float_list
#        }).rename(columns={"data": "cmd_vel"})

        data_frames = [
            episode,
            odom,
            #            cmd_vel,
            start_goal
        ]
        if not laserscan.empty:
            data_frames.insert(1, laserscan)
        return data_frames

    def __init__(self, dir: str):

        self.dir = dir

        self.robot_params = self._get_robot_params()

        data = pd.concat(self._load_data(), axis=1, join="inner")
        data = data.loc[:, ~data.columns.duplicated()].copy()

        i = 0

        episode_data = self._episode_data = {}

        while True:
            current_episode = data[data["episode"] == i]

            if len(current_episode) < Config.MIN_EPISODE_LENGTH:
                break

            episode_data[i] = self._analyze_episode(current_episode, i)
            i = i + 1

    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self._episode_data).transpose().set_index("episode")

    def _analyze_episode(self, episode: pd.DataFrame, index) -> Metric:

        episode["time"] /= 10**10

        positions = np.array([frame["position"] for frame in episode["odom"]])
        # `velocity`, not `position`.  Reading position here made `velocity`,
        # `acceleration` and `jerk` functions of the robot's DISTANCE FROM THE
        # ORIGIN: every delivered metrics.csv has velocity[i] == |position[i]|,
        # e.g. a first sample of 5.257 for a robot standing at
        # (3.926, -2.012, -2.86), which is a pose, not a speed.
        velocities = np.array([frame["velocity"] for frame in episode["odom"]])

        curvature, normalized_curvature = Math.curvature(positions)
        roughness = Math.roughness(positions)

        # Linear speed only.  `velocity` is [linear.x, linear.y, angular.z], so
        # norming all three added rad/s in quadrature with m/s.  This matches
        # `social_metrics` and `vln_task_metrics`, which both take hypot(vx, vy).
        vel_absolute = np.linalg.norm(Math.planar(velocities), axis=1)
        acceleration = Math.acceleration(vel_absolute)
        jerk = Math.jerk(vel_absolute)

        if "laserscan" in episode.columns:
            collisions, collision_amount = self._get_collisions(
                episode["laserscan"],
                self.robot_params["robot_radius"]
            )
        else:
            # Some visual-navigation robots do not publish LaserScan. Do not
            # synthesize scan data; mark collision metrics as unavailable while
            # still emitting odometry/path metrics for the episode.
            collisions, collision_amount = [], None

        # Sample timestamps, in acquisition order, used to detect recording
        # time-segment boundaries.  Without this the polyline is charged for the
        # recorder's backwards /clock rebase as if the robot had teleported, which
        # is why metrics.csv disagreed with social_metrics.path_length_m.
        sample_times = np.array(episode["time"], dtype=float)
        path_length = Math.path_length(positions, sample_times)
        turn = Math.turn(positions[:, 2], sample_times)

        time = list(episode["time"])[-1] - list(episode["time"])[0]

        start_position = self._get_mean_position(episode, "start")
        goal_position = self._get_mean_position(episode, "goal")

        # print("PATH LENGTH", path_length, path_length_per_step)

        return Metric(
            curvature=Math.round_values(curvature),
            normalized_curvature=Math.round_values(normalized_curvature),
            roughness=Math.round_values(roughness),
            path_length_values=Math.round_values(path_length),
            path_length=path_length.sum(),
            acceleration=Math.round_values(acceleration),
            jerk=Math.round_values(jerk),
            velocity=Math.round_values(vel_absolute),
            collision_amount=collision_amount,
            collisions=list(collisions),
            path=[list(p) for p in positions],
            angle_over_length=np.abs(turn.sum() / path_length.sum()),
            #            action_type = list(self._get_action_type(episode["cmd_vel"])),
            time_diff=time,  # Ros time in ns
            time=list(map(int, episode["time"].tolist())),
            episode=index,
            result=self._get_success(time, collision_amount),
            #            cmd_vel = list(map(list, episode["cmd_vel"].to_list())),
            goal=goal_position,
            start=start_position
        )

    def _get_robot_params(self):

        params_path = os.path.join(self.dir, "params.yaml")
        content: typing.Dict[str, typing.Any] = {}
        if os.path.exists(params_path):
            with open(params_path, "r") as file:
                content = yaml.safe_load(file) or {}

        # Determine robot model.
        # Preferred: explicit 'model' or 'robot_model' stored in params.yaml.
        model = (content.get("model") or content.get("robot_model") or "").strip()
        if not model:
            # Common recorder format: namespace: task_generator_node/<robot>
            namespace = (content.get("namespace") or "").strip()
            if namespace and "/" in namespace:
                model = namespace.split("/")[-1]

        if not model:
            # Last resort: keep behavior explicit instead of silently using the wrong model.
            raise ValueError(
                f"Robot model could not be determined. "
                f"Set 'model' in '{params_path}' (e.g. model: jackal) or provide a namespace ending with '/<robot>'."
            )

        # In arena5, robot model parameters live in arena_robots/robots/<model>/model_params.yaml
        candidates = []
        try:
            candidates.append(
                os.path.join(
                    get_package_share_directory("arena_robots"),
                    "robots",
                    model,
                    "model_params.yaml",
                )
            )
        except Exception:
            pass

        # Fallbacks (legacy layouts)
        try:
            candidates.append(
                os.path.join(
                    get_package_share_directory("arena_simulation_setup"),
                    "entities",
                    "robots",
                    model,
                    "model_params.yaml",
                )
            )
        except Exception:
            pass

        robot_model_params_file = next((p for p in candidates if p and os.path.exists(p)), None)
        if not robot_model_params_file:
            tried = ", ".join([p for p in candidates if p])
            raise FileNotFoundError(
                f"Robot model params file not found for model '{model}'. Tried: {tried}"
            )

        with open(robot_model_params_file, "r") as file:
            robot_model_param = yaml.safe_load(file) or {}

        # Support both styles:
        # 1) ROS params YAML: { '/**': { ros__parameters: {...} } }
        # 2) Plain YAML mapping: { robot_radius: 0.267, ... }
        if (
            isinstance(robot_model_param, dict)
            and "/**" in robot_model_param
            and isinstance(robot_model_param.get("/**"), dict)
            and "ros__parameters" in robot_model_param["/**"]
        ):
            params = robot_model_param["/**"]["ros__parameters"] or {}
        else:
            params = robot_model_param

        if not isinstance(params, dict):
            raise ValueError(
                f"Invalid robot params in '{robot_model_params_file}': expected a mapping of parameters."
            )

        if "robot_radius" not in params:
            raise KeyError(
                f"robot_radius missing in '{robot_model_params_file}'. "
                f"This parameter is required for collision distance thresholding."
            )

        return params

    def _get_mean_position(self, episode, key):

        positions = episode[key].to_list()
        counter = {}

        for p in positions:
            hash = ":".join([str(pos) for pos in p])

            counter[hash] = counter.get(hash, 0) + 1

        sorted_positions = dict(sorted(counter.items(), key=lambda x: x))

        return [float(r) for r in list(sorted_positions.keys())[0].split(":")]

    def _get_success(self, time, collisions):

        if time >= Config.TIMEOUT_TRESHOLD:
            return DoneReason.TIMEOUT

        if collisions is not None and collisions >= Config.MAX_COLLISIONS:
            return DoneReason.COLLISION

        return DoneReason.GOAL_REACHED

    def _get_collisions(self, laser_scans, lower_bound):
        """
        Calculates the collisions. Therefore,
        the laser scans is examinated and all values below a
        specific range are marked as collision.

        Argument:
            - Array laser scans representing the scans over
            time
            - the lower bound for which a collisions are counted

        Returns tupel of:
            - Array of tuples with indexs and time in which
            a collision happened
        """
        collisions = []
        collisions_marker = []

        for i, scan in enumerate(laser_scans):

            # scan can be a numpy array (preferred) or an empty list (from empty CSV cells).
            scan_arr = np.asarray(scan, dtype=float)
            if scan_arr.size == 0:
                is_collision = False
            else:
                is_collision = bool(np.any(scan_arr <= float(lower_bound)))

            collisions_marker.append(is_collision)

            if is_collision:
                collisions.append(i)

        collision_amount = 0

        for i, coll in enumerate(collisions_marker[1:]):
            prev_coll = collisions_marker[i]

            if coll - prev_coll > 0:
                collision_amount += 1

        return collisions, collision_amount

    def _get_action_type(self, actions):
        action_type = []

        for action in actions:
            if sum(action) == 0:
                action_type.append(Action.STOP.value)
            elif action[0] == 0 and action[1] == 0:
                action_type.append(Action.ROTATE.value)
            else:
                action_type.append(Action.MOVE.value)

        return action_type

    def _get_position_for_collision(self, collisions, positions):
        for i, collision in enumerate(collisions):
            collisions[i][2] = positions[collision[0]]

        return collisions


class PedsimMetric(Metric, typing.TypedDict):

    num_pedestrians: int

    avg_velocity_in_personal_space: float
    total_time_in_personal_space: int
    time_in_personal_space: typing.List[int]

    total_time_looking_at_pedestrians: int
    time_looking_at_pedestrians: typing.List[int]

    total_time_looked_at_by_pedestrians: int
    time_looked_at_by_pedestrians: typing.List[int]


class PedsimMetrics(Metrics):

    def _load_data(self) -> List[DataFrame]:
        pedsim_data = pd.read_csv(
            os.path.join(self.dir, "pedsim_agents_data.csv"),
            converters={"data": Utils.parse_pedsim}
        ).rename(columns={"data": "peds"})

        return super()._load_data() + [pedsim_data]

    def __init__(self, dir: str, **kwargs):
        super().__init__(dir=dir, **kwargs)

    def _analyze_episode(self, episode: pd.DataFrame, index):

        max_peds = max(*(len(peds) for peds in episode['peds']), 0)
        episode = episode.iloc[[len(peds) == max_peds for peds in episode['peds']]].copy()

        super_analysis = super()._analyze_episode(episode, index)

        robot_position = np.array([odom["position"][:2] for odom in episode["odom"]])
        peds_position = np.array([[ped.position for ped in peds] for peds in episode["peds"]])

        # list of (timestamp, ped) indices, duplicate timestamps allowed
        personal_space_frames = np.linalg.norm(peds_position - robot_position[:, None], axis=-1) <= Config.PERSONAL_SPACE_RADIUS
        # list of timestamp indices, no duplicates
        is_personal_space = personal_space_frames.max(axis=1)

        # time in personal space
        time = np.diff(np.array(episode["time"]), prepend=0)
        total_time_in_personal_space = time[is_personal_space].sum()
        time_in_personal_space = [time[frames].sum(axis=0).astype(np.integer) for frames in personal_space_frames.T]

        # v_avg in personal space
        velocity = np.array(super_analysis["velocity"])
        velocity = velocity[is_personal_space]
        avg_velocity_in_personal_space = velocity.mean() if velocity.size else 0

        # gazes
        robot_direction = np.array([odom["position"][2] for odom in episode["odom"]])
        peds_direction = np.array([[ped.theta for ped in peds] for peds in episode["peds"]])
        angle_robot_peds = np.squeeze(np.angle(np.array(peds_position - robot_position[:, np.newaxis]).view(np.complex128)))

        # time looking at pedestrians
        robot_gaze = Math.angle_difference(robot_direction[:, np.newaxis], angle_robot_peds)
        looking_at_frames = np.abs(robot_gaze) <= Config.ROBOT_GAZE_ANGLE
        total_time_looking_at_pedestrians = time[looking_at_frames.max(axis=1)].sum()
        time_looking_at_pedestrians = [time[frames].sum(axis=0).astype(np.integer) for frames in looking_at_frames.T]

        # time being looked at by pedestrians
        ped_gaze = Math.angle_difference(peds_direction, np.pi - angle_robot_peds)
        looked_at_frames = np.abs(ped_gaze) <= Config.PEDESTRIAN_GAZE_ANGLE
        total_time_looked_at_by_pedestrians = time[looked_at_frames.max(axis=1)].sum()
        time_looked_at_by_pedestrians = [time[frames].sum(axis=0).astype(np.integer) for frames in looked_at_frames.T]

        return PedsimMetric(
            **super_analysis,
            avg_velocity_in_personal_space=avg_velocity_in_personal_space,
            total_time_in_personal_space=total_time_in_personal_space,
            time_in_personal_space=time_in_personal_space,
            total_time_looking_at_pedestrians=total_time_looking_at_pedestrians,
            time_looking_at_pedestrians=time_looking_at_pedestrians,
            total_time_looked_at_by_pedestrians=total_time_looked_at_by_pedestrians,
            time_looked_at_by_pedestrians=time_looked_at_by_pedestrians,
            num_pedestrians=peds_position.shape[1]
        )


class SubjectAwareMetrics(Metrics):
    def _load_data(self) -> List[DataFrame]:
        subjects_data = pd.read_csv(
            os.path.join(self.dir, "pedsim_agents_data.csv"),
            converters={"data": Utils.parse_pedsim}
        ).rename(columns={"data": "subjects"})

        return super()._load_data() + [subjects_data]

    def __init__(self, dir: str, **kwargs):
        self.world_name = kwargs.get("world_name", "arena_hospital_small")
        parent_kwargs = {k: v for k, v in kwargs.items() if k != "world_name"}

        self.subjects_config = self._load_subjects_config()

        super().__init__(dir=dir, **parent_kwargs)

    def _load_subjects_config(self):
        config_path = os.path.join(
            get_package_share_directory("arena_simulation_setup"),
            "worlds",
            self.world_name,
            "metrics",
            "default.yaml",
        )

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Subject metrics config not found: '{config_path}'. "
                f"Create '<world>/metrics/default.yaml' or pass a valid world_name."
            )

        with open(config_path, "r") as file:
            config = yaml.safe_load(file) or {}

        subjects_config = config.get("subjects", {})
        if subjects_config is None:
            subjects_config = {}
        if not isinstance(subjects_config, dict):
            raise ValueError(
                f"Invalid subjects config in '{config_path}': expected a mapping under 'subjects'."
            )

        # Preferred: subjects.by_name (e.g. "hunav_20").
        by_name = subjects_config.get("by_name", {})
        if by_name is None:
            by_name = {}
        if not isinstance(by_name, dict):
            raise ValueError(
                f"Invalid subjects.by_name config in '{config_path}': expected a mapping under 'subjects.by_name'."
            )

        # Legacy compatibility: subjects.by_id (e.g. "20").
        by_id = subjects_config.get("by_id", {})
        if by_id is None:
            by_id = {}
        if not isinstance(by_id, dict):
            raise ValueError(
                f"Invalid subjects.by_id config in '{config_path}': expected a mapping under 'subjects.by_id'."
            )

        default_cfg = subjects_config.get("default")
        if default_cfg is not None and not isinstance(default_cfg, dict):
            raise ValueError(
                f"Invalid subjects.default config in '{config_path}': expected a mapping under 'subjects.default'."
            )

        return subjects_config

    def _get_subject_config_for_key(self, subject_key: str):
        by_name = self.subjects_config.get("by_name", {}) or {}
        if subject_key in by_name:
            return by_name[subject_key] or {}

        # Legacy fallback: allow configuring by numeric ID.
        by_id = self.subjects_config.get("by_id", {}) or {}
        if subject_key in by_id:
            return by_id[subject_key] or {}

        default_cfg = self.subjects_config.get("default")
        if default_cfg is None:
            return None
        return default_cfg or {}

    def _subject_key_from_entry(self, subj) -> typing.Optional[str]:
        # parse_pedsim returns Pedestrian(id, type, ...). For HuNav recordings,
        # Utils.parse_pedsim maps dict key 'name' into the Pedestrian 'type' field.
        if subj is None:
            return None

        if isinstance(subj, dict):
            for k in ("name", "type", "id"):
                v = subj.get(k)
                if v is not None and str(v) != "":
                    return str(v)
            return None

        for attr in ("name", "type", "id"):
            if hasattr(subj, attr):
                v = getattr(subj, attr)
                if v is not None and str(v) != "":
                    return str(v)

        return None

    def _collect_observed_subject_keys(self, subjects_data_timeline) -> typing.Set[str]:
        observed: typing.Set[str] = set()
        for subjects_data in subjects_data_timeline:
            if not subjects_data:
                continue
            for subj in subjects_data:
                key = self._subject_key_from_entry(subj)
                if key:
                    observed.add(key)
        return observed

    def _analyze_episode(self, episode: pd.DataFrame, index) -> SubjectAwareMetric:
        super_analysis = super()._analyze_episode(episode, index)

        overall_subject_score = 0.0
        subject_violation_count = 0
        actual_subject_count = 0
        total_actual_weight = 0.0

        if "subjects" in episode.columns:
            robot_positions = np.array([odom["position"][:2] for odom in episode["odom"]])
            subjects_data_timeline = episode["subjects"].tolist()

            configured_keys = set((self.subjects_config.get("by_name", {}) or {}).keys())
            # Legacy: include any explicitly configured by_id entries too.
            configured_keys.update(set((self.subjects_config.get("by_id", {}) or {}).keys()))
            observed_keys = self._collect_observed_subject_keys(subjects_data_timeline)
            subject_keys_to_check = sorted(configured_keys.union(observed_keys))

            for subject_key in subject_keys_to_check:
                subject_info = self._get_subject_config_for_key(subject_key)
                if subject_info is None:
                    continue

                subject_metric = self._analyze_subject_timeline(
                    subject_key,
                    subject_info,
                    subjects_data_timeline,
                    robot_positions,
                    episode,
                )

                if subject_metric:
                    weight = subject_info.get("average_weight", 1.0)
                    overall_subject_score += subject_metric["total_subject_score"] * weight
                    total_actual_weight += weight
                    actual_subject_count += 1

                    violation_count = self._count_subject_violations(
                        subject_metric["robot_subject_distances"],
                        subject_info,
                    )
                    subject_violation_count += violation_count

        if self.subjects_config.get("scoring", {}).get("normalize_by_episode_time", False):
            episode_time = super_analysis["time_diff"]
            if episode_time > 0:
                overall_subject_score /= episode_time

        average_subject_score = 0.0
        if actual_subject_count > 0 and total_actual_weight > 0:
            average_subject_score = (overall_subject_score / total_actual_weight) * 100

        return SubjectAwareMetric(
            **super_analysis,
            overall_subject_score=overall_subject_score,
            average_subject_score=average_subject_score,
            subject_violation_count=subject_violation_count,
        )

    def _analyze_subject_timeline(self, subject_key, subject_config, subjects_data_timeline, robot_positions, episode):
        try:
            robot_subject_distances = []
            subject_positions_timeline = []
            subject_type = "unknown"

            for timestep_idx, subjects_data in enumerate(subjects_data_timeline):
                if not subjects_data or len(subjects_data) == 0:
                    continue

                subject_data = None
                for subj in subjects_data:
                    match_key = self._subject_key_from_entry(subj)
                    if match_key is not None and match_key == str(subject_key):
                        subject_data = subj
                        break

                if not subject_data:
                    continue

                subject_position = None
                if hasattr(subject_data, "position"):
                    subject_position = subject_data.position
                elif hasattr(subject_data, "positions"):
                    subject_position = subject_data.positions[0] if subject_data.positions else None

                if subject_position and len(subject_position) >= 2 and timestep_idx < len(robot_positions):
                    robot_pos = robot_positions[timestep_idx]
                    distance = np.linalg.norm(robot_pos - np.array(subject_position[:2]))
                    robot_subject_distances.append(distance)
                    subject_positions_timeline.append(subject_position)

                    if subject_type == "unknown":
                        if isinstance(subject_data, dict):
                            subject_type = subject_data.get("type", "unknown")
                        else:
                            subject_type = getattr(subject_data, "type", "unknown")

            if not robot_subject_distances:
                return None

            scoring_type = subject_config.get("scoring", "distance_penalty")
            scoring_params = subject_config.get("params", {})

            subject_scores = self._calculate_subject_scores(
                scoring_type,
                scoring_params,
                robot_subject_distances,
            )

            total_subject_score = np.mean(subject_scores) if subject_scores else 0.0

            return SubjectMetric(
                subject_id=str(subject_key),
                subject_type=subject_type,
                subject_position=subject_positions_timeline,
                robot_subject_distances=robot_subject_distances,
                subject_scores=subject_scores,
                total_subject_score=total_subject_score,
                scoring_function_type=scoring_type,
            )
        except Exception as e:
            print(f"Error analyzing subject {subject_id}: {e}")
            return None

    def _calculate_subject_scores(self, scoring_type, params, distances):
        if not distances:
            return []

        distances = np.array(distances)

        if scoring_type == "u_curve":
            optimal_distance = params.get("optimal_distance", 1.0)
            curve_width = params.get("curve_width", 0.5)
            scores = Math.u_curve_score(distances, optimal_distance, curve_width)
        elif scoring_type == "distance_penalty":
            safe_distance = params.get("safe_distance", 1.0)
            penalty_weight = params.get("penalty_weight", 1.0)
            scores = Math.distance_penalty_score(distances, safe_distance, penalty_weight)
        else:
            scores = np.zeros_like(distances, dtype=float)

        return scores.tolist()

    def _count_subject_violations(self, distances, subject_config):
        if not distances:
            return 0

        distances = np.array(distances)
        scoring_type = subject_config.get("scoring", "distance_penalty")
        params = subject_config.get("params", {})
        tolerance = params.get("tolerance", 0.5)

        if scoring_type == "u_curve":
            optimal_distance = params.get("optimal_distance", 1.0)
            violations = np.logical_or(
                distances < (optimal_distance - tolerance),
                distances > (optimal_distance + tolerance),
            )
        elif scoring_type == "distance_penalty":
            safe_distance = params.get("safe_distance", 1.0)
            violations = distances < (safe_distance + tolerance)
        else:
            violations = np.zeros_like(distances, dtype=bool)

        violation_count = 0
        was_in_violation = False

        for is_violation in violations:
            if is_violation and not was_in_violation:
                violation_count += 1
                was_in_violation = True
            elif not is_violation:
                was_in_violation = False

        return violation_count


class ZoneAwareMetrics(Metrics):
    def _load_data(self) -> List[DataFrame]:
        return super()._load_data()

    def __init__(self, dir: str, **kwargs):
        self.world_name = kwargs.get("world_name", "small_warehouse")
        parent_kwargs = {k: v for k, v in kwargs.items() if k != "world_name"}

        self.zones_scoring_config = {}
        self.zones_config = self._load_zones_config()

        super().__init__(dir=dir, **parent_kwargs)

    def _load_zones_config(self):
        base = get_package_share_directory("arena_simulation_setup")
        metrics_config_path = os.path.join(base, "worlds", self.world_name, "metrics", "default.yaml")
        world_yaml_path = os.path.join(base, "worlds", self.world_name, "world.yaml")

        if not os.path.exists(metrics_config_path):
            raise FileNotFoundError(
                f"Zone metrics config not found: '{metrics_config_path}'. "
                f"Create '<world>/metrics/default.yaml' or pass a valid world_name."
            )

        if not os.path.exists(world_yaml_path):
            raise FileNotFoundError(
                f"World definition file not found: '{world_yaml_path}'. "
                f"Create '<world>/world.yaml' (with zones/corners) or pass a valid world_name."
            )

        with open(metrics_config_path, "r") as file:
            metrics_config = yaml.safe_load(file) or {}

        zones_config = metrics_config.get("zones", {})
        if not isinstance(zones_config, dict):
            raise ValueError(
                f"Invalid zones config in '{metrics_config_path}': expected a mapping under 'zones'."
            )

        # Support legacy structure where a "scoring" block is included under "zones".
        self.zones_scoring_config = zones_config.get("scoring", {}) or {}
        zones_config = {k: v for k, v in zones_config.items() if k != "scoring"}

        with open(world_yaml_path, "r") as file:
            world_config = yaml.safe_load(file) or {}

        world_zones = world_config.get("zones", [])
        if not isinstance(world_zones, list):
            world_zones = []

        # Build a quick lookup by zone name from world.yaml
        world_zone_by_name = {}
        for zone in world_zones:
            if not isinstance(zone, dict):
                continue
            name = zone.get("name")
            if not name:
                continue
            world_zone_by_name[name] = zone

        missing_world_zones = [name for name in zones_config.keys() if name not in world_zone_by_name]
        if missing_world_zones:
            missing_str = ", ".join(sorted(missing_world_zones))
            raise KeyError(
                f"Zone(s) configured in '{metrics_config_path}' but missing from '{world_yaml_path}': {missing_str}"
            )

        # Attach polygons from world.yaml corners for zones defined in default.yaml
        for zone_name, zone_cfg in zones_config.items():
            world_zone = world_zone_by_name.get(zone_name)
            corners = world_zone.get("corners", [])
            if not corners or not isinstance(corners, list):
                raise ValueError(
                    f"Zone '{zone_name}' in '{world_yaml_path}' has no valid 'corners' list."
                )

            polygon = []
            for c in corners:
                if not isinstance(c, dict):
                    raise ValueError(
                        f"Zone '{zone_name}' in '{world_yaml_path}' has an invalid corner entry: {c}"
                    )
                if "x" not in c or "y" not in c:
                    raise ValueError(
                        f"Zone '{zone_name}' in '{world_yaml_path}' corner is missing x/y: {c}"
                    )
                polygon.append([float(c["x"]), float(c["y"])])

            if len(polygon) < 3:
                raise ValueError(
                    f"Zone '{zone_name}' in '{world_yaml_path}' has < 3 corners; cannot form a polygon."
                )

            # Keep structure compatible with existing loop: a list of polygons.
            zone_cfg["polygons"] = [polygon]

        return zones_config

    def _analyze_episode(self, episode: pd.DataFrame, index) -> ZoneAwareMetric:
        super_analysis = super()._analyze_episode(episode, index)

        robot_positions = np.array([odom["position"][:2] for odom in episode["odom"]])

        zone_violations = []
        for zone_label, zone_config in self.zones_config.items():
            zone_violations.extend(
                self._check_zone_violations(zone_label, zone_config, robot_positions, episode)
            )

        zone_violation_count = len(zone_violations)
        zone_violation_time = sum(violation["duration"] for violation in zone_violations)

        overall_zone_score = 100.0
        if zone_violations:
            total_severity = sum(violation["severity"] for violation in zone_violations)
            overall_zone_score = 100.0 + total_severity / len(zone_violations)

        if self.zones_scoring_config.get("normalize_by_episode_time", False):
            episode_time = super_analysis["time_diff"]
            if episode_time > 0:
                overall_zone_score /= episode_time

        return ZoneAwareMetric(
            **super_analysis,
            overall_zone_score=overall_zone_score,
            zone_violation_count=zone_violation_count,
            zone_violation_time=zone_violation_time,
        )

    def _check_zone_violations(self, zone_type, zone_config, robot_positions, episode):
        violations = []

        polygons = zone_config.get("polygons", [])
        if not polygons:
            return violations

        was_in_violation = False
        violation_start_time = None
        violation_start_index = None

        for i, robot_pos in enumerate(robot_positions):
            is_in_zone = False
            for polygon_points in polygons:
                if len(polygon_points) >= 3:
                    polygon = np.array(polygon_points)
                    if self._is_point_in_polygon(robot_pos, polygon):
                        is_in_zone = True
                        break

            if is_in_zone and not was_in_violation:
                violation_start_time = int(episode["time"].iloc[i]) if i < len(episode["time"]) else 0
                violation_start_index = i
                was_in_violation = True
            elif not is_in_zone and was_in_violation:
                if violation_start_index is not None:
                    duration = i - violation_start_index
                    start_robot_pos = robot_positions[violation_start_index]
                    severity = self._calculate_violation_severity(
                        zone_type,
                        zone_config,
                        start_robot_pos,
                        violation_start_index,
                        duration,
                    )

                    violation = ZoneViolation(
                        timestamp=violation_start_time,
                        position=start_robot_pos.tolist(),
                        duration=duration,
                        severity=severity,
                        violation_type=zone_config.get("penalty_type", "zone_violation"),
                    )
                    violations.append(violation)

                was_in_violation = False
                violation_start_time = None
                violation_start_index = None

        if was_in_violation and violation_start_index is not None:
            duration = len(robot_positions) - violation_start_index
            start_robot_pos = robot_positions[violation_start_index]
            severity = self._calculate_violation_severity(
                zone_type,
                zone_config,
                start_robot_pos,
                violation_start_index,
                duration,
            )

            violation = ZoneViolation(
                timestamp=violation_start_time,
                position=start_robot_pos.tolist(),
                duration=duration,
                severity=severity,
                violation_type=zone_config.get("penalty_type", "zone_violation"),
            )
            violations.append(violation)

        return violations

    def _is_point_in_polygon(self, point, polygon):
        if not SHAPELY_AVAILABLE:
            return Math.point_in_polygon(point, polygon)

        polygon_shapely = Polygon(polygon)
        point_shapely = Point(point)
        return polygon_shapely.contains(point_shapely)

    def _calculate_violation_severity(self, zone_type, zone_config, robot_pos, time_index, duration):
        params = zone_config.get("params", {})
        base_penalty = params.get("violation_penalty", 0.0)
        penalty_weight = params.get("penalty_weight", 1.0)

        severity = base_penalty * penalty_weight

        if params.get("continuous_violation", False):
            time_multiplier = params.get("time_multiplier", 0.1)
            time_penalty = duration * time_multiplier
            severity = severity + (base_penalty * time_penalty)

        max_penalty = params.get("max_penalty", float("-inf"))
        if max_penalty > float("-inf"):
            severity = max(severity, max_penalty)

        return severity
