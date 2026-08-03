#!/usr/bin/env python3

import argparse
import csv
import math
import os
import re
import time
import traceback
from datetime import datetime

import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from hunav_msgs.msg import Agents
from nav_msgs.msg import Odometry
from rclpy.clock import Clock as RclpyClock
from rclpy.clock_type import ClockType
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from rclpy.serialization import serialize_message
from rosbag2_py import (ConverterOptions, SequentialWriter, StorageOptions,
                        TopicMetadata)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int16
# for transformations
from tf_transformations import euler_from_quaternion

# from arena_evaluation.scripts.utils import Pedestrian
# import pedsim_msgs.msg           as pedsim_msgs
import arena_evaluation_msgs.srv as arena_evaluation_srvs


def pose_stamped_to_xyyaw(msg: PoseStamped) -> list[float]:
    pose = msg.pose
    _, _, yaw = euler_from_quaternion([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ])
    return [
        round(pose.position.x, 3),
        round(pose.position.y, 3),
        round(yaw, 3),
    ]


class DataCollector(Node):

    def __init__(self, topic, unique_name):

        super().__init__(f'data_collector{unique_name}')

        topic_callbacks = [
            ("scan", self.laserscan_callback),
            ("odom", self.odometry_callback),
            ("cmd_vel", self.action_callback),
            ("human_states", self.hunav_agents_callback),
            ("pedsim_agents_data", self.hunav_agents_callback),
            # ("pedsim_agents_data", self.pedsim_callback)
        ]

        # raise ValueError(topic, unique_name)

        try:
            # Recorder() passes: [topic_name, output_name, msg_type]
            # BagRecorder() passes: [topic_name, topic_id, msg_type] where topic_id == topic_name
            # We therefore match callbacks against both the real topic basename and the output basename.
            self.topic_name = topic[0]
            self.output_name = topic[1]

            topic_basenames = {
                os.path.basename(str(self.topic_name).strip('/')),
                os.path.basename(str(self.output_name).strip('/')),
            }
            matches = (t[1] for t in topic_callbacks if t[0] in topic_basenames)
            type_callback = next(matches, lambda x: None)

            def callback(msg):
                self.msg = msg
                return type_callback(msg)
        except BaseException as e:
            self.get_logger().error(f"Error in callback setup: {e}")
            traceback.print_exc()
            return

        # In Recorder(): this is the output CSV file stem.
        # In BagRecorder(): this equals the real topic name.
        self.full_topic_name = topic[1]
        self.msg = None
        self.data = None

        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.subscription = self.create_subscription(
            topic[2],
            topic[0],
            callback,
            self.qos
        )

    def laserscan_callback(self, msg_laserscan: LaserScan):

        self.data = [msg_laserscan.range_max if math.isnan(val) else round(val, 3) for val in msg_laserscan.ranges]

    def odometry_callback(self, msg_odometry: Odometry):

        pose3d = msg_odometry.pose.pose
        twist = msg_odometry.twist.twist

        roll, pitch, yaw = euler_from_quaternion([
            pose3d.orientation.x,
            pose3d.orientation.y,
            pose3d.orientation.z,
            pose3d.orientation.w
        ])

        self.data = {
            "position": [
                round(pose3d.position.x, 3),
                round(pose3d.position.y, 3),
                round(yaw, 3)
            ],
            "velocity": [
                round(twist.linear.x, 3),
                round(twist.linear.y, 3),
                round(twist.angular.z, 3)
            ],
        }

    def action_callback(self, msg_action: Twist):  # variables will be written to csv whenever an action is published

        self.data = [
            round(msg_action.linear.x, 3),
            round(msg_action.linear.y, 3),
            round(msg_action.angular.z, 3)
        ]

    def hunav_agents_callback(self, msg_agents: Agents):

        # Convert hunav Agents to the schema expected by Utils.parse_pedsim:
        # a stringified list of dicts with keys matching scripts/utils.py::Pedestrian.
        pedestrians = []

        for agent in msg_agents.agents:
            # Only persons are relevant for pedestrian metrics.
            if hasattr(agent, 'PERSON') and agent.type != agent.PERSON:
                continue

            x = round(agent.position.position.x, 3)
            y = round(agent.position.position.y, 3)

            # Use yaw directly; it is already part of hunav Agent.
            theta = round(float(agent.yaw), 3)

            # Destination: use the first goal if present, otherwise current position.
            if getattr(agent, 'goals', None):
                gx = round(agent.goals[0].position.x, 3)
                gy = round(agent.goals[0].position.y, 3)
            else:
                gx, gy = x, y

            # Map behavior to a stable, readable string.
            beh_type = getattr(getattr(agent, 'behavior', None), 'type', None)
            beh_state = getattr(getattr(agent, 'behavior', None), 'state', None)
            social_state = f"beh_type={beh_type},beh_state={beh_state}"

            # For reporting: encode the configured HuNav agent name into the legacy
            # pedsim schema's `type` field.
            agent_name = getattr(agent, 'name', '')
            if agent_name:
                m = re.match(r"^hunav_(\d+)$", agent_name)
                if m:
                    agent_name = f"hunav_{int(m.group(1)):02d}"
                ped_type = agent_name
            else:
                ped_type = "PERSON"

            pedestrians.append({
                "id": str(agent.id),
                "name": ped_type,
                "social_state": social_state,
                "position": [x, y],
                "theta": theta,
                "destination": [gx, gy],
            })

        self.data = pedestrians

    def get_data(self):
        return (
            self.full_topic_name,
            self.data
        )

    def episode_callback(self, msg_scenario_reset):

        print(msg_scenario_reset)

        self.data = msg_scenario_reset.data

    # def pedsim_callback(self, msg_pedsim: pedsim_msgs.PedsimAgentsDataframe):
    #     self.data = [
    #         Pedestrian(
    #             id = agent.id,
    #             type = agent.type,
    #             social_state = agent.social_state,
    #             position = [agent.pose.position.x, agent.pose.position.y],
    #             theta = np.arctan2(agent.forces.force.y, agent.forces.force.x),
    #             destination = [agent.destination.x, agent.destination.y]
    #         )._asdict()
    #         for agent
    #         in msg_pedsim.agent_states
    #     ]


class TimeSegmentTracker:
    """Makes a backwards /clock step observable instead of silently absorbing it.

    THE DEFECT THIS EXISTS FOR
    --------------------------
    While /clock is stalled -- which happens for several seconds during scene load
    -- ``wall_clock_fallback_callback`` keeps recording by FABRICATING sim
    timestamps that advance by one record period each tick.  Those stamps run well
    past the last value /clock actually published.  When /clock resumes it
    therefore looks like time jumped backwards, and ``_record_tick`` used to
    absorb that by adopting the lower value as its new baseline and carrying on
    appending to the same CSV.

    Nothing recorded the event: no file rotation, no marker, no log line, and the
    ``episode`` column stayed at 0.  The result is a CSV that is several
    concatenated *time segments* rather than one monotonic series, with the only
    observable boundary being the backwards step itself.  Every consumer that
    sorted rows by `time` was therefore shuffling them, which inflated pedestrian
    travel, robot path length and the legacy path length -- by up to x41 in the
    delivered runs.

    WHAT THIS CHANGES
    -----------------
    The splice is still allowed to happen, because refusing it would mean dropping
    data, but it is no longer silent:

      * a WARNING names both stamps, the size of the step, and the classification
      * a sidecar ``recording_segments.csv`` records every boundary, so a consumer
        reads the segment structure instead of having to infer it
      * fabricated stamps are counted and announced when they start, so the rows
        that carry a lie about sim time can be identified

    Boundaries are classified, because the two causes need different responses:

      ``synthetic_overshoot`` -- the resumed stamp is at or above the last value
          /clock actually published, so sim time never went backwards at all; our
          own fabricated stamps overshot.  This is the cause of every boundary in
          the delivered runs.
      ``sim_clock_reset`` -- the resumed stamp is below the last real /clock
          reading, i.e. the simulator's clock genuinely restarted.

    Deliberately NOT changed: the fabrication strategy itself.  Holding the stamp
    at the last real /clock value instead of advancing it would remove the
    overshoot at the source, but it also changes which rows get written and how
    they are stamped, and that cannot be validated without a full eval run.  The
    counts and the classification recorded here are what a future lane needs to
    make that call on evidence.

    Nothing here assumes how many segments a run will have; the count tracks how
    often /clock stalls.
    """

    SEGMENT_MARKER_FILE = "recording_segments"
    SEGMENT_MARKER_HEADER = [
        "segment",
        "previous_time",
        "resumed_time",
        "backwards_by",
        "kind",
        "synthetic_rows_before",
    ]

    def _init_time_segment_state(self):
        # last stamp that came from /clock rather than from the fallback
        self._last_real_clock_time = None
        # True while the fallback is fabricating stamps ahead of /clock
        self._synthetic_time_active = False
        # rows written carrying a fabricated stamp
        self._synthetic_row_count = 0
        # 0-based index of the time segment currently being written
        self._recording_segment = 0
        self._segment_marker_started = False

    def _classify_backwards_step(self, resumed_time) -> str:
        if self._last_real_clock_time is not None and resumed_time >= self._last_real_clock_time:
            return "synthetic_overshoot"
        return "sim_clock_reset"

    def _note_time_segment_boundary(self, previous_time, resumed_time) -> None:
        """Log and persist a boundary.  Never raises: recording must not stop."""
        self._recording_segment += 1
        kind = self._classify_backwards_step(resumed_time)
        backwards_by = int(previous_time) - int(resumed_time)
        try:
            if not self._segment_marker_started:
                self.write_data(self.SEGMENT_MARKER_FILE, self.SEGMENT_MARKER_HEADER, mode="w")
                self._segment_marker_started = True
            self.write_data(
                self.SEGMENT_MARKER_FILE,
                [
                    self._recording_segment,
                    int(previous_time),
                    int(resumed_time),
                    backwards_by,
                    kind,
                    self._synthetic_row_count,
                ],
            )
        except BaseException as exc:  # pragma: no cover - marker must never stop recording
            self.get_logger().error(f"Could not write the time-segment marker: {exc}")
        self.get_logger().warn(
            f"/clock stepped BACKWARDS: {int(previous_time)} -> {int(resumed_time)} "
            f"({backwards_by} units, {kind}). Recording continues in the same files, so "
            f"they now hold {self._recording_segment + 1} concatenated time segments and "
            f"`time` is NOT a valid global sort key. "
            f"{self._synthetic_row_count} rows so far carry a fabricated stamp. "
            f"Boundaries are listed in {self.SEGMENT_MARKER_FILE}.csv."
        )

    def _note_synthetic_time_started(self, step_units) -> None:
        if self._synthetic_time_active:
            return
        self._synthetic_time_active = True
        self.get_logger().warn(
            f"/clock has not progressed for two record periods, so recording continues with "
            f"FABRICATED sim timestamps advancing {step_units} units per tick. These stamps do "
            f"not reflect simulator time; when /clock resumes they will appear as a forward "
            f"jump followed by a backwards step."
        )


class Recorder(TimeSegmentTracker, Node):

    def __init__(self, result_dir):

        super().__init__("data_recorder_node")

        self.declare_parameter("data_recorder_autoprefix", "")
        self.result_dir = self.get_directory(result_dir)

        self.declare_parameter("model", "")
        self.model = self.get_parameter("model").value

        self.base_dir = get_package_share_directory("arena_evaluation")
        self.result_dir = os.path.join(self.base_dir, "data", self.result_dir)
        # current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # self.base_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..', '..', 'src', 'arena', 'evaluation', 'arena_evaluation'))
        # self.result_dir = os.path.join(self.base_dir, "data", self.result_dir)
        os.makedirs(self.result_dir, exist_ok=True)

        self.write_params()

        topics_to_monitor = self.get_topics_to_monitor()
        published_topics = [topic[0] for topic in self.get_topic_names_and_types()]  # self.get_topic_names_and_types() is a list of tuples each tuple contain the topic name and a list of types

        topic_matcher = re.compile(f"({'|'.join([t[0] for t in topics_to_monitor])})$")

        topics_to_sub = []

        for topic_name in published_topics:

            match = re.search(topic_matcher, topic_name)

            if not match:
                continue

            if (topic_class := self.get_class_for_topic_name(topic_name)) is not None:
                topics_to_sub.append([topic_name, *topic_class])

        self.data_collectors = []

        self.declare_parameter('start', [0.0, 0.0, 0.0])
        self.declare_parameter('goal', [0.0, 0.0, 0.0])

        for topic in topics_to_sub:
            topic_name = topic[0]
            unique_name = topic_name.replace('/', '_')
            data_collector = DataCollector(topic, unique_name)
            self.data_collectors.append(data_collector)
            self.write_data(
                topic[1],
                ["time", "data"],
                mode="w"
            )

        self.write_data("episode", ["time", "episode"], mode="w")
        self.write_data("start_goal", ["episode", "start", "goal"], mode="w")

        self.config = self.read_config()
        self._record_period_sec = max(float(self.config.get("record_frequency", 400)) / 1000.0, 0.05)
        self._last_clock_progress_wall_time = 0.0
        self._last_clock_sim_time = None
        self._last_record_wall_time = 0.0
        self._init_time_segment_state()

        self.current_episode = 0
        self.current_time = None

        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.clock_sub = self.create_subscription(
            Clock,
            "/clock",
            self.clock_callback,
            self.qos
        )

        self.scenario_reset_sub = self.create_subscription(
            Int16,
            "/scenario_reset",
            self.scenario_reset_callback,
            self.qos
        )
        self.wall_clock_timer = self.create_timer(
            self._record_period_sec,
            self.wall_clock_fallback_callback,
            clock=RclpyClock(clock_type=ClockType.STEADY_TIME),
        )

        # Define the service for changing directory
        self.change_directory_service = self.create_service(
            arena_evaluation_srvs.ChangeDirectory,
            'change_directory',
            self.change_directory_callback
        )

    def get_directory(self, directory: str):
        AUTO_PREFIX = "auto:/"
        PARAM_AUTO_PREFIX = "data_recorder_autoprefix"

        if directory.startswith(AUTO_PREFIX):
            set_prefix = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            print(f"Generated timestamp: {set_prefix}")

            param_value = self.get_parameter(PARAM_AUTO_PREFIX).value

            if param_value == "":
                self.set_parameters([rclpy.parameter.Parameter(PARAM_AUTO_PREFIX, rclpy.Parameter.Type.STRING, set_prefix)])
            else:
                set_prefix = param_value

            directory = os.path.join(str(set_prefix), directory[len(AUTO_PREFIX):])

        return directory

    def write_params(self):

        with open(self.result_dir + "/params.yaml", "w") as file:

            # Declare the parameters locally in the method
            self.declare_parameter("map_file", "")
            self.declare_parameter("scenario_file", "")
            self.declare_parameter("inter_planner", "")
            self.declare_parameter("local_planner", "")
            self.declare_parameter("agent_name", "")

            # Get the parameter values
            map_file = self.get_parameter("map_file").value
            scenario_file = self.get_parameter("scenario_file").value
            inter_planner = self.get_parameter("inter_planner").value
            local_planner = self.get_parameter("local_planner").value
            agent_name = self.get_parameter("agent_name").value
            namespace = self.get_namespace().strip('/')

            yaml.dump({
                "model": self.model,
                "map_file": map_file,
                "scenario_file": scenario_file,
                "inter_planner": inter_planner,
                "local_planner": local_planner,
                "agent_name": agent_name,
                "namespace": namespace
            }, file)

    def get_topics_to_monitor(self):

        namespace = self.get_namespace().strip("/")

        return [
            (f"{namespace}/scan", LaserScan),
            (f"{namespace}/scenario_reset", Int16),
            (f"{namespace}/odom", Odometry),
            (f"{namespace}/cmd_vel", Twist),
            (f"{namespace}/human_states", Agents),
            ("/human_states", Agents),
            # ("/pedsim_simulator/pedsim_agents_data", pedsim_msgs.PedsimAgentsDataframe)
        ]

    def get_class_for_topic_name(self, topic_name):
        if "/scan" in topic_name:
            return ["scan", LaserScan]
        if "/odom" in topic_name:
            return ["odom", Odometry]
        if "/cmd_vel" in topic_name:
            return ["cmd_vel", Twist]
        if "/human_states" in topic_name:
            # Keep legacy file name expected by PedsimMetrics.
            return ["pedsim_agents_data", Agents]
        # if "/pedsim_agents_data" in topic_name:
        #     return ["pedsim_agents_data", pedsim_msgs.PedsimAgentsDataframe]

    def write_data(self, file_name, data, mode="a"):
        with open(f"{self.result_dir}/{file_name}.csv", mode, newline="") as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(data)
            file.close()

    def read_config(self):
        with open(self.base_dir + "/config" + "/data_recorder_config.yaml") as file:
            return yaml.safe_load(file)

    def clock_callback(self, clock: Clock):
        current_simulation_action_time = int(clock.clock.sec * 10_000_000_000 + clock.clock.nanosec)

        if self._last_clock_sim_time != current_simulation_action_time:
            self._last_clock_sim_time = current_simulation_action_time
            self._last_clock_progress_wall_time = time.monotonic()
            self._synthetic_time_active = False

        self._record_tick(current_simulation_action_time)

    def wall_clock_fallback_callback(self):
        if self._last_clock_sim_time is None or self.current_time is None:
            return
        if self._last_record_wall_time and time.monotonic() - self._last_record_wall_time < self._record_period_sec * 2.0:
            return
        step_ns = int(float(self.config["record_frequency"]) * 1_000_000)
        synthetic_time = self.current_time + step_ns
        self._note_synthetic_time_started(step_ns)
        self._record_tick(synthetic_time, synthetic=True)

    def _record_tick(self, current_simulation_action_time, synthetic=False):

        if self.current_time is None:
            self.current_time = current_simulation_action_time
        elif current_simulation_action_time < self.current_time:
            # A backwards step splices two time segments into one set of files.
            # Report it before rebasing so no consumer is handed spliced data
            # without a marker (see TimeSegmentTracker).
            self._note_time_segment_boundary(self.current_time, current_simulation_action_time)
            self.current_time = current_simulation_action_time

        if not synthetic:
            self._last_real_clock_time = current_simulation_action_time

        time_diff = (current_simulation_action_time - self.current_time) / 1e6  # in ms

        if time_diff < self.config["record_frequency"]:
            return

        self.current_time = current_simulation_action_time
        self._last_record_wall_time = time.monotonic()
        if synthetic:
            self._synthetic_row_count += 1

        for collector in self.data_collectors:

            topic_name, data = collector.get_data()

            self.write_data(topic_name, [self.current_time, data])

        self.write_data("episode", [self.current_time, self.current_episode])
        self.write_data("start_goal", [
            self.current_episode,
            self.get_parameter('start').value,
            self.get_parameter('goal').value
        ])

    def scenario_reset_callback(self, data: Int16):
        self.current_episode = data.data

    def change_directory_callback(self, request, response):  # ROS2: Change parameters and update configurations on the fly without needing to restart the node
        new_directory = request.data
        self.result_dir = self.get_directory(new_directory)
        response.success = True
        response.message = "Directory changed successfully"
        return response


class BagRecorder(TimeSegmentTracker, Node):
    def __init__(self, result_dir: str):
        super().__init__("bag_recorder_node")

        self.declare_parameter("data_recorder_autoprefix", "")
        self.result_dir = self.get_directory(result_dir)

        self.declare_parameter("model", "")
        self.model = self.get_parameter("model").value

        self.declare_parameter("world", "")
        self.world = self.get_parameter("world").value

        self.base_dir = get_package_share_directory("arena_evaluation")
        self.result_dir = os.path.join(self.base_dir, "data", self.result_dir)
        os.makedirs(self.result_dir, exist_ok=True)

        self.write_params()

        topics_to_monitor = self.get_topics_to_monitor()

        # Resolve any remappings so we record under the actual topic names.
        # This avoids rosbag metadata listing topics that never existed on the graph
        # (e.g. when a subscription is remapped at launch time).
        resolved_topics_to_monitor = []
        resolved_seen = set()
        for topic_name, msg_type in topics_to_monitor:
            try:
                resolved_name = self.resolve_topic_name(topic_name)
            except BaseException:
                resolved_name = topic_name
            if resolved_name in resolved_seen:
                continue
            resolved_seen.add(resolved_name)
            resolved_topics_to_monitor.append((resolved_name, msg_type))
        topics_to_monitor = resolved_topics_to_monitor
        published_topics = [t[0] for t in topics_to_monitor]

        topic_matcher = re.compile(f"({'|'.join([t[0] for t in topics_to_monitor])})$")

        topics_to_sub = []
        for topic_name in published_topics:
            match = re.search(topic_matcher, topic_name)
            if not match:
                continue
            # Append a list: [full_topic_name, topic_id, topic_type]
            if (topic_class := self.get_class_for_topic_name(topic_name)) is not None:
                topics_to_sub.append([topic_name, topic_name, topic_class[1]])

        self.data_collectors = []

        # When using BagRecorder, we also export CSV in the same format as Recorder
        # so that the existing metrics pipeline can be reused.
        self._csv_topic_to_file = {}
        human_csv_created = False

        self.declare_parameter('start', [0.0, 0.0, 0.0])
        self.declare_parameter('goal', [0.0, 0.0, 0.0])
        self.declare_parameter('scenario_reset_topic', '/scenario_reset')
        self.declare_parameter('start_topic', 'episode_start_pose')
        self.declare_parameter('goal_topic', 'episode_goal_pose_metadata')
        self.current_start = [0.0, 0.0, 0.0]
        self.current_goal = [0.0, 0.0, 0.0]
        for topic in topics_to_sub:
            topic_name = topic[0]
            unique_name = topic_name.replace('/', '_')
            collector = DataCollector(topic, unique_name)
            self.data_collectors.append(collector)

            # Only create CSV outputs for topics supported by DataCollector callbacks.
            basename = os.path.basename(topic_name)
            if basename in ("scan", "odom", "cmd_vel"):
                self._csv_topic_to_file[topic_name] = basename
                self.write_data(basename, ["time", "data"], mode="w")
            elif basename in ("lidar",):
                # Jackal's default scan topic in some Arena configs is <ns>/lidar; metrics expects scan.csv.
                self._csv_topic_to_file[topic_name] = "scan"
                self.write_data("scan", ["time", "data"], mode="w")
            elif basename in ("human_states",):
                # Keep legacy PedsimMetrics output and the benchmark artifact name in sync.
                # If multiple human_states topics are configured (e.g. robot namespace + parent namespace),
                # only export one of them to CSV to avoid duplicated rows.
                if not human_csv_created:
                    self._csv_topic_to_file[topic_name] = ["human_states", "pedsim_agents_data"]
                    self.write_data("human_states", ["time", "data"], mode="w")
                    self.write_data("pedsim_agents_data", ["time", "data"], mode="w")
                    human_csv_created = True

        # Write extra information as needed (episode and start_goal can be recorded as parameters or in a separate bag topic)
        self.current_episode = 0
        self.current_time = None

        # Keep the same additional CSV files as Recorder
        self.write_data("episode", ["time", "episode"], mode="w")
        self.write_data("start_goal", ["episode", "start", "goal"], mode="w")

        # --- Setup rosbag2 writer ---

        bag_uri = os.path.join(self.result_dir, "recording")
        storage_options = StorageOptions(uri=bag_uri, storage_id='sqlite3')
        converter_options = ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        self.writer = SequentialWriter()
        self.writer.open(storage_options, converter_options)
        # Create topic metadata for each topic that will be recorded.
        self.topics_metadata = {}
        for topic in topics_to_sub:
            topic_name = topic[0]
            msg_type = topic[2]
            # Construct the type string. This follows the convention "package/msg/MessageType"
            type_str = f"{os.path.dirname(msg_type.__module__.replace('.', '/'))}/{msg_type.__name__}"
            topic_metadata_kwargs = {
                'name': topic_name.strip('/'),
                'type': type_str,
                'serialization_format': 'cdr',
            }
            try:
                metadata = TopicMetadata(
                    **topic_metadata_kwargs,
                    offered_qos_profiles='',
                )
            except TypeError:
                # ROS 2 Jazzy rosbag2_py expects an explicit topic id and a QoS list.
                metadata = TopicMetadata(
                    id=len(self.topics_metadata) + 1,
                    **topic_metadata_kwargs,
                    offered_qos_profiles=[],
                )
            self.writer.create_topic(metadata)
            self.topics_metadata[topic_name] = metadata

        # Setup QoS for clock and scenario reset subscriptions
        self.qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=10,
        )

        self.clock_sub = self.create_subscription(
            Clock,
            "/clock",
            self.clock_callback,
            self.qos
        )

        scenario_reset_topic = str(self.get_parameter('scenario_reset_topic').value or '/scenario_reset')
        self.scenario_reset_sub = self.create_subscription(
            Int16,
            self.resolve_topic_name(scenario_reset_topic),
            self.scenario_reset_callback,
            self.qos
        )
        pose_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.start_sub = self.create_subscription(
            PoseStamped,
            self.resolve_topic_name(str(self.get_parameter('start_topic').value or 'episode_start_pose')),
            self.start_callback,
            pose_qos,
        )
        self.goal_sub = self.create_subscription(
            PoseStamped,
            self.resolve_topic_name(str(self.get_parameter('goal_topic').value or 'episode_goal_pose_metadata')),
            self.goal_callback,
            pose_qos,
        )
        self.config = self.read_config()
        self._record_period_sec = max(float(self.config.get("record_frequency", 400)) / 1000.0, 0.05)
        self._last_clock_progress_wall_time = 0.0
        self._last_clock_sim_time = None
        self._last_record_wall_time = 0.0
        self._init_time_segment_state()
        self.wall_clock_timer = self.create_timer(
            self._record_period_sec,
            self.wall_clock_fallback_callback,
            clock=RclpyClock(clock_type=ClockType.STEADY_TIME),
        )

        self.change_directory_service = self.create_service(
            arena_evaluation_srvs.ChangeDirectory,
            'change_directory',
            self.change_directory_callback
        )

        self.get_logger().info(f"Started recording to rosbag at: {bag_uri}")

    def write_data(self, file_name, data, mode="a"):
        with open(f"{self.result_dir}/{file_name}.csv", mode, newline="") as file:
            writer = csv.writer(file, delimiter=',')
            writer.writerow(data)
            file.close()

    def get_directory(self, directory: str) -> str:
        AUTO_PREFIX = "auto:/"
        PARAM_AUTO_PREFIX = "data_recorder_autoprefix"
        if directory.startswith(AUTO_PREFIX):
            set_prefix = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            self.get_logger().info(f"Generated timestamp: {set_prefix}")
            param_value = self.get_parameter(PARAM_AUTO_PREFIX).value
            if param_value == "":
                self.set_parameters([Parameter(PARAM_AUTO_PREFIX, Parameter.Type.STRING, set_prefix)])
            else:
                set_prefix = param_value
            directory = os.path.join(str(set_prefix), directory[len(AUTO_PREFIX):])
        return directory

    def write_params(self):
        # Write runtime parameters to a YAML file (for record keeping)
        params_path = os.path.join(self.result_dir, "params.yaml")
        with open(params_path, "w") as file:
            self.declare_parameter("map_file", "")
            self.declare_parameter("scenario_file", "")
            self.declare_parameter("inter_planner", "")
            self.declare_parameter("local_planner", "")
            self.declare_parameter("agent_name", "")

            map_file = self.get_parameter("map_file").value
            scenario_file = self.get_parameter("scenario_file").value
            inter_planner = self.get_parameter("inter_planner").value
            local_planner = self.get_parameter("local_planner").value
            agent_name = self.get_parameter("agent_name").value
            namespace = self.get_namespace().strip('/')
            yaml.dump({
                "model": self.model,
                "world": self.world,
                "map_file": map_file,
                "scenario_file": scenario_file,
                "inter_planner": inter_planner,
                "local_planner": local_planner,
                "agent_name": agent_name,
                "namespace": namespace
            }, file)

    def get_topics_to_monitor(self):
        namespace = self.get_namespace().rstrip('/')

        # Workaround for setups where the robot runs in a child namespace (e.g. /task_generator_node/jackal)
        # but HuNav publishes human_states in the parent namespace (e.g. /task_generator_node).
        parts = namespace.strip('/').split('/') if namespace not in ('', '/') else []
        parent_namespace = '/' + '/'.join(parts[:-1]) if len(parts) > 1 else namespace

        topics = [
            (f"{namespace}/scan", LaserScan),
            (f"{namespace}/front/scan", LaserScan),
            ("/front/scan", LaserScan),
            (f"{namespace}/lidar", LaserScan),
            ("/lidar", LaserScan),
            (f"{namespace}/scenario_reset", Int16),
            (f"{namespace}/odom", Odometry),
            (f"{namespace}/cmd_vel", Twist),
        ]

        if parent_namespace and parent_namespace != namespace:
            topics.append((f"{parent_namespace}/human_states", Agents))

        topics.extend([
            (f"{namespace}/human_states", Agents),
            ("/human_states", Agents),
        ])

        return topics

    def get_class_for_topic_name(self, topic_name: str):
        if "/scan" in topic_name or "/lidar" in topic_name:
            return ["scan", LaserScan]
        if "/odom" in topic_name:
            return ["odom", Odometry]
        if "/cmd_vel" in topic_name:
            return ["cmd_vel", Twist]
        if "/scenario_reset" in topic_name:
            return ["scenario_reset", Int16]
        if "/human_states" in topic_name:
            return ["human_states", Agents]  # hunav topic
        # if "/pedsim_agents_data" in topic_name:
        #     return ["pedsim_agents_data", pedsim_msgs.PedsimAgentsDataframe]

    def clock_callback(self, clock: Clock):
        # Keep the same time scaling as Recorder to stay compatible with existing metrics converters.
        # NOTE: use an integer scale to keep timestamps as int for rosbag2 writer.
        current_simulation_action_time = clock.clock.sec * 10_000_000_000 + clock.clock.nanosec

        if self._last_clock_sim_time != current_simulation_action_time:
            self._last_clock_sim_time = current_simulation_action_time
            self._last_clock_progress_wall_time = time.monotonic()
            self._synthetic_time_active = False

        self._record_tick(current_simulation_action_time)

    def wall_clock_fallback_callback(self):
        if self._last_clock_sim_time is None or self.current_time is None:
            return
        if self._last_record_wall_time and time.monotonic() - self._last_record_wall_time < self._record_period_sec * 2.0:
            return
        step_ns = int(float(self.config["record_frequency"]) * 1_000_000)
        synthetic_time = self.current_time + step_ns
        self._note_synthetic_time_started(step_ns)
        self._record_tick(synthetic_time, synthetic=True)

    def _record_tick(self, current_simulation_action_time, synthetic=False):
        if self.current_time is None:
            self.current_time = current_simulation_action_time
        elif current_simulation_action_time < self.current_time:
            # A backwards step splices two time segments into one set of files and
            # into the rosbag.  Report it before rebasing so no consumer is handed
            # spliced data without a marker (see TimeSegmentTracker).
            self._note_time_segment_boundary(self.current_time, current_simulation_action_time)
            self.current_time = current_simulation_action_time

        if not synthetic:
            self._last_real_clock_time = current_simulation_action_time

        # Record at the configured frequency (in ms) from the configuration file
        time_diff = (current_simulation_action_time - self.current_time) / 1e6  # in ms
        if time_diff < self.config["record_frequency"]:
            return

        self.current_time = current_simulation_action_time
        self._last_record_wall_time = time.monotonic()
        if synthetic:
            self._synthetic_row_count += 1

        # For each DataCollector, retrieve the last message and record it into the rosbag.
        for collector in self.data_collectors:
            topic_name = collector.topic_name
            msg = collector.msg
            # self.get_logger().warn(f"collected {topic_name}: {msg}")

            if msg is None:
                continue
            try:
                serialized_msg = serialize_message(msg)
                self.writer.write(topic_name.strip('/'), serialized_msg, int(self.current_time))
            except BaseException as e:
                self.get_logger().error(f"Error writing message on topic {topic_name}: {e}")

            # Additionally export CSV for supported topics
            if topic_name in self._csv_topic_to_file:
                _, data = collector.get_data()
                csv_targets = self._csv_topic_to_file[topic_name]
                if isinstance(csv_targets, str):
                    csv_targets = [csv_targets]
                for csv_target in csv_targets:
                    self.write_data(csv_target, [self.current_time, data])

        # Export episode and start/goal parameters in the same format as Recorder
        self.write_data("episode", [self.current_time, self.current_episode])
        self.write_data("start_goal", [
            self.current_episode,
            self.current_start,
            self.current_goal
        ])

    def scenario_reset_callback(self, data: Int16):
        self.current_episode = data.data

    def start_callback(self, data: PoseStamped):
        self.current_start = pose_stamped_to_xyyaw(data)

    def goal_callback(self, data: PoseStamped):
        self.current_goal = pose_stamped_to_xyyaw(data)

    def change_directory_callback(self, request, response):
        new_directory = request.data
        self.result_dir = self.get_directory(new_directory)
        response.success = True
        response.message = "Directory changed successfully"
        return response

    def read_config(self):
        config_path = os.path.join(self.base_dir, "config", "data_recorder_config.yaml")
        with open(config_path, "r") as file:
            return yaml.safe_load(file)


def main(args=None):

    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", default="auto:/")
    arguments, extra_args = parser.parse_known_args()  # Parse the known arguments and ignore the extra_args

    try:
        recorder = BagRecorder(arguments.dir)

        executor = MultiThreadedExecutor()
        executor.add_node(recorder)

        for collector in recorder.data_collectors:
            executor.add_node(collector)

        executor.spin()

    except BaseException as e:
        print(f"Exception in main: {e}")
        traceback.print_exc()
        raise e
    finally:
        # recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
