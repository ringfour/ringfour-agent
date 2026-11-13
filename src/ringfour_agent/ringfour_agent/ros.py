"""The agent's ROS interface: publishers, subscriptions, TF.

The ONLY module (besides main.py) that imports rclpy. Everything robot-
specific (topic names, frame names) comes from the profile; there is no
robot knowledge here. Handlers talk to this object through the small
method surface below — blocking methods (TF lookups) must be called via
asyncio.to_thread so they never stall the RPC loop.

Thread model: the node spins on a background MultiThreadedExecutor
(main.py); subscription callbacks update state under a lock; the asyncio
thread reads snapshots and publishes commands (rclpy publishers are safe
to call from any thread).
"""

from __future__ import annotations

import math
import threading
import time
import xml.etree.ElementTree as ET
from collections import deque
from typing import Any

import numpy as np
import yaml
from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import (
    ConfigureController,
    ListControllers,
    LoadController,
    SwitchController,
    UnloadController,
)
from diagnostic_msgs.msg import DiagnosticArray
from geometry_msgs.msg import Pose2D
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist as TwistMsg
from geometry_msgs.msg import TwistStamped as TwistStampedMsg
from geometry_msgs.msg import WrenchStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from nav_msgs.msg import Odometry
from rcl_interfaces.srv import ListParameters
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.time import Time
from sensor_msgs.msg import (
    BatteryState,
    CameraInfo,
    FluidPressure,
    Illuminance,
    Image,
    Imu,
    JointState,
    MagneticField,
    Range,
    Temperature,
)
from sensor_msgs.msg import LaserScan as LaserScanMsg
from slam_toolbox.srv import (
    DeserializePoseGraph,
    Pause,
    Reset,
    SerializePoseGraph,
)
from std_msgs.msg import Float64MultiArray, String
from trajectory_msgs.msg import (
    JointTrajectory,
    JointTrajectoryPoint,
)
import tf2_ros

from ringfour.description import sha_of
from ringfour.enums import Controller

# Where the composed description is published. Never configurable at the
# robot level: spec 26.2 is explicit that there is one live source and no
# file path. A GROUP may override it (namespaced launches), which is a
# different thing — that is still a live topic.
DEFAULT_DESCRIPTION_TOPIC = "/robot_description"

# Odometry QoS: BEST_EFFORT + VOLATILE, keep-last depth 10. We always want
# the latest pose, so dropped samples don't matter; more importantly a
# BEST_EFFORT subscription is compatible with BOTH best-effort and reliable
# publishers, while a RELIABLE subscription silently receives NOTHING from a
# best-effort publisher (which is how diff_drive_controller publishes odom).
# A RELIABLE+VOLATILE sub looked fine in theory and got zero messages in
# practice — see docs/worklogs/ringfour_phase3.md.
_ODOM_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

# Sensor streams (images, scans, joint states, battery): same rule as
# odometry — BEST_EFFORT receives from both best-effort and reliable
# publishers, RELIABLE silently receives nothing from best-effort ones.
_SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

# /robot_description is published once, latched (TRANSIENT_LOCAL); a
# volatile subscription joins too late and never sees it.
_LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

from .profile import Profile
from .vendor import VendorSensors, _ros_type

# RGB/depth messages buffered per camera for stamp pairing (get_synced).
#
# This is a TIME WINDOW, not a frame count: it has to be deep enough to still
# contain a matching pair after one stream stalls. Real robots stall — a
# thermally throttled SBC, a VLM eating memory, a GPU busy elsewhere — and a
# sensor that misses a burst of frames is normal, not exceptional. At 10
# frames (~400 ms at 25 Hz) a routine ~400 ms depth stall pushed the last
# pairable rgb off the end of the deque and get_synced failed outright, even
# though a perfectly good pair had existed a moment earlier.
#
# Deeper costs memory: each entry holds the ROS Image, ~0.9 MiB rgb and
# ~1.2 MiB depth at VGA. 30 entries is ~2 s at 15 Hz for ~63 MiB per camera,
# and subscriptions are lazy, so only cameras actually in use pay it.
_CAMERA_BUFFER = 30

# Below these speeds the robot reads as "not moving". Guessed: sim odom
# noise at standstill is ~1e-4; revisit on hardware.
_MOVING_LINEAR = 0.01   # m/s
_MOVING_ANGULAR = 0.01  # rad/s


class TfLookupError(Exception):
    """A TF lookup failed (unknown frame or timeout waiting for data)."""


class _CameraState:
    """Per-camera subscriptions and message buffers."""

    def __init__(self) -> None:
        self.rgb: deque = deque(maxlen=_CAMERA_BUFFER)    # (stamp, Image)
        self.depth: deque = deque(maxlen=_CAMERA_BUFFER)  # (stamp, Image)
        self.info: CameraInfo | None = None
        self.subscribed = False


class RosInterface:
    """Profile-driven ROS plumbing for one robot."""

    def __init__(self, node: Node, profile: Profile) -> None:
        self._node = node
        self._spec = profile.spec
        self._frames = profile.spec.frames
        self._lock = threading.RLock()
        self._odom_received = None
        self._odom: Odometry | None = None
        self._distance = 0.0
        self._last_xy: tuple[float, float] | None = None

        self._cmd_vel_pub = None
        self._odom_topic = None
        self._cmd_vel_stamped = True
        mobility = profile.spec.mobility
        if mobility is not None and mobility.odom_topic:
            self._odom_topic = mobility.odom_topic
        if mobility is not None and mobility.cmd_vel:
            # On Jazzy, twist_mux (use_stamped:true) and diff_drive_controller
            # both subscribe TwistStamped, so the cmd_vel sink type is a
            # profile detail (mobility.cmd_vel_stamped). Publishing the wrong
            # type means the messages are silently never delivered.
            self._cmd_vel_stamped = mobility.cmd_vel_stamped
            msg_type = TwistStampedMsg if self._cmd_vel_stamped else TwistMsg
            self._cmd_vel_pub = node.create_publisher(
                msg_type, mobility.cmd_vel, 10)
        if mobility is not None and mobility.odom_topic:
            node.create_subscription(
                Odometry, mobility.odom_topic, self._on_odom, _ODOM_QOS)

        self._tf_buffer = tf2_ros.Buffer(node=node)
        self._tf_listener = tf2_ros.TransformListener(
            self._tf_buffer, node, spin_thread=False)

        # Autonomy plumbing (Phase 5): Nav2 action client, slam_toolbox
        # service clients, /map subscription. Created eagerly when the
        # profile declares the capability; all calls are blocking and go
        # through asyncio.to_thread (futures resolve on the executor
        # thread, the worker thread waits on an Event).
        self._nav_submission = None
        self._traj_submission = None
        self._nav_client = None
        self._nav_goal_handle = None
        # Bumped by nav_cancel. A goal being sent captures this before the
        # round-trip and re-checks it after acceptance, so a stop() that
        # lands mid-send is not lost. See nav_send_goal.
        self._nav_stop_epoch = 0
        self._nav_feedback: dict[str, Any] | None = None
        self._nav_result_future = None
        if mobility is not None and mobility.nav2_navigate_to_pose:
            self._nav_client = ActionClient(
                node, NavigateToPose, mobility.nav2_navigate_to_pose)
        self._costmap_clients = [
            node.create_client(ClearEntireCostmap, name)
            for name in ("/global_costmap/clear_entirely_global_costmap",
                         "/local_costmap/clear_entirely_local_costmap")
        ]
        self._slam_clients: dict[str, Any] = {}
        self._map_msg: OccupancyGridMsg | None = None
        if profile.spec.slam is not None:
            self._slam_clients = {
                "serialize": node.create_client(
                    SerializePoseGraph, "/slam_toolbox/serialize_map"),
                "deserialize": node.create_client(
                    DeserializePoseGraph, "/slam_toolbox/deserialize_map"),
                "pause": node.create_client(
                    Pause, "/slam_toolbox/pause_new_measurements"),
                "reset": node.create_client(
                    Reset, "/slam_toolbox/reset"),
            }
            # slam_toolbox publishes /map transient_local at
            # map_update_interval; latch the latest for robot.map.
            node.create_subscription(
                OccupancyGridMsg, "/map", self._on_map,
                QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                ))

        # Sensing state (Phase 4). Camera subscriptions are created
        # lazily on first use (4 cameras x 30 fps of Python-side image
        # callbacks is real CPU; don't pay it for cameras nobody reads).
        self._sub_lock = threading.Lock()
        self._cameras: dict[str, _CameraState] = {}
        self._scan: LaserScanMsg | None = None
        self._joints: dict[str, tuple[float, float, float]] = {}
        self._joint_stamps: dict[str, float] = {}
        self._joint_received: dict[str, float] = {}
        self._joint_history = deque(maxlen=256)
        self._battery: BatteryState | None = None
        self._description: str | None = None
        self._description_sha: str | None = None
        self._joint_limits: dict[str, tuple[float, float] | None] | None = None

        # Cheap state subscriptions are eager: joint states, battery and
        # the low-rate sensor family feed the watch sampler, which must
        # see data before any client asks for it. The robot description
        # arrives once, latched. Only image topics are lazy.
        if self._spec.groups:
            node.create_subscription(
                JointState, self._spec.joint_states,
                self._on_joint_state, _SENSOR_QOS)
        status = self._spec.status
        if status is not None and status.battery:
            node.create_subscription(
                BatteryState, status.battery, self._on_battery, _SENSOR_QOS)
        if status is not None and status.diagnostics:
            node.create_subscription(
                DiagnosticArray, status.diagnostics,
                self._on_diagnostics, _SENSOR_QOS)
        node.create_subscription(
            String, DEFAULT_DESCRIPTION_TOPIC,
            self._on_description, _LATCHED_QOS)
        # Namespaced launches publish a description per arm instead of one
        # composed tree (spec 8.6). Each distinct override gets its own
        # latched subscription; the group is validated against the topic
        # it declares, not against the default one.
        self._group_descriptions: dict[str, str] = {}
        self._group_topics: dict[str, str] = {
            name: group.description_topic
            for name, group in (self._spec.groups or {}).items()
            if group.description_topic
        }
        for topic in set(self._group_topics.values()):
            node.create_subscription(
                String, topic,
                lambda msg, t=topic: self._on_group_description(t, msg),
                _LATCHED_QOS)

        # controller_manager introspection, used once at startup to
        # validate the profile's groups (6.1) and again at control-session
        # entry (6.4). Created unconditionally: a robot with no groups
        # still has a controller_manager, and creating a client costs
        # nothing until it is called.
        self._cm_list = node.create_client(
            ListControllers, "/controller_manager/list_controllers")
        self._cm_params = node.create_client(
            ListParameters, "/controller_manager/list_parameters")
        # The session lifecycle (6.4): load -> configure -> switch on
        # entry, switch back -> unload on exit. Four calls, each a
        # failure point, and the teardown is the half that matters —
        # a crashed mission that skips it leaves the arm on a
        # controller nobody is driving.
        self._cm_load = node.create_client(
            LoadController, "/controller_manager/load_controller")
        self._cm_configure = node.create_client(
            ConfigureController, "/controller_manager/configure_controller")
        self._cm_switch = node.create_client(
            SwitchController, "/controller_manager/switch_controller")
        self._cm_unload = node.create_client(
            UnloadController, "/controller_manager/unload_controller")
        # One trajectory action client per controller, created on demand:
        # the controller does not exist until a session loads it.
        self._traj_clients: dict[str, ActionClient] = {}
        self._traj_goal: Any = None
        self._traj_result: dict[str, Any] | None = None
        self._traj_feedback: Any = None
        # Command publishers for position controllers, created EAGERLY at
        # startup rather than on first use. A publisher created and
        # published to in the same breath loses its first messages: DDS
        # matching with the controller's subscription has not completed
        # yet, and the write goes nowhere with no error. That cost an hour
        # and a wrong entry in found_along_the_way (item 11) — the gripper
        # looked dead when it was only unmatched.
        self._position_publishers: dict[str, Any] = {}
        for group_config in (self._spec.groups or {}).values():
            for role in (Controller.TOOL, Controller.STREAM):
                controller = group_config.controllers.get(role)
                if controller and controller not in self._position_publishers:
                    self._position_publishers[controller] = (
                        node.create_publisher(
                            Float64MultiArray,
                            f"/{controller}/commands", 10))

        # Vendor escape hatches (profile `sensors:` / `actions:`). Types
        # are resolved at runtime, so no per-vendor code lives here.
        self._vendor = VendorSensors(node, self._spec.sensors or {},
                                     _SENSOR_QOS)
        self._vendor_publishers: dict[str, Any] = {}
        self._vendor_types: dict[str, Any] = {}
        for name, action in (self._spec.actions or {}).items():
            try:
                from rosidl_runtime_py.utilities import get_message

                message_type = get_message(_ros_type(action.type))
            except Exception as exc:
                node.get_logger().warn(
                    f"actions.{name}: cannot resolve type "
                    f"{action.type!r} ({exc})")
                continue
            self._vendor_types[name] = message_type
            self._vendor_publishers[name] = node.create_publisher(
                message_type, action.topic, 10)
            node.get_logger().info(
                f"  actions.{name} -> {action.topic} ({action.type}, "
                f"motion={action.is_motion})")

        # latest-value sensors: one slot per source,
        # read back as wire dicts by the matching *_reading() method.
        self._latest: dict[str, Any] = {}
        self._latest_received: dict[str, float] = {}
        self._battery_received = None
        for name, range_cfg in (self._spec.range_sensors or {}).items():
            self._subscribe_latest(Range, range_cfg.topic, f"range.{name}")
        if self._spec.imu is not None:
            self._subscribe_latest(Imu, self._spec.imu.topic, "imu")
            if self._spec.imu.mag_topic:
                self._subscribe_latest(
                    MagneticField, self._spec.imu.mag_topic, "mag")
        for arm_name, ft_cfg in (self._spec.force_torque or {}).items():
            self._subscribe_latest(
                WrenchStamped, ft_cfg.topic, f"wrench.{arm_name}")
        env = self._spec.environment
        if env is not None:
            for kind, msg_type in (("temperature", Temperature),
                                   ("pressure", FluidPressure),
                                   ("illuminance", Illuminance)):
                topic = getattr(env, kind)
                if topic:
                    self._subscribe_latest(msg_type, topic, f"env.{kind}")

    def _subscribe_latest(self, msg_type: Any, topic: str, key: str) -> None:
        def store(msg: Any, key: str = key) -> None:
            with self._lock:
                self._latest[key] = msg
                self._latest_received[key] = time.monotonic()
        self._node.create_subscription(msg_type, topic, store, _SENSOR_QOS)

    def _get_latest(self, key: str) -> Any | None:
        with self._lock:
            return self._latest.get(key)

    def signal_provenance(self, path):
        """Read under the value lock; header-less signals have unknown source age."""
        with self._lock:
            msg = received = None
            if path.startswith(('pose.', 'velocity.')) or path == 'distance_traveled':
                msg, received = self._odom, self._odom_received
            elif path.startswith('battery.'):
                msg, received = self._battery, self._battery_received
            elif path.startswith('arms.') and ('.effort.' in path or '.joint_positions.' in path):
                joint = path.rsplit('.', 1)[1]
                return {'source_stamp': self._joint_stamps.get(joint),
                        'received': self._joint_received.get(joint),
                        'source_now': self._node.get_clock().now().nanoseconds*1e-9,
                        'clock_domain': 'ros'}
            else:
                parts = path.split('.')
                key = ('range.'+parts[1] if parts[0]=='range' else
                       'imu' if parts[0]=='imu' else
                       'env.'+parts[1] if parts[0]=='environment' else
                       'wrench.'+parts[1] if parts[0]=='arms' and '.wrench.' in path else None)
                if key:
                    msg, received = self._latest.get(key), self._latest_received.get(key)
            header = getattr(msg, 'header', None)
            stamp = None if header is None else header.stamp.sec+header.stamp.nanosec*1e-9
            return {'source_stamp': stamp, 'received': received,
                    'source_now': self._node.get_clock().now().nanoseconds*1e-9,
                    'clock_domain': 'ros' if header is not None else None}

    # -- frame aliases ---------------------------------------------------------

    def resolve_frame(self, name: str) -> str:
        """Resolve "@map"/"@odom"/"@base" to the profile's frame names;
        anything else passes through unchanged."""
        aliases = {"@map": self._frames.map, "@odom": self._frames.odom,
                   "@base": self._frames.base}
        return aliases.get(name, name)

    # -- velocity commands -------------------------------------------------------

    def publish_twist(self, linear: float, angular: float,
                      lateral: float = 0.0) -> None:
        """Send one velocity command (m/s, rad/s). Caller clamps."""
        if self._cmd_vel_pub is None:
            raise TfLookupError("profile declares no cmd_vel topic")
        if self._cmd_vel_stamped:
            msg = TwistStampedMsg()
            msg.header.stamp = self._node.get_clock().now().to_msg()
            msg.header.frame_id = self._frames.base
            twist = msg.twist
        else:
            msg = TwistMsg()
            twist = msg
        twist.linear.x = float(linear)
        twist.linear.y = float(lateral)
        twist.angular.z = float(angular)
        self._cmd_vel_pub.publish(msg)

    def publish_zero(self) -> None:
        """Halt: zero velocity command."""
        self.publish_twist(0.0, 0.0, 0.0)

    # -- odometry state -----------------------------------------------------------

    def _on_odom(self, msg: Odometry) -> None:
        with self._lock:
            if self._odom is None:
                self._node.get_logger().info(
                    f"first odometry received on {self._odom_topic!r}")
            self._odom = msg
            self._odom_received = time.monotonic()
            xy = (msg.pose.pose.position.x, msg.pose.pose.position.y)
            if self._last_xy is not None:
                self._distance += math.hypot(xy[0] - self._last_xy[0],
                                             xy[1] - self._last_xy[1])
            self._last_xy = xy

    def mobility_stop_snapshot(self):
        """Atomic full 3D odometry evidence, with source and receipt provenance."""
        with self._lock:
            msg = self._odom
            if msg is None: raise ValueError("odometry unavailable")
            p, q, v = msg.pose.pose.position, msg.pose.pose.orientation, msg.twist.twist
            return {"position": (p.x, p.y, p.z), "orientation": (q.x, q.y, q.z, q.w),
                    "linear": (v.linear.x, v.linear.y, v.linear.z),
                    "angular": (v.angular.x, v.angular.y, v.angular.z),
                    "source_stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                    "received": self._odom_received,
                    "source_now": self._node.get_clock().now().nanoseconds * 1e-9,
                    "receive_now": time.monotonic()}

    def navigation_pose_snapshot(self, frame):
        # Use the measured odometry time, never an arbitrary latest TF sample.
        evidence = self.mobility_stop_snapshot()
        transform = self.tf_lookup(frame, self._frames.base, .5, evidence['source_stamp'])
        return {"position": tuple(transform['translation'][a] for a in ('x','y','z')),
                "orientation": tuple(transform['rotation'][a] for a in ('x','y','z','w')),
                "source_stamp": evidence['source_stamp'], "received": evidence['received'],
                "source_now": self._node.get_clock().now().nanoseconds * 1e-9,
                "receive_now": time.monotonic()}

    def get_state(self) -> dict[str, Any]:
        """Pose / velocity / is_moving / distance_traveled, wire-shaped.

        Pose prefers the map frame (when TF can localize base in map),
        falls back to the odom frame; ``pose.frame`` says which you got.
        Raises TfLookupError if no odometry has arrived yet.
        """
        with self._lock:
            odom = self._odom
            distance = self._distance
        if odom is None:
            raise TfLookupError("no odometry received yet")

        pose = self._best_pose()
        twist = odom.twist.twist
        speed = math.hypot(twist.linear.x, twist.linear.y)
        moving = speed > _MOVING_LINEAR or abs(twist.angular.z) > _MOVING_ANGULAR
        p = odom.pose.pose
        return {
            "stamp": _stamp(odom.header),
            "odometry_pose": {"x": p.position.x, "y": p.position.y, "z": p.position.z,
                              "q": {"x": p.orientation.x, "y": p.orientation.y,
                                    "z": p.orientation.z, "w": p.orientation.w},
                              "frame": odom.header.frame_id},
            "pose": pose,
            "velocity": {
                "linear": {"x": twist.linear.x, "y": twist.linear.y,
                           "z": twist.linear.z},
                "angular": {"x": twist.angular.x, "y": twist.angular.y,
                            "z": twist.angular.z},
            },
            "is_moving": moving,
            "distance_traveled": distance,
        }

    def _best_pose(self) -> dict[str, Any]:
        """base pose in map frame if available, else odom frame."""
        base = self._frames.base
        for frame in (self._frames.map, self._frames.odom):
            if self._tf_buffer.can_transform(frame, base, Time()):
                t = self._tf_buffer.lookup_transform(frame, base, Time())
                tr, rot = t.transform.translation, t.transform.rotation
                return {
                    "x": tr.x, "y": tr.y, "z": tr.z,
                    "q": {"x": rot.x, "y": rot.y, "z": rot.z, "w": rot.w},
                    "frame": frame,
                }
        raise TfLookupError(
            f"cannot locate {base!r} in {self._frames.map!r} or "
            f"{self._frames.odom!r} (TF not up yet?)"
        )

    # -- TF ---------------------------------------------------------------------

    def tf_lookup(self, parent: str, child: str,
                  timeout: float, stamp: float | None = None) -> dict[str, Any]:
        """Latest or timestamp-aligned transform child->parent coordinates, wire-shaped.
        Blocking up to ``timeout``; call via asyncio.to_thread."""
        parent = self.resolve_frame(parent)
        child = self.resolve_frame(child)
        try:
            t = self._tf_buffer.lookup_transform(
                parent, child, Time() if stamp is None else Time(seconds=stamp),
                timeout=Duration(seconds=timeout))
        except tf2_ros.TransformException as exc:
            raise TfLookupError(
                f"no transform {parent!r} <- {child!r}: {exc}") from exc
        tr, rot = t.transform.translation, t.transform.rotation
        return {
            "translation": {"x": tr.x, "y": tr.y, "z": tr.z},
            "rotation": {"x": rot.x, "y": rot.y, "z": rot.z, "w": rot.w},
            "parent": parent,
            "child": child,
            "stamp": _stamp(t.header),
        }

    def tf_frames(self) -> list[str]:
        """All frame names currently in the TF buffer."""
        return self._frame_names()

    def _frame_names(self) -> list[str]:
        data = yaml.safe_load(self._tf_buffer.all_frames_as_yaml()) or {}
        names = set(data.keys())
        for info in data.values():
            names.add(info.get("parent", ""))
        names.discard("")
        return sorted(names)

    def tf_wait_for(self, frame: str, timeout: float) -> None:
        """Block until ``frame`` exists in TF. Raises TfLookupError on
        timeout. Call via asyncio.to_thread."""
        frame = self.resolve_frame(frame)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if frame in self._frame_names():
                return
            time.sleep(0.1)
        raise TfLookupError(f"frame {frame!r} did not appear in {timeout}s")

    # -- cameras (Phase 4) -------------------------------------------------------

    def ensure_camera(self, name: str) -> None:
        """Create the camera's subscriptions on first use. Thread-safe;
        raises TfLookupError for a camera the profile does not declare."""
        cfg = (self._spec.cameras or {}).get(name)
        if cfg is None:
            raise TfLookupError(f"profile declares no camera {name!r}")
        with self._sub_lock:
            state = self._cameras.setdefault(name, _CameraState())
            if state.subscribed:
                return
            state.subscribed = True
            if cfg.rgb:
                self._node.create_subscription(
                    Image, cfg.rgb,
                    lambda msg, s=state: self._on_image(s.rgb, msg),
                    _SENSOR_QOS)
            if cfg.depth:
                self._node.create_subscription(
                    Image, cfg.depth,
                    lambda msg, s=state: self._on_image(s.depth, msg),
                    _SENSOR_QOS)
            if cfg.info:
                self._node.create_subscription(
                    CameraInfo, cfg.info,
                    lambda msg, s=state: setattr(s, "info", msg),
                    _SENSOR_QOS)

    def _on_image(self, buffer: deque, msg: Image) -> None:
        with self._lock:
            buffer.append((_stamp(msg.header), msg))

    @staticmethod
    def _newest(entries: list) -> tuple:
        """The entry with the largest stamp.

        NOT ``entries[-1]``: the buffers are filled by ROS callbacks in
        ARRIVAL order, and arrival order is not stamp order — a message can
        overtake an older one across the gz bridge under load. Indexing the
        tail then hands back a frame older than the one served a moment ago,
        which surfaced as a camera stream delivering out-of-order stamps
        during a full-suite run (2026-08-24). Buffers hold 30 entries, so
        scanning is free.
        """
        return max(entries, key=lambda entry: entry[0])

    def camera_sample(
        self,
        name: str,
        need_rgb: bool,
        need_depth: bool,
        sync_tolerance: float = 0.05,
        wait: float = 2.0,
    ) -> dict[str, Any]:
        """Latest camera data, decoded to numpy.

        Returns {stamp, rgb, depth, intrinsics, frame_id}; rgb/depth are
        None when not requested or not configured. When both are
        requested, returns the pair with the closest stamps (within
        ``sync_tolerance``); when only rgb is requested but the camera
        has depth, the closest depth within tolerance rides along for
        the deprojection record (None if nothing close exists).

        Blocks up to ``wait`` for the first messages after a fresh
        subscription; call via asyncio.to_thread.
        """
        self.ensure_camera(name)
        cfg = (self._spec.cameras or {})[name]
        state = self._cameras[name]
        want_rgb = need_rgb and bool(cfg.rgb)
        want_depth = need_depth and bool(cfg.depth)
        if need_rgb and not cfg.rgb:
            raise TfLookupError(f"camera {name!r} has no rgb topic")
        if need_depth and not cfg.depth:
            raise TfLookupError(
                f"camera {name!r} has no depth topic; deprojection and "
                "clouds need a depth-capable camera")

        deadline = time.monotonic() + wait
        while True:
            with self._lock:
                rgb = list(state.rgb)
                depth = list(state.depth)
            ready = ((not want_rgb or rgb)
                     and (not want_depth or depth))
            if ready:
                break
            if time.monotonic() > deadline:
                missing = []
                if want_rgb and not rgb:
                    missing.append(cfg.rgb)
                if want_depth and not depth:
                    missing.append(cfg.depth)
                raise TfLookupError(
                    f"no data from camera {name!r} on {missing} within "
                    f"{wait}s (topic remapped? sim paused?)")
            time.sleep(0.02)

        if want_rgb and (want_depth or (cfg.depth and depth)):
            # Per-camera override, else the caller's default. A slow or
            # frame-dropping sensor needs a looser window than a lockstep
            # simulator does.
            tol = getattr(cfg, "sync_tolerance", None) or sync_tolerance
            pair = _closest_pair(rgb, depth)
            if pair is None or abs(pair[0][0] - pair[1][0]) > tol:
                if want_depth:
                    # Say WHY. "no pair within 0.05s" alone cannot distinguish
                    # a stalled stream from a mis-stamped one from a genuinely
                    # unsynced pair, and that difference is the whole diagnosis.
                    newest_rgb = self._newest(rgb)[0] if rgb else 0.0
                    newest_depth = self._newest(depth)[0] if depth else 0.0
                    now = max(newest_rgb, newest_depth)
                    best = ("none" if pair is None
                            else f"{abs(pair[0][0] - pair[1][0]) * 1e3:.0f} ms")
                    rgb_age = (f"{(now - newest_rgb) * 1e3:.0f} ms"
                               if rgb else "no data")
                    dep_age = (f"{(now - newest_depth) * 1e3:.0f} ms"
                               if depth else "no data")
                    raise TfLookupError(
                        f"no rgb/depth pair within {tol * 1e3:.0f} ms for "
                        f"camera {name!r}: closest pair {best}, newest rgb is "
                        f"{rgb_age} old, newest depth is {dep_age} old "
                        f"(buffer holds {len(rgb)} rgb / {len(depth)} depth). "
                        f"A large age gap means that stream stalled — check "
                        f"sensor rate and CPU/GPU load; raise the camera's "
                        f"sync_tolerance if the sensors are genuinely unsynced."
                    )
                pair = (self._newest(rgb), None)
            rgb_entry, depth_entry = pair
        elif want_rgb:
            rgb_entry, depth_entry = self._newest(rgb), None
        else:
            rgb_entry, depth_entry = None, self._newest(depth)

        info = state.info
        intrinsics = _intrinsics_dict(info)
        frame_id = cfg.frame or (
            info.header.frame_id if info is not None else None)
        if frame_id is None:
            anchor = rgb_entry or depth_entry
            frame_id = anchor[1].header.frame_id
        return {
            "stamp": (rgb_entry or depth_entry)[0],
            "rgb": _image_to_numpy(rgb_entry[1]) if rgb_entry else None,
            "rgb_stamp": rgb_entry[0] if rgb_entry else None,
            "depth": _image_to_numpy(depth_entry[1]) if depth_entry else None,
            "depth_stamp": depth_entry[0] if depth_entry else None,
            "intrinsics": intrinsics,
            "frame_id": frame_id,
        }

    def camera_meta(self, name: str, wait: float = 2.0) -> dict[str, Any]:
        """Intrinsics + frame id without pulling an image. Blocks up to
        ``wait`` for camera_info after a fresh subscription (None fields
        if the camera has no info topic)."""
        self.ensure_camera(name)
        cfg = (self._spec.cameras or {})[name]
        state = self._cameras[name]
        info = None
        if cfg.info:
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline:
                info = state.info
                if info is not None:
                    break
                time.sleep(0.02)
        frame_id = cfg.frame or (
            info.header.frame_id if info is not None else None)
        return {"intrinsics": _intrinsics_dict(info), "frame_id": frame_id}

    def snapshot_transforms(self, camera_frame: str) -> dict[str, dict]:
        """Latest transforms target <- camera for map/odom/base, for the
        deprojection record. Frames TF cannot resolve right now are
        simply absent."""
        out: dict[str, dict] = {}
        for target in (self._frames.map, self._frames.odom,
                       self._frames.base):
            if target == camera_frame or target in out:
                continue
            if not self._tf_buffer.can_transform(
                    target, camera_frame, Time()):
                continue
            t = self._tf_buffer.lookup_transform(target, camera_frame, Time())
            tr, rot = t.transform.translation, t.transform.rotation
            out[target] = {
                "translation": {"x": tr.x, "y": tr.y, "z": tr.z},
                "rotation": {"x": rot.x, "y": rot.y, "z": rot.z, "w": rot.w},
                "parent": target,
                "child": camera_frame,
            }
        return out

    # -- lidar (Phase 4) -----------------------------------------------------------

    def ensure_lidar(self) -> None:
        if self._spec.lidar is None:
            raise TfLookupError("profile declares no lidar")
        with self._sub_lock:
            if getattr(self, "_lidar_subscribed", False):
                return
            self._lidar_subscribed = True
            self._node.create_subscription(
                LaserScanMsg, self._spec.lidar.scan,
                self._on_scan, _SENSOR_QOS)

    def _on_scan(self, msg: LaserScanMsg) -> None:
        with self._lock:
            self._scan = msg

    def lidar_sample(self, wait: float = 2.0) -> dict[str, Any]:
        """Latest scan as {stamp, frame, angle_*, ranges (np.float32)}.
        Blocks up to ``wait`` after a fresh subscription; call via
        asyncio.to_thread."""
        self.ensure_lidar()
        deadline = time.monotonic() + wait
        while True:
            with self._lock:
                msg = self._scan
            if msg is not None:
                break
            if time.monotonic() > deadline:
                raise TfLookupError(
                    f"no scan on {self._spec.lidar.scan!r} within {wait}s")
            time.sleep(0.02)
        intensities = np.asarray(msg.intensities, dtype=np.float32)
        return {
            "stamp": _stamp(msg.header),
            "frame": msg.header.frame_id,
            "angle_min": msg.angle_min,
            "angle_max": msg.angle_max,
            "angle_increment": msg.angle_increment,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
            "ranges": np.asarray(msg.ranges, dtype=np.float32),
            "intensities": intensities if intensities.size else None,
        }

    # -- joint states / battery (Phase 4) ----------------------------------------

    def _on_joint_state(self, msg: JointState) -> None:
        with self._lock:
            from .feedback import update_joint_feedback
            update_joint_feedback(self._joints, self._joint_stamps, msg, _stamp(msg.header))
            for name in msg.name[:len(msg.position)]:
                self._joint_received[name] = time.monotonic()
            stamps = set(self._joint_stamps.values())
            if len(stamps) == 1:
                stamp = next(iter(stamps))
                if self._joint_history and stamp < self._joint_history[-1][0]:
                    self._joint_history.clear()  # simulation/ROS clock reset
                self._joint_history.append((stamp, dict(self._joints)))

    def joint_snapshot(self) -> dict[str, tuple[float, float, float]]:
        """{joint: (position, velocity, effort)} from the last message."""
        with self._lock:
            return dict(self._joints)

    def verified_joint_snapshot(self, names, max_age):
        """Required positions with independently bounded ROS and receive ages."""
        from .feedback import check_freshness
        with self._lock:
            check_freshness(names, self._joint_stamps, self._joint_received,
                            self._node.get_clock().now().nanoseconds / 1e9,
                            time.monotonic(), max_age)
            return dict(self._joints)

    def stop_snapshot(self, names):
        """Position/velocity evidence and provenance from the same locked cache."""
        with self._lock:
            return {"positions": {n: self._joints[n][0] for n in names if n in self._joints},
                    "velocities": {n: self._joints[n][1] for n in names if n in self._joints},
                    "source_stamps": {n: self._joint_stamps[n] for n in names if n in self._joint_stamps},
                    "received": {n: self._joint_received[n] for n in names if n in self._joint_received},
                    "source_now": self._node.get_clock().now().nanoseconds / 1e9,
                    "receive_now": time.monotonic()}

    def joint_snapshot_with_stamp(self, stamp=None):
        """Positions and their common ROS timestamp; None for mixed samples."""
        if stamp is not None:
            from .feedback import snapshot_at
            deadline = time.monotonic()+.5
            while True:
                try:
                    with self._lock:
                        return snapshot_at(self._joint_history, stamp)
                except LookupError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(.005)  # subscriber may trail TF by one callback
        with self._lock:
            stamps = set(self._joint_stamps.values())
            stamp = next(iter(stamps)) if len(stamps) == 1 else None
            return dict(self._joints), stamp

    def joint_type(self, name):
        """URDF measurement units; group labels do not establish joint type."""
        with self._lock:
            description = self._description
        if not description:
            return None
        for joint in ET.fromstring(description).findall("joint"):
            if joint.get("name") == name:
                return joint.get("type")
        return None

    def joint_limits(self) -> dict[str, tuple[float, float] | None]:
        """{joint: (lower, upper) | None} parsed from /robot_description.
        Empty until the latched description arrives."""
        with self._lock:
            description = self._description
            if self._joint_limits is not None:
                return self._joint_limits
        if description is None:
            return {}
        limits: dict[str, tuple[float, float] | None] = {}
        try:
            root = ET.fromstring(description)
        except ET.ParseError:
            return {}
        # findall, not iter: <ros2_control> blocks also contain <joint>
        # elements (that is where base101's tool joint arm_6 lives, and
        # ONLY there - see found_along_the_way.md item 7). They carry no
        # type= and were being skipped by accident; being explicit means
        # the next tool joint that grows a type attribute cannot leak in.
        for joint in root.findall("joint"):
            name = joint.get("name")
            kind = joint.get("type")
            if name is None or kind in (None, "fixed"):
                continue
            limit = joint.find("limit")
            if kind == "continuous" or limit is None:
                limits[name] = None
                continue
            lower, upper = limit.get("lower"), limit.get("upper")
            if lower is None or upper is None:
                limits[name] = None
            else:
                limits[name] = (float(lower), float(upper))
        with self._lock:
            self._joint_limits = limits
        return limits

    def _on_description(self, msg: String) -> None:
        with self._lock:
            self._description = msg.data
            self._description_sha = sha_of(msg.data)
            self._joint_limits = None  # reparse lazily

    def _on_group_description(self, topic: str, msg: String) -> None:
        with self._lock:
            self._group_descriptions[topic] = msg.data

    def group_description(self, group: str) -> str | None:
        """The description for one group, when its profile overrides the
        topic. None when it declares no override or nothing arrived yet;
        callers fall back to the composed description."""
        topic = self._group_topics.get(group)
        if topic is None:
            return None
        with self._lock:
            return self._group_descriptions.get(topic)

    def group_description_topics(self) -> dict[str, str]:
        """{group: topic} for groups that override the description topic."""
        return dict(self._group_topics)

    def description_sha(self) -> str | None:
        """sha256 of the current description, or None before it arrives.

        Cheap enough to poll: computed once per received message, not per
        call. The model watcher compares it to spot a robot_state_publisher
        that restarted with a different xacro.
        """
        with self._lock:
            return self._description_sha

    def _on_battery(self, msg: BatteryState) -> None:
        with self._lock:
            self._battery = msg
            self._battery_received = time.monotonic()

    # sensor_msgs/BatteryState.power_supply_status values.
    _POWER_STATUS = {1: "charging", 2: "discharging", 3: "not_charging",
                     4: "full"}

    def battery_state(self) -> dict[str, Any] | None:
        """Wire-shaped BatteryState dict, or None before data.

        sensor_msgs/BatteryState.percentage is 0..1 per the message
        definition, but real drivers (and `ros2 topic pub` users)
        routinely publish 0..100; values above 1.5 are passed through
        as-is, smaller ones are scaled.
        """
        with self._lock:
            msg = self._battery
        if msg is None:
            return None
        level: float | None = float(msg.percentage)
        if math.isnan(level):
            level = None
        elif level <= 1.5:
            level *= 100.0
        return {
            "level": level,
            "voltage": _none_if_nan(msg.voltage),
            "current": _none_if_nan(msg.current),
            "temperature": _none_if_nan(msg.temperature),
            "is_charging": msg.power_supply_status == 1,
            "power_supply_status": self._POWER_STATUS.get(
                msg.power_supply_status, "unknown"),
            "stamp": _stamp(msg.header),
        }

    # -- latest-value sensor family -----------------------------------------------

    def range_reading(self, name: str) -> dict[str, Any] | None:
        msg = self._get_latest(f"range.{name}")
        if msg is None:
            return None
        return {
            "range": float(msg.range),
            "min_range": float(msg.min_range),
            "max_range": float(msg.max_range),
            "field_of_view": float(msg.field_of_view),
            "radiation_type": "infrared" if msg.radiation_type == 1
            else "ultrasonic",
            "stamp": _stamp(msg.header),
            "frame": msg.header.frame_id,
        }

    def imu_reading(self) -> dict[str, Any] | None:
        msg = self._get_latest("imu")
        if msg is None:
            return None
        o, av, la = msg.orientation, msg.angular_velocity, \
            msg.linear_acceleration
        return {
            "orientation": (
                None if msg.orientation_covariance[0] == -1.0
                else {"x": o.x, "y": o.y, "z": o.z, "w": o.w}),
            "angular_velocity": {"x": av.x, "y": av.y, "z": av.z},
            "linear_acceleration": {"x": la.x, "y": la.y, "z": la.z},
            "orientation_covariance": _covariance(
                msg.orientation_covariance),
            "angular_velocity_covariance": _covariance(
                msg.angular_velocity_covariance),
            "linear_acceleration_covariance": _covariance(
                msg.linear_acceleration_covariance),
            "stamp": _stamp(msg.header),
            "frame": msg.header.frame_id,
        }

    def mag_reading(self) -> dict[str, Any] | None:
        msg = self._get_latest("mag")
        if msg is None:
            return None
        f = msg.magnetic_field
        return {
            "magnetic_field": {"x": f.x, "y": f.y, "z": f.z},
            "covariance": _covariance(msg.magnetic_field_covariance),
            "stamp": _stamp(msg.header),
            "frame": msg.header.frame_id,
        }

    def wrench_reading(self, arm: str) -> dict[str, Any] | None:
        msg = self._get_latest(f"wrench.{arm}")
        if msg is None:
            return None
        f, t = msg.wrench.force, msg.wrench.torque
        return {
            "force": {"x": f.x, "y": f.y, "z": f.z},
            "torque": {"x": t.x, "y": t.y, "z": t.z},
            "stamp": _stamp(msg.header),
            "frame": msg.header.frame_id,
        }

    def environment_reading(self, kind: str) -> dict[str, Any] | None:
        msg = self._get_latest(f"env.{kind}")
        if msg is None:
            return None
        value_field = {"temperature": "temperature",
                       "pressure": "fluid_pressure",
                       "illuminance": "illuminance"}[kind]
        return {
            kind if kind != "pressure" else "pressure":
                float(getattr(msg, value_field)),
            "variance": float(msg.variance),
            "stamp": _stamp(msg.header),
            "frame": msg.header.frame_id,
        }

    def _on_diagnostics(self, msg: DiagnosticArray) -> None:
        with self._lock:
            self._latest["diagnostics"] = msg

    _DIAG_LEVELS = {0: "ok", 1: "warn", 2: "error", 3: "stale"}

    def diagnostics_report(self) -> dict[str, Any] | None:
        msg = self._get_latest("diagnostics")
        if msg is None:
            return None
        return {
            "stamp": _stamp(msg.header),
            "subsystems": [{
                "name": s.name,
                "level": self._DIAG_LEVELS.get(
                    int.from_bytes(s.level, "little")
                    if isinstance(s.level, bytes) else int(s.level),
                    "stale"),
                "message": s.message,
                "values": {kv.key: kv.value for kv in s.values},
            } for s in msg.status],
        }


    # -- navigation (Phase 5) ------------------------------------------------------

    def _pose_stamped(self, pose: dict[str, Any]) -> PoseStamped:
        """Wire Pose dict -> PoseStamped, resolving frame aliases."""
        msg = PoseStamped()
        msg.header.frame_id = self.resolve_frame(
            str(pose.get("frame", "@map")))
        # stamp stays zero = "latest": a now() stamp can run ahead of
        # bt_navigator's TF buffer and abort the goal on arrival.
        msg.pose.position.x = float(pose.get("x", 0.0))
        msg.pose.position.y = float(pose.get("y", 0.0))
        msg.pose.position.z = float(pose.get("z", 0.0))
        q = pose.get("q") or {}
        msg.pose.orientation.x = float(q.get("x", 0.0))
        msg.pose.orientation.y = float(q.get("y", 0.0))
        msg.pose.orientation.z = float(q.get("z", 0.0))
        msg.pose.orientation.w = float(q.get("w", 1.0))
        return msg

    def _on_map(self, msg: OccupancyGridMsg) -> None:
        with self._lock:
            self._map_msg = msg

    def map_grid(self, wait: float = 5.0) -> dict[str, Any]:
        """Latest occupancy grid as {width, height, resolution, origin,
        stamp, data (H,W) int8}. Blocks up to ``wait`` for the first map;
        call via asyncio.to_thread."""
        deadline = time.monotonic() + wait
        while True:
            with self._lock:
                msg = self._map_msg
            if msg is not None:
                break
            if time.monotonic() > deadline:
                raise TfLookupError(
                    f"no map on /map within {wait}s (SLAM running?)")
            time.sleep(0.05)
        info = msg.info
        o = info.origin
        data = np.asarray(msg.data, dtype=np.int8).reshape(
            info.height, info.width)
        return {
            "width": int(info.width),
            "height": int(info.height),
            "resolution": float(info.resolution),
            "origin": {
                "x": o.position.x, "y": o.position.y, "z": o.position.z,
                "q": {"x": o.orientation.x, "y": o.orientation.y,
                      "z": o.orientation.z, "w": o.orientation.w},
                "frame": msg.header.frame_id or self._frames.map,
            },
            "stamp": _stamp(msg.header),
            "data": data,
        }

    def nav_available(self, timeout: float = 2.0) -> bool:
        """True when the NavigateToPose action server is up."""
        if self._nav_client is None:
            return False
        return self._nav_client.wait_for_server(timeout_sec=timeout)

    def nav_start(self, pose: dict[str, Any], timeout: float = 5.0) -> None:
        """Send one NavigateToPose goal; blocks until accepted.

        Raises TfLookupError if the server is missing or rejects the
        goal. One active goal at a time (motion is single-writer).
        Call via asyncio.to_thread.
        """
        if self._nav_client is None:
            raise TfLookupError(
                "profile declares no nav2_navigate_to_pose action")
        if not self._nav_client.wait_for_server(timeout_sec=timeout):
            raise TfLookupError(
                "NavigateToPose action server not available "
                "(is the nav stack running?)")
        goal = NavigateToPose.Goal()
        goal.pose = self._pose_stamped(pose)
        with self._lock:
            self._nav_feedback = None
            self._nav_result_future = None
            self._nav_goal_handle = None

        # Captured BEFORE the send. Sending is a round-trip to
        # bt_navigator, and for its whole duration `_nav_goal_handle` is
        # None — so a nav_cancel arriving in that window would find nothing
        # to cancel and return silently while Nav2 went on to accept the
        # goal and drive away. The asyncio task cancel does not save us
        # either: this runs inside asyncio.to_thread, which cannot be
        # interrupted. So we notice after the fact instead.
        with self._lock:
            epoch = self._nav_stop_epoch

        from .action_submission import ActionSubmission
        guard = getattr(self, "check_motion_dispatch", None)
        if guard is not None: guard()
        send_future = self._nav_client.send_goal_async(goal, feedback_callback=self._on_nav_feedback)
        def on_accept(handle, result):
            with self._lock:
                self._nav_goal_handle = handle
                self._nav_result_future = result
        submission = ActionSubmission(send_future, on_accept)
        with self._lock: self._nav_submission = submission
        submission.wait(timeout)
        with self._lock: superseded = self._nav_stop_epoch != epoch
        if superseded: submission.stop()

    def _on_nav_feedback(self, msg: Any) -> None:
        fb = msg.feedback
        p = fb.current_pose.pose
        with self._lock:
            self._nav_feedback = {
                "distance_remaining": float(fb.distance_remaining),
                "navigation_time": fb.navigation_time.sec
                + fb.navigation_time.nanosec * 1e-9,
                "estimated_time_remaining": fb.estimated_time_remaining.sec
                + fb.estimated_time_remaining.nanosec * 1e-9,
                "recoveries": int(fb.number_of_recoveries),
                "pose": {
                    "x": p.position.x, "y": p.position.y, "z": p.position.z,
                    "q": {"x": p.orientation.x, "y": p.orientation.y,
                          "z": p.orientation.z, "w": p.orientation.w},
                    "frame": fb.current_pose.header.frame_id
                    or self._frames.map,
                },
            }

    def nav_feedback(self) -> dict[str, Any] | None:
        with self._lock:
            return self._nav_feedback

    def nav_wait(self, poll: float = 0.1) -> dict[str, Any] | None:
        """One bounded wait step for the active goal's result.

        Returns None while still running (call again), else
        {status: succeeded|canceled|aborted, error_code, error_msg}.
        Call via asyncio.to_thread in a loop so task cancellation can
        interleave.
        """
        with self._lock:
            future = self._nav_result_future
        if future is None:
            raise TfLookupError("no active navigation goal")
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if future.done():
            done.set()
        if not done.wait(poll):
            return None
        wrapped = future.result()
        status = {
            GoalStatus.STATUS_SUCCEEDED: "succeeded",
            GoalStatus.STATUS_CANCELED: "canceled",
            GoalStatus.STATUS_ABORTED: "aborted",
        }.get(wrapped.status, "aborted")
        result = wrapped.result
        with self._lock:
            self._nav_goal_handle = None
            self._nav_result_future = None
        return {
            "status": status,
            "error_code": int(getattr(result, "error_code", 0)),
            "error_msg": str(getattr(result, "error_msg", "")),
        }

    def nav_cancel(self, timeout: float = 2.0) -> None:
        """Cancel the active/pending goal and require terminal acknowledgement.

        Also valid *before* there is a goal to cancel: the epoch bump makes
        an in-flight send cancel itself the moment it is accepted, which is
        the only way to stop a robot whose goal has not been acknowledged
        yet.
        """
        with self._lock:
            self._nav_stop_epoch += 1
            submission = self._nav_submission
        if submission is not None: submission.stop(timeout)

    def clear_costmaps(self, timeout: float = 2.0) -> int:
        """Clear global+local costmaps (housekeeping before each goal).
        Returns how many cleared; missing services are skipped quietly —
        the nav stack may simply not be up yet."""
        cleared = 0
        for client in self._costmap_clients:
            if not client.wait_for_service(timeout_sec=timeout):
                continue
            if self._call_service(client, ClearEntireCostmap.Request(),
                                  timeout) is not None:
                cleared += 1
        return cleared

    # -- SLAM (Phase 5) -------------------------------------------------------------

    def slam_serialize(self, filename: str, timeout: float = 10.0) -> None:
        """Serialize the pose graph to ``filename`` (slam_toolbox adds
        .data/.posegraph). Raises TfLookupError on failure."""
        request = SerializePoseGraph.Request()
        request.filename = filename
        response = self._slam_call("serialize", request, timeout)
        if response.result != 0:
            raise TfLookupError(
                f"slam_toolbox failed to serialize to {filename!r} "
                f"(result {response.result})")

    def slam_deserialize(self, filename: str, match_type: int,
                         x: float = 0.0, y: float = 0.0, theta: float = 0.0,
                         timeout: float = 20.0) -> None:
        """Load a pose graph. match_type: 1 = continue mapping from the
        first node, 3 = localization at the given pose."""
        request = DeserializePoseGraph.Request()
        request.filename = filename
        request.match_type = match_type
        request.initial_pose = Pose2D(x=float(x), y=float(y),
                                      theta=float(theta))
        self._slam_call("deserialize", request, timeout)

    def slam_pause_toggle(self, timeout: float = 5.0) -> bool:
        """Toggle slam_toolbox's measurement processing; returns the NEW
        paused state (slam_toolbox's Pause service is a toggle)."""
        response = self._slam_call("pause", Pause.Request(), timeout)
        return bool(response.status)

    def slam_reset(self, pause: bool = False, timeout: float = 10.0) -> None:
        """Clear the pose graph and start a fresh map."""
        request = Reset.Request()
        request.pause_new_measurements = pause
        self._slam_call("reset", request, timeout)

    def _slam_call(self, key: str, request: Any, timeout: float) -> Any:
        client = self._slam_clients.get(key)
        if client is None:
            raise TfLookupError("profile declares no slam capability")
        if not client.wait_for_service(timeout_sec=min(timeout, 5.0)):
            raise TfLookupError(
                f"slam_toolbox service {client.srv_name!r} not available "
                "(is the slam stack running?)")
        response = self._call_service(client, request, timeout)
        if response is None:
            raise TfLookupError(
                f"slam_toolbox service {client.srv_name!r} timed out")
        return response

    @staticmethod
    def _call_service(client: Any, request: Any,
                      timeout: float) -> Any | None:
        """Blocking service call from a worker thread (futures resolve
        on the executor thread). None on timeout."""
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout):
            return None
        return future.result()

    # -- controller_manager introspection -----------------------------------

    def cancel_all_goals(self, action_name, timeout=2.0):
        """Operator recovery only: cancel orphan goals from an earlier process."""
        from action_msgs.srv import CancelGoal
        client = self._node.create_client(CancelGoal, action_name.rstrip('/')+'/_action/cancel_goal')
        try:
            if not client.wait_for_service(timeout_sec=timeout):
                raise TfLookupError(f'orphan cancellation service unavailable: {action_name}')
            response = self._call_service(client, CancelGoal.Request(), timeout)
            if response is None or response.return_code != 0:
                raise TfLookupError(f'orphan cancellation was not acknowledged: {action_name}')
        finally:
            self._node.destroy_client(client)

    def loaded_controllers(
        self, timeout: float = 5.0,
    ) -> dict[str, dict[str, Any]]:
        """{name: {"state", "type", "joints"}} for every LOADED controller.

        ``joints`` is read off ``claimed_interfaces`` (``joint/position``,
        ...) with the order ros2_control reports, deduplicated. That order
        is the command vector's order, which is the only reason to prefer
        it over the profile.

        Raises TfLookupError when the controller_manager is not up - a
        robot whose controllers cannot be listed is one whose profile
        cannot be validated, and guessing is worse than failing.
        """
        if not self._cm_list.wait_for_service(timeout_sec=min(timeout, 5.0)):
            raise TfLookupError(
                "/controller_manager/list_controllers not available "
                "(is ros2_control running?)")
        response = self._call_service(
            self._cm_list, ListControllers.Request(), timeout)
        if response is None:
            raise TfLookupError(
                "/controller_manager/list_controllers timed out")
        out: dict[str, dict[str, Any]] = {}
        for controller in response.controller:
            joints: dict[str, None] = {}
            for interface in controller.claimed_interfaces:
                joints[interface.split("/")[0]] = None
            out[controller.name] = {
                "state": controller.state,
                "type": controller.type,
                "joints": tuple(joints),
            }
        return out

    def declared_controllers(self, timeout: float = 5.0) -> frozenset[str]:
        """Controller names the controller_manager knows a type for.

        A superset of the loaded ones. base101 spawns only the position
        controller of each exclusive pair and leaves the trajectory twin
        declared-but-unloaded (6.0b), so this - not the loaded list - is
        what says whether ``load_controller`` can succeed later. Derived
        from the ``<name>.type`` parameters, which is where a declared
        controller lives before it is loaded.
        """
        if not self._cm_params.wait_for_service(timeout_sec=min(timeout, 5.0)):
            raise TfLookupError(
                "/controller_manager/list_parameters not available "
                "(is ros2_control running?)")
        request = ListParameters.Request()
        request.depth = 0          # 0 = unlimited, all nested names
        response = self._call_service(self._cm_params, request, timeout)
        if response is None:
            raise TfLookupError(
                "/controller_manager/list_parameters timed out")
        return frozenset(
            name[: -len(".type")]
            for name in response.result.names
            if name.endswith(".type")
        )

    # -- controller lifecycle (6.4) ------------------------------------------

    def load_controller(self, name: str, timeout: float = 10.0) -> None:
        """Bring a declared-but-unloaded controller up to `unconfigured`."""
        self._cm_call(self._cm_load, LoadController.Request(name=name),
                      timeout, f"load {name}")

    def configure_controller(self, name: str, timeout: float = 10.0) -> None:
        """`unconfigured` -> `inactive`, so it can be switched to."""
        self._cm_call(self._cm_configure,
                      ConfigureController.Request(name=name),
                      timeout, f"configure {name}")

    def unload_controller(self, name: str, timeout: float = 10.0) -> None:
        """Remove a controller entirely. The last half of teardown."""
        self._cm_call(self._cm_unload, UnloadController.Request(name=name),
                      timeout, f"unload {name}")

    def switch_controllers(self, activate: list[str], deactivate: list[str],
                           timeout: float = 10.0) -> None:
        """One switch: deactivate and activate in a single call.

        STRICT on purpose. BEST_EFFORT would report success having done
        only half the switch, which is exactly the state that leaves an
        arm claimed by nobody. An exclusive pair claims the same joints,
        so the deactivate must land for the activate to be possible —
        doing both in one call is what keeps the joints claimed
        throughout, and on position-interface hardware the arm holds
        through it.
        """
        request = SwitchController.Request(
            activate_controllers=list(activate),
            deactivate_controllers=list(deactivate),
            strictness=SwitchController.Request.STRICT,
            activate_asap=True,
        )
        request.timeout = Duration(seconds=timeout).to_msg()
        if not self._cm_switch.wait_for_service(timeout_sec=min(timeout, 5.0)):
            raise TfLookupError(
                "/controller_manager/switch_controller not available")
        response = self._call_service(self._cm_switch, request, timeout + 2.0)
        if response is None:
            raise TfLookupError(
                f"switch_controller timed out (activate={activate}, "
                f"deactivate={deactivate})")
        if not response.ok:
            raise TfLookupError(
                f"switch_controller refused: activate={activate}, "
                f"deactivate={deactivate}. Usually means a controller is "
                f"not loaded, or another controller already claims those "
                f"joints.")

    def _cm_call(self, client: Any, request: Any, timeout: float,
                 what: str) -> None:
        if not client.wait_for_service(timeout_sec=min(timeout, 5.0)):
            raise TfLookupError(
                f"{client.srv_name} not available (is ros2_control up?)")
        response = self._call_service(client, request, timeout)
        if response is None:
            raise TfLookupError(f"{what} timed out")
        if not response.ok:
            raise TfLookupError(f"{what} failed")

    # -- trajectory execution -------------------------------------------------

    def _traj_client(self, controller: str) -> Any:
        client = self._traj_clients.get(controller)
        if client is None:
            client = ActionClient(
                self._node, FollowJointTrajectory,
                f"/{controller}/follow_joint_trajectory")
            self._traj_clients[controller] = client
        return client

    def follow_trajectory(self, controller: str, joints: list[str],
                          points: list[tuple[list[float], float]],
                          timeout: float = 5.0) -> None:
        """Send a JointTrajectory goal. ``points`` is [(positions, t), ...].

        Returns as soon as the goal is accepted; poll `traj_wait`. The SDK
        generates nothing — JointTrajectoryController interpolates between
        the waypoints, which is the whole reason `follow` takes waypoints
        and not a dense path (spec 8.2).
        """
        client = self._traj_client(controller)
        if not client.wait_for_server(timeout_sec=timeout):
            raise TfLookupError(
                f"/{controller}/follow_joint_trajectory not available. The "
                f"controller must be ACTIVE before its action server "
                f"appears.")
        trajectory = JointTrajectory()
        trajectory.joint_names = list(joints)
        for positions, when in points:
            point = JointTrajectoryPoint()
            point.positions = [float(v) for v in positions]
            point.velocities = [0.0] * len(positions)
            point.time_from_start = Duration(seconds=float(when)).to_msg()
            trajectory.points.append(point)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        from .action_submission import ActionSubmission
        self._traj_result = None
        self._traj_feedback = None
        guard = getattr(self, "check_motion_dispatch", None)
        if guard is not None: guard()
        send = client.send_goal_async(goal, feedback_callback=self._on_traj_feedback)
        def on_accept(handle, result):
            with self._lock: self._traj_goal = handle
            result.add_done_callback(self._on_traj_result)
        submission = ActionSubmission(send, on_accept)
        with self._lock: self._traj_submission = submission
        submission.wait(timeout)

    def _on_traj_feedback(self, message: Any) -> None:
        with self._lock:
            self._traj_feedback = message.feedback

    def _on_traj_result(self, future: Any) -> None:
        try:
            result = future.result()
        except Exception as exc:              # pragma: no cover
            with self._lock:
                self._traj_result = {"status": "error", "message": str(exc)}
            return
        wrapped = result.result
        with self._lock:
            self._traj_result = {
                "status": ("succeeded"
                           if result.status == GoalStatus.STATUS_SUCCEEDED
                           else "aborted"),
                "error_code": int(getattr(wrapped, "error_code", 0)),
                "message": str(getattr(wrapped, "error_string", "")),
            }

    def traj_wait(self, poll: float = 0.1) -> dict[str, Any] | None:
        """The terminal result, or None while still running."""
        time.sleep(poll)
        with self._lock:
            return self._traj_result

    def traj_feedback(self) -> dict[str, Any] | None:
        """{joint: (desired, actual, error)} from the controller, or None."""
        with self._lock:
            feedback = self._traj_feedback
        if feedback is None:
            return None
        names = list(feedback.joint_names)
        return {
            "joints": names,
            "desired": list(feedback.desired.positions),
            "actual": list(feedback.actual.positions),
            "error": list(feedback.error.positions),
        }

    def traj_cancel(self, timeout: float = 2.0) -> None:
        """Cancel the pending/running goal and require a terminal result. Idempotent."""
        with self._lock: submission = self._traj_submission
        if submission is not None:
            submission.stop(timeout)
            with self._lock:
                if self._traj_submission is submission: self._traj_goal = None

    def command_positions(self, controller: str, joints: list[str],
                          positions: list[float]) -> None:
        """One Float64MultiArray to a JointGroupPositionController.

        The STREAM role's write path, and what `hold()` uses on a position
        controller — there is no action to send a single point to.
        """
        publisher = self._position_publishers.get(controller)
        if publisher is None:
            publisher = self._node.create_publisher(
                Float64MultiArray, f"/{controller}/commands", 10)
            self._position_publishers[controller] = publisher
        message = Float64MultiArray()
        message.data = [float(v) for v in positions]
        publisher.publish(message)

    # -- vendor escape hatches -------------------------------------------------

    def vendor_read(self, group: str, name: str) -> dict[str, Any] | None:
        """{value, stamp, units} for one declared signal, or None."""
        return self._vendor.read(group, name)

    def vendor_value(self, group: str, name: str) -> float | None:
        """Just the number — what the watch sampler polls."""
        return self._vendor.value(group, name)

    def vendor_snapshot(self) -> dict[str, dict[str, Any]]:
        return self._vendor.snapshot()

    def vendor_unavailable(self) -> dict[str, str]:
        """Signals whose ROS type could not be resolved, and why."""
        return self._vendor.unavailable

    def vendor_ready(self, action: str) -> bool:
        return action in self._vendor_publishers

    def vendor_publish(self, action: str, payload: dict[str, Any]) -> None:
        """Build and send one declared action message."""
        message = self._vendor_types[action]()
        if payload:
            from rosidl_runtime_py.set_message import set_message_fields

            set_message_fields(message, payload)
        self._vendor_publishers[action].publish(message)

    def description(self, wait: float = 10.0) -> str | None:
        """The latched /robot_description, waiting up to ``wait`` for it.

        Latched publishers deliver almost immediately once the
        subscription matches, but "almost" is not "before the agent
        finishes booting" - startup validation has to wait rather than
        conclude the robot has no description.
        """
        deadline = time.monotonic() + wait
        while True:
            with self._lock:
                description = self._description
            if description is not None or time.monotonic() >= deadline:
                return description
            time.sleep(0.1)

    def map_odom_correction(self) -> tuple[float, float, float] | None:
        """Current map->odom transform as (x, y, yaw), or None when the
        TF is not available (not localized / SLAM down). Used by the
        localization-quality monitor: in a well-localized run this drifts
        smoothly; jumps mean relocalization events."""
        try:
            t = self._tf_buffer.lookup_transform(
                self._frames.map, self._frames.odom, Time())
        except tf2_ros.TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                         1 - 2 * (q.y * q.y + q.z * q.z))
        tr = t.transform.translation
        return (tr.x, tr.y, yaw)


def _stamp(header: Any) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _none_if_nan(value: float) -> float | None:
    value = float(value)
    return None if math.isnan(value) else value


def _covariance(cov: Any) -> list[float] | None:
    """ROS covariance convention: -1 in element [0] means unknown."""
    cov = list(cov)
    return None if not cov or cov[0] == -1.0 else cov


def _intrinsics_dict(info: CameraInfo | None) -> dict[str, Any] | None:
    if info is None:
        return None
    return {
        "fx": info.k[0], "fy": info.k[4],
        "cx": info.k[2], "cy": info.k[5],
        "width": info.width, "height": info.height,
        "distortion_model": info.distortion_model or "plumb_bob",
        "distortion_coeffs": list(info.d),
    }


def _closest_pair(
    rgb: list, depth: list
) -> tuple[tuple, tuple] | None:
    """The (rgb, depth) entry pair with minimal stamp distance."""
    if not rgb or not depth:
        return None
    best = None
    for r in rgb:
        for d in depth:
            delta = abs(r[0] - d[0])
            if best is None or delta < best[0]:
                best = (delta, r, d)
    return (best[1], best[2])


def _image_to_numpy(msg: Image) -> np.ndarray:
    """Decode a sensor_msgs/Image without cv_bridge.

    Supports what cameras actually publish here: rgb8/bgr8 color,
    32FC1 float meters and 16UC1 millimeter depth.
    """
    h, w, step = msg.height, msg.width, msg.step
    enc = msg.encoding.lower()
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        arr = data.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)
    if enc == "32fc1":
        arr = data.reshape(h, step).view(np.float32)[:, :w]
        return np.ascontiguousarray(arr)
    if enc in ("16uc1", "mono16"):
        arr = data.reshape(h, step).view(np.uint16)[:, :w]
        return np.ascontiguousarray(arr.astype(np.float32) / 1000.0)
    raise TfLookupError(f"unsupported image encoding {msg.encoding!r}")
