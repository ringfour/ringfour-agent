"""Wire method handlers.

Each handler is ``async def name(session, params) -> result``. Handlers
signal client-visible failures by raising server.RpcError with the name of
a ringfour exception class. New methods register in build_registry and in
ringfour/scripts/gen_protocol.py so protocol.json stays truthful.

Capability-dependent methods are only registered when the profile declares
the capability, so a chassis-only robot answers method-not-found for arm
calls (the client SDK raises CapabilityNotSupported before the wire).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import time
import uuid
from typing import Any

from pydantic import ValidationError

from ringfour.description import sha_of
from ringfour.enums import AuditKind, Controller, Outcome
from ringfour.models import (
    CameraIntrinsics,
    GroupInfo,
    Hello,
    Point,
    Pose,
    ToolInfo,
    Transform,
    Welcome,
)
from ringfour.version import PROTOCOL_VERSION

from . import sensing
from .context import AgentContext
from .control import checked_arrival_residual, arrival_failure_message, worst_joint
from .server import Handler, RpcError, Session
from .state_paths import validate_path
from .tasks import TaskHandle

# debug.send_payload is test plumbing, not robot data; keep it small.
_MAX_DEBUG_PAYLOAD = 16 * 1024 * 1024

# Server-side cap on lidar stream rates.
_MAX_LIDAR_HZ = 20.0


def _require_ros(ctx: AgentContext) -> Any:
    """Handlers that touch the robot need the ROS interface."""
    if ctx.ros is None:
        raise RpcError("RingfourError",
                       "agent is running without a ROS interface (test mode)")
    return ctx.ros


def _require_teleop(ctx: AgentContext) -> Any:
    """Teleop handlers need the manager (absent without ROS)."""
    if ctx.teleop is None:
        raise RpcError("RingfourError",
                       "agent is running without a ROS interface (test mode)")
    return ctx.teleop


# Per (group, joint): the last commanded position, and where the joint was
# standing when that command went out. The gripper is a position
# controller — no action result, no effort interface — so "am I holding
# something" cannot be read off the hardware and has to be inferred
# (decision 27).
#
# The starting position is kept because commanded-vs-actual alone is not
# enough. A jaw that never moved at all reads exactly like a jaw that
# closed onto an object: both are stationary at a value that is not what
# was asked for. Requiring that the joint MOVED and then stopped tells the
# two apart, and it is the difference between "holding" and "the gripper
# is not actuating" — which is precisely the case seen on base101's sim,
# where arm_6 does not respond to anyone (found_along_the_way item 11).
#
# The trade is a false negative: closing on an object the jaw is already
# touching reports not-holding. That is the safer direction to be wrong in.
_TOOL_COMMANDS: dict[tuple[str, str], tuple[float, float]] = {}
# Commanded-vs-actual gap that counts as "did not get there", and the
# speed below which the joint counts as stopped. Both in the tool joint's
# own units (radians on base101's jaws, whose range is 0 .. 2.14).
_TOOL_ERROR_EPS = 0.02
_TOOL_STOPPED_EPS = 0.01


def _hold_at(ros: Any, session: Any, joints: list[str],
             positions: list[float]) -> None:
    """Command a standing position on whatever controller is active.

    A JointTrajectoryController has no "go here now" call, so holding is a
    one-point trajectory arriving almost immediately. A position
    controller takes the vector directly. Which one is in play depends on
    the session's role, and the session knows.
    """
    if session.role is Controller.TRAJECTORY:
        ros.follow_trajectory(session.controller, joints,
                              [(positions, 0.2)])
    elif session.role is Controller.VELOCITY:
        ros.command_positions(session.controller, joints, [0.] * len(joints))
    else:
        ros.command_positions(session.controller, joints, positions)


def _description_of(ctx: AgentContext) -> str | None:
    """The composed description, without waiting.

    ``RosInterface.description()`` blocks up to 10 s so startup validation
    can wait for a latched publisher. The handshake must not: a client
    connecting before robot_state_publisher is up gets ``None`` and a
    ``model.changed`` when it appears.
    """
    if ctx.ros is None:
        return None
    getter = getattr(ctx.ros, "description", None)
    return None if getter is None else getter(0.0)


def _signal_inventory(ctx):
    """Discover units/frames at the integration boundary, never from morphology."""
    out={}
    for path in (ctx.watch_paths or {}):
        unit=None;frame=None
        if path.startswith('battery.'):
            unit={'level':'percent','voltage':'V','current':'A'}.get(path.split('.')[-1])
        elif path.startswith('velocity.'):
            unit='rad/s' if '.angular.' in path else 'm/s'
        elif path.startswith('pose.'):
            unit='m';frame=ctx.profile.spec.frames.map
        elif path=='distance_traveled' or path.startswith('range.'):unit='m'
        elif path.startswith('imu.'):
            unit='rad/s' if 'angular_velocity' in path else 'm/s²'
        elif path.startswith('arms.') and '.joint_positions.' in path:
            kind=ctx.ros.joint_type(path.rsplit('.',1)[-1]) if hasattr(ctx.ros,"joint_type") else None
            unit={'revolute':'rad','continuous':'rad','prismatic':'m'}.get(kind)
        elif path.startswith('arms.') and '.effort.' in path:
            kind=ctx.ros.joint_type(path.rsplit('.',1)[-1]) if hasattr(ctx.ros,"joint_type") else None
            unit={'revolute':'N m','continuous':'N m','prismatic':'N'}.get(kind)
        elif path.startswith('environment.'):
            unit={'temperature':'°C','pressure':'Pa','illuminance':'lux'}.get(path.split('.')[-1])
        elif path.startswith('sensors.'):
            _,group,signal=path.split('.',2)
            unit=ctx.profile.spec.sensors[group][signal].units
        out[path]={'unit':unit,'frame':frame}
    return out


def _group_infos(ctx: AgentContext) -> dict[str, GroupInfo]:
    """The static shape of every group, for the handshake.

    Joint names come from the resolved runtime, not the profile - they
    were read off the controller at startup, which is the whole point of
    6.1. The tool joint is excluded, same as `arm.joint_states`: this is
    the command vector's shape.
    """
    out: dict[str, GroupInfo] = {}
    for name, group in ctx.groups.items():
        config = group.config
        tool = None
        if config.tool is not None:
            tool = ToolInfo(type=config.tool.type, joint=config.tool.joint,
                            open=config.tool.open, closed=config.tool.closed)
        workspace = None
        if config.workspace is not None and config.workspace.box:
            workspace = dict(config.workspace.box)
        out[name] = GroupInfo(
            kind=config.kind,
            tip=config.tip,
            dof_mask=config.dof_mask,
            joints=group.joints,
            tool=tool,
            pointing_frame=config.pointing_frame,
            workspace=workspace,
            control_modes=tuple(mode for mode in config.controllers if mode is not Controller.TOOL),
        )
    return out


def _group_description_shas(ctx: AgentContext) -> dict[str, str]:
    """{group: sha} for groups that publish their own description."""
    getter = getattr(ctx.ros, "group_description", None)
    if getter is None:
        return {}
    out: dict[str, str] = {}
    for name in ctx.groups:
        xml = getter(name)
        if xml is not None:
            out[name] = sha_of(xml)
    return out


def build_registry(ctx: AgentContext) -> dict[str, Handler]:
    """Build the method table for one loaded profile."""

    profile = ctx.profile

    # -- handshake / diagnostics ----------------------------------------------

    async def hello(session: Session, params: dict[str, Any]) -> Any:
        from .admission import DESCRIPTORS
        try:
            request = Hello.model_validate(params)
        except ValidationError as exc:
            raise RpcError("ProtocolMismatch", f"bad hello: {exc}") from exc
        if request.protocol != PROTOCOL_VERSION:
            raise RpcError(
                "ProtocolMismatch",
                f"client speaks protocol {request.protocol}, "
                f"agent speaks {PROTOCOL_VERSION}",
            )
        if session.handshaken:
            raise RpcError("ProtocolMismatch", "connection has already completed its handshake")
        lease_policy = profile.spec.safety.lease
        if lease_policy is not None and "operate" in session.principal.permissions:
            from .leases import Lease
            session.lease = Lease.issue(ctx.boot_epoch, session.principal.identity,
                                        session.client_id, lease_policy.lifetime)
        session.handshaken = True
        # The description rides along in the handshake (spec 26.2) so a
        # client can build its chain with no second round trip. It is
        # ~80 KB on base101 against a 64 MiB frame cap, once per
        # connection.
        description = sha = None
        group_shas: dict[str, str] = {}
        if ctx.ros is not None:
            description = _description_of(ctx)
            sha = getattr(ctx.ros, "description_sha", lambda: None)()
            group_shas = _group_description_shas(ctx)
        stop_policies = {name: group.config.stop_verification or profile.spec.safety.stop_verification
                         for name, group in ctx.groups.items()}
        return Welcome(
            protocol=PROTOCOL_VERSION,
            profile=profile.info,
            capabilities=profile.info.capabilities,
            instances=profile.instances,
            features={"boot_epoch": ctx.boot_epoch,
                      "commands": ctx.commands.feature([name for name in registry if DESCRIPTORS[name].effect != 'read']),
                      "connection_lease": {"id": session.lease.id, "epoch": ctx.boot_epoch,
                          "robot": profile.info.name, "principal": session.principal.identity,
                          "lifetime": lease_policy.lifetime, "renew_interval": lease_policy.renew_interval}
                          if session.lease is not None else None,
                      "payload_transports": ["shm", "net"],
                      "strict_measured_stops": not ctx.allow_unverified_stops,
                      "measured_stop_groups": [name for name, policy in stop_policies.items() if policy is not None],
                      "body_stop_policy": profile.spec.safety.stop_verification.model_dump(mode="json") if profile.spec.safety.stop_verification else None,
                      "durable_quarantine": hasattr(ctx.lock.quarantine,"finish_boot"),
                      "local_mobility_recovery": session.principal.authenticated and "recover" in session.principal.permissions and profile.spec.mobility is not None,
                      "local_group_recovery": session.principal.authenticated and "recover" in session.principal.permissions,
                      "access": {"principal": session.principal.identity,
                                 "authenticated": session.principal.authenticated,
                                 "mode": "local-peer" if session.principal.authenticated else "insecure-development" if "operate" in session.principal.permissions else "unverified-observer",
                                 "permissions": sorted(session.principal.permissions)},
                      "verified_operations": not ctx.allow_unverified_stops, "typed_trajectories": True,
                      "operation_cleanup_timeout":60., "persistent_guards":True,
                      "integration":{"slam_backend":getattr(profile.spec.slam,"backend",None),"controllers":{name:dict(group.controllers) for name,group in ctx.groups.items()}},
                      "signals":_signal_inventory(ctx), "control_session_ids": True, "task_status": True, "vendor_dispatch_receipts": True,
                      "task_retention_seconds": ctx.tasks.retention,
                      "task_record_capacity": ctx.tasks.max_records,
                      "group_stop_policy": {name: policy.model_dump(mode="json")
                          for name, policy in stop_policies.items() if policy is not None},
                      "group_feedback_policy": {name: {
                          "max_age": group.config.feedback_max_age,
                          "goal_tolerance": group.goal_tolerance,
                          "goal_tolerances": group.config.goal_tolerances}
                          for name,group in ctx.groups.items()}},
            description=description,
            description_sha=sha,
            group_description_shas=group_shas,
            groups=_group_infos(ctx),
            sensors={
                group: {name: {"units": signal.units, "type": signal.type}
                        for name, signal in signals.items()}
                for group, signals in (profile.spec.sensors or {}).items()},
            actions={
                name: {"is_motion": action.is_motion,
                       "max_rate": action.max_rate,
                       "parameters": {
                           field: {"type": spec.type, "min": spec.min,
                                   "max": spec.max, "default": spec.default}
                           for field, spec in action.parameters.items()}}
                for name, action in (profile.spec.actions or {}).items()},
        ).model_dump(mode="json")

    async def model_description(session: Session,
                                params: dict[str, Any]) -> Any:
        """The description for one group, or the composed one.

        Only groups whose profile overrides `description_topic` have their
        own; everything else answers with the composed description, so a
        client can ask uniformly without first checking which kind it has.
        """
        group = params.get("group")
        if group is not None and not isinstance(group, str):
            raise RpcError("RingfourError", "group must be a string")
        ros = _require_ros(ctx)
        xml = None
        if group is not None:
            if group not in ctx.groups:
                raise RpcError(
                    "RingfourError",
                    f"unknown group {group!r} (profile has "
                    f"{sorted(ctx.groups)})")
            xml = getattr(ros, "group_description", lambda _g: None)(group)
        if xml is None:
            xml = _description_of(ctx)
        if xml is None:
            raise RpcError(
                "RingfourError",
                "robot model is unavailable; model-dependent calculations cannot run",
                {"diagnostics": {"model_description": "unavailable"}})
        return {"description": xml, "sha": sha_of(xml)}

    async def ping(session: Session, params: dict[str, Any]) -> Any:
        return {}

    # -- tasks -------------------------------------------------------------------

    async def task_status(session: Session, params: dict[str, Any]) -> Any:
        return ctx.tasks.status(params.get("task_id"), session)

    async def task_cancel(session: Session, params: dict[str, Any]) -> Any:
        task_id = params.get("task_id")
        if not isinstance(task_id, str):
            raise RpcError("RingfourError", "task_id must be a string")
        record=ctx.tasks.owned(task_id,session)
        terminal=record.finished_at is not None
        await ctx.tasks.cancel(task_id, session)
        return {"accepted":not terminal,"already_terminal":terminal}

    # -- audit -------------------------------------------------------------------

    async def audit_query(session: Session, params: dict[str, Any]) -> Any:
        events = ctx.audit.query(
            since=params.get("since"),
            kind=params.get("kind"),
            task_id=params.get("task_id"),
            limit=params.get("limit", 500),
        )
        return {"events": [event.model_dump(mode="json") for event in events]}

    async def audit_subscribe(session: Session, params: dict[str, Any]) -> Any:
        session.audit_subscribed = True
        return {}

    async def audit_unsubscribe(session: Session,
                                params: dict[str, Any]) -> Any:
        session.audit_subscribed = False
        return {}

    # -- mobility state / TF (Phase 3) ----------------------------------------------

    async def mobility_get_state(session: Session,
                                 params: dict[str, Any]) -> Any:
        ros = _require_ros(ctx)
        try:
            return await asyncio.to_thread(ros.get_state)
        except Exception as exc:
            raise RpcError("RingfourError", str(exc)) from exc

    async def tf_lookup(session: Session, params: dict[str, Any]) -> Any:
        ros = _require_ros(ctx)
        try:
            return await asyncio.to_thread(
                ros.tf_lookup,
                str(params["parent"]), str(params["child"]),
                float(params.get("timeout", 2.0)),
                **({"stamp": float(params["stamp"])} if params.get("stamp") is not None else {}),
            )
        except KeyError as exc:
            raise RpcError("RingfourError", f"missing param {exc}") from exc
        except Exception as exc:
            raise RpcError("TimeoutError", str(exc)) from exc

    async def tf_frames(session: Session, params: dict[str, Any]) -> Any:
        ros = _require_ros(ctx)
        return {"frames": await asyncio.to_thread(ros.tf_frames)}

    async def tf_wait_for(session: Session, params: dict[str, Any]) -> Any:
        ros = _require_ros(ctx)
        try:
            await asyncio.to_thread(
                ros.tf_wait_for, str(params["frame"]),
                float(params.get("timeout", 5.0)),
            )
        except Exception as exc:
            raise RpcError("TimeoutError", str(exc)) from exc
        return {}

    async def tf_transform(session: Session, params: dict[str, Any]) -> Any:
        # Re-express a Pose or Point in another frame: one TF lookup, then
        # the shared spatial math. kind selects the entity type.
        ros = _require_ros(ctx)
        kind = params.get("kind")
        if kind not in ("pose", "point"):
            raise RpcError("RingfourError", "kind must be 'pose' or 'point'")
        model = Pose if kind == "pose" else Point
        try:
            entity = model.model_validate(params["entity"])
        except (KeyError, ValidationError) as exc:
            raise RpcError("RingfourError", f"bad entity: {exc}") from exc
        try:
            raw = await asyncio.to_thread(
                ros.tf_lookup, str(params["to_frame"]), entity.frame,
                float(params.get("timeout", 2.0)),
            )
        except Exception as exc:
            raise RpcError("TimeoutError", str(exc)) from exc
        transform = Transform.model_validate(raw)
        if kind == "point":
            moved = transform.apply(entity)
        else:
            x, y, z = transform.rotation.rotate(entity.x, entity.y, entity.z)
            moved = Pose(
                x=x + transform.translation.x,
                y=y + transform.translation.y,
                z=z + transform.translation.z,
                q=transform.rotation.multiply(entity.q),
                frame=transform.parent,
            )
        return {"entity": moved.model_dump(mode="json")}

    # -- cameras (Phase 4) -------------------------------------------------------

    def _camera_cfg(params: dict[str, Any]) -> tuple[str, Any]:
        cameras = profile.spec.cameras or {}
        name = params.get("camera")
        if not isinstance(name, str) or name not in cameras:
            raise RpcError(
                "RingfourError",
                f"unknown camera {name!r} (profile has {sorted(cameras)})",
            )
        return name, cameras[name]

    async def _capture(name: str, cfg: Any, need_rgb: bool,
                       need_depth: bool) -> tuple[Any, dict[str, Any]]:
        """Sample the camera, snapshot TF, store the deprojection record.

        Returns (record, sample). The record id is the frame handle the
        client gets; it carries depth + capture-time transforms so
        deproject stays paired with THIS frame.
        """
        ros = _require_ros(ctx)

        def grab() -> tuple[dict[str, Any], dict[str, dict]]:
            sample = ros.camera_sample(name, need_rgb, need_depth)
            return sample, ros.snapshot_transforms(sample["frame_id"])

        try:
            sample, raw_transforms = await asyncio.to_thread(grab)
        except RpcError:
            raise
        except Exception as exc:
            raise RpcError("RingfourError", str(exc)) from exc
        record = sensing.FrameRecord(
            camera=name,
            stamp=sample["stamp"],
            tf_frame=sample["frame_id"],
            optical=cfg.optical,
            intrinsics=(None if sample["intrinsics"] is None
                        else CameraIntrinsics.model_validate(
                            sample["intrinsics"])),
            depth=sample["depth"],
            rgb=sample["rgb"],
            transforms={target: Transform.model_validate(raw)
                        for target, raw in raw_transforms.items()},
        )
        ctx.frames.store(record, ttl=cfg.frame_ttl)
        return record, sample

    async def _payload_put(session, array: Any, ttl: float | None = None) -> dict[str, Any]:
        if session.payload_transport == "net":
            import numpy as np
            from ringfour.aio.transport import MAX_MESSAGE_BYTES
            array = np.ascontiguousarray(array)
            if array.dtype.kind not in "buif" or array.nbytes > MAX_MESSAGE_BYTES - 8:
                raise RpcError("RingfourError", "network array has unsupported dtype or exceeds the payload limit")
            meta = {"dtype": array.dtype.str, "shape": list(array.shape), "size": array.nbytes}
            payload_id = await session.send_payload("array/raw", meta, array.tobytes())
            return {"transport": "net", "name": str(payload_id), **meta}
        if ctx.shm is None:
            raise RpcError("RingfourError", "agent has no shared-memory store")
        ref = ctx.shm.put(array) if ttl is None else ctx.shm.put(array, ttl)
        return ref.model_dump(mode="json")

    async def camera_info(session: Session, params: dict[str, Any]) -> Any:
        name, cfg = _camera_cfg(params)
        ros = _require_ros(ctx)
        try:
            meta = await asyncio.to_thread(ros.camera_meta, name)
        except Exception as exc:
            raise RpcError("RingfourError", str(exc)) from exc
        return {
            "camera": name,
            "has_rgb": bool(cfg.rgb),
            "has_depth": bool(cfg.depth),
            "frame_ttl": cfg.frame_ttl,
            "max_stream_fps": cfg.max_stream_fps,
            "intrinsics": meta["intrinsics"],
            "frame": meta["frame_id"],
        }

    async def camera_get_frame(session: Session,
                               params: dict[str, Any]) -> Any:
        name, cfg = _camera_cfg(params)
        record, sample = await _capture(name, cfg, need_rgb=True,
                                        need_depth=False)
        return {
            "frame_id": record.id,
            "camera": name,
            "stamp": sample["rgb_stamp"],
            "payload": await _payload_put(session, sample["rgb"]),
        }

    async def camera_get_depth(session: Session,
                               params: dict[str, Any]) -> Any:
        name, cfg = _camera_cfg(params)
        record, sample = await _capture(name, cfg, need_rgb=False,
                                        need_depth=True)
        return {
            "frame_id": record.id,
            "camera": name,
            "stamp": sample["depth_stamp"],
            "payload": await _payload_put(session, sample["depth"]),
        }

    async def camera_get_synced(session: Session,
                                params: dict[str, Any]) -> Any:
        name, cfg = _camera_cfg(params)
        record, sample = await _capture(name, cfg, need_rgb=True,
                                        need_depth=True)
        return {
            "frame": {
                "frame_id": record.id,
                "camera": name,
                "stamp": sample["rgb_stamp"],
                "payload": await _payload_put(session, sample["rgb"]),
            },
            "depth": {
                "frame_id": record.id,
                "camera": name,
                "stamp": sample["depth_stamp"],
                "payload": await _payload_put(session, sample["depth"]),
            },
        }

    async def camera_get_cloud(session: Session,
                               params: dict[str, Any]) -> Any:
        name, cfg = _camera_cfg(params)
        voxel = params.get("voxel")
        if not isinstance(voxel, (int, float)) or voxel <= 0:
            raise RpcError(
                "RingfourError",
                "voxel is mandatory and must be > 0 (meters) — the API's "
                "size budget; pick the coarsest voxel your task tolerates",
            )
        record, sample = await _capture(name, cfg, need_rgb=False,
                                        need_depth=True)
        if record.intrinsics is None:
            raise RpcError(
                "RingfourError",
                f"camera {name!r} has no intrinsics (no camera_info topic "
                "in the profile); cannot build a cloud",
            )
        rgb = record.rgb
        if rgb is None and cfg.rgb:
            # Colors are best-effort: use the synced rgb if one exists.
            ros = _require_ros(ctx)
            try:
                extra = await asyncio.to_thread(
                    ros.camera_sample, name, True, True)
                rgb = extra["rgb"]
            except Exception:
                rgb = None
        points, colors = await asyncio.to_thread(
            sensing.make_cloud, sample["depth"], rgb, record.intrinsics,
            float(voxel), record.optical,
        )
        result = {
            "stamp": sample["depth_stamp"],
            "frame": record.tf_frame,
            "count": int(points.shape[0]),
            "points": await _payload_put(session, points),
            "colors": None if colors is None else await _payload_put(session, colors),
        }
        return result

    async def camera_deproject(session: Session,
                               params: dict[str, Any]) -> Any:
        ros = _require_ros(ctx)
        frame_id = params.get("frame_id")
        if not isinstance(frame_id, str):
            raise RpcError("RingfourError", "frame_id must be a string")
        pixel = params.get("pixel")
        if (not isinstance(pixel, (list, tuple)) or len(pixel) != 2
                or not all(isinstance(c, (int, float)) for c in pixel)):
            raise RpcError("RingfourError", "pixel must be [u, v]")
        record = ctx.frames.get(frame_id)
        in_frame = params.get("in_frame")
        if in_frame is not None:
            in_frame = ros.resolve_frame(str(in_frame))
        point = sensing.deproject(record, (pixel[0], pixel[1]), in_frame)
        return {"point": point}

    async def camera_stream_start(session: Session,
                                  params: dict[str, Any]) -> Any:
        name, cfg = _camera_cfg(params)
        _require_ros(ctx)   # guard only; the tick closure takes its own
        fps = float(params.get("fps", 5.0))
        fps = min(max(fps, 0.1), cfg.max_stream_fps)
        shm_ttl = max(2.0, 3.0 / fps)

        # Ticks are paced on the WALL clock while stamps are sensor time.
        # Whenever the source is slower than the requested fps — a low
        # simulator real-time factor, a thermally throttled SBC, a camera
        # that dropped to half rate — the same frame comes round twice, and
        # sending it again costs a fresh /dev/shm segment for pixels the
        # client already has. Skipping is the established behavior for a
        # tick with nothing new (see the RpcError branch).
        last_stamp: float | None = None

        async def tick() -> dict[str, Any] | None:
            nonlocal last_stamp
            try:
                record, sample = await _capture(name, cfg, need_rgb=True,
                                                need_depth=False)
            except RpcError:
                return None  # sensor warming up / TF gap: skip this tick
            stamp = sample["rgb_stamp"]
            if last_stamp is not None and stamp <= last_stamp:
                return None  # no new frame since the last tick
            last_stamp = stamp
            return {
                "kind": "camera.frame",
                "camera": name,
                "frame_id": record.id,
                "stamp": stamp,
                "payload": await _payload_put(session, sample["rgb"], ttl=shm_ttl),
            }

        stream_id = ctx.streams.start(session, 1.0 / fps, tick)
        return {"stream_id": stream_id, "fps": fps}

    async def camera_stream_stop(session: Session,
                                 params: dict[str, Any]) -> Any:
        ctx.streams.stop(str(params.get("stream_id")), session)
        return {}

    # -- lidar (Phase 4) ---------------------------------------------------------

    async def _lidar_tick(session) -> dict[str, Any]:
        ros = _require_ros(ctx)
        try:
            sample = await asyncio.to_thread(ros.lidar_sample)
        except RpcError:
            raise
        except Exception as exc:
            raise RpcError("RingfourError", str(exc)) from exc
        sample["ranges"] = await _payload_put(session, sample.pop("ranges"), ttl=10.0)
        intensities = sample.pop("intensities")
        sample["intensities"] = (
            None if intensities is None else await _payload_put(session, intensities, ttl=10.0))
        return sample

    async def lidar_scan(session: Session, params: dict[str, Any]) -> Any:
        return await _lidar_tick(session)

    async def lidar_stream_start(session: Session,
                                 params: dict[str, Any]) -> Any:
        hz = float(params.get("hz", 5.0))
        hz = min(max(hz, 0.1), _MAX_LIDAR_HZ)

        # Same reasoning as the camera stream: wall-clock ticks against a
        # sensor-time source re-send the same sweep when the source is
        # slower than the requested rate, and each resend allocates a
        # /dev/shm segment for ranges the client already has.
        last_stamp: float | None = None

        async def tick() -> dict[str, Any] | None:
            nonlocal last_stamp
            try:
                sample = await _lidar_tick(session)
            except RpcError:
                return None
            stamp = sample.get("stamp")
            if (last_stamp is not None and stamp is not None
                    and stamp <= last_stamp):
                return None  # no new sweep since the last tick
            last_stamp = stamp
            sample["kind"] = "lidar.scan"
            return sample

        stream_id = ctx.streams.start(session, 1.0 / hz, tick)
        return {"stream_id": stream_id, "hz": hz}

    async def lidar_stream_stop(session: Session,
                                params: dict[str, Any]) -> Any:
        ctx.streams.stop(str(params.get("stream_id")), session)
        return {}


    # -- vendor escape hatches (profile sensors: / actions:) ----------------------

    async def sensor_read(session: Session, params: dict[str, Any]) -> Any:
        """One declared signal, or the whole lot."""
        ros = _require_ros(ctx)
        group = params.get("group")
        if group is None:
            return {"sensors": await asyncio.to_thread(ros.vendor_snapshot)}
        declared = profile.spec.sensors or {}
        if group not in declared:
            raise RpcError(
                "RingfourError",
                f"unknown sensor group {group!r} (this robot has "
                f"{sorted(declared)})")
        name = params.get("signal")
        if name is None:
            return {"signals": {
                signal: await asyncio.to_thread(ros.vendor_read, group, signal)
                for signal in declared[group]}}
        if name not in declared[group]:
            raise RpcError(
                "RingfourError",
                f"sensors.{group} has no signal {name!r} (it has "
                f"{sorted(declared[group])})")
        reading = await asyncio.to_thread(ros.vendor_read, group, name)
        if reading is None:
            # Declared but silent: the difference between "no such
            # signal" (above) and "nothing has published yet" matters to
            # a mission deciding whether to wait or to give up.
            return {"value": None, "stamp": None,
                    "units": declared[group][name].units}
        return reading

    async def action_invoke(session: Session, params: dict[str, Any]) -> Any:
        """Command one declared action. Audited; motion ones take the lock."""
        if ctx.actions is None:
            raise RpcError("CapabilityNotSupported",
                           "this robot declares no actions")
        name = params.get("action")
        if not isinstance(name, str):
            raise RpcError("RingfourError", "action must be a string")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise RpcError("RingfourError", "arguments must be an object")
        return await asyncio.to_thread(
            ctx.actions.invoke, name, arguments, session.client_id)

    # -- control sessions (spec 8.2 / plan 6.4) -----------------------------------

    def _require_control(ctx: AgentContext) -> Any:
        if ctx.control is None:
            raise RpcError(
                "CapabilityNotSupported",
                "this robot declares no groups, so there is nothing to "
                "open a control session on")
        return ctx.control

    def _session_id(params):
        value = params.get("session_id")
        if not isinstance(value, str) or not value:
            raise RpcError("SafetyViolation", "a current control session ID is required",
                           {"reason": "stale session"})
        return value

    def _fresh_snapshot(ros, names, group):
        if not hasattr(ros, "verified_joint_snapshot"):
            raise RpcError("CompletionUnverified", "integration cannot establish joint feedback freshness")
        try:
            return ros.verified_joint_snapshot(names, group.config.feedback_max_age)
        except ValueError as exc:
            raise RpcError("CompletionUnverified", str(exc), {"arrived": False}) from exc

    async def group_control_start(session: Session,
                                  params: dict[str, Any]) -> Any:
        control = _require_control(ctx)
        raw = params.get("raw")
        role = params.get("role")
        if raw is not None and not isinstance(raw, str):
            raise RpcError("RingfourError", "raw must be a controller name")
        parsed_role = None
        if role is not None:
            try:
                parsed_role = Controller(role)
            except ValueError:
                raise RpcError(
                    "RingfourError",
                    f"unknown controller role {role!r}; expected one of "
                    f"{[str(r) for r in Controller]}") from None
        opened = await asyncio.to_thread(
            control.start, session.client_id, params.get("group"),
            parsed_role, raw)
        return {"session_id": opened.id, "controller": opened.controller,
                "loaded": opened.loaded_here,
                "deactivated": opened.deactivated}

    async def group_control_stop(session: Session,
                                 params: dict[str, Any]) -> Any:
        control = _require_control(ctx)
        name = params.get("group")
        current = await asyncio.to_thread(control.prepare_close, str(name), session.client_id, _session_id(params))
        if current is not None:
            ctx.tasks.cancel_resource(current.id)
            try:
                await ctx.tasks.drain_resource(current.id)
            except RpcError as exc:
                ctx.lock.quarantine[str(name)] = {"reason": "operation cleanup timeout", "error": str(exc)}
                raise
        await asyncio.to_thread(control.end, str(name), client_id=session.client_id,
                                session_id=_session_id(params))
        return {}

    async def group_start_trajectory(session: Session, params: dict[str, Any]) -> Any:
        from ringfour.models.execution import JointTrajectory
        control=_require_control(ctx);ros=_require_ros(ctx)
        try:trajectory=JointTrajectory.model_validate(params['trajectory'])
        except (ValueError,TypeError) as exc:raise RpcError('RingfourError','invalid typed trajectory',{'error':str(exc)}) from exc
        first=trajectory.configurations[0];group=control.group(params['group'])
        if first.robot!=ctx.profile.info.name or first.group!=group.name:
            raise RpcError('SafetyViolation','trajectory belongs to another robot or group')
        revision=await asyncio.to_thread(control.model_revision,group)
        if revision is None or first.model_revision!=revision:
            raise RpcError('ModelChanged','trajectory does not match the current robot model')
        current=await asyncio.to_thread(control.session,group.name,session.client_id,params['session_id'])
        if current.model_revision!=revision:raise RpcError('ModelChanged','control session belongs to an old robot model')
        limits=await asyncio.to_thread(ros.joint_limits)
        for configuration in trajectory.configurations:
            if set(configuration.positions)!=set(group.joints):raise RpcError('RingfourError','trajectory joint coverage differs from the group')
            for joint in configuration.joints:
                kind=await asyncio.to_thread(ros.joint_type,joint.name)
                unit={'revolute':'rad','continuous':'rad','prismatic':'m'}.get(kind)
                bounds=limits.get(joint.name) or (None,None)
                if joint.unit!=unit or (joint.lower,joint.upper)!=tuple(bounds):
                    raise RpcError('ModelChanged','trajectory joint units/bounds differ from the current model')
        return await group_follow(session,{'group':group.name,'session_id':params['session_id'],
            'waypoints':[q.vector(group.joints) for q in trajectory.configurations], '_times':trajectory.times,'dt':1.})

    async def group_follow(session: Session, params: dict[str, Any]) -> Any:
        """Run a joint trajectory as a task, mirroring mobility.navigate_to.

        The SDK generates no trajectory: waypoints go to the
        JointTrajectoryController with time_from_start and it interpolates
        (spec 8.2). Same task contract as navigation, so break-cancels and
        exceptions propagate identically.
        """
        ros = _require_ros(ctx)
        control = _require_control(ctx)
        name = str(params.get("group"))
        open_session = await asyncio.to_thread(control.session, name, session.client_id, _session_id(params))
        group = control.group(name)

        if not control.allow_unverified_stops and open_session.role is not Controller.TRAJECTORY:
            raise RpcError("CapabilityNotSupported", "follow requires a declared trajectory controller role")
        waypoints = params.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise RpcError("RingfourError", "waypoints must be a non-empty "
                                            "list of joint vectors")
        width = len(group.joints)
        for index, point in enumerate(waypoints):
            if not isinstance(point, list) or len(point) != width:
                raise RpcError(
                    "RingfourError",
                    f"waypoint {index} has {len(point) if isinstance(point, list) else '?'} "
                    f"values, group {name!r} has {width} joints "
                    f"({', '.join(group.joints)})")
        duration = params.get("dt", 2.0)
        if isinstance(duration, bool) or not isinstance(duration, (int,float)):
            raise RpcError("RingfourError", "trajectory step duration must be numeric")
        dt = float(duration)
        if not math.isfinite(dt) or dt <= 0:
            raise RpcError("RingfourError", "trajectory step duration must be positive and finite")
        limits = await asyncio.to_thread(ros.joint_limits)
        for index, point in enumerate(waypoints):
            for joint, value in zip(group.joints, point):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise RpcError("RingfourError", f"waypoint {index}: {joint} requires a finite number")
                bound = limits.get(joint)
                if bound is not None and not bound[0] <= value <= bound[1]:
                    raise RpcError("SafetyViolation", f"waypoint {index}: {joint} exceeds its positional bounds",
                                   {"reason": "joint limits", "joint": joint, "bounds": bound})
        times=params.get("_times")
        points = [(list(point), times[index] if times is not None else dt * (index + 1))
                  for index, point in enumerate(waypoints)]
        if not all(math.isfinite(when) for _,when in points):
            raise RpcError("RingfourError", "trajectory time horizon must be finite")

        open_session = await asyncio.to_thread(control.reserve_trajectory, name, session.client_id, _session_id(params))
        async def body(handle: TaskHandle) -> dict[str, Any]:
            try:
                submission = asyncio.create_task(asyncio.to_thread(
                    control.dispatch, open_session,
                    lambda: ros.follow_trajectory(open_session.controller, list(group.joints), points)))
                try:
                    await asyncio.shield(submission)
                except asyncio.CancelledError:
                    # A worker cannot be killed. Await acceptance before cancelling,
                    # so cancellation cannot return before a late goal dispatch.
                    await submission
                    handle.mark_dispatched()
                    await handle.verify_stop(control.stop_work, open_session)
                    raise
                handle.mark_dispatched()
            except RpcError:
                if not control.allow_unverified_stops:
                    await handle.verify_stop(control.stop_work, open_session)
                raise
            except Exception as exc:
                if not control.allow_unverified_stops:
                    await asyncio.to_thread(control.stop_work, open_session)
                open_session.running = False
                raise RpcError("ControllerError", str(exc),
                               {"controller": open_session.controller}) from exc
            try:
                last_report = 0.0
                started = time.monotonic()
                total = points[-1][1] if points else 0.0
                while True:
                    outcome = await asyncio.to_thread(ros.traj_wait, 0.1)
                    if outcome is not None:
                        break
                    now = time.monotonic()
                    if now - last_report >= 0.2:
                        last_report = now
                        feedback = ros.traj_feedback()
                        # Progress is elapsed-over-planned, not distance
                        # covered: the controller owns interpolation and
                        # reports no completion fraction, and a
                        # joint-distance estimate would read as progress
                        # while the arm sat still against a limit. Capped
                        # just below 1.0 so only ARRIVED reports done.
                        progress = (min(0.99, (now - started) / total)
                                    if total > 0 else 0.0)
                        await handle.report(progress=progress,
                                            data=feedback)
            except asyncio.CancelledError:
                await handle.verify_stop(control.stop_work, open_session)
                raise
            except Exception:
                await handle.verify_stop(control.stop_work, open_session)
                raise
            stop_report=None
            if not control.allow_unverified_stops:
                stop_report=await handle.verify_stop(control.stop_work, open_session)
            if outcome["status"] == "succeeded":
                # THE CONTROLLER SAYING "succeeded" MEANS IT FINISHED ITS
                # PLAN, not that the joints are where the plan ended. Look
                # before agreeing: a trajectory given less time than the
                # joints need stops part way and still succeeds, and
                # reporting that as ARRIVED is how a mission ends up
                # reasoning about an arm that is somewhere else entirely.
                snapshot = await asyncio.to_thread(_fresh_snapshot, ros, list(group.joints), group)
                try:
                    residual = checked_arrival_residual(list(group.joints), points[-1][0], snapshot)
                except ValueError as exc:
                    raise RpcError("CompletionUnverified", str(exc), {"controller":open_session.controller,"arrived":False}) from exc
                failures = {j: value for j,value in residual.items()
                            if abs(value) > group.config.goal_tolerances.get(j, group.goal_tolerance)}
                worst = max(failures, key=lambda j: abs(failures[j]) / group.config.goal_tolerances.get(j, group.goal_tolerance)) if failures else None
                gap = abs(failures[worst]) if worst is not None else 0.
                if worst is not None:
                    tolerance = group.config.goal_tolerances.get(worst, group.goal_tolerance)
                    joint_type = await asyncio.to_thread(ros.joint_type, worst) if hasattr(ros,"joint_type") else None
                    raise RpcError(
                        "ExecutionFailed",
                        arrival_failure_message(worst,gap,tolerance,joint_type),
                        {"controller": open_session.controller,
                         "arrived": False,
                         "residual": residual,
                         "goal_tolerance": tolerance,
                         "goal_tolerances": {j: group.config.goal_tolerances.get(j, group.goal_tolerance) for j in group.joints}})
                return {"outcome": "arrived", "arrived": True,
                        "residual": residual,"positions":{j:snapshot[j][0] for j in group.joints},
                        "model_revision":open_session.model_revision,"stop_verification":stop_report,
                        "goal_tolerances":{j:group.config.goal_tolerances.get(j,group.goal_tolerance) for j in group.joints}}
            raise RpcError(
                "ExecutionFailed",
                outcome.get("message") or "trajectory aborted",
                {"error_code": outcome.get("error_code"),
                 "controller": open_session.controller})

        try:
            task_id = ctx.tasks.start("group.start_trajectory", session, body, resource=open_session.id)
        except Exception:
            open_session.running = False
            raise
        ctx.tasks._tasks[task_id].aio_task.add_done_callback(lambda _: setattr(open_session, "running", False))
        return {"task_id": task_id}

    async def group_hold(session: Session, params: dict[str, Any]) -> Any:
        """Command the current position and stay in the session."""
        ros = _require_ros(ctx)
        control = _require_control(ctx)
        name = str(params.get("group"))
        open_session = await asyncio.to_thread(control.session, name, session.client_id, _session_id(params))
        group = control.group(name)
        if any(s.running for s in control.active().values()):
            raise RpcError("ResourceBusy", "hold cannot replace an active trajectory")
        snapshot = await asyncio.to_thread(_fresh_snapshot, ros, list(group.joints), group)
        missing = [j for j in group.joints if j not in snapshot]
        if missing:
            raise RpcError("RingfourError",
                           f"no joint state for {missing}")
        positions = [snapshot[j][0] for j in group.joints]
        if not all(math.isfinite(value) for value in positions):
            raise RpcError("CompletionUnverified", "hold requires finite joint feedback")
        limits = await asyncio.to_thread(ros.joint_limits)
        outside = [joint for joint,value in zip(group.joints, positions)
                   if limits.get(joint) is not None and not limits[joint][0] <= value <= limits[joint][1]]
        if outside:
            raise RpcError("SafetyViolation", "hold position exceeds reported joint limits",
                           {"reason": "joint limits", "joints": outside})
        if open_session.role is not Controller.TRAJECTORY:
            await asyncio.to_thread(control.dispatch, open_session,
                                   lambda: _hold_at(ros, open_session, list(group.joints), positions))
            if not control.allow_unverified_stops:
                await asyncio.to_thread(control.stop_work, open_session)
            return {"positions": dict(zip(group.joints, positions))}
        open_session = await asyncio.to_thread(control.reserve_trajectory, name, session.client_id, _session_id(params))
        submitted = False
        try:
            submission = asyncio.create_task(asyncio.to_thread(
                control.dispatch, open_session,
                lambda: _hold_at(ros, open_session, list(group.joints), positions)))
            try:
                await asyncio.shield(submission)
            except asyncio.CancelledError:
                await submission
                submitted = True
                raise
            submitted = True
            deadline = time.monotonic()+5.0
            while True:
                ctx.safety.check_motion(session.client_id, "group.hold")
                outcome = await asyncio.to_thread(ros.traj_wait, .05)
                if outcome is not None:
                    if outcome["status"] != "succeeded":
                        raise RpcError("ExecutionFailed", "hold controller did not complete its command")
                    break
                if time.monotonic() >= deadline:
                    raise RpcError("CompletionUnverified", "hold controller did not complete before its deadline")
            if not control.allow_unverified_stops:
                await asyncio.to_thread(control.stop_work, open_session)
            return {"positions": dict(zip(group.joints, positions))}
        except BaseException as original:
            if submitted:
                try:
                    await asyncio.to_thread(control.stop_work, open_session)
                except RpcError as cleanup:
                    if isinstance(original, asyncio.CancelledError):
                        raise cleanup from original
                    if isinstance(original, RpcError):
                        original.extra["cleanup"] = cleanup.extra
                    original.add_note(f"Additional cleanup failure: {cleanup}")
            raise
        finally:
            open_session.running = False

    async def group_stop(session: Session, params: dict[str, Any]) -> Any:
        """Cancel any running trajectory. The session stays open."""
        ros = _require_ros(ctx)
        control = _require_control(ctx)
        name = str(params.get("group"))
        open_session = await asyncio.to_thread(control.session, name, session.client_id, _session_id(params))
        cancelled = ctx.tasks.cancel_resource(open_session.id)
        try:
            await ctx.tasks.drain_resource(open_session.id)
        except RpcError as exc:
            ctx.lock.quarantine[name] = {"reason": "operation cleanup timeout", "error": str(exc)}
            raise
        # Do not cancel the adapter's unrelated group after this group is idle.
        if open_session.running or open_session.role is not Controller.TRAJECTORY:
            await asyncio.to_thread(control.stop_work, open_session)
        elif control.stop_policy(group := control.group(name)) is not None:
            from .stopping import verify_joint_stop
            try:
                await asyncio.to_thread(verify_joint_stop, ros, group.joints, control.stop_policy(group), group.config.feedback_max_age)
            except RpcError as exc:
                ctx.lock.quarantine[name] = {"reason": "group stop unverified", "evidence": exc.extra}
                raise
        if name in ctx.lock.quarantine:
            raise RpcError("CleanupFailed", "group operation cleanup remains unverified",
                           {"cleanup": ctx.lock.quarantine[name]})
        open_session.running = False
        ctx.audit.record(AuditKind.COMMAND, client=session.client_id,
                         call="group.stop", outcome=Outcome.SUCCESS,
                         detail={"group": name, "cancelled": cancelled})
        return {"cancelled": cancelled}

    async def group_tool_set(session: Session, params: dict[str, Any]) -> Any:
        """Command the tool joint. Needs no switch: the tool controller
        claims a disjoint joint, so it is active throughout an arm
        session (6.0b)."""
        handle=params.get("_handle")
        stop_evidence=None
        ros = _require_ros(ctx)
        control = _require_control(ctx)
        name = str(params.get("group"))
        group = control.group(name)
        tool = group.config.tool
        controller = group.controllers.get(Controller.TOOL)
        if tool is None or controller is None or group.tool_joint is None:
            raise RpcError(
                "CapabilityNotSupported",
                f"group {name!r} declares no tool")
        position = params.get("position")
        if isinstance(position, bool) or not isinstance(position, (int, float)) or not math.isfinite(position):
            raise RpcError("RingfourError", "position must be a number")
        snapshot = await asyncio.to_thread(_fresh_snapshot, ros, [group.tool_joint], group)
        if group.tool_joint not in snapshot or not math.isfinite(snapshot[group.tool_joint][0]):
            raise RpcError("CompletionUnverified", "tool position feedback is unavailable")
        limits = await asyncio.to_thread(ros.joint_limits)
        bounds = limits.get(group.tool_joint)
        if bounds is not None and not bounds[0] <= position <= bounds[1]:
            raise RpcError("SafetyViolation", "tool target exceeds positional bounds", {"reason": "joint limits"})
        await asyncio.to_thread(control.require_stop, group, (group.tool_joint,))
        def check_tool_controller():
            states = ros.loaded_controllers()
            found = states.get(controller, {})
            if found.get("state") != "active" or set(found.get("joints", ())) != {group.tool_joint}:
                raise RpcError("ControllerError", "tool controller is inactive or claims a different resource")
        if not control.allow_unverified_stops:
            await asyncio.to_thread(check_tool_controller)
        from threading import Event
        revoked = Event()
        def dispatch_tool():
            nonlocal stop_evidence
            with ctx.lock.mutex:
                ctx.safety.check_motion(session.client_id, "group.tool_set")
                acquisition = "tool:" + name + ":" + uuid.uuid4().hex
                if not ctx.lock.acquire(session.client_id, acquisition,
                                        resources=(f"joint:{group.tool_joint}", f"controller:{controller}")):
                    raise RpcError("ResourceBusy", "tool is owned by another client")
            try:
                with ctx.lock.mutex:
                    ctx.safety.check_motion(session.client_id, "group.tool_set")
                    control.require_stop(group, (group.tool_joint,))
                    if not control.allow_unverified_stops:
                        check_tool_controller()
                    if handle:handle.mark_dispatched()
                    ros.command_positions(controller, [group.tool_joint], [float(position)])
                if not control.allow_unverified_stops:
                    from .stopping import verify_joint_stop
                    policy = control.stop_policy(group)
                    deadline = time.monotonic() + params.get("execution_timeout", 30.)
                    while True:
                        if revoked.is_set():
                            raise RpcError("SafetyViolation", "tool request was revoked")
                        ctx.safety.check_motion(session.client_id, "group.tool_set")
                        current = _fresh_snapshot(ros, [group.tool_joint], group)
                        if abs(current[group.tool_joint][0] - position) <= group.config.goal_tolerances.get(group.tool_joint, group.goal_tolerance):
                            break
                        if time.monotonic() >= deadline:
                            raise RpcError("CompletionUnverified", "tool target was not reached within its operation budget")
                        time.sleep(.02)
                    stop_evidence=verify_joint_stop(ros, [group.tool_joint], policy, group.config.feedback_max_age,
                                      abort=lambda: revoked.is_set() or ctx.state.estop_engaged or ctx.state.closing or bool(ctx.state.guard_inhibits))
                    current = _fresh_snapshot(ros, [group.tool_joint], group)
                    if abs(current[group.tool_joint][0] - position) > group.config.goal_tolerances.get(group.tool_joint, group.goal_tolerance):
                        raise RpcError("CompletionUnverified", "tool target drifted during measured-rest verification")
            except Exception:
                try:
                    current = _fresh_snapshot(ros, [group.tool_joint], group)
                    ros.command_positions(controller, [group.tool_joint], [current[group.tool_joint][0]])
                    if not control.allow_unverified_stops:
                        from .stopping import verify_joint_stop
                        stop_evidence=verify_joint_stop(ros, [group.tool_joint], control.stop_policy(group), group.config.feedback_max_age)
                        if handle:handle.record_stop(stop_evidence)
                except Exception as cleanup:
                    ctx.lock.quarantine['tool:' + name] = {"reason": "tool stop unverified", "error": str(cleanup),
                            "expected_controllers": {controller: "active"}}
                raise
            finally:
                ctx.lock.release(session.client_id, acquisition)
        submission = asyncio.create_task(asyncio.to_thread(dispatch_tool))
        try:
            await asyncio.shield(submission)
        except asyncio.CancelledError:
            revoked.set()
            # Worker holds current measured position and verifies rest before
            # releasing its acquisition.
            try:
                await submission
            except Exception as cleanup:
                if "tool:"+name in ctx.lock.quarantine:
                    raise RpcError("StopUnverified","tool cancellation cleanup failed",{"cleanup":ctx.lock.quarantine["tool:"+name]}) from cleanup
            raise
        was_at = snapshot[group.tool_joint][0]
        _TOOL_COMMANDS[(name, group.tool_joint)] = (float(position), was_at)
        ctx.audit.record(AuditKind.COMMAND, client=session.client_id,
                         call="group.tool_set", outcome=Outcome.SUCCESS,
                         detail={"group": name, "position": float(position)})
        current=await asyncio.to_thread(_fresh_snapshot,ros,[group.tool_joint],group)
        unit={"revolute":"rad","continuous":"rad","prismatic":"m"}.get(ros.joint_type(group.tool_joint))
        if unit is None:raise RpcError("CapabilityNotSupported","tool joint units unavailable")
        return {"group":name,"joint":group.tool_joint,"position":current[group.tool_joint][0],
                "target":float(position),"residual":current[group.tool_joint][0]-position,"unit":unit,
                "tolerance":group.config.goal_tolerances.get(group.tool_joint,group.goal_tolerance),"stop_verification":stop_evidence}

    async def group_start_tool(session,params):
        # The task retains submission and cleanup; a lost observer does not own it.
        async def body(handle):
            return await group_tool_set(session,{**params,'_handle':handle})
        return {'task_id':ctx.tasks.start('group.start_tool',session,body,resource='tool:'+params['group'])}

    async def group_tool_state(session: Session,
                               params: dict[str, Any]) -> Any:
        """Position, velocity, and the is_holding inference (decision 27).

        The gripper is a position controller with no effort interface, so
        force cannot be read. `arm_6` does expose velocity, which makes a
        better rule than position error alone: commanded differs from
        actual AND the joint has stopped moving means it is pressing on
        something. Position error by itself cannot tell a grasp from a jaw
        still travelling.
        """
        ros = _require_ros(ctx)
        control = _require_control(ctx)
        name = str(params.get("group"))
        group = control.group(name)
        if group.tool_joint is None:
            raise RpcError("CapabilityNotSupported",
                           f"group {name!r} declares no tool")
        snapshot = await asyncio.to_thread(ros.joint_snapshot)
        sample = snapshot.get(group.tool_joint)
        if sample is None:
            raise RpcError("RingfourError",
                           f"no joint state for {group.tool_joint!r}")
        position, velocity, _effort = sample
        record = _TOOL_COMMANDS.get((name, group.tool_joint))
        commanded = None if record is None else record[0]
        holding = False
        moved = None
        if record is not None:
            commanded, was_at = record
            moved = abs(position - was_at) > _TOOL_ERROR_EPS
            holding = (abs(commanded - position) > _TOOL_ERROR_EPS
                       and abs(velocity) < _TOOL_STOPPED_EPS
                       and moved)
        return {"position": position, "velocity": velocity,
                "commanded": commanded, "moved": moved,
                "is_holding": holding}

    # -- joint states (Phase 4, controller-level) -----------------------------------

    async def arm_joint_states(session: Session,
                               params: dict[str, Any]) -> Any:
        """State of one group's joints. Still named arm.* on the wire: the
        method is shipped, and 6.4 is what renames the surface."""
        ros = _require_ros(ctx)
        name = params.get("arm")
        group = ctx.groups.get(name) if isinstance(name, str) else None
        if group is None:
            raise RpcError(
                "RingfourError",
                f"unknown group {name!r} (profile has "
                f"{sorted(ctx.groups)})",
            )
        # The group's OWN joints, tool excluded. group.all_joints (which
        # includes it) is for the watch-path table, where every commandable
        # joint is watchable; here the mapping is the command vector's
        # shape, and the tool is reached through group.tool instead
        # (spec 8.1 / 8.3).
        joints = list(group.joints)
        if not joints:
            raise RpcError(
                "RingfourError",
                f"group {name!r} has no joints: none of its controllers "
                "was loaded at startup and the profile sets no "
                f"groups.{name}.joints override",
            )
        include_external = params.get("include_external", False)
        if not isinstance(include_external, bool):
            raise RpcError("RingfourError", "include_external must be a boolean")
        stamp = None
        if (include_external or params.get("stamp") is not None) and hasattr(ros, "joint_snapshot_with_stamp"):
            requested_stamp = params.get("stamp")
            try:
                snapshot, stamp = await asyncio.to_thread(ros.joint_snapshot_with_stamp, requested_stamp)
            except LookupError as exc:
                raise RpcError("FrameExpired", str(exc)) from exc
        else:
            snapshot = await asyncio.to_thread(ros.joint_snapshot)
        limits = await asyncio.to_thread(ros.joint_limits)
        missing = [j for j in joints if j not in snapshot]
        if missing:
            raise RpcError(
                "RingfourError",
                f"no joint state yet for {missing} (joint_state_broadcaster "
                "running? profile joint_states topic correct?)",
            )
        result = {"joints": {
            j: {
                "position": snapshot[j][0],
                "velocity": snapshot[j][1],
                "effort": snapshot[j][2],
                "limits": limits.get(j),
            }
            for j in joints
        }}
        if include_external:
            # Preserve command-vector shape while exposing the same measured
            # snapshot for independently controlled mounts and collision links.
            result["positions"] = {j: value[0] for j, value in snapshot.items()}
            result["stamp"] = stamp
        return result


    # -- watches (Phase 4) -------------------------------------------------------

    async def watch_start(session: Session, params: dict[str, Any]) -> Any:
        if params.get("stop"):
            raise RpcError("RingfourError","stop watches are retired; install a persistent guard")
        path = params.get("path")
        if not isinstance(path, str):
            raise RpcError("RingfourError", "path must be a string")
        validate_path(ctx.watch_paths or {}, path)
        outside = params.get("outside")
        if outside is not None:
            if (not isinstance(outside, (list, tuple)) or len(outside) != 2):
                raise RpcError("RingfourError", "outside must be [lo, hi]")
            outside = (float(outside[0]), float(outside[1]))
        above = params.get("above")
        below = params.get("below")
        watch_id = ctx.watches.start(
            owner=session.client_id,
            path=path,
            above=None if above is None else float(above),
            below=None if below is None else float(below),
            outside=outside,
            stop=bool(params.get("stop", False)),
            debounce=float(params.get("debounce", 0.0)),
            lifetime=params.get("lifetime"),
        )
        ctx.audit.record(
            AuditKind.COMMAND, client=session.client_id, call="watch.start",
            detail={"watch_id": watch_id, "path": path, "above": above, "below": below,
                    "outside": list(outside) if outside else None,
                    "stop": bool(params.get("stop", False))},
        )
        return {"watch_id": watch_id}

    async def watch_poll(session: Session, params: dict[str, Any]) -> Any:
        return ctx.watches.poll(str(params.get("watch_id")),
                                session.client_id)

    async def watch_stop(session: Session, params: dict[str, Any]) -> Any:
        ctx.watches.stop(str(params.get("watch_id")), session.client_id)
        return {}

    # -- latest-value sensors -------------------------------------------------------

    def _read_latest(reader: Any, what: str) -> dict[str, Any]:
        """Run a RosInterface *_reading method; map no-data to an error."""
        _require_ros(ctx)
        result = reader()
        if result is None:
            raise RpcError(
                "TimeoutError",
                f"no {what} data received yet (topic publishing? QoS?)",
            )
        return result

    async def range_read(session: Session, params: dict[str, Any]) -> Any:
        sensors = profile.spec.range_sensors or {}
        name = params.get("name")
        if not isinstance(name, str) or name not in sensors:
            raise RpcError(
                "RingfourError",
                f"unknown range sensor {name!r} (profile has "
                f"{sorted(sensors)})",
            )
        ros = _require_ros(ctx)
        return _read_latest(lambda: ros.range_reading(name),
                            f"range sensor {name!r}")

    async def imu_read(session: Session, params: dict[str, Any]) -> Any:
        ros = _require_ros(ctx)
        return _read_latest(ros.imu_reading, "imu")

    async def imu_mag(session: Session, params: dict[str, Any]) -> Any:
        if profile.spec.imu is None or not profile.spec.imu.mag_topic:
            raise RpcError("RingfourError",
                           "profile declares no magnetometer (imu.mag_topic)")
        ros = _require_ros(ctx)
        return _read_latest(ros.mag_reading, "magnetometer")

    async def arm_wrench(session: Session, params: dict[str, Any]) -> Any:
        sensors = profile.spec.force_torque or {}
        name = params.get("arm")
        if not isinstance(name, str) or name not in sensors:
            raise RpcError(
                "RingfourError",
                f"no force/torque sensor for arm {name!r} (profile has "
                f"{sorted(sensors)})",
            )
        ros = _require_ros(ctx)
        return _read_latest(lambda: ros.wrench_reading(name),
                            f"force/torque ({name})")

    async def env_read(session: Session, params: dict[str, Any]) -> Any:
        env = profile.spec.environment
        kind = params.get("kind")
        if kind not in ("temperature", "pressure", "illuminance"):
            raise RpcError(
                "RingfourError",
                "kind must be temperature | pressure | illuminance")
        if env is None or not getattr(env, kind):
            raise RpcError(
                "RingfourError",
                f"profile declares no {kind} sensor (environment.{kind})")
        ros = _require_ros(ctx)
        return _read_latest(lambda: ros.environment_reading(kind), kind)

    async def status_diagnostics(session: Session,
                                 params: dict[str, Any]) -> Any:
        status = profile.spec.status
        if status is None or not status.diagnostics:
            raise RpcError(
                "RingfourError",
                "profile declares no diagnostics topic (status.diagnostics)")
        ros = _require_ros(ctx)
        return _read_latest(ros.diagnostics_report, "diagnostics")

    # -- status / events (Phase 4) -------------------------------------------------

    async def status_get(session: Session, params: dict[str, Any]) -> Any:
        battery = None
        if (ctx.ros is not None and profile.spec.status is not None
                and profile.spec.status.battery):
            battery = ctx.ros.battery_state()
        return {"battery": battery, "estop": ctx.state.estop_engaged}

    async def events_subscribe(session: Session,
                               params: dict[str, Any]) -> Any:
        session.events_subscribed = True
        return {}

    async def events_unsubscribe(session: Session,
                                 params: dict[str, Any]) -> Any:
        session.events_subscribed = False
        return {}

    # -- teleop (Phase 3) --------------------------------------------------------------

    async def teleop_start(session: Session, params: dict[str, Any]) -> Any:
        teleop = _require_teleop(ctx)
        granted = teleop.start(
            session.client_id, float(params.get("watchdog", 0.5)))
        return {"watchdog": granted}

    async def teleop_drive(session: Session, params: dict[str, Any]) -> Any:
        teleop = _require_teleop(ctx)
        teleop.drive(
            session.client_id,
            linear=float(params.get("linear", 0.0)),
            angular=float(params.get("angular", 0.0)),
            lateral=float(params.get("lateral", 0.0)),
        )
        return {}

    async def teleop_stop(session: Session, params: dict[str, Any]) -> Any:
        report = await asyncio.to_thread(_require_teleop(ctx).stop, session.client_id)
        return {"stop_verification": report}

    async def teleop_end(session: Session, params: dict[str, Any]) -> Any:
        report = await asyncio.to_thread(_require_teleop(ctx).end, session.client_id)
        return {"stop_verification": report}

    # -- mobility: navigation (Phase 5) -----------------------------------------------

    def _resolve_goal(params: dict[str, Any]) -> dict[str, Any]:
        """A goal is a wire Pose dict or {"place": name} (spec 17 sugar)."""
        goal = params.get("goal")
        if not isinstance(goal, dict):
            raise RpcError("RingfourError",
                           "goal must be a Pose object or {place: name}")
        try:
            parsed = Pose.model_validate(goal).model_dump(mode="json")
            components = [parsed[k] for k in ("x","y","z")] + list(parsed["q"].values())
            if not all(math.isfinite(value) for value in components):
                raise RpcError("RingfourError", "navigation goal must contain finite geometry")
            return parsed
        except ValidationError as exc:
            raise RpcError("RingfourError", f"bad goal pose: {exc}") from exc

    # Nav2 error_code ranges -> NavigationFailed.reason. bt_navigator
    # propagates the failing action's code: 2xx = ComputePathToPose
    # (planning), 1xx = FollowPath (control). Anything else: "aborted".
    def _nav_reason(error_code: int) -> str:
        if 200 <= error_code < 300:
            return "no_path"
        if 100 <= error_code < 200:
            return "stuck"
        return "aborted"

    async def _cancel_navigation(ros, *, verify=True):
        failures = []
        try:
            await asyncio.to_thread(ros.nav_cancel)
        except Exception as exc:
            failures.append({"step": "cancel navigation", "error": str(exc)})
        if verify:
            try:
                stop_report=await asyncio.to_thread(body_stop)
            except Exception as exc:
                failures.append({"step": "verify body stop", "error": str(exc)})
        if failures:
            report = {"reason": "navigation cancellation unverified", "failures": failures,
                      "physical_stop": "unverified"}
            with ctx.lock.mutex: ctx.lock.quarantine["mobility"] = report
            raise RpcError("CleanupFailed", "navigation cancellation was not verified", {"cleanup": report})

        return stop_report if verify else None

    def body_stop():
        try:
            ros = _require_ros(ctx)
            ros.publish_zero()
            if not ctx.allow_unverified_stops:
                from .stopping import verify_mobility_stop
                return verify_mobility_stop(ros, ctx.profile.spec.safety.stop_verification,
                                           ctx.profile.spec.safety.feedback_max_age)
        except Exception as exc:
            report = {"reason": "body stop unverified", "error": str(exc), "physical_stop": "unverified"}
            if isinstance(exc, RpcError): report["evidence"] = exc.extra
            with ctx.lock.mutex: ctx.lock.quarantine["mobility"] = report
            raise RpcError("StopUnverified", "body stop was not verified", {"cleanup": report}) from exc

    async def navigate_to(session: Session, params: dict[str, Any]) -> Any:
        ros = _require_ros(ctx)
        goals = params.get("_route_goals") or [_resolve_goal(params)]
        goals = [{**goal,"frame":({"@map":ctx.profile.spec.frames.map,"@odom":ctx.profile.spec.frames.odom,"@base":ctx.profile.spec.frames.base}.get(goal["frame"],goal["frame"]))} for goal in goals]
        if not ctx.allow_unverified_stops:
            from .stopping import require_mobility_stop
            await asyncio.to_thread(require_mobility_stop, ros, ctx.profile.spec.safety.stop_verification,
                                    ctx.profile.spec.safety.feedback_max_age)
        acquisition = "navigation:" + uuid.uuid4().hex
        if not await asyncio.to_thread(ctx.lock.acquire, session.client_id, acquisition, resources=("adapter:mobility",)):
            raise RpcError("ResourceBusy", "robot motion belongs to another client")

        def dispatch_navigation(goal):
            with ctx.lock.mutex:
                ctx.safety.check_motion(session.client_id, "mobility.navigate_to")
                if not ctx.allow_unverified_stops:
                    from .stopping import require_mobility_stop
                    require_mobility_stop(ros, ctx.profile.spec.safety.stop_verification, ctx.profile.spec.safety.feedback_max_age)
                ros.nav_start(goal)

        async def single_goal(handle: TaskHandle, goal,index) -> dict[str, Any]:
            # Housekeeping per the nav restructure: clear both costmaps
            # before every goal (there is no behavior_server to do it).
            await asyncio.to_thread(ros.clear_costmaps)
            try:
                submission = asyncio.create_task(asyncio.to_thread(dispatch_navigation,goal))
                try:
                    await asyncio.shield(submission)
                except asyncio.CancelledError:
                    await submission
                    handle.mark_dispatched()
                    handle.record_stop(await _cancel_navigation(ros))
                    raise
                handle.mark_dispatched()
            except RpcError:
                if not ctx.allow_unverified_stops:
                    handle.record_stop(await _cancel_navigation(ros))
                raise
            except Exception as exc:
                if not ctx.allow_unverified_stops:
                    handle.record_stop(await _cancel_navigation(ros))
                raise RpcError("NavigationFailed", str(exc),
                               {"reason": "no_path"}) from exc
            try:
                last_report = 0.0
                while True:
                    outcome = await asyncio.to_thread(ros.nav_wait, 0.2)
                    if outcome is not None:
                        break
                    now = time.monotonic()
                    if now - last_report >= 0.25:
                        last_report = now
                        feedback = ros.nav_feedback()
                        if feedback is not None:
                            await handle.report(data=({**feedback,"waypoint_index":index,"waypoint_count":len(goals)} if params.get("_route_goals") else feedback))
            except asyncio.CancelledError:
                handle.record_stop(await _cancel_navigation(ros))
                raise
            except Exception:
                handle.record_stop(await _cancel_navigation(ros))
                raise
            stop_report=await asyncio.to_thread(body_stop)
            handle.record_stop(stop_report)
            if outcome["status"] == "succeeded":
                if not ctx.allow_unverified_stops:
                    from .stopping import verify_navigation_arrival
                    report = await asyncio.to_thread(verify_navigation_arrival, ros, goal,
                        ctx.profile.spec.mobility, ctx.profile.spec.safety.feedback_max_age)
                    return {"outcome": "arrived", "arrival_verification": report,"stop_verification":stop_report}
                return {"outcome": "arrived"}
            if outcome["status"] == "canceled":
                raise asyncio.CancelledError()
            reason = _nav_reason(outcome["error_code"])
            raise RpcError(
                "NavigationFailed",
                outcome["error_msg"] or f"navigation {outcome['status']}",
                {"reason": reason, "error_code": outcome["error_code"]},
            )

        async def body(handle):
            results=[]
            for index,goal in enumerate(goals):
                ctx.safety.check_motion(session.client_id,"mobility.navigate_to")
                results.append(await single_goal(handle,goal,index))
            return {"arrivals":results} if params.get("_route_goals") else results[0]

        try:
            task_id = ctx.tasks.start("mobility.start_route" if params.get("_route_goals") else "mobility.navigate_to", session, body, resource="mobility")
        except Exception:
            ctx.lock.release(session.client_id, acquisition)
            raise
        ctx.tasks._tasks[task_id].aio_task.add_done_callback(
            lambda _: ctx.lock.release(session.client_id, acquisition))
        return {"task_id": task_id}

    async def start_route(session,params):
        goals=params['goals']
        if not isinstance(goals,list) or not goals:raise RpcError('RingfourError','route requires framed poses')
        from .admission import schema
        for goal in goals:schema(('goal',),goal='pose').validate({'goal':goal})
        return await navigate_to(session,{'_route_goals':goals})

    async def mobility_stop(session: Session, params: dict[str, Any]) -> Any:
        """Immediate halt: cancel active navigation + zero the velocity.
        (resume() is dropped per Q3: re-send the goal instead.)"""
        ros = _require_ros(ctx)
        if ctx.teleop is not None:
            await asyncio.to_thread(ctx.teleop.end_all)
        cancelled = ctx.tasks.cancel_kind("mobility.")
        failures = []
        try:
            await _cancel_navigation(ros, verify=False)
        except RpcError as exc:
            failures.append(exc.extra)
        # Attempt both independent stop paths even if cancellation fails.
        try:
            stop_report = await asyncio.to_thread(body_stop)
        except Exception as exc:
            failures.append({"step": "publish_zero", "error": str(exc)})
        try:
            await ctx.tasks.drain_resource("mobility")
        except RpcError as exc:
            failures.append(exc.extra)
        if failures:
            report = {"reason": "mobility stop unverified", "failures": failures,
                      "physical_stop": "unverified"}
            with ctx.lock.mutex:
                ctx.lock.quarantine["mobility"] = report
            raise RpcError("CleanupFailed", "mobility stop could not be acknowledged", {"cleanup": report})
        ctx.audit.record(AuditKind.COMMAND, client=session.client_id,
                         call="mobility.stop",
                         detail={"cancelled": cancelled})
        return {"cancelled": cancelled, "stop_verification": stop_report}

    # -- SLAM (Phase 5) ----------------------------------------------------------------

    def _require_slam(ctx: AgentContext) -> Any:
        if ctx.slam is None:
            raise RpcError("RingfourError",
                           "agent is running without SLAM (test mode "
                           "or profile lacks slam)")
        return ctx.slam

    async def slam_get_state(session: Session,
                             params: dict[str, Any]) -> Any:
        return _require_slam(ctx).state()

    async def slam_start(session: Session, params: dict[str, Any]) -> Any:
        slam = _require_slam(ctx)
        preserve = params.get("preserve_graph", False)
        if not isinstance(preserve, bool):
            raise RpcError("RingfourError", "preserve_graph must be a boolean")
        return {"mode": await asyncio.to_thread(slam.start, preserve)}

    async def slam_stop(session: Session, params: dict[str, Any]) -> Any:
        slam = _require_slam(ctx)
        return {"mode": await asyncio.to_thread(slam.stop)}

    async def slam_save(session: Session, params: dict[str, Any]) -> Any:
        slam = _require_slam(ctx)
        name = params.get("name")
        if not isinstance(name, str):
            raise RpcError("RingfourError", "name must be a string")
        info = await asyncio.to_thread(slam.save, name)
        ctx.audit.record(AuditKind.COMMAND, client=session.client_id,
                         call="slam.save", detail={"name": name})
        return {"info": info}

    async def slam_load(session: Session, params: dict[str, Any]) -> Any:
        slam = _require_slam(ctx)
        name = params.get("name")
        if not isinstance(name, str):
            raise RpcError("RingfourError", "name must be a string")
        await asyncio.to_thread(slam.load, name)
        return {"mode": slam.mode, "active_map": slam.active_map}

    async def slam_localize(session: Session,
                            params: dict[str, Any]) -> Any:
        slam = _require_slam(ctx)
        pose = params.get("pose")
        name = params.get("name")
        await asyncio.to_thread(slam.localize, pose, name)
        return {"mode": slam.mode, "active_map": slam.active_map}

    async def slam_relocalize(session: Session,
                              params: dict[str, Any]) -> Any:
        slam = _require_slam(ctx)
        await asyncio.to_thread(slam.relocalize)
        return {"mode": slam.mode}

    async def slam_wait_for_localization(session: Session,
                                         params: dict[str, Any]) -> Any:
        slam = _require_slam(ctx)
        timeout = float(params.get("timeout", 10.0))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if slam.quality >= 0.5 and not slam.is_lost:
                return {"localization_quality": slam.quality}
            await asyncio.sleep(0.1)
        raise RpcError(
            "TimeoutError",
            f"not localized within {timeout}s "
            f"(quality {slam.quality:.2f})",
        )

    async def slam_map_info(session: Session,
                            params: dict[str, Any]) -> Any:
        slam = _require_slam(ctx)
        return {"info": slam.map_info(str(params.get("name")))}

    async def slam_delete(session: Session, params: dict[str, Any]) -> Any:
        slam = _require_slam(ctx)
        await asyncio.to_thread(slam.delete, str(params.get("name")))
        return {}

    async def slam_get_map(session: Session, params: dict[str, Any]) -> Any:
        ros = _require_ros(ctx)
        try:
            grid = await asyncio.to_thread(ros.map_grid)
        except RpcError:
            raise
        except Exception as exc:
            raise RpcError("TimeoutError", str(exc)) from exc
        data = grid.pop("data")
        if grid.get("width", 0) <= 0 or grid.get("height", 0) <= 0 or data.size == 0:
            raise RpcError("CompletionUnverified", "map publisher has not produced a nonempty occupancy grid",
                           {"reason": "map unavailable", "width": grid.get("width"), "height": grid.get("height")})
        grid["payload"] = await _payload_put(session, data, ttl=30.0)
        return grid


    # -- named places (Phase 5) ---------------------------------------------------------

    def _require_places(ctx: AgentContext) -> Any:
        if ctx.places is None:
            raise RpcError("RingfourError",
                           "profile declares no places capability")
        return ctx.places

    async def places_save(session: Session, params: dict[str, Any]) -> Any:
        places = _require_places(ctx)
        name = params.get("name")
        if not isinstance(name, str):
            raise RpcError("RingfourError", "name must be a string")
        pose = params.get("pose")
        if pose is None:
            # spec: save("dock") captures the robot's current pose
            ros = _require_ros(ctx)
            try:
                state = await asyncio.to_thread(ros.get_state)
            except Exception as exc:
                raise RpcError("RingfourError",
                               f"cannot read current pose: {exc}") from exc
            pose = state["pose"]
        else:
            try:
                pose = Pose.model_validate(pose).model_dump(mode="json")
            except ValidationError as exc:
                raise RpcError("RingfourError",
                               f"bad pose: {exc}") from exc
        await asyncio.to_thread(places.save, name, pose)
        return {"pose": pose}

    async def places_get(session: Session, params: dict[str, Any]) -> Any:
        places = _require_places(ctx)
        return {"pose": places.get(str(params.get("name")))}

    async def places_list(session: Session, params: dict[str, Any]) -> Any:
        places = _require_places(ctx)
        return {"names": places.list()}

    async def places_delete(session: Session,
                            params: dict[str, Any]) -> Any:
        places = _require_places(ctx)
        await asyncio.to_thread(places.delete, str(params.get("name")))
        return {}

    # -- debug (test plumbing, not robot API) ----------------------------------------

    async def debug_send_payload(session: Session,
                                 params: dict[str, Any]) -> Any:
        # Exercises the binary payload channel until real image methods
        # exist (Phase 4). Sends `size` random bytes, returns the payload
        # id and sha256 so the client can verify integrity.
        size = params.get("size", 1024)
        if not isinstance(size, int) or not 0 <= size <= _MAX_DEBUG_PAYLOAD:
            raise RpcError(
                "RingfourError",
                f"size must be an int in [0, {_MAX_DEBUG_PAYLOAD}]",
            )
        data = os.urandom(size)
        payload_id = await session.send_payload(
            kind="debug/random", meta={"size": size}, data=data
        )
        return {
            "payload_id": payload_id,
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    async def debug_set_estop(session: Session,
                              params: dict[str, Any]) -> Any:
        # Simulated e-stop for profiles with no hardware e-stop topic.
        engaged = params.get("engaged")
        if not isinstance(engaged, bool):
            raise RpcError("RingfourError", "engaged must be a bool")
        if not engaged and ctx.lock.quarantine:
            raise RpcError("SafetyViolation", "recover quarantined resources before releasing the inhibit",
                           {"reason": "quarantine", "resources": ctx.lock.quarantine})
        ctx.state.estop_engaged = engaged
        if engaged:
            from .halt import halt_existing
            await halt_existing(ctx, reason="operator inhibit", client=session.client_id)
        ctx.audit.record(AuditKind.ESTOP, client=session.client_id,
                         outcome=Outcome.SUCCESS,
                         detail={"engaged": engaged})
        ctx.events.emit("estop", {"engaged": engaged})
        return {"engaged": engaged}

    async def debug_run_task(session: Session,
                             params: dict[str, Any]) -> Any:
        # Exercises the task lifecycle until real long-running verbs exist:
        # ticks progress updates over `duration` seconds, cancellable,
        # optionally failing on purpose.
        duration = float(params.get("duration", 1.0))
        ticks = int(params.get("ticks", 10))
        fail = bool(params.get("fail", False))
        if not 0 < duration <= 600 or not 1 <= ticks <= 1000:
            raise RpcError("RingfourError",
                           "need 0 < duration <= 600 and 1 <= ticks <= 1000")

        async def body(handle: TaskHandle) -> dict[str, Any]:
            for tick in range(ticks):
                await asyncio.sleep(duration / ticks)
                await handle.report(progress=(tick + 1) / ticks)
            if fail:
                raise RpcError("ExecutionFailed", "deliberate test failure")
            return {"ticks": ticks}

        task_id = ctx.tasks.start("debug.run_task", session, body)
        return {"task_id": task_id}

    async def command_get(session, params):
        return ctx.commands.get(session, params['command_id'])

    async def operation_get(session, params):
        operation_id = params['operation_id']
        record = ctx.tasks._tasks.get(operation_id)
        if record is not None:
            ctx.commands.authorize(session, record.session.principal.identity, record.session.reconciliation_key)
            if record.finished_at is None:
                unknown = record.aio_task is not None and record.aio_task.done()
                return {'operation_id': operation_id, 'epoch': ctx.boot_epoch, 'state': 'unknown' if unknown else 'active',
                        'snapshot': ctx.tasks.status(operation_id, record.session),
                        'retained_until': None, 'physical_outcome': 'unknown' if unknown else 'recorded'}
        return ctx.commands.operation(session, operation_id)

    async def safety_recover(session: Session, params: dict[str, Any]) -> Any:
        if not session.principal.authenticated:
            raise RpcError("PermissionDenied", "recovery requires an authenticated local operator")
        if ctx.tasks.has_running():
            raise RpcError("ResourceBusy", "wait for accepted work and cleanup before recovery")
        if params['resource'] == 'restart-model':
            from .recovery import recover_missing_model
            report = await asyncio.to_thread(recover_missing_model, ctx)
        elif params['resource'] in ('restart', 'journal'):
            from .recovery import recover_restart
            report = await asyncio.to_thread(recover_restart, ctx, params['resource'])
        elif params['resource'] == 'mobility':
            from .recovery import recover_mobility
            report = await asyncio.to_thread(recover_mobility, ctx)
        else:
            control = _require_control(ctx)
            report = await asyncio.to_thread(control.recover, params['resource'])
        ctx.audit.record(AuditKind.SAFETY, client=session.client_id,
                         call="safety.recover", outcome=Outcome.SUCCESS,
                         detail={"principal": session.principal.identity, "recovery": report})
        return report

    async def measurement_get(session,params):
        from .measurements import bounded_resource
        return (await bounded_resource(ctx,params['kind'],params.get('group'),params.get('max_age',.5),
            params.get('timeout',1.),params.get('require_valid',True),params.get('frame'))).model_dump(mode='json')

    async def signal_get(session, params):
        from .measurements import bounded_signal
        return (await bounded_signal(ctx,params['path'],params.get('max_age',.5),
                                     params.get('timeout',1.),params.get('require_valid',True))).model_dump(mode='json')

    async def guard_open(session, params):
        return (await ctx.guards.install(session.client_id,params['condition'])).model_dump(mode='json')

    async def guard_status(session, params):
        return ctx.guards.status(params['guard_id'],session.client_id).model_dump(mode='json')

    async def guard_reset(session, params):
        return (await ctx.guards.reset(params['guard_id'],session.client_id)).model_dump(mode='json')

    async def guard_remove(session, params):
        return await ctx.guards.remove(params['guard_id'],session.client_id)

    async def set_payload_transport(session: Session, params: dict[str, Any]) -> Any:
        transport = params["transport"]
        if transport not in ("shm", "net"):
            raise RpcError("CapabilityNotSupported", f"unsupported payload transport {transport!r}")
        session.payload_transport = transport
        return {"transport": transport}

    async def renew_lease(session: Session, params: dict[str, Any]) -> Any:
        if session.lease is None:
            raise RpcError("SafetyViolation", "this connection has no lease", {"reason": "lease unavailable"})
        session.lease.renew(params["lease_id"], params["epoch"], session.principal.identity, session.client_id)
        return {"lease_id": session.lease.id, "epoch": ctx.boot_epoch, "lifetime": session.lease.lifetime}

    registry: dict[str, Handler] = {
        "lease.renew": renew_lease,
        "payload.set_transport": set_payload_transport,
        "hello": hello,
        "measurement.get":measurement_get,"signal.get":signal_get, "guard.open":guard_open,"guard.status":guard_status,
        "guard.reset":guard_reset,"guard.remove":guard_remove,
        "ping": ping,
        "safety.recover": safety_recover,
        "task.cancel": task_cancel,
        "task.status": task_status,
        "audit.query": audit_query,
        "audit.subscribe": audit_subscribe,
        "audit.unsubscribe": audit_unsubscribe,
        "debug.send_payload": debug_send_payload,
        "debug.set_estop": debug_set_estop,
        "debug.run_task": debug_run_task,
        # TF is not capability-gated: every ROS robot has a TF tree.
        "tf.lookup": tf_lookup,
        "tf.frames": tf_frames,
        "tf.wait_for": tf_wait_for,
        "tf.transform": tf_transform,
        # Events are not capability-gated: estop/watch events apply to
        # every robot.
        "events.subscribe": events_subscribe,
        "events.unsubscribe": events_unsubscribe,
    }
    if "mobility" in profile.info.capabilities:
        registry["mobility.navigate_to"] = navigate_to
        registry["mobility.start_route"] = start_route
        registry["mobility.get_state"] = mobility_get_state
        registry["mobility.stop"] = mobility_stop
    if "slam" in profile.info.capabilities:
        registry["slam.get_state"] = slam_get_state
        registry["slam.start"] = slam_start
        registry["slam.stop"] = slam_stop
        registry["slam.save"] = slam_save
        registry["slam.load"] = slam_load
        registry["slam.localize"] = slam_localize
        registry["slam.relocalize"] = slam_relocalize
        registry["slam.wait_for_localization"] = slam_wait_for_localization
        registry["slam.map_info"] = slam_map_info
        registry["slam.delete"] = slam_delete
        registry["slam.get_map"] = slam_get_map
    if "places" in profile.info.capabilities:
        registry["places.save"] = places_save
        registry["places.get"] = places_get
        registry["places.list"] = places_list
        registry["places.delete"] = places_delete
    if "teleop" in profile.info.capabilities:
        registry["teleop.start"] = teleop_start
        registry["teleop.drive"] = teleop_drive
        registry["teleop.stop"] = teleop_stop
        registry["teleop.end"] = teleop_end
    if "cameras" in profile.info.capabilities:
        registry["camera.info"] = camera_info
        registry["camera.get_frame"] = camera_get_frame
        registry["camera.get_depth"] = camera_get_depth
        registry["camera.get_synced"] = camera_get_synced
        registry["camera.get_cloud"] = camera_get_cloud
        registry["camera.deproject"] = camera_deproject
        registry["camera.stream_start"] = camera_stream_start
        registry["camera.stream_stop"] = camera_stream_stop
    if "lidar" in profile.info.capabilities:
        registry["lidar.scan"] = lidar_scan
        registry["lidar.stream_start"] = lidar_stream_start
        registry["lidar.stream_stop"] = lidar_stream_stop
    registry["model.description"] = model_description
    if "sensors" in profile.info.capabilities:
        registry["sensor.read"] = sensor_read
    if "actions" in profile.info.capabilities:
        registry["action.invoke"] = action_invoke
    if "groups" in profile.info.capabilities:
        registry["arm.joint_states"] = arm_joint_states
        registry["group.control_start"] = group_control_start
        registry["group.control_stop"] = group_control_stop
        registry["group.start_trajectory"] = group_start_trajectory
        registry["group.hold"] = group_hold
        registry["group.stop"] = group_stop
        registry["group.start_tool"] = group_start_tool
        registry["group.tool_state"] = group_tool_state
    if "range_sensors" in profile.info.capabilities:
        registry["range.read"] = range_read
    if "imu" in profile.info.capabilities:
        registry["imu.read"] = imu_read
        registry["imu.mag"] = imu_mag
    if "force_torque" in profile.info.capabilities:
        registry["arm.wrench"] = arm_wrench
    if "environment" in profile.info.capabilities:
        registry["env.read"] = env_read
    if "watches" in profile.info.capabilities:
        registry["watch.start"] = watch_start
        registry["watch.poll"] = watch_poll
        registry["watch.stop"] = watch_stop
    if "status" in profile.info.capabilities:
        registry["status.get"] = status_get
        registry["status.diagnostics"] = status_diagnostics
    registry['commands.get'] = command_get
    registry['operations.get'] = operation_get
    return registry
