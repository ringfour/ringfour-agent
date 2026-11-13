"""Wires a loaded profile into a ready-to-start AgentServer.

Used by main.py and by the engine's tests (which run the same stack
in-process, minus rclpy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .audit import AuditLog
from .context import AgentContext
from .events import EventHub
from .control import ControlManager
from .groups import resolve_groups
from .vendor import VendorActions
from .handlers import build_registry
from .profile import Profile
from .safety import AgentState, MotionLock, SafetyLayer
from .sensing import FrameCache
from .server import AgentServer
from .places import PlaceStore
from .shm import ShmStore
from .slam import SlamManager
from .state_paths import build_path_table
from .streams import StreamManager
from .tasks import TaskManager
from .teleop import TeleopManager
from .watches import WatchRegistry


def build_agent(
    profile: Profile,
    *,
    unix_path: str | None,
    host: str | None = None,
    port: int | None = None,
    audit_dir: str | Path | None = None,
    ros: Any = None,
    access_policy=None,
    allow_unverified_stops=False,
    state_dir=None,
) -> tuple[AgentServer, AgentContext]:
    """Assemble audit + safety + tasks + handlers + server for one profile.

    ``audit_dir`` overrides the profile's audit directory (tests use a tmp
    dir). ``ros`` is the RosInterface (main.py) or a fake (tests); None
    runs the protocol stack without a robot. Call ``await server.start()``
    on the returned server; call ``ctx.audit.close()`` after shutdown.
    """
    # Before anything is wired: a group that does not match the robot is
    # a startup failure, not a runtime surprise (spec 8.6). Raises
    # GroupValidationError, which main.py turns into exit 1.
    groups = resolve_groups(profile.spec, ros)

    audit = AuditLog(profile.info.name, profile.spec.audit,
                     directory=audit_dir)
    state = AgentState()
    lock = MotionLock()
    if state_dir is not None:
        from .durability import DurableQuarantine
        revision = ros.description_sha() if ros is not None and hasattr(ros, 'description_sha') else None
        lock.quarantine = DurableQuarantine(state_dir, profile.info.name, revision)
        if lock.quarantine:
            state.estop_engaged = True
    safety = SafetyLayer(state, lock)
    if ros is not None:
        def check_ros_dispatch():
            if lock.owner is None:
                from .server import RpcError
                raise RpcError("SafetyViolation", "ROS motion dispatch requires an acquisition")
            safety.check_motion(lock.owner, "ROS action dispatch")
        ros.check_motion_dispatch = check_ros_dispatch
    teleop = None
    if "teleop" in profile.info.capabilities and ros is not None:
        teleop = TeleopManager(lock, audit, ros, profile, safety, allow_unverified_stops=allow_unverified_stops)
    ctx = AgentContext(
        profile=profile,
        allow_unverified_stops=allow_unverified_stops,
        audit=audit,
        tasks=TaskManager(audit),
        safety=safety,
        state=state,
        lock=lock,
        watches=WatchRegistry(),
        events=EventHub(),
        frames=FrameCache(),
        streams=StreamManager(),
        ros=ros,
        teleop=teleop,
        shm=ShmStore(profile.info.name),
        groups=groups,
    )
    from .commands import CommandLedger
    import hashlib
    def journal_failure(error):
        state.estop_engaged = True
        lock.quarantine['journal'] = {'reason': 'durable command storage failure', 'error': error}
        # Storage failure revokes admission immediately and initiates the same
        # bounded halt used by an operator inhibit, without blocking dispatch.
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # Startup has no accepted work; the boot will fail closed.
        ctx.tasks.cancel_running()
        async def halt_after_storage_failure():
            from .halt import halt_existing
            from .server import RpcError
            try:
                await halt_existing(ctx, reason='durable command storage failure')
            except RpcError:
                pass  # Retained quarantine truthfully prevents confirmed recovery.
        halt = loop.create_task(halt_after_storage_failure())
        ctx.background_halts.add(halt)
        halt.add_done_callback(ctx.background_halts.discard)
    journal_path = Path(state_dir) / (hashlib.sha256(profile.info.name.encode()).hexdigest()+'.sqlite3') if state_dir is not None else None
    ctx.commands = CommandLedger(ctx.boot_epoch, path=journal_path, on_failure=journal_failure, audit=audit)
    ctx.tasks.ledger = ctx.commands
    if ros is not None and groups:
        ctx.control = ControlManager(lock, audit, ros, groups, ctx.safety, stop_policy=profile.spec.safety.stop_verification, allow_unverified_stops=allow_unverified_stops)
    if ros is not None and profile.spec.actions:
        ctx.actions = VendorActions(ros, profile.spec.actions, audit,
                                    ctx.safety, lock, profile=profile, groups=groups, allow_unverified_stops=allow_unverified_stops)
    if ros is not None and "slam" in profile.info.capabilities:
        ctx.slam = SlamManager(ros, profile.spec.slam.map_dir)
    ctx.watch_paths = build_path_table(ctx) if ros is not None else {}
    from .guards import GuardRegistry
    ctx.guards = GuardRegistry(ctx)
    server = AgentServer(
        registry=build_registry(ctx),
        ctx=ctx,
        unix_path=unix_path,
        host=host,
        port=port,
        access_policy=access_policy,
    )
    if ctx.slam is not None and "places" in profile.info.capabilities:
        ctx.places = PlaceStore(ctx.slam)
    return server, ctx
