"""Control sessions: nothing moves outside one (spec 8.2).

A session resolves a `Controller` role to the profile's controller name,
takes the motion lock for that group, brings the controller up, and hands
back a handle. Leaving it puts the robot back exactly as it was found.

**Teardown is the point of this module.** Entry is four service calls and
they are easy; the interesting half is that every one of them has to be
undone, including on the paths the client does not control - a crashed
mission, a dropped WiFi link, a laptop that went to sleep. Spec 8.2 is
explicit: the restore belongs in the agent's disconnect hook, not in the
caller's ``with`` block, because a ``with`` block cannot run when the
process holding it is gone.

The state that must not be left behind (proved reachable by hand on
2026-08-23, and again on 2026-08-24):

    switch --deactivate arm_controller --activate arm_trajectory_controller
    <mission dies here>
    -> arm on the trajectory controller, position controller inactive,
       motion lock released. The next client commands a controller
       nobody is driving, and nothing reports an error.

So `end()` is idempotent, runs in reverse order, and keeps going when a
step fails - a half-completed teardown is worse than a noisy one.

**One switch, on entry, and nowhere else.** No autoswitching (spec 8.2):
that removes mid-call switching, the `is_moving` guard, cross-group
deactivation hazards and the controller-manager lock, all by construction.
`raw=` exists for vendors whose controllers fall outside the role
vocabulary; it bypasses resolution, is audited, and is not portable.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import OrderedDict
from functools import wraps
from dataclasses import dataclass, field
from typing import Any

from ringfour.enums import AuditKind, Controller, Outcome

from .audit import AuditLog
from .groups import GroupRuntime
from .safety import MotionLock
from .server import RpcError

log = logging.getLogger("ringfour_agent.control")


def synchronized(method):
    """Serialize controller lifecycle mutations across dispatcher worker threads."""
    @wraps(method)
    def run(self, *args, **kwargs):
        with self._lifecycle_mutex:
            return method(self, *args, **kwargs)
    return run


@dataclass
class Session:
    """One open control session.

    ``loaded_here`` and ``deactivated`` are the undo list. Each is
    recorded before adapter dispatch, covering effects whose acknowledgements
    were lost. Entry failure restores the measured prior controller states.
    """

    client_id: int
    group: str
    role: Controller | None          # None for a raw= session
    controller: str
    # Controllers this session brought up, to unload on exit. Empty when
    # the controller was already loaded - someone else's, not ours to
    # remove.
    loaded_here: list[str] = field(default_factory=list)
    # Controllers this session deactivated, to put back on exit.
    deactivated: list[str] = field(default_factory=list)
    opened_at: float = field(default_factory=time.monotonic)
    # True while a trajectory is in flight, so stop() and teardown know to
    # cancel before switching anything.
    running: bool = False
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    activated_here: bool = False
    configured_here: bool = False
    prior_states: dict = field(default_factory=dict)
    model_revision: str | None = None


class ControlManager:
    """Owns every open control session (one per group, one lock overall)."""

    def __init__(self, lock: MotionLock, audit: AuditLog, ros: Any,
                 groups: dict[str, GroupRuntime], safety=None, *, stop_policy=None, allow_unverified_stops=False) -> None:
        from threading import RLock
        self._lifecycle_mutex = RLock()
        self._lock = lock
        self._audit = audit
        self._ros = ros
        self._groups = groups
        self._safety = safety
        self._stop_policy = stop_policy
        self.allow_unverified_stops = allow_unverified_stops
        self._sessions: dict[str, Session] = {}
        self._closed: OrderedDict[str, dict] = OrderedDict()

    def model_revision(self, group):
        from ringfour.description import sha_of
        if hasattr(self._ros,'group_description'):
            xml=self._ros.group_description(group.name)
            if xml:return sha_of(xml)
        return self._ros.description_sha() if hasattr(self._ros,'description_sha') else None

    def stop_policy(self, group):
        return group.config.stop_verification or self._stop_policy

    def require_stop(self, group, names=None):
        if self.allow_unverified_stops and self.stop_policy(group) is None:
            return
        from .stopping import require_joint_stop
        require_joint_stop(self._ros, names or group.joints, self.stop_policy(group), group.config.feedback_max_age)

    # -- lookup ---------------------------------------------------------------

    def group(self, name: Any) -> GroupRuntime:
        if not isinstance(name, str) or name not in self._groups:
            raise RpcError(
                "RingfourError",
                f"unknown group {name!r} (robot has {sorted(self._groups)})")
        return self._groups[name]

    @synchronized
    def session(self, group: str, client_id: int, session_id=None) -> Session:
        current = self._sessions.get(group)
        if current is None:
            raise RpcError(
                "RingfourError",
                f"no control session open on group {group!r}; motion only "
                f"happens inside one (spec 8.2)")
        if current.client_id != client_id:
            raise RpcError(
                "SafetyViolation",
                f"group {group!r} is held by client {current.client_id}",
                {"reason": "motion lock held"})
        if session_id is not None and session_id != current.id:
            raise RpcError("SafetyViolation", "control session is stale",
                           {"reason": "stale session", "group": group})
        return current

    def active(self) -> dict[str, Session]:
        return dict(self._sessions)

    @synchronized
    def prepare_close(self, group, client_id, session_id):
        if session_id in self._closed:
            previous = self._closed[session_id]
            if previous["client_id"] != client_id or previous["group"] != group:
                raise RpcError("SafetyViolation", "closed session belongs to another owner")
            return None
        return self.session(group, client_id, session_id)

    @synchronized
    def reserve_trajectory(self, group, client_id, session_id):
        current = self.session(group, client_id, session_id)
        if any(s.running for s in self._sessions.values()):
            raise RpcError("ResourceBusy", "the integration supports one active trajectory at a time",
                           {"reason": "shared trajectory adapter"})
        current.running = True
        return current

    @synchronized
    def dispatch(self, current, action):
        """Revalidate authority at dispatch, atomic with session revocation."""
        self.session(current.group, current.client_id, current.id)
        if self._safety is not None:
            self._safety.check_motion(current.client_id, "group motion dispatch")
        group=self._groups[current.group]
        if current.model_revision is not None and current.model_revision!=self.model_revision(group):
            raise RpcError("ModelChanged", "control session belongs to an old robot model")
        self.require_stop(group)
        return action()

    # -- entry -----------------------------------------------------------------

    @synchronized
    def start(self, client_id: int, group_name: str,
              role: Controller | None = None,
              raw: str | None = None) -> Session:
        """Open a session: resolve, lock, load, configure, switch.

        Raises RpcError naming the resolved controller. Anything done
        before a failure is undone before the raise.
        """
        group = self.group(group_name)
        if self._safety is not None:
            self._safety.check_motion(client_id, "group.control_start")
        self.require_stop(group)
        if (role is None) == (raw is None):
            raise RpcError("RingfourError",
                           "give exactly one of role= or raw=")

        if raw is not None:
            controller = raw.lstrip("/")
            if not self.allow_unverified_stops:
                matches = [r for r,c in group.controllers.items() if c == controller]
                if len(matches) != 1:
                    raise RpcError("StopUnverified", "raw controller has no declared command/stop role")
                role = matches[0]
        else:
            controller = group.controllers.get(role)
            if controller is None:
                raise RpcError(
                    "CapabilityNotSupported",
                    f"group {group_name!r} declares no {role} controller "
                    f"(it has {sorted(str(r) for r in group.controllers)})")

        if group_name in self._sessions:
            holder = self._sessions[group_name].client_id
            raise RpcError(
                "SafetyViolation",
                f"group {group_name!r} already has a control session "
                f"(client {holder})",
                {"reason": "motion lock held"})
        for current in self._sessions.values():
            if current.controller == controller or set(group.joints) & set(self._groups[current.group].joints):
                raise RpcError("ResourceBusy", "control scopes share a physical resource",
                               {"reason": "shared resource", "groups": [group_name, current.group]})
        if role is Controller.TOOL:
            raise RpcError("CapabilityNotSupported", "tool controllers use the separate tool joint API")
        session = Session(client_id=client_id, group=group_name, role=role,
                          controller=controller,model_revision=self.model_revision(group))
        footprint = [f"joint:{joint}" for joint in group.joints] + [f"controller:{controller}"]
        if not self._lock.acquire(client_id, session.id, resources=footprint,
                                  exclusive=raw is not None and controller not in group.controllers.values()):
            raise RpcError(
                "SafetyViolation",
                f"motion lock held by client {self._lock.owner}",
                {"reason": "motion lock held"})

        try:
            self._bring_up(session, group)
            self._check_entry(session)
        except Exception as exc:
            report = self._unwind(session)
            self._lock.release(client_id, session.id)
            if not report["confirmed"]:
                if isinstance(exc, RpcError):
                    exc.extra["cleanup"] = report
                else:
                    exc.add_note(f"Cleanup failed: {report}")
            if isinstance(exc, RpcError):
                raise
            raise RpcError("ControllerError", f"could not acquire control of {group_name!r}: {exc}",
                           {"cleanup": report, "controller": controller}) from exc

        self._sessions[group_name] = session
        self._audit.record(
            AuditKind.COMMAND, client=client_id, call="group.control",
            outcome=Outcome.SUCCESS,
            detail={"group": group_name, "role": str(role) if role else "raw",
                    "controller": controller,
                    "loaded": list(session.loaded_here),
                    "deactivated": list(session.deactivated)})
        log.info("control session open: %s -> %s (loaded=%s, deactivated=%s)",
                 group_name, controller, session.loaded_here or "-",
                 session.deactivated or "-")
        return session

    def _check_entry(self, session):
        if self._safety is not None:
            self._safety.check_motion(session.client_id, "controller acquisition dispatch")

    def _bring_up(self, session: Session, group: GroupRuntime) -> None:
        """load -> configure -> switch, skipping steps already done."""
        controller = session.controller
        states = self._ros.loaded_controllers()
        session.prior_states = {k: v["state"] for k,v in states.items()}

        if controller not in states:
            # A service may apply a change before losing its reply. Record
            # attempted effects before dispatch so rollback covers uncertainty.
            session.loaded_here.append(controller)
            self._check_entry(session)
            self._ros.load_controller(controller)
            self._check_entry(session)
            self._ros.configure_controller(controller)
        elif states[controller]["state"] == "unconfigured":
            session.configured_here = True
            self._check_entry(session)
            self._ros.configure_controller(controller)

        if states.get(controller, {}).get("state") == "active":
            self._verify_raw_footprint(session, group, states)
            return                       # already active; no switch needed

        # Anything active that claims a joint we need has to go. On an
        # exclusive pair that is the twin. The tool controller claims a
        # disjoint joint and is deliberately left alone - which is what
        # lets tool.close() work during an arm session without a switch.
        wanted = set(group.joints)
        conflicting = [
            name for name, state in states.items()
            if state["state"] == "active" and wanted & set(state["joints"])
        ]
        occupied = {s.controller for s in self._sessions.values()}
        if occupied & set(conflicting):
            raise RpcError("ResourceBusy", "acquisition would deactivate another scope's controller",
                           {"reason": "shared controller footprint"})
        self._check_entry(session)
        session.activated_here = True
        session.deactivated.extend(conflicting)
        self._ros.switch_controllers(activate=[controller],
                                     deactivate=conflicting)
        if session.role is None:
            self._verify_raw_footprint(session, group, self._ros.loaded_controllers())

    def _verify_raw_footprint(self, session, group, states):
        if session.role is not None:
            return
        claims = set(states.get(session.controller, {}).get("joints", ()))
        if claims != set(group.joints):
            raise RpcError("SafetyViolation", "raw controller claims do not match the resolved group",
                           {"reason": "raw controller footprint", "controller": session.controller,
                            "expected_joints": list(group.joints), "claimed_joints": sorted(claims)})

    # -- exit ------------------------------------------------------------------

    @synchronized
    def end(self, group_name: str, *, reason: str = "closed",
            client_id: int | None = None, session_id: str | None = None) -> dict | None:
        """Restore and release. Idempotent, and safe to call twice."""
        if session_id is not None and session_id in self._closed:
            previous = self._closed[session_id]
            if previous["client_id"] != client_id or previous["group"] != group_name:
                raise RpcError("SafetyViolation", "closed session belongs to another owner")
            report = previous["report"]
            if not report["confirmed"]:
                raise RpcError("CleanupFailed", "previous cleanup failed", {"cleanup": report})
            return report
        current = self._sessions.get(group_name)
        if current is not None and client_id is not None and current.client_id != client_id:
            raise RpcError("SafetyViolation", "cannot release another client's control scope")
        if session_id is not None and (current is None or current.id != session_id):
            raise RpcError("SafetyViolation", "control session is stale", {"reason": "stale session"})
        session = self._sessions.pop(group_name, None)
        if session is None:
            return
        report = self._unwind(session)
        self._closed[session.id] = {"client_id": session.client_id, "group": group_name, "report": report}
        if len(self._closed) > 1024:
            self._closed.popitem(last=False)
        self._lock.release(session.client_id, session.id)
        self._audit.record(
            AuditKind.COMMAND, client=client_id or session.client_id,
            call="group.control_stop", outcome=Outcome.SUCCESS if report["confirmed"] else Outcome.ERROR,
            detail={"group": group_name, "controller": session.controller,
                    "reason": reason, "cleanup": report,
                    "held_for": round(time.monotonic() - session.opened_at,
                                      3)})
        if not report["confirmed"]:
            raise RpcError("CleanupFailed", f"control cleanup for {group_name!r} could not be verified",
                           {"cleanup": report, "group": group_name})
        log.info("control session closed: %s (%s)", group_name, reason)
        return report

    def _request_stop(self, session):
        group = self._groups[session.group]
        if session.role is Controller.TRAJECTORY or session.role is None:
            if session.running:
                self._ros.traj_cancel()
        elif session.role is Controller.VELOCITY:
            self._ros.command_positions(session.controller, list(group.joints), [0.] * len(group.joints))
        else:
            snapshot = self._ros.verified_joint_snapshot(group.joints, group.config.feedback_max_age)
            positions = [snapshot[n][0] for n in group.joints]
            if not all(math.isfinite(v) for v in positions):
                raise RpcError("StopUnverified", "holding requires finite current positions")
            self._ros.command_positions(session.controller, list(group.joints), positions)

    def _unwind(self, session: Session) -> dict:
        """Continue every undo step, verify controller state, retain failures."""
        steps = []
        def attempt(name, action):
            try:
                action()
                steps.append({"step": name, "status": "confirmed"})
            except Exception as exc:
                steps.append({"step": name, "status": "failed", "error": str(exc)})
                log.exception("teardown step failed: %s", name)
        if session.running or (not self.allow_unverified_stops and session.activated_here):
            attempt("request controller stop", lambda: self._request_stop(session))
            session.running = False
        group = self._groups[session.group]
        stop_report = None
        if self.stop_policy(group) is not None or not self.allow_unverified_stops:
            def verify_stop():
                nonlocal stop_report
                from .stopping import verify_joint_stop
                stop_report = verify_joint_stop(self._ros, group.joints,
                    self.stop_policy(group), group.config.feedback_max_age)
            attempt("verify measured joint stop", verify_stop)
        if session.activated_here:
            attempt("restore controllers", lambda: self._ros.switch_controllers(
                activate=list(session.deactivated), deactivate=[session.controller]))
        for controller in reversed(session.loaded_here):
            attempt(f"unload {controller}", lambda c=controller: self._ros.unload_controller(c))
        if session.configured_here:
            attempt("restore unconfigured controller", lambda: self._ros.unload_controller(session.controller))
            attempt("reload unconfigured controller", lambda: self._ros.load_controller(session.controller))
        def verify():
            states = self._ros.loaded_controllers()
            names = set(session.deactivated) | {session.controller}
            wrong = {name: {"expected": session.prior_states.get(name),
                            "actual": states.get(name, {}).get("state")}
                     for name in names
                     if states.get(name, {}).get("state") != session.prior_states.get(name)}
            if wrong:
                raise RuntimeError(f"controller restoration differs: {wrong}")
        attempt("read back controller states", verify)
        before_restore_stop = stop_report
        if self.stop_policy(group) is not None or not self.allow_unverified_stops:
            attempt("verify restored measured joint stop", verify_stop)
        if session.group in self._lock.quarantine:
            steps.append({"step": "prior operation cleanup", "status": "failed",
                          "error": self._lock.quarantine[session.group]})
        report = {"group": session.group, "session_id": session.id,
                  "confirmed": all(s["status"] == "confirmed" for s in steps),
                  "steps": steps,
                  "stop_verification": stop_report,
                  "pre_restore_stop_verification": before_restore_stop,
                  "expected_controllers": {n: session.prior_states.get(n) for n in set(session.deactivated) | {session.controller}},
                  "physical_stop": "uncommissioned" if stop_report and stop_report.get("verified") and before_restore_stop and before_restore_stop.get("verified") else "unverified"}
        if not report["confirmed"]:
            self._lock.quarantine[session.group] = report
        return report

    @synchronized
    def stop_work(self, current):
        """Stop this adapter reservation and verify any declared measured criterion."""
        self.session(current.group, current.client_id, current.id)
        group = self._groups[current.group]
        try:
            self._request_stop(current)
            if self.stop_policy(group) is not None or not self.allow_unverified_stops:
                from .stopping import verify_joint_stop
                return verify_joint_stop(self._ros, group.joints,
                    self.stop_policy(group), group.config.feedback_max_age)
            return None  # legacy acknowledgement-only policy, never verified rest
        except Exception as exc:
            report = {"reason": "group stop unverified", "error": str(exc),
                      "physical_stop": "unverified"}
            if isinstance(exc, RpcError): report["evidence"] = exc.extra
            self._lock.quarantine[current.group] = report
            if isinstance(exc, RpcError): raise
            raise RpcError("CleanupFailed", "trajectory cancellation could not be acknowledged",
                           {"group": current.group, "cleanup": report}) from exc

    @synchronized
    def recover(self, group_name):
        """Privileged caller supplies no evidence; reverify integration state."""
        tool = group_name.startswith('tool:')
        group = self.group(group_name.removeprefix('tool:') if tool else group_name)
        names = (group.tool_joint,) if tool else group.joints
        if tool and group.tool_joint is None:
            raise RpcError('CapabilityNotSupported','group declares no tool')
        prior = self._lock.quarantine.get(group_name)
        if prior is None:
            raise RpcError("SafetyViolation", "resource is not quarantined")
        if self._lock.owner is not None or self._sessions:
            raise RpcError("ResourceBusy", "close all motion acquisitions before recovery")
        if self._safety is None or not self._safety._state.estop_engaged:
            raise RpcError("SafetyViolation", "engage the inhibit before recovery", {"reason": "recovery requires inhibit"})
        if self.stop_policy(group) is None:
            raise RpcError("StopUnverified", "recovery requires a reviewed measured-stop policy")
        expected = prior.get("expected_controllers")
        if not isinstance(expected, dict) or not expected:
            raise RpcError("CleanupFailed", "quarantine lacks a restorable controller checkpoint")
        try:
            self._ros.traj_cancel()
            def check_states():
                actual = self._ros.loaded_controllers()
                mismatches = {n: {"expected": state, "actual": actual.get(n, {}).get("state")}
                              for n, state in expected.items() if actual.get(n, {}).get("state") != state}
                if mismatches:
                    raise RpcError("CleanupFailed", "controller state is not restored", {"controllers": mismatches})
            check_states()
            from .stopping import verify_joint_stop
            stopped = verify_joint_stop(self._ros, names,
                self.stop_policy(group), group.config.feedback_max_age)
            check_states()
        except RpcError:
            raise
        except Exception as exc:
            raise RpcError("CleanupFailed", "recovery state verification failed", {"error": str(exc)}) from exc
        if not self._safety._state.estop_engaged:
            raise RpcError("SafetyViolation", "inhibit changed during recovery")
        del self._lock.quarantine[group_name]
        return {"resource": group_name, "recovered": True,
                "stop_verification": stopped, "inhibit_engaged": True,
                "physical_stop": "uncommissioned"}

    # -- the crash path --------------------------------------------------------

    def on_disconnect(self, client_id: int) -> int:
        """A vanished client must not leave the arm on a controller nobody
        is driving. Returns how many sessions were closed."""
        stale = [name for name, s in self._sessions.items()
                 if s.client_id == client_id]
        for name in stale:
            try:
                self.end(name, reason="disconnect", client_id=client_id)
            except RpcError:
                log.exception("disconnect cleanup left %s quarantined", name)
                continue
            self._audit.record(
                AuditKind.SAFETY, client=client_id, call="group.control",
                outcome=Outcome.SUCCESS,
                detail={"reason": "disconnect", "group": name,
                        "message": "session closed, controller restored"})
        return len(stale)

    def end_all(self, reason: str = "shutdown") -> int:
        count = 0
        for name in list(self._sessions):
            try:
                self.end(name, reason=reason)
            except RpcError:
                log.exception("shutdown cleanup left %s quarantined", name)
            count += 1
        return count


def arrival_residual(joints: "list[str]", goal: "list[float]",
                     snapshot: "dict[str, tuple]") -> "dict[str, float]":
    """Final joint minus commanded, per joint, for a finished trajectory.

    Pulled out of the handler so it can be tested without a robot: the
    handler around it is all ROS and asyncio, and this is the part with
    an answer that can be wrong.

    Joints missing from the snapshot are omitted rather than assumed to
    be at zero — no reading is not the same as a reading of no error.
    """
    return {
        joint: float(snapshot[joint][0]) - float(want)
        for joint, want in zip(joints, goal)
        if joint in snapshot
    }


def worst_joint(residual: "dict[str, float]") -> "tuple[str | None, float]":
    """The joint furthest from its target, and by how much (absolute).

    ("", 0.0) for an empty residual, which happens when no joint in the
    group reported state at all — a different problem, and not one to
    report as a perfect arrival.
    """
    if not residual:
        return None, 0.0
    name = max(residual, key=lambda j: abs(residual[j]))
    return name, abs(residual[name])


def checked_arrival_residual(joints, goal, snapshot):
    """Arrival needs actual finite feedback for every commanded joint."""
    import math
    missing = [name for name in joints if name not in snapshot]
    invalid = [name for name in joints if name in snapshot and not math.isfinite(snapshot[name][0])]
    if missing or invalid:
        raise ValueError(f'Cannot verify trajectory arrival: missing positions {missing}, non-finite positions {invalid}')
    return arrival_residual(joints, goal, snapshot)


def arrival_failure_message(joint, gap, tolerance, joint_type=None):
    import math
    unit = 'm' if joint_type == 'prismatic' else 'rad' if joint_type in ('revolute','continuous') else 'joint units'
    measured = f'{gap:.3f} {unit}'
    if unit == 'rad':measured += f' ({math.degrees(gap):.1f} deg)'
    return (f'trajectory finished with {joint} {measured} from its final waypoint, '
            f'over the {tolerance:.3f} {unit} goal tolerance. The controller '
            'completed its schedule; the joint did not keep up. Allow more '
            'time (a larger dt), command a smaller move, and check feasibility.')
