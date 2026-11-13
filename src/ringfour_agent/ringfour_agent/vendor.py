"""Vendor escape hatches: any topic, bound in the profile (sensors/actions).

The capability registry covers what most robots have — a battery, a lidar,
an IMU. It cannot cover what one vendor decided to put on a topic: servo
temperature, a bus voltage, a front LED, an outrigger. This is where those
live, declared by name in the profile and reached as
``robot.sensors.<group>.<signal>`` and ``robot.actions.<name>()``.

**Read and write are not symmetric**, which is why they are two sections
rather than one with a direction flag:

* a **sensor** is subscribed, cached, and registered as a watch path. It
  is free, idempotent, and safe to expose to anyone;
* an **action** publishes. In this system that means audited, rate
  limited, and — if the profile says so — holding the motion lock and
  refusing under e-stop, like every other thing that moves a robot.

No per-vendor code is needed either way. ``get_message`` resolves a type
name at runtime (how ``ros2 topic echo`` handles a type it has never
seen) and ``set_message_fields`` builds one from a dict (how ``ros2 topic
pub`` does). The profile supplies the names; rclpy supplies the rest.
"""

from __future__ import annotations

import logging
import time
import math
from typing import Any

from ringfour.enums import AuditKind, Outcome
from ringfour_agent.models.profile import ActionConfig, SensorSignal

from .server import RpcError

log = logging.getLogger("ringfour_agent.vendor")

# The field a single-value std_msgs carries. Declaring `field:` overrides
# it; almost no telemetry needs to.
DEFAULT_FIELD = "data"


class VendorSensors:
    """Every declared sensor signal, subscribed and cached."""

    def __init__(self, node: Any, groups: dict[str, dict[str, SensorSignal]],
                 qos: Any) -> None:
        self._values: dict[tuple[str, str], tuple[float, float]] = {}
        self._groups = groups
        self._failed: dict[str, str] = {}
        for group, signals in groups.items():
            for name, signal in signals.items():
                self._subscribe(node, group, name, signal, qos)

    def _subscribe(self, node: Any, group: str, name: str,
                   signal: SensorSignal, qos: Any) -> None:
        try:
            from rosidl_runtime_py.utilities import get_message

            message_type = get_message(_ros_type(signal.type))
        except Exception as exc:
            # A bad type name is a profile error, but not one worth
            # refusing to boot over: the rest of the robot still works,
            # and the signal reports itself as unavailable.
            self._failed[f"{group}.{name}"] = str(exc)
            log.warning("sensors.%s.%s: cannot resolve type %r (%s)",
                        group, name, signal.type, exc)
            return

        field = signal.field or DEFAULT_FIELD

        def store(message: Any, key: tuple[str, str] = (group, name),
                  path: str = field) -> None:
            value = _dig(message, path)
            if value is not None:
                self._values[key] = (float(value), time.time())

        node.create_subscription(message_type, signal.topic, store, qos)
        log.info("  sensors.%s.%s <- %s (%s)", group, name, signal.topic,
                 signal.type)

    def read(self, group: str, name: str) -> dict[str, Any] | None:
        """{value, stamp, units} or None before the first message."""
        signal = self._groups.get(group, {}).get(name)
        if signal is None:
            return None
        found = self._values.get((group, name))
        if found is None:
            return None
        value, stamp = found
        return {"value": value, "stamp": stamp, "units": signal.units}

    def value(self, group: str, name: str) -> float | None:
        """Just the number, for the watch sampler."""
        found = self._values.get((group, name))
        return None if found is None else found[0]

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            group: {name: self.read(group, name) for name in signals}
            for group, signals in self._groups.items()
        }

    @property
    def unavailable(self) -> dict[str, str]:
        """Signals whose type could not be resolved, and why."""
        return dict(self._failed)


class VendorActions:
    """Every declared action, as a publisher plus its guard rails."""

    def __init__(self, ros: Any, actions: dict[str, ActionConfig],
                 audit: Any, safety: Any, lock: Any, *, profile=None, groups=None, allow_unverified_stops=False) -> None:
        # Policy lives here; the publishers live in RosInterface with
        # every other publisher, created at construction so the first
        # message is not lost to DDS matching (found_along_the_way 11).
        self._profile = profile
        self._groups = groups or {}
        self.allow_unverified_stops = allow_unverified_stops
        self._held = {}
        self._stopping = set()
        self._ros = ros
        self._actions = actions
        self._audit = audit
        self._safety = safety
        self._lock = lock
        self._last: dict[str, float] = {}

    def invoke(self, name: str, arguments: dict[str, Any],
               client_id: int) -> dict[str, Any]:
        return self._invoke(name, arguments, client_id)

    def _invoke(self, name, arguments, client_id):
        action = self._actions.get(name)
        if action is None:
            raise RpcError(
                "RingfourError",
                f"unknown action {name!r} (this robot has "
                f"{sorted(self._actions)})")
        if not self._ros.vendor_ready(name):
            raise RpcError(
                "CapabilityNotSupported",
                f"actions.{name} declares type {action.type!r}, which this "
                f"agent could not resolve")

        # 2. Rate limit. An arbitrary topic is otherwise a hole straight
        #    through to the robot's DDS.
        now = time.monotonic()
        gap = now - self._last.get(name, 0.0)
        if action.max_rate > 0 and gap < 1.0 / action.max_rate:
            raise RpcError(
                "RingfourError",
                f"actions.{name} is limited to {action.max_rate:g} Hz "
                f"(last call {gap * 1000:.0f} ms ago)")

        payload = self._build(name, action, arguments)
        token = "action:" + name
        held_before = self._lock.owner == client_id and token in self._lock._acquisitions
        # 1. Safety, if the profile says this moves the robot.
        if action.is_motion:
            # Dynamic effect comes from the validated profile; use the
            # same motion gate as built-in RPC descriptors.
            self._safety.check_motion(client_id, f"actions.{name}")
            if not self.allow_unverified_stops:
                self._contract(name, action, verify=False)
            if not self._lock.acquire(client_id, token, exclusive=True):
                raise RpcError(
                    "SafetyViolation",
                    f"actions.{name} moves the robot and the motion lock "
                    f"is held by client {self._lock.owner}",
                    {"reason": "motion lock held"})

        if action.is_motion:
            self._held[(client_id, name)] = action
        try:
            with self._lock.mutex:
                if name in self._stopping:
                    raise RpcError("ResourceBusy", "vendor stop verification is in progress")
                if action.is_motion:
                    self._safety.check_motion(client_id, f"actions.{name}")
                self._ros.vendor_publish(name, payload)
        except RpcError:
            if action.is_motion and not self.allow_unverified_stops:
                self._stop(client_id, name, action)
            if action.is_motion and not held_before:
                self._lock.release(client_id, token)
            raise
        except Exception as exc:
            if action.is_motion and not self.allow_unverified_stops:
                self._stop(client_id, name, action)
            if action.is_motion and not held_before:
                self._lock.release(client_id, token)
            raise RpcError(
                "RingfourError",
                f"actions.{name}: cannot build a {action.type} from "
                f"{payload}: {exc}") from exc
        self._last[name] = now

        self._audit.record(
            AuditKind.COMMAND, client=client_id, call=f"actions.{name}",
            outcome=Outcome.SUCCESS,
            detail={"topic": action.topic, "type": action.type,
                    "is_motion": action.is_motion, "arguments": payload})
        return {"action": name, "dispatched": True, "completion": "unverified"}

    @staticmethod
    def _build(name: str, action: ActionConfig,
               arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate the caller's arguments against the declaration."""
        unknown = set(arguments) - set(action.parameters)
        if unknown:
            takes = sorted(action.parameters) or "no arguments"
            raise RpcError(
                "RingfourError",
                f"actions.{name} takes {takes}; got unexpected "
                f"{sorted(unknown)}")
        payload: dict[str, Any] = {}
        for field, spec in action.parameters.items():
            if field in arguments:
                value = arguments[field]
            elif spec.default is not None:
                value = spec.default
            else:
                raise RpcError(
                    "RingfourError",
                    f"actions.{name} needs {field!r} (no default declared)")
            payload[field] = _coerce(name, field, spec, value)
        return payload

    def _contract(self, name, action, *, verify):
        from .stopping import require_joint_stop, verify_joint_stop, require_mobility_stop, verify_mobility_stop
        if action.stop_arguments is None:
            raise RpcError("StopUnverified", f"actions.{name} needs explicit stop_arguments")
        # Validate the stop payload before dispatching the first motion.
        self._build(name, action, action.stop_arguments)
        if self._profile is None:
            raise RpcError("StopUnverified", "vendor motion has no bound stop policy")
        policy = self._profile.spec.safety.stop_verification
        if action.motion_resource == 'mobility':
            method = verify_mobility_stop if verify else require_mobility_stop
            return method(self._ros, policy, self._profile.spec.safety.feedback_max_age)
        resource = action.motion_resource or ''
        group = self._groups.get(resource.removeprefix('group:')) if resource.startswith('group:') else None
        if group is None:
            raise RpcError("StopUnverified", "vendor motion requires mobility or a declared group footprint")
        policy = group.config.stop_verification or policy
        method = verify_joint_stop if verify else require_joint_stop
        return method(self._ros, group.joints, policy, group.config.feedback_max_age)

    def _stop(self, client_id, name, action):
        with self._lock.mutex:
            self._stopping.add(name)
        try:
            if not self.allow_unverified_stops:
                self._ros.vendor_publish(name, self._build(name, action, action.stop_arguments or {}))
                report = self._contract(name, action, verify=True)
                self._audit.record(AuditKind.SAFETY, client=client_id, call=f"actions.{name}.stop",
                                   outcome=Outcome.SUCCESS, detail={"stop_verification": report})
        except Exception as exc:
            report = {"reason": "vendor stop unverified", "error": str(exc), "physical_stop": "unverified"}
            with self._lock.mutex: self._lock.quarantine['action:' + name] = report
            raise RpcError("StopUnverified", f"actions.{name} stop was not verified", {"cleanup": report}) from exc
        finally:
            with self._lock.mutex:
                self._stopping.discard(name)
                self._held.pop((client_id, name), None)
                self._lock.release(client_id, "action:" + name)

    def release(self, client_id: int) -> None:
        """Stop and verify every motion action before releasing its acquisition."""
        failures = []
        with self._lock.mutex:
            held = [(name, action) for (owner, name), action in self._held.items() if owner == client_id]
        for name, action in held:
            try: self._stop(client_id, name, action)
            except RpcError as exc: failures.append(exc.extra)
        if failures:
            raise RpcError("CleanupFailed", "vendor motion cleanup failed", {"cleanup": failures})

    def end_all(self):
        failures = []
        for client in {owner for owner, _ in tuple(self._held)}:
            try: self.release(client)
            except RpcError as exc: failures.append(exc.extra)
        if failures:
            raise RpcError("CleanupFailed", "vendor shutdown failed", {"cleanup": failures})

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._actions)


def _coerce(action: str, field: str, spec: Any, value: Any) -> Any:
    kinds = {"float": float, "int": int, "bool": bool, "string": str}
    want = kinds[spec.type]
    try:
        coerced = want(value)
    except (TypeError, ValueError) as exc:
        raise RpcError(
            "RingfourError",
            f"actions.{action}.{field} wants {spec.type}, got "
            f"{value!r}") from exc
    if spec.type in ("float", "int"):
        if isinstance(value, bool) or not math.isfinite(coerced):
            raise RpcError("RingfourError", f"actions.{action}.{field} needs a finite {spec.type}")
        if spec.min is not None and coerced < spec.min:
            raise RpcError(
                "RingfourError",
                f"actions.{action}.{field} = {coerced} is below the "
                f"declared minimum {spec.min}")
        if spec.max is not None and coerced > spec.max:
            raise RpcError(
                "RingfourError",
                f"actions.{action}.{field} = {coerced} is above the "
                f"declared maximum {spec.max}")
    return coerced


def _ros_type(name: str) -> str:
    """`std_msgs/Int32` and `std_msgs/msg/Int32` both work."""
    parts = name.split("/")
    if len(parts) == 2:
        return f"{parts[0]}/msg/{parts[1]}"
    return name


def _dig(message: Any, path: str) -> Any:
    value: Any = message
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value if isinstance(value, (int, float, bool)) else None
