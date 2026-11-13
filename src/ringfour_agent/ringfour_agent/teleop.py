"""Teleop sessions (spec section 14): streamed intent with a server-side
watchdog.

A session holds the motion lock exclusively. The client streams drive
commands at its own rate; if they stop arriving (WiFi drop, crashed
client, sleeping laptop) the watchdog zeroes the velocity within the
watchdog window. The client is never trusted to stop the robot.

No rclpy here: velocity goes out through the RosInterface handed in at
construction (tests inject a fake).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from ringfour.enums import AuditKind, Locomotion, Outcome

from .audit import AuditLog
from .profile import Profile
from .safety import MotionLock
from .server import RpcError

log = logging.getLogger("ringfour_agent.teleop")

# A watchdog below this is unenforceable (scheduler jitter); reject it.
_MIN_WATCHDOG = 0.05  # seconds


@dataclass
class _Session:
    client_id: int
    watchdog: float
    last_command: float = field(default_factory=time.monotonic)
    halted_by_watchdog: bool = False
    watchdog_task: asyncio.Task | None = None


class TeleopManager:
    """Owns all teleop sessions (one per client, one lock for all)."""

    def __init__(self, lock: MotionLock, audit: AuditLog, ros: Any,
                 profile: Profile, safety: Any = None, *, allow_unverified_stops=False) -> None:
        self.allow_unverified_stops = allow_unverified_stops
        self._policy = profile.spec.safety.stop_verification
        self._max_age = profile.spec.safety.feedback_max_age
        from threading import Event
        self._stop_done = Event()
        self._stop_done.set()
        self._stop_error = self._stop_report = None
        self._stopping = False
        self._safety = safety
        self._lock = lock
        self._audit = audit
        self._ros = ros
        self._mobility = profile.spec.mobility
        self._teleop = profile.spec.teleop
        self._sessions: dict[int, _Session] = {}

    # -- session lifecycle ---------------------------------------------------

    def start(self, client_id: int, requested_watchdog: float) -> float:
        """Acquire the motion lock and start the watchdog. Returns the
        granted watchdog period (server-clamped, spec: clients may request
        shorter than the profile max, never longer)."""
        if self._safety is not None:
            self._safety.check_motion(client_id, "teleop.start")
        if self._stopping:
            raise RpcError("ResourceBusy", "body stop verification is in progress")
        if not self.allow_unverified_stops:
            from .stopping import require_mobility_stop
            require_mobility_stop(self._ros, self._policy, self._max_age)
        if client_id in self._sessions:
            raise RpcError("RingfourError", "teleop session already active")
        if not math.isfinite(requested_watchdog):
            raise RpcError("RingfourError", "watchdog must be finite")
        if not self._lock.acquire(client_id, "teleop", resources=("adapter:mobility",)):
            raise RpcError(
                "SafetyViolation",
                f"motion lock held by client {self._lock.owner}",
                {"reason": "motion lock held"},
            )
        if self._stopping:
            raise RpcError("ResourceBusy", "body stop verification is in progress")
        if not self.allow_unverified_stops:
            from .stopping import require_mobility_stop
            require_mobility_stop(self._ros, self._policy, self._max_age)
        if client_id in self._sessions:
            raise RpcError("RingfourError", "teleop session already active")
        max_watchdog = self._teleop.max_watchdog if self._teleop else 1.0
        watchdog = min(float(requested_watchdog), max_watchdog)
        if watchdog < _MIN_WATCHDOG:
            self._lock.release(client_id, "teleop")
            raise RpcError("RingfourError",
                           f"watchdog must be >= {_MIN_WATCHDOG}s")
        session = _Session(client_id=client_id, watchdog=watchdog)
        session.watchdog_task = asyncio.ensure_future(self._watch(session))
        self._sessions[client_id] = session
        return watchdog

    def end(self, client_id: int) -> None:
        """Zero velocity, stop the watchdog, release the lock. Idempotent."""
        session = self._sessions.pop(client_id, None)
        if session is None:
            return
        if session.watchdog_task is not None:
            session.watchdog_task.get_loop().call_soon_threadsafe(session.watchdog_task.cancel)
        try:
            return self._publish_zero()
        finally:
            self._lock.release(client_id, "teleop")

    def on_disconnect(self, client_id: int) -> None:
        """Server hook: a vanished client must not leave the robot moving
        or locked."""
        if client_id in self._sessions:
            stop_report = self.end(client_id)
            self._audit.record(
                AuditKind.SAFETY, client=client_id, call="teleop",
                outcome=Outcome.SUCCESS,
                detail={"reason": "disconnect",
                        "stop_verification": stop_report},
            )

    # -- commands ----------------------------------------------------------------

    def drive(self, client_id: int, linear: float, angular: float,
              lateral: float = 0.0) -> None:
        """One velocity command, clamped by the profile's limits."""
        session = self._session(client_id)
        if self._safety is not None:
            self._safety.check_motion(client_id, "teleop.drive")
        if lateral != 0.0:
            locomotion = (self._mobility.locomotion if self._mobility
                          else None)
            if locomotion is not Locomotion.OMNI:
                raise RpcError(
                    "CapabilityNotSupported",
                    f"lateral velocity needs omni locomotion, "
                    f"this robot is {locomotion or 'unknown'}",
                )
        linear = self._clamp(linear, self._max_linear())
        lateral = self._clamp(lateral, self._max_linear())
        angular = self._clamp(angular, self._max_angular())
        session.last_command = time.monotonic()
        session.halted_by_watchdog = False
        with self._lock.mutex:
            if self._stopping:
                raise RpcError("ResourceBusy", "body stop verification is in progress")
            if client_id not in self._sessions:
                raise RpcError("SafetyViolation", "teleop session was revoked")
            if self._safety is not None:
                self._safety.check_motion(client_id, "teleop.drive")
            self._ros.publish_twist(linear, angular, lateral)

    def stop(self, client_id: int) -> None:
        """Immediate zero, session stays alive."""
        session = self._session(client_id)
        session.last_command = time.monotonic()
        return self._publish_zero()

    def zero_all(self) -> None:
        """Emergency hook (e-stop engaged): zero regardless of sessions."""
        self._publish_zero()

    def end_all(self) -> int:
        """Halt hook (stop-watch fired): end every session — zero
        command, measured rest verified before locks release. A hung owner's next drive gets
        SafetyViolation("no teleop session"), which is the point.
        Returns how many sessions were ended."""
        clients = list(self._sessions)
        failures = []
        for client_id in clients:
            try:
                self.end(client_id)
            except RpcError as exc:
                failures.append(exc.extra)
        if failures:
            raise RpcError("CleanupFailed", "teleop shutdown could not publish all stops", {"cleanup": failures})
        return len(clients)

    # -- internals -----------------------------------------------------------------

    def _session(self, client_id: int) -> _Session:
        session = self._sessions.get(client_id)
        if session is None:
            raise RpcError(
                "SafetyViolation", "no active teleop session; call "
                "teleop.start first", {"reason": "no teleop session"},
            )
        return session

    def _max_linear(self) -> float | None:
        return self._mobility.max_linear if self._mobility else None

    def _max_angular(self) -> float | None:
        return self._mobility.max_angular if self._mobility else None

    @staticmethod
    def _clamp(value: float, limit: float | None) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise RpcError("SafetyViolation", "velocity commands must be finite", {"reason": "invalid velocity"})
        if limit is None:
            return value
        return max(-limit, min(limit, value))

    async def _watch(self, session: _Session) -> None:
        """Zero the velocity when commands stop arriving for > watchdog."""
        interval = session.watchdog / 4
        while True:
            await asyncio.sleep(interval)
            silent = time.monotonic() - session.last_command
            if silent > session.watchdog and not session.halted_by_watchdog:
                session.halted_by_watchdog = True
                try:
                    stop_report = await asyncio.to_thread(self._publish_zero)
                except RpcError:
                    # Stop eligibility immediately and retain quarantine even
                    # when the disconnected owner's acquisition is released.
                    self._sessions.pop(session.client_id, None)
                    self._lock.release(session.client_id, "teleop")
                    return
                self._audit.record(
                    AuditKind.SAFETY, client=session.client_id, call="teleop",
                    outcome=Outcome.SUCCESS,
                    detail={"reason": "teleop watchdog",
                            "silent_for": round(silent, 3), "stop_verification": stop_report},
                )
                log.warning("teleop watchdog completed stop contract (client %d silent "
                            "%.2fs)", session.client_id, silent)

    def _publish_zero(self) -> None:
        with self._lock.mutex:
            waiting = self._stopping
            if not waiting:
                self._stopping = True
                self._stop_done.clear()
                self._stop_error = self._stop_report = None
        if waiting:
            budget = self._policy.timeout + 1 if self._policy else 3.
            if not self._stop_done.wait(budget):
                report = {"reason": "concurrent body stop cleanup timed out", "physical_stop": "unverified"}
                with self._lock.mutex: self._lock.quarantine['mobility'] = report
                raise RpcError("StopUnverified", "body cleanup remains in flight", {"cleanup": report})
            if self._stop_error is not None: raise self._stop_error
            return self._stop_report
        try:
            self._ros.publish_zero()
            if not self.allow_unverified_stops:
                from .stopping import verify_mobility_stop
                self._stop_report = verify_mobility_stop(self._ros, self._policy, self._max_age)
                return self._stop_report
        except Exception as exc:
            report = {"reason": "body stop contract failed", "error": str(exc),
                      "physical_stop": "unverified"}
            if isinstance(exc, RpcError):
                report["evidence"] = exc.extra
                report["error_type"] = exc.error_type
            with self._lock.mutex:
                self._lock.quarantine["mobility"] = report
            self._audit.record(AuditKind.SAFETY, call="teleop.stop", outcome=Outcome.ERROR,
                               detail={"cleanup": report})
            self._stop_error = RpcError("CleanupFailed", "teleop stop was not verified", {"cleanup": report})
            raise self._stop_error from exc
        finally:
            with self._lock.mutex:
                self._stopping = False
                self._stop_done.set()
