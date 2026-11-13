"""Safety layer skeleton (spec section 21) and the motion lock.

Every motion-classed wire method passes ``SafetyLayer.check`` before its
handler runs — server-side, non-bypassable, regardless of which client
sent it. Phase 2 checks: e-stop state and the motion lock. Velocity
clamps, workspace bounds, geofence etc. land with their phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from .server import RpcError



@dataclass
class AgentState:
    """Mutable agent-wide state the safety layer reads."""

    estop_engaged: bool = False
    closing: bool = False
    guard_inhibits: dict[str, str] = field(default_factory=dict)


@dataclass
class MotionLock:
    """Motion is single-writer (spec section 14 resolution): one client id
    may hold the lock; teleop sessions (Phase 3) hold it exclusively.
    Read access is never locked."""

    owner: int | None = field(default=None)
    revoked_clients: set[int] = field(default_factory=set, repr=False)
    _acquisitions: set[str] = field(default_factory=set, repr=False)
    _resources: dict[str, frozenset[str]] = field(default_factory=dict, repr=False)
    _exclusive: set[str] = field(default_factory=set, repr=False)
    mutex: RLock = field(default_factory=RLock, repr=False)
    quarantine: dict[str, dict] = field(default_factory=dict)

    def acquire(self, client_id: int, acquisition: str = "legacy", *, resources=(), exclusive=False) -> bool:
        """Take the lock. True on success or if already held by this
        client; False if another client holds it."""
        with self.mutex:
            if client_id in self.revoked_clients:
                raise RpcError("SafetyViolation", "connection authority was revoked", {"reason": "lease expired"})
            if self.quarantine:
                raise RpcError("SafetyViolation", "motion is inhibited after unverified cleanup",
                               {"reason": "quarantine", "resources": self.quarantine})
            if self.owner is None or self.owner == client_id:
                footprint = frozenset(resources)
                if acquisition in self._acquisitions and (self._resources.get(acquisition) != footprint or
                                                          (acquisition in self._exclusive) != exclusive):
                    raise RpcError("SafetyViolation", "an acquisition cannot change its resource footprint")
                others = self._acquisitions - {acquisition}
                conflicts = {token: sorted(self._resources.get(token, frozenset()) & footprint)
                             for token in others if self._resources.get(token, frozenset()) & footprint}
                if others and (exclusive or self._exclusive - {acquisition}):
                    conflicts.update({token: ["opaque robot-wide effect"] for token in others})
                if conflicts:
                    raise RpcError("ResourceBusy", "motion acquisitions have conflicting resources",
                                   {"reason": "resource conflict", "conflicts": conflicts})
                self._resources[acquisition] = footprint
                if exclusive: self._exclusive.add(acquisition)
                self.owner = client_id
                self._acquisitions.add(acquisition)
                return True
            return False

    def release(self, client_id: int, acquisition: str = "legacy") -> bool:
        """Release if held by this client. True if released."""
        with self.mutex:
            if self.owner != client_id or acquisition not in self._acquisitions:
                return False
            self._acquisitions.remove(acquisition)
            self._resources.pop(acquisition, None)
            self._exclusive.discard(acquisition)
            if not self._acquisitions:
                self.owner = None
            return True

    def release_all(self, client_id: int) -> bool:
        """Revoke a disconnected owner after its cleanup; retain quarantine."""
        with self.mutex:
            if self.owner != client_id:
                return False
            self._acquisitions.clear()
            self._resources.clear()
            self._exclusive.clear()
            self.owner = None
            return True

    def held_by_other(self, client_id: int) -> bool:
        return self.owner is not None and self.owner != client_id


class SafetyLayer:
    """Validates motion commands before they reach ROS."""

    def __init__(self, state: AgentState, lock: MotionLock) -> None:
        self._state = state
        self._lock = lock

    def check(self, client_id: int, method: str, *, descriptor=None) -> None:
        """Raise RpcError(SafetyViolation) if ``method`` must not run now.

        Non-motion methods always pass: reads are never safety-gated.
        Every method must have an admission descriptor. Dynamic vendor
        effects additionally call `check_motion` from their validated profile.
        """
        from .admission import DESCRIPTORS
        descriptor = descriptor or DESCRIPTORS.get(method)
        if descriptor is None:
            raise RpcError("SafetyViolation", f"unclassified method {method!r}",
                           {"reason": "unclassified method"})
        if descriptor.effect != "motion":
            return
        self.check_motion(client_id, method)

    def check_motion(self, client_id: int, what: str) -> None:
        """The motion gate itself: e-stop, then the lock.

        Split out so that a caller which already knows it is about to move
        the robot cannot accidentally bypass the rule by not appearing in
        the static admission inventory. A profile author writing `is_motion: true` has to
        get the real gate, or the field is decoration — and decoration
        that looks like a safety property is worse than no field at all.
        """
        if client_id in self._lock.revoked_clients:
            raise RpcError("SafetyViolation", "connection authority was revoked", {"reason": "lease expired"})
        if self._state.closing:
            raise RpcError("SafetyViolation", "agent is shutting down; new motion is inhibited",
                           {"reason": "agent shutdown"})
        if self._lock.quarantine:
            raise RpcError("SafetyViolation", "motion is inhibited after unverified cleanup",
                           {"reason": "quarantine", "resources": self._lock.quarantine})
        if self._state.guard_inhibits:
            raise RpcError("SafetyViolation", "motion is inhibited by a persistent guard",
                           {"reason": "guard", "guards": dict(self._state.guard_inhibits)})
        if self._state.estop_engaged:
            raise RpcError(
                "SafetyViolation",
                f"e-stop is engaged; {what} rejected",
                {"reason": "estop"},
            )
        if self._lock.held_by_other(client_id):
            raise RpcError(
                "SafetyViolation",
                f"motion lock held by client {self._lock.owner}",
                {"reason": "motion lock held"},
            )
