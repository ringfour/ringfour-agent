"""Finite connection authority, independent of teleop command freshness."""
from dataclasses import dataclass
import time
import uuid

from .server import RpcError


@dataclass
class Lease:
    id: str
    epoch: str
    principal: str
    client_id: int
    deadline: float
    lifetime: float
    revoked: bool = False

    @classmethod
    def issue(cls, epoch, principal, client_id, lifetime, *, now=None):
        now = time.monotonic() if now is None else now
        return cls(uuid.uuid4().hex, epoch, principal, client_id,
                   now + lifetime, lifetime)

    def check(self, *, now=None):
        now = time.monotonic() if now is None else now
        if self.revoked or now >= self.deadline:
            self.revoked = True
            raise RpcError("SafetyViolation", "connection lease has expired",
                           {"reason": "lease expired", "epoch": self.epoch})

    def renew(self, lease_id, epoch, principal, client_id, *, now=None):
        self.check(now=now)
        if (lease_id, epoch, principal, client_id) != (self.id, self.epoch, self.principal, self.client_id):
            raise RpcError("SafetyViolation", "lease identity does not match this connection",
                           {"reason": "lease identity"})
        now = time.monotonic() if now is None else now
        self.deadline = now + self.lifetime
