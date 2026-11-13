"""Agent-side shared-memory payload store (the v1 frame transport).

Architecture ruling (Cristi, 2026-06-12): the agent writes raw sensor
arrays to files under /dev/shm and sends a ``PayloadRef`` descriptor over
the RPC channel; the local client mmaps the file as numpy. Zero copy on
the read side; the agent pays one memcpy (ROS message -> segment).

Lifetime: segments expire after their TTL or, oldest first, when the
store exceeds its byte budget. Expiring means unlink — a client that
already mapped the segment keeps valid memory (POSIX semantics), so
aggressive expiry is safe; a client that waits past the TTL to fetch
gets FrameExpired from its transport layer.

No ROS imports here: unit-testable anywhere with a /dev/shm.
"""

from __future__ import annotations

import logging
import os
import re
import secrets
import time

import numpy as np

from ringfour.models.sensing import PayloadRef

log = logging.getLogger("ringfour_agent.shm")

SHM_DIR = "/dev/shm"

# Default byte budget across all live segments. At VGA RGB (~0.9 MiB per
# frame) this holds ~4.5 min of 1 Hz frames or ~37 s of a 5 fps stream
# alongside everything else; oldest segments evict first.
DEFAULT_BUDGET = 192 * 1024 * 1024

DEFAULT_TTL = 30.0  # seconds; per-call override for stream frames

# Permission bits for a payload segment: owner writes, everyone reads.
#
# 0600 was wrong. The whole design is "the agent writes, ANOTHER PROCESS
# mmaps" — a handoff across a process boundary — and 0600 only works when
# both processes happen to run as the same user. That held on a dev box and
# broke the moment the agent ran in a container: every camera call handed the
# client a segment name it could not open (PermissionError, errno 13).
#
# World-readable is the right call rather than a shared group, because it
# matches the surrounding threat model instead of pretending to exceed it:
# these frames are already on ROS topics that any local process can subscribe
# to, and /dev/shm is not shared off the machine. Adding a group would mean
# deployment-time group setup for privacy the rest of the stack does not
# provide anyway. Segments stay owner-WRITE, so no one else can corrupt one.
PAYLOAD_MODE = 0o644

# Startup sweep threshold. Comfortably past DEFAULT_TTL so a segment another
# live agent is actively serving is never yanked, while genuine orphans from
# a killed run go on the next start.
STALE_AFTER = 300.0  # seconds


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "-", name)


class ShmStore:
    """Owns every shm segment one agent process creates."""

    def __init__(self, robot_name: str, budget: int = DEFAULT_BUDGET) -> None:
        # A RANDOM run token, not os.getpid(). The PID was wrong twice over
        # once the agent moved into a container: PIDs restart low in a fresh
        # namespace, so a restarted agent reuses a previous run's prefix and
        # collides on O_EXCL below; and the /proc liveness check that decided
        # which siblings were stale is namespace-local, so it read some
        # unrelated local PID as "that run is still alive" and never swept.
        # /dev/shm is shared host-wide (ipc: host is required so the client
        # can mmap at all), so those orphans accumulate until every payload
        # call fails. Randomness makes collisions vanish; age handles cleanup.
        self._prefix = f"rc-{_sanitize(robot_name)}-{secrets.token_hex(4)}"
        self._seq = 0
        # name -> (deadline, size); insertion order == creation order.
        self._live: dict[str, tuple[float, int]] = {}
        self._bytes = 0
        self._budget = budget
        self._sweep_stale_siblings(robot_name)

    def put(self, array: np.ndarray, ttl: float = DEFAULT_TTL) -> PayloadRef:
        """Write one array into a fresh segment; returns its descriptor."""
        array = np.ascontiguousarray(array)
        # O_EXCL is deliberate: silently reusing a name would hand a client a
        # segment someone else still owns. A collision should be impossible
        # now, but retry rather than fail the caller's frame if one happens.
        for attempt in range(3):
            self._seq += 1
            name = f"{self._prefix}-{self._seq}"
            try:
                fd = os.open(f"{SHM_DIR}/{name}",
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                             PAYLOAD_MODE)
                # os.open's mode argument is masked by the process umask, so
                # a 077 umask would silently give 0600 back and reintroduce
                # the bug in a way that only shows on someone else's machine.
                # fchmod is not masked; set the mode explicitly.
                os.fchmod(fd, PAYLOAD_MODE)
                break
            except FileExistsError:
                log.warning("shm name collision on %s; retrying", name)
        else:
            raise RuntimeError(
                f"could not allocate an shm segment after 3 attempts "
                f"(prefix {self._prefix}); /dev/shm may be full or polluted")
        try:
            os.write(fd, array.data.cast("B"))
        finally:
            os.close(fd)
        self._live[name] = (time.monotonic() + ttl, array.nbytes)
        self._bytes += array.nbytes
        self._expire()
        return PayloadRef(
            transport="shm",
            name=name,
            dtype=array.dtype.name,
            shape=tuple(array.shape),
            size=array.nbytes,
        )

    def expire(self) -> int:
        """Drop segments past their TTL; returns how many were freed.

        Public because expiry used to happen ONLY inside put(), which meant
        reclamation was driven entirely by new traffic: a mission that pulled
        frames hard and then stopped left the store pinned at its full budget
        (192 MiB) until the next payload call or agent shutdown. On an SBC
        that is real memory pressure held by an idle robot, so the agent
        calls this on a timer too.
        """
        before = len(self._live)
        self._expire()
        return before - len(self._live)

    def close(self) -> None:
        """Unlink every live segment (agent shutdown)."""
        for name in list(self._live):
            self._unlink(name)

    # -- internals -------------------------------------------------------------

    def _expire(self) -> None:
        now = time.monotonic()
        for name, (deadline, _size) in list(self._live.items()):
            if deadline <= now:
                self._unlink(name)
        while self._bytes > self._budget and self._live:
            self._unlink(next(iter(self._live)))  # oldest first

    def _unlink(self, name: str) -> None:
        _deadline, size = self._live.pop(name)
        self._bytes -= size
        try:
            os.unlink(f"{SHM_DIR}/{name}")
        except OSError:
            pass  # already gone; nothing to free

    def _sweep_stale_siblings(self, robot_name: str) -> None:
        """Unlink this robot's segments left behind by earlier agent runs.

        A SIGKILLed agent cannot clean up after itself, and under a
        container it cannot even be recognised: the old scheme encoded the
        PID in the name and checked /proc to decide liveness, which is
        meaningless across PID namespaces and left orphans forever.

        Age is the honest signal instead. Any segment older than
        STALE_AFTER cannot belong to a live exchange — DEFAULT_TTL is 30 s
        and a client that has not fetched by then already gets
        FrameExpired. Our own prefix is random and brand new, so it can
        never match here; only previous runs are touched.
        """
        prefix = f"rc-{_sanitize(robot_name)}-"
        cutoff = time.time() - STALE_AFTER
        removed = 0
        try:
            entries = os.listdir(SHM_DIR)
        except OSError:
            return
        for entry in entries:
            if not entry.startswith(prefix) or entry.startswith(self._prefix):
                continue
            path = f"{SHM_DIR}/{entry}"
            try:
                if os.stat(path).st_mtime > cutoff:
                    continue      # young enough that another agent may own it
                os.unlink(path)
                removed += 1
            except OSError:
                pass
        if removed:
            log.info("removed %d stale shm segment(s) from earlier runs",
                     removed)
