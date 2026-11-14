"""Task lifecycle on the agent (spec sections 5 and 23).

A task is an asyncio coroutine with an id, an owning client session, and a
state machine (pending → running → succeeded | failed | cancelled). Every
state change is pushed to the owner as a ``task.update`` notification and
recorded in the audit log.

Task bodies signal user-visible failure by raising server.RpcError (the
error type name reaches the client); any other exception becomes a generic
RingfourError-typed failure.
"""

from __future__ import annotations

import asyncio
import uuid
import logging
import time
from typing import Any, Awaitable, Callable

from ringfour import wire
from ringfour.enums import AuditKind, Outcome
from ringfour.models import TaskError, TaskUpdate

from .audit import AuditLog
from .server import RpcError, Session

log = logging.getLogger("ringfour_agent.tasks")

# A task's terminal state and an audit outcome are different vocabularies:
# the state says what happened to the task, the outcome says how the call
# ended. The state itself is kept in the record's ``detail``.
_TASK_OUTCOME = {
    "succeeded": Outcome.SUCCESS,
    "failed": Outcome.ERROR,
    "cancelled": Outcome.CANCELLED,
}
_MOTION_TASKS = frozenset({'group.start_trajectory', 'group.start_tool',
                           'mobility.navigate_to', 'mobility.start_route'})


class TaskHandle:
    """Passed to a task body so it can report progress."""

    def __init__(self, record: "TaskRecord", manager: "TaskManager") -> None:
        self._record = record
        self._manager = manager

    def mark_dispatched(self):self._record.dispatched=True

    def record_stop(self,evidence):self._record.cleanup=evidence

    async def verify_stop(self, callback, *args):
        """Join a stop worker even when cancellation arrives during verification.

        Cancelling an asyncio wait cannot cancel its controller/feedback thread.
        Do not publish a terminal cancellation before that worker's evidence or
        failure is recorded. A verification failure takes precedence over cancel.
        """
        pending = asyncio.create_task(asyncio.to_thread(callback, *args))
        try:
            evidence = await asyncio.shield(pending)
        except asyncio.CancelledError:
            self.record_stop(await pending)
            raise
        self.record_stop(evidence)
        return evidence

    async def report(self, progress: float | None = None,
                     message: str | None = None,
                     data: dict[str, Any] | None = None) -> None:
        """Push a "running" task.update with optional progress (0..1)
        and task-kind-specific feedback ``data``."""
        await self._manager._push(self._record, "running",
                                  progress=progress, message=message,
                                  data=data)


# A task body: takes the handle, returns the result dict (or None).
TaskBody = Callable[[TaskHandle], Awaitable[dict[str, Any] | None]]


class TaskRecord:
    """One task's bookkeeping."""


    def __init__(self, kind: str, session: Session, resource=None) -> None:
        self.id = "t" + uuid.uuid4().hex
        self.kind = kind
        from .commands import current_command
        self.command_id = current_command.get()
        self.session = session
        self.state = "pending"
        self.aio_task: asyncio.Task | None = None
        self.resource = resource
        self.update = None
        self.finished_at = None
        self.cancel_requested = False
        self.dispatched=False
        self.cleanup=None
        self.sequence=0
        self.updated_at=time.monotonic()


class TaskManager:
    """Creates, tracks, cancels tasks; emits updates + audit records."""

    def __init__(self, audit: AuditLog, *, clock=time.monotonic, retention=300.0, max_records=1024) -> None:
        self._audit = audit
        self._clock = clock
        self.retention = retention
        self.max_records = max_records
        self.ledger = None
        self.motion_recorder = None
        self._tasks: dict[str, TaskRecord] = {}

    def start(self, kind: str, session: Session, body: TaskBody, *, resource=None) -> str:
        """Create a task and schedule it. Returns the task id immediately."""
        self._expire()
        if resource is not None and any(r.resource == resource and r.finished_at is None
                                        for r in self._tasks.values()):
            raise RpcError("ResourceBusy", f"resource {resource!r} already has accepted work")
        if len(self._tasks) >= self.max_records:
            raise RpcError("ResourceBusy", "operation record capacity reached; wait for advertised retention expiry")
        record = TaskRecord(kind, session, resource)
        if self.ledger is not None:
            self.ledger.task_started(record)
        self._tasks[record.id] = record
        record.aio_task = asyncio.ensure_future(self._run(record, body))
        if self.motion_recorder is not None and kind in _MOTION_TASKS:
            self.motion_recorder.emit('motion.accepted', operation_id=record.id,
                                      session_id=session.reconciliation_key, owner=session.principal.identity)
        self._audit.record(
            AuditKind.COMMAND, client=session.client_id, call=kind,
            task_id=record.id, detail={"event": "started", "command_id": record.command_id, "principal": session.principal.identity, "session": session.reconciliation_key, "resource": resource},
        )
        return record.id

    def _expire(self):
        for key, record in list(self._tasks.items()):
            if record.finished_at is not None and self._clock()-record.finished_at >= self.retention:
                del self._tasks[key]

    def owned(self, task_id, session):
        self._expire()
        record = self._tasks.get(task_id)
        if record is None:
            raise RpcError("RingfourError", f"unknown or expired operation {task_id!r}")
        if record.session.client_id != session.client_id:
            raise RpcError("SafetyViolation", "operation belongs to another client",
                           {"reason": "operation ownership"})
        return record

    def status(self, task_id, session):
        record = self.owned(task_id, session)
        update = record.update or TaskUpdate(task_id=record.id, state=record.state)
        return update.model_copy(update={"report_age": max(0.0, time.monotonic()-record.updated_at)}).model_dump(mode="json")

    @staticmethod
    def _cancel(record):
        if record.finished_at is not None or record.cancel_requested:
            return False
        record.cancel_requested = True
        if record.state != "pending" and record.aio_task is not None and not record.aio_task.done():
            record.aio_task.cancel()
        return True

    async def cancel(self, task_id: str, session: Session) -> None:
        """Owner-only, idempotent cancellation; terminal results never change."""
        self._cancel(self.owned(task_id, session))

    async def _run(self, record: TaskRecord, body: TaskBody) -> None:
        try:
            if record.cancel_requested:
                raise asyncio.CancelledError()
            await self._push(record, "running")
            result = await body(TaskHandle(record, self))
        except asyncio.CancelledError:
            if not record.dispatched and record.cleanup is None:
                record.cleanup={"verified":True,"criterion":"not-dispatched","physical_stop":"unverified"}
            await self._finish(record, "cancelled")
            return
        except RpcError as exc:
            await self._finish(record, "failed", error=TaskError(
                type=exc.error_type, message=str(exc), data=exc.extra))
            return
        except Exception as exc:  # task body bug: fail the task, log it
            log.exception("task %s (%s) crashed", record.id, record.kind)
            await self._finish(record, "failed", error=TaskError(
                type="RingfourError", message=f"internal error: {exc}"))
            return
        await self._finish(record, "succeeded", result=result or {})

    async def _finish(
        self,
        record: TaskRecord,
        state: str,
        result: dict[str, Any] | None = None,
        error: TaskError | None = None,
    ) -> None:
        if record.finished_at is not None:
            return
        if self.ledger is not None:
            terminal = TaskUpdate(task_id=record.id, state=state, result=result, error=error,
                                  sequence=record.sequence+1, cleanup=record.cleanup,
                                  progress=1.0 if state == 'succeeded' else None)
            self.ledger.task_finished(record, terminal.model_dump(mode='json'))
        record.finished_at = self._clock()
        if self.motion_recorder is not None and record.kind in _MOTION_TASKS:
            fields = {'operation_id': record.id, 'session_id': record.session.reconciliation_key,
                      'owner': record.session.principal.identity}
            if (isinstance(record.cleanup, dict) and record.cleanup.get('verified') is True
                    and record.cleanup.get('criterion') in ('joint-velocity-dwell', 'body-velocity-dwell')):
                self.motion_recorder.emit('motion.stop_verified', stop_verified=True, **fields)
            self.motion_recorder.emit({'succeeded': 'motion.completed', 'failed': 'motion.failed',
                                       'cancelled': 'motion.cancelled'}[state], **fields)
        detail: dict[str, Any] = {"state": state, "command_id": record.command_id, "principal": record.session.principal.identity, "session": record.session.reconciliation_key, "resource": record.resource}
        if error is not None:
            detail["type"] = error.type
            detail["message"] = error.message
        self._audit.record(
            AuditKind.RESULT, client=record.session.client_id,
            call=record.kind, task_id=record.id,
            outcome=_TASK_OUTCOME[state], detail=detail,
        )
        await self._push(record, state, result=result, error=error,
                         progress=1.0 if state == "succeeded" else None)

    async def _push(
        self,
        record: TaskRecord,
        state: str,
        progress: float | None = None,
        message: str | None = None,
        data: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: TaskError | None = None,
    ) -> None:
        """Send a task.update to the owning client. Best effort: a gone
        client must not kill the task."""
        if record.update is not None and record.update.state in ("succeeded", "failed", "cancelled"):
            return
        record.state = state
        record.sequence+=1
        record.updated_at=time.monotonic()
        update = TaskUpdate(task_id=record.id, state=state, progress=progress,
                            message=message, data=data, result=result,
                            error=error,sequence=record.sequence,cleanup=record.cleanup)
        record.update = update
        try:
            await record.session.websocket.send(
                wire.notification("task.update",
                                  update.model_dump(mode="json"))
            )
        except Exception:
            pass  # owner disconnected; task continues, audit has the trail

    def cancel_for_session(self, session: Session) -> int:
        """Cancel every running task owned by one client (the
        stop_on_disconnect path). Returns how many were cancelled."""
        cancelled = 0
        for record in list(self._tasks.values()):
            if (record.session is session and record.aio_task is not None
                    and not record.aio_task.done()):
                cancelled += int(self._cancel(record))
        return cancelled

    async def cancel_all(self) -> None:
        """Shutdown helper: cancel everything still running."""
        self.cancel_running()
        pending = [r.aio_task for r in self._tasks.values() if r.aio_task is not None and not r.aio_task.done()]
        if pending:
            _, unfinished = await asyncio.wait(pending, timeout=15.0)
            if unfinished:
                raise RpcError("CleanupFailed", "shutdown task cleanup exceeded its deadline")

    def has_running(self) -> bool:
        """Any accepted work still executing or cleaning up, regardless of owner."""
        return any(r.aio_task is not None and not r.aio_task.done() for r in self._tasks.values())

    def cancel_running(self) -> int:
        """Cancel every running task regardless of owner (watch halt /
        e-stop path). Returns how many were cancelled."""
        return self.cancel_kind("")

    def cancel_kind(self, prefix: str) -> int:
        """Cancel running tasks whose kind starts with ``prefix``
        (e.g. "mobility." for robot.stop()). Returns the count."""
        cancelled = 0
        for record in list(self._tasks.values()):
            if (record.kind.startswith(prefix)
                    and record.aio_task is not None
                    and not record.aio_task.done()):
                cancelled += int(self._cancel(record))
        return cancelled

    def cancel_resource(self, resource):
        return sum(int(self._cancel(record)) for record in list(self._tasks.values())
                   if record.resource == resource)

    async def drain_resource(self, resource, timeout=15.0):
        pending = [r.aio_task for r in self._tasks.values()
                   if r.resource == resource and r.aio_task is not None and not r.aio_task.done()]
        if pending:
            done, unfinished = await asyncio.wait(pending, timeout=timeout)
            if unfinished:
                raise RpcError("CleanupFailed", "operation cleanup did not finish before controller restoration",
                               {"resource": resource})

    async def drain_session(self, session, timeout=15.0):
        pending = [r.aio_task for r in self._tasks.values()
                   if r.session is session and r.aio_task is not None and not r.aio_task.done()]
        if pending:
            _, unfinished = await asyncio.wait(pending, timeout=timeout)
            if unfinished:
                raise RpcError("CleanupFailed", "disconnected client's operations did not finish cleanup")
