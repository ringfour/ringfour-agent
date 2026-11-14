"""The agent's WebSocket JSON-RPC server.

Serves the same protocol on a unix socket (local clients) and TCP (remote
clients) from one asyncio loop. Pure transport plus the two chokepoints
the spec demands of it (sections 18 and 21): every request is audited, and
motion-classed methods pass the safety layer before their handler runs.

Method behavior lives in handlers.py; this module never imports rclpy so
the whole wire stack stays unit-testable without a ROS environment.

Failure modes: a malformed frame gets a JSON-RPC parse/invalid-request
error back; an unknown method gets method-not-found; a handler exception
becomes an application error carrying the exception type name in
data["type"] (which the client maps back to a ringfour exception).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import websockets

from ringfour import wire
from ringfour.aio.transport import MAX_MESSAGE_BYTES
from ringfour.enums import AuditKind, Outcome

if TYPE_CHECKING:
    from .context import AgentContext

log = logging.getLogger("ringfour_agent.server")
_MOTION_STARTS = frozenset({'group.start_trajectory', 'group.start_tool', 'mobility.navigate_to',
                            'mobility.start_route', 'teleop.start'})


class RpcError(Exception):
    """A handler-signaled error to send back to the client.

    ``error_type`` must name a ringfour exception class (errors.py in the
    client SDK) so the client raises the right thing.
    """

    def __init__(self, error_type: str, message: str,
                 extra: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.extra = extra or {}


class Session:
    """One connected client. Handlers receive this to send payloads and to
    check/flip per-connection state (handshake, audit subscription)."""

    _ids = itertools.count(1)

    def __init__(self, websocket: Any, *, principal=None) -> None:
        from .access import Principal, PERMISSIONS
        # Directly constructed sessions are trusted in-process callers. Network
        # sessions always receive a principal from the listener access policy.
        self.principal = principal or Principal('in-process', PERMISSIONS, True)
        self.websocket = websocket
        self.client_id = next(self._ids)
        import uuid
        self.reconciliation_key = uuid.uuid4().hex
        self.payload_transport = "shm"
        self.lease = None
        self.cleanup_task = None
        self.handshaken = False
        self.audit_subscribed = False
        self.events_subscribed = False
        self.motion_teleop_active = False
        self._payload_ids = itertools.count(1)

    async def send_payload(self, kind: str, meta: dict[str, Any],
                           data: bytes) -> int:
        """Announce and send one binary payload; returns its payload id."""
        payload_id = next(self._payload_ids)
        await self.websocket.send(wire.payload_header(payload_id, kind, meta))
        await self.websocket.send(wire.encode_payload_frame(payload_id, data))
        return payload_id


# A handler takes (session, params) and returns the JSON-RPC result.
Handler = Callable[[Session, dict[str, Any]], Awaitable[Any]]


class AgentServer:
    """Owns the unix + TCP listeners and dispatches requests to handlers."""

    def __init__(
        self,
        registry: dict[str, Handler],
        ctx: "AgentContext",
        unix_path: str | None,
        host: str | None,
        port: int | None,
        access_policy=None,
    ) -> None:
        if unix_path is None and port is None:
            raise ValueError("need at least one of unix_path / port")
        from .admission import validate_registry
        validate_registry(registry)
        from .access import AccessPolicy
        self.access_policy = access_policy or AccessPolicy()
        self._registry = registry
        self._ctx = ctx
        self._reaper = None
        self._robot_name = ctx.profile.info.name
        self._unix_path = unix_path
        self._host = host or "0.0.0.0"
        self._port = port
        self._servers: list[Any] = []
        self._sessions: set[Session] = set()
        self._sampler: asyncio.Task | None = None
        self._model_watcher: asyncio.Task | None = None
        self._lease_monitor: asyncio.Task | None = None
        self._lease_clock = time.monotonic
        ctx.audit.subscribers.append(self._fan_out_audit_event)
        ctx.events.subscribers.append(self._fan_out_event)

    async def start(self) -> None:
        """Bind the listeners. Raises OSError if a bind fails."""
        if self._ctx.motion_recorder is not None:
            await self._ctx.motion_recorder.start()
        if self._ctx.motion_handoff is not None:
            await self._ctx.motion_handoff.start()
        # max_size must match the client transport's MAX_MESSAGE_BYTES;
        # websockets' 1 MiB default aborts connections on big payloads.
        if self._port is not None:
            self._servers.append(
                await websockets.serve(self._handle, self._host, self._port,
                                       max_size=MAX_MESSAGE_BYTES)
            )
            log.info("listening on ws://%s:%s", self._host, self._port)
        if self._unix_path is not None:
            self._servers.append(
                await websockets.unix_serve(self._handle, self._unix_path,
                                            max_size=MAX_MESSAGE_BYTES)
            )
            os.chmod(self._unix_path, 0o600)
            log.info("listening on unix://%s", self._unix_path)
        if self._ctx.profile.spec.safety.lease is not None:
            self._lease_monitor = asyncio.create_task(self._monitor_leases())
        # The watch sampler lives and dies with the server: watches are
        # safety surface, so they must run whenever clients can connect.
        from .sampler import run_watch_sampler
        self._sampler = asyncio.ensure_future(
            run_watch_sampler(self._ctx, self._ctx.watch_paths or {}))
        from .model import run_model_watcher
        self._model_watcher = asyncio.ensure_future(
            run_model_watcher(self._ctx))
        if self._ctx.slam is not None:
            from .slam import run_quality_monitor
            self._slam_monitor = asyncio.ensure_future(
                run_quality_monitor(self._ctx.slam))
        # Reap expired payload segments while idle. The shm store otherwise
        # only reclaims inside put(), so a mission that pulls frames hard and
        # then stops leaves /dev/shm pinned at the full budget (192 MiB) until
        # the next payload call or shutdown — real memory pressure held by an
        # idle robot, which matters most on the boards that can least spare it.
        if self._ctx.shm is not None:
            self._reaper = asyncio.ensure_future(self._reap_shm())

    async def _reap_shm(self, interval: float = 5.0) -> None:
        """Expire timed-out shm segments on a timer.

        Well under the 30 s default TTL, and cheap: it walks a dict of at
        most a few hundred entries. Never allowed to die — a reaper that
        raises would silently stop reclaiming.
        """
        while True:
            await asyncio.sleep(interval)
            try:
                freed = self._ctx.shm.expire()
                if freed:
                    log.debug("reaped %d expired shm segment(s)", freed)
            except Exception:
                log.exception("shm reaper failed")

    async def close(self) -> None:
        self._ctx.state.closing = True
        if self._lease_monitor is not None:
            self._lease_monitor.cancel()
            self._lease_monitor = None
        if self._reaper is not None:
            self._reaper.cancel()
            self._reaper = None
        if self._sampler is not None:
            self._sampler.cancel()
            self._sampler = None
        if self._model_watcher is not None:
            self._model_watcher.cancel()
            self._model_watcher = None
        self._ctx.tasks.cancel_running()
        if self._ctx.teleop is not None:
            try:
                await asyncio.to_thread(self._ctx.teleop.end_all)
            except RpcError:
                log.exception("shutdown teleop cleanup failed; continuing other cleanup")
        if self._ctx.actions is not None:
            try:
                await asyncio.to_thread(self._ctx.actions.end_all)
            except RpcError:
                log.exception("shutdown vendor cleanup failed; continuing")
        try:
            await self._ctx.commands.drain()
        except RpcError as exc:
            self._ctx.lock.quarantine['shutdown'] = {'reason': 'accepted command drain failed', 'error': str(exc)}
            log.exception('shutdown command drain failed; continuing stop cleanup')
        try:
            await self._ctx.tasks.cancel_all()
        except RpcError as exc:
            self._ctx.lock.quarantine["shutdown"] = {"error": str(exc)}
        if self._ctx.background_halts:
            _, unfinished = await asyncio.wait(tuple(self._ctx.background_halts), timeout=15.)
            if unfinished:
                self._ctx.lock.quarantine["shutdown"] = {"reason": "background halt cleanup remains in flight"}
        if self._ctx.control is not None and "shutdown" not in self._ctx.lock.quarantine:
            await asyncio.to_thread(self._ctx.control.end_all, reason="agent shutdown")
        if getattr(self, "_slam_monitor", None) is not None:
            self._slam_monitor.cancel()
            self._slam_monitor = None
        self._ctx.streams.close()
        for server in self._servers:
            server.close()
            await server.wait_closed()
        self._servers.clear()
        if self._ctx.motion_recorder is not None:
            await self._ctx.motion_recorder.close()
        if self._ctx.motion_handoff is not None:
            await self._ctx.motion_handoff.close()
        if self._ctx.shm is not None:
            self._ctx.shm.close()
        checkpoint = getattr(self._ctx.lock.quarantine, 'finish_boot', None)
        if checkpoint is not None and not self._ctx.commands.pending and not self._ctx.tasks.has_running() and not any(not t.done() for t in self._ctx.background_halts):
            checkpoint()

    # -- per-connection ------------------------------------------------------

    async def _handle(self, websocket: Any) -> None:
        session = Session(websocket, principal=self.access_policy.principal(websocket))
        self._sessions.add(session)
        self._ctx.audit.record(AuditKind.CONNECTION, client=session.client_id,
                               outcome=Outcome.SUCCESS,
                               detail={"event": "connected", "principal": session.principal.identity,
                                       "authenticated": session.principal.authenticated,
                                       "permissions": sorted(session.principal.permissions)})
        # Keep ordinary requests ordered, but never put renewals or stops behind
        # a slow read/adapter wait on the same connection. Both lanes are bounded.
        requests = asyncio.Queue(maxsize=32)
        priority = set()
        worker = asyncio.create_task(self._request_worker(session, requests))
        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    log.warning("client %d sent a binary frame; ignored", session.client_id)
                    continue
                try:
                    parsed = wire.parse_message(message)
                except ValueError:
                    parsed = {}
                method = parsed.get("method")
                if method == "lease.renew":
                    await self._serve_request(session, message)
                    continue
                from .admission import DESCRIPTORS
                descriptor = DESCRIPTORS.get(method) if isinstance(method, str) else None
                is_stop = descriptor is not None and descriptor.effect == "stop"
                params = parsed.get("params")
                is_stop |= method == "debug.set_estop" and isinstance(params, dict) and params.get("engaged") is True
                if is_stop and len(priority) < 8:
                    task = asyncio.create_task(self._serve_request(session, message))
                    priority.add(task)
                    task.add_done_callback(priority.discard)
                elif is_stop or requests.full():
                    await websocket.send(wire.error_response(parsed.get("id"), wire.ERR_APPLICATION,
                        "connection request capacity reached", self._error_data("ResourceBusy", method, parsed.get("id"))))
                else:
                    requests.put_nowait(message)
        except websockets.ConnectionClosed:
            pass
        finally:
            # No queued request or worker may acquire authority after disconnect.
            if session.lease is not None:
                session.lease.revoked = True
            with self._ctx.lock.mutex:
                self._ctx.lock.revoked_clients.add(session.client_id)
            pending = {worker, *priority}
            for task in pending:
                task.cancel()
            _, unfinished = await asyncio.wait(pending, timeout=15.)
            if unfinished:
                self._ctx.lock.quarantine[f"client:{session.client_id}"] = {
                    "reason": "request dispatch did not drain after disconnect"}
            await self._cleanup_session(session)

    async def _serve_request(self, session, message):
        try:
            reply = await self._dispatch(session, message)
            if reply is not None:
                await session.websocket.send(reply)
        except websockets.ConnectionClosed:
            pass
        except Exception:
            log.exception("request dispatch failed; closing the connection")
            await session.websocket.close(code=1011, reason="request dispatch failed")

    async def _request_worker(self, session, requests):
        while True:
            message = await requests.get()
            try:
                await self._serve_request(session, message)
            finally:
                requests.task_done()

    async def _cleanup_session(self, session):
        if session.cleanup_task is None:
            session.cleanup_task = asyncio.create_task(self._cleanup_session_once(session))
        await asyncio.shield(session.cleanup_task)

    async def _cleanup_session_once(self, session):
        # Reject worker-thread dispatch/acquisition before the first cleanup await.
        if session.lease is not None:
            session.lease.revoked = True
        with self._ctx.lock.mutex:
            self._ctx.lock.revoked_clients.add(session.client_id)
        self._sessions.discard(session)
        try:
            await self._ctx.commands.drain(session)
        except RpcError as exc:
            self._ctx.lock.quarantine[f'client:{session.client_id}'] = {'reason': 'command drain failed', 'error': str(exc)}
        # A vanished client must not leave the robot moving or locked.
        self._ctx.streams.on_disconnect(session)
        self._ctx.watches.on_disconnect(session.client_id)
        if self._ctx.teleop is not None:
            stop_report = None
            failed = True
            try:
                stop_report = await asyncio.to_thread(self._ctx.teleop.on_disconnect, session.client_id)
                failed = False
            except RpcError:
                log.exception("disconnect teleop cleanup failed; continuing other cleanup")
            finally:
                if session.motion_teleop_active and self._ctx.motion_recorder is not None:
                    fields = {'operation_id': 'teleop:' + session.reconciliation_key,
                              'session_id': session.reconciliation_key, 'owner': session.principal.identity}
                    if (isinstance(stop_report, dict) and stop_report.get('verified') is True
                            and stop_report.get('criterion') in ('joint-velocity-dwell', 'body-velocity-dwell')):
                        self._ctx.motion_recorder.emit('motion.stop_verified', stop_verified=True, **fields)
                    self._ctx.motion_recorder.emit('motion.failed' if failed else 'motion.cancelled', **fields)
                    session.motion_teleop_active = False
        if self._ctx.actions is not None:
            try:
                await asyncio.to_thread(self._ctx.actions.release, session.client_id)
            except RpcError:
                log.exception("disconnect vendor cleanup failed; continuing")
        # Revoke operations before restoring the controllers they use.
        # v1 has no detached motion, even when an old profile disabled cleanup.
        cancelled = self._ctx.tasks.cancel_for_session(session)
        clean = True
        try:
            await self._ctx.tasks.drain_session(session)
        except RpcError as exc:
            clean = False
            self._ctx.lock.quarantine[f"client:{session.client_id}"] = {
                "reason": "disconnect cleanup unverified", "error": str(exc)}
            log.exception("disconnect cleanup left motion quarantined")
        if clean and self._ctx.control is not None:
            closed = await asyncio.to_thread(self._ctx.control.on_disconnect, session.client_id)
            if closed:
                log.warning("client %d vanished holding %d scope(s); inspect cleanup reports",
                            session.client_id, closed)
        if cancelled:
            self._ctx.audit.record(
                AuditKind.SAFETY, client=session.client_id,
                outcome=Outcome.SUCCESS if clean else Outcome.ERROR,
                detail={"reason": "stop_on_disconnect", "cancelled": cancelled,
                        "cleanup_confirmed": clean})
        if clean and self._ctx.guards is not None:
            self._ctx.guards.disconnected_after_cleanup(session.client_id)
        if self._ctx.lock.release_all(session.client_id):
            log.info("released motion lock of client %d",
                     session.client_id)
        self._ctx.audit.record(AuditKind.CONNECTION,
                               client=session.client_id,
                               outcome=Outcome.SUCCESS,
                               detail={"event": "disconnected"})

    async def _monitor_leases(self):
        policy = self._ctx.profile.spec.safety.lease
        while True:
            await asyncio.sleep(min(policy.renew_interval / 2, 0.25))
            self._expire_leases()

    def _expire_leases(self):
        """Revoke first; cleanup and socket close never block monitoring."""
        for session in tuple(self._sessions):
            lease = session.lease
            if lease is None or session.cleanup_task is not None:
                continue
            if lease.revoked or self._lease_clock() >= lease.deadline:
                lease.revoked = True
                with self._ctx.lock.mutex:
                    self._ctx.lock.revoked_clients.add(session.client_id)
                self._ctx.tasks.cancel_for_session(session)
                task = asyncio.create_task(self._expire_session(session))
                self._ctx.background_halts.add(task)
                task.add_done_callback(self._ctx.background_halts.discard)
                # Claim cleanup immediately, before another monitor tick.
                session.cleanup_task = asyncio.create_task(self._cleanup_session_once(session))

    async def _expire_session(self, session):
        try:
            await self._cleanup_session(session)
        except Exception as exc:
            with self._ctx.lock.mutex:
                self._ctx.lock.quarantine[f"client:{session.client_id}"] = {
                    "reason": "lease expiry cleanup failed", "error": str(exc)}
            log.exception("lease expiry cleanup failed")
        finally:
            await session.websocket.close(code=1008, reason="connection lease expired")

    async def _dispatch(self, session: Session, text: str) -> str | None:
        """Bind validated mutations before any handler can dispatch an effect."""
        from .admission import DESCRIPTORS, resolve_descriptor
        try:
            msg = wire.parse_message(text)
        except ValueError:
            return await self._execute(session, text)
        method, params = msg.get('method'), msg.get('params', {})
        if method not in self._registry or method not in DESCRIPTORS or not session.handshaken or msg.get('id') is None:
            return await self._execute(session, text)
        try:
            descriptor = DESCRIPTORS[method]
            descriptor.schema.validate(params)
            descriptor = resolve_descriptor(method, params, self._ctx.profile)
            session.principal.require(descriptor.permission)
            if descriptor.effect != 'read':
                if self._ctx.commands.failed:
                    raise RpcError('CleanupFailed', 'command storage is inhibited; operator recovery required')
                return await self._ctx.commands.execute(session, msg.get('command_id'), method, params,
                    lambda: self._execute(session, text), msg['id'])
        except RpcError as exc:
            # Storage faults must never block the independently authorized stop path.
            if self._ctx.commands.failed and (descriptor.effect == 'stop' or (method == 'debug.set_estop' and params.get('engaged') is True)):
                return await self._execute(session, text)
            self._motion_refused(session, method)
            data = self._error_data(exc.error_type, method, msg.get('id'))
            data.update(exc.extra)
            return wire.error_response(msg.get('id'), wire.ERR_APPLICATION, str(exc), data)
        return await self._execute(session, text)

    def _motion_refused(self, session, method):
        recorder = self._ctx.motion_recorder
        if recorder is not None and method in _MOTION_STARTS:
            recorder.emit('motion.refused', session_id=session.reconciliation_key,
                          owner=session.principal.identity)

    def _motion_stop_failed(self, session, method):
        recorder = self._ctx.motion_recorder
        if recorder is not None and method in ('teleop.stop', 'teleop.end', 'mobility.stop'):
            recorder.emit('motion.failed', operation_id='teleop:' + session.reconciliation_key if method.startswith('teleop.') else None,
                          session_id=session.reconciliation_key, owner=session.principal.identity)
            if method == 'teleop.end':
                session.motion_teleop_active = False

    async def _execute(self, session: Session, text: str) -> str | None:
        try:
            msg = wire.parse_message(text)
        except ValueError as exc:
            return wire.error_response(None, wire.ERR_PARSE, str(exc))

        request_id = msg.get("id")
        method = msg.get("method")
        if method is None:
            if request_id is None:
                return None  # stray notification from client; ignore
            return wire.error_response(
                request_id, wire.ERR_INVALID_REQUEST, "missing method"
            )
        if request_id is None:
            return None  # client notifications are not part of protocol 1

        handler = self._registry.get(method)
        if handler is None:
            return wire.error_response(
                request_id, wire.ERR_METHOD_NOT_FOUND,
                f"unknown method {method!r}",
                self._error_data("RingfourError", method, request_id),
            )
        if method != "hello" and not session.handshaken:
            return wire.error_response(
                request_id, wire.ERR_INVALID_REQUEST,
                "handshake required: call hello first",
                self._error_data("ProtocolMismatch", method, request_id),
            )

        params = msg.get("params", {})
        if not isinstance(params, dict):
            return wire.error_response(request_id, wire.ERR_INVALID_REQUEST,
                                       "RPC params must be an object")
        audit = self._ctx.audit
        try:
            from .admission import DESCRIPTORS, resolve_descriptor
            descriptor = DESCRIPTORS[method]
            descriptor.schema.validate(params)
            descriptor = resolve_descriptor(method, params, self._ctx.profile)
            session.principal.require(descriptor.permission)
            if session.lease is not None and (descriptor.effect == "motion" or descriptor.ownership == "configuration-reservation"):
                try:
                    session.lease.check(now=self._lease_clock())
                except RpcError:
                    with self._ctx.lock.mutex:
                        self._ctx.lock.revoked_clients.add(session.client_id)
                    self._expire_leases()
                    raise
            from .commands import current_command
            audit.record(AuditKind.COMMAND, client=session.client_id, call=method,
                         detail={"event": "admitted", "command_id": current_command.get(),
                                 "principal": session.principal.identity, "session": session.reconciliation_key,
                                 "params": params, "admission": {"effect": descriptor.effect,
                                 "resources": descriptor.resources(params),
                                 "ownership": descriptor.ownership,
                                 "permission": descriptor.permission,
                                 "completion": descriptor.completion}})
            self._ctx.safety.check(session.client_id, method, descriptor=descriptor)
        except RpcError as exc:
            self._motion_refused(session, method)
            audit.record(AuditKind.SAFETY, client=session.client_id,
                         call=method, outcome=Outcome.REJECTED,
                         detail={"reason": exc.extra.get("reason"),
                                 "message": str(exc)})
            data = self._error_data(exc.error_type, method, request_id)
            data.update(exc.extra)
            return wire.error_response(
                request_id, wire.ERR_APPLICATION, str(exc), data
            )
        reservation = None
        try:
            if descriptor.ownership == 'configuration-reservation':
                import uuid
                reservation = 'configuration:' + uuid.uuid4().hex
                acquiring = asyncio.create_task(asyncio.to_thread(self._ctx.lock.acquire,
                    session.client_id, reservation, resources=descriptor.resources(params)))
                try:
                    granted = await asyncio.shield(acquiring)
                except asyncio.CancelledError:
                    if await acquiring:
                        await asyncio.to_thread(self._ctx.lock.release, session.client_id, reservation)
                    raise
                if not granted:
                    raise RpcError("ResourceBusy", "configuration conflicts with the robot's motion owner")
            result = await handler(session, params)
        except RpcError as exc:
            self._motion_refused(session, method)
            self._motion_stop_failed(session, method)
            audit.record(AuditKind.RESULT, client=session.client_id,
                         call=method, outcome=Outcome.ERROR,
                         detail={"type": exc.error_type, "message": str(exc)})
            data = self._error_data(exc.error_type, method, request_id)
            data.update(exc.extra)
            return wire.error_response(
                request_id, wire.ERR_APPLICATION, str(exc), data
            )
        except Exception as exc:  # handler bug: report, don't kill the link
            self._motion_refused(session, method)
            self._motion_stop_failed(session, method)
            log.exception("handler %s failed", method)
            audit.record(AuditKind.RESULT, client=session.client_id,
                         call=method, outcome=Outcome.ERROR,
                         detail={"type": "RingfourError",
                                 "message": f"internal: {exc}"})
            return wire.error_response(
                request_id, wire.ERR_APPLICATION,
                f"internal error in {method}: {exc}",
                self._error_data("RingfourError", method, request_id),
            )
        finally:
            if reservation is not None:
                release = asyncio.create_task(asyncio.to_thread(self._ctx.lock.release,
                    session.client_id, reservation))
                await asyncio.shield(release)
        audit.record(AuditKind.RESULT, client=session.client_id,
                     call=method, outcome=Outcome.SUCCESS,
                     detail={'event': 'handler-acknowledgement', 'command_id': current_command.get(),
                             'principal': session.principal.identity})
        recorder = self._ctx.motion_recorder
        if recorder is not None and method in ('teleop.start', 'teleop.stop', 'teleop.end', 'mobility.stop'):
            fields = {'operation_id': 'teleop:' + session.reconciliation_key if method.startswith('teleop.') else None,
                      'session_id': session.reconciliation_key, 'owner': session.principal.identity}
            if method == 'teleop.start':
                recorder.emit('motion.accepted', **fields)
                session.motion_teleop_active = True
            else:
                evidence = result.get('stop_verification') if isinstance(result, dict) else None
                if (isinstance(evidence, dict) and evidence.get('verified') is True
                        and evidence.get('criterion') in ('joint-velocity-dwell', 'body-velocity-dwell')):
                    recorder.emit('motion.stop_verified', stop_verified=True, **fields)
                if method == 'teleop.end' and session.motion_teleop_active:
                    recorder.emit('motion.completed', **fields)
                    session.motion_teleop_active = False
        return wire.response(request_id, result)

    # -- audit tail fan-out ------------------------------------------------------

    def _fan_out_audit_event(self, event: Any) -> None:
        """AuditLog subscriber: push audit.event to subscribed sessions.

        Called synchronously from record() on the server's loop; the sends
        are scheduled, never awaited, so a slow client cannot stall
        dispatch.
        """
        subscribed = [s for s in self._sessions if s.audit_subscribed]
        if not subscribed:
            return
        text = wire.notification("audit.event", event.model_dump(mode="json"))
        for session in subscribed:
            asyncio.ensure_future(self._send_quietly(session, text))

    def _fan_out_event(self, event: dict) -> None:
        """EventHub subscriber: push system events (estop, watch fires)
        to sessions that called events.subscribe."""
        subscribed = [s for s in self._sessions if s.events_subscribed]
        if not subscribed:
            return
        text = wire.notification("event", event)
        for session in subscribed:
            asyncio.ensure_future(self._send_quietly(session, text))

    @staticmethod
    async def _send_quietly(session: Session, text: str) -> None:
        try:
            await session.websocket.send(text)
        except Exception:
            pass  # client gone; the disconnect path cleans up

    def _error_data(self, error_type: str, call: str,
                    request_id: int | str | None) -> dict[str, Any]:
        return {
            "type": error_type,
            "robot": self._robot_name,
            "call": call,
            "request_id": request_id,
        }
