"""Durable acceptance/result ledger. A crash never causes automatic replay.

Command IDs encode their immutable epoch and expiry. Eviction cannot turn an old
ID into fresh work. Request fingerprints, not request payloads, are retained.
"""
from __future__ import annotations
import asyncio
from contextvars import ContextVar
import os
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time
from .server import RpcError

current_command = ContextVar('current_command', default=None)

_ID = re.compile(r'^([0-9a-f]{32})\.([0-9]{1,16})\.([0-9a-f]{32})$')


class CommandLedger:
    validity = 300.0
    retention = 300.0
    capacity = 10000

    def __init__(self, epoch, *, path=None, clock=None, on_failure=None, audit=None):
        self.epoch = epoch
        self.durable = path is not None
        self.on_failure = on_failure
        self.audit = audit
        self.failed = False
        self.pending = {}
        self._clock = clock or time.time
        self._anchor = self._clock()
        self._monotonic = time.monotonic()
        self._real_clock = clock is None
        self._last = self._anchor
        if path is not None:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.db = sqlite3.connect(str(path) if path else ':memory:', check_same_thread=False)
            if path is not None:
                os.chmod(path, 0o600)
            self.db.execute('PRAGMA journal_mode=WAL')
            self.db.execute('PRAGMA synchronous=FULL')
            self.db.executescript('''
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS commands (
                    principal TEXT NOT NULL, id TEXT NOT NULL, owner TEXT NOT NULL,
                    epoch TEXT NOT NULL, method TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    expires REAL NOT NULL, retain_until REAL NOT NULL,
                    state TEXT NOT NULL, reply TEXT,
                    PRIMARY KEY(principal,id));
                CREATE TABLE IF NOT EXISTS operations (
                    id TEXT PRIMARY KEY, principal TEXT NOT NULL, owner TEXT NOT NULL,
                    epoch TEXT NOT NULL, state TEXT NOT NULL, snapshot TEXT NOT NULL,
                    retain_until REAL);
            ''')
            row = self.db.execute("SELECT value FROM metadata WHERE key='clock_floor'").fetchone()
            if row:
                self._last = max(self._last, row[0])
            now = self.now()
            with self.db:
                self.db.execute("UPDATE commands SET state='unknown', retain_until=max(retain_until,?) WHERE state='pending'", (now+self.retention,))
                self.db.execute("UPDATE operations SET state='unknown', retain_until=? WHERE state='active'", (now+self.retention,))
            self.prune()
        except (sqlite3.Error, OSError) as exc:
            self.storage_failure(exc)

    def storage_failure(self, exc):
        already_failed = self.failed
        self.failed = True
        if self.on_failure and not already_failed:
            self.on_failure(str(exc))
        raise RpcError('CleanupFailed', 'durable command/result storage unavailable; new effects inhibited',
                       {'reason': 'journal unavailable', 'retry': 'operator-recovery'}) from exc

    def now(self):
        advancing = self._anchor + time.monotonic()-self._monotonic if self._real_clock else self._clock()
        self._last = max(self._last, self._clock(), advancing)
        return self._last

    def prune(self):
        now = self.now()
        try:
            with self.db:
                self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('clock_floor',?)", (now,))
                self.db.execute("DELETE FROM commands WHERE state!='pending' AND retain_until<?", (now,))
                self.db.execute("DELETE FROM operations WHERE retain_until IS NOT NULL AND retain_until<?", (now,))
        except sqlite3.Error as exc:
            self.storage_failure(exc)

    @staticmethod
    def parse(command_id):
        match = _ID.fullmatch(command_id) if isinstance(command_id, str) else None
        if not match:
            raise RpcError('CommandRequired', 'a client-generated epoch.expiry.uuid command ID is required')
        return match[1], int(match[2])/1000

    @staticmethod
    def authorize(session, principal, owner):
        if session.principal.identity != principal or (not session.principal.authenticated and session.reconciliation_key != owner):
            raise RpcError('PermissionDenied', 'record belongs to another principal or unverified connection')

    @staticmethod
    def canonical(value):
        if isinstance(value, dict):
            return {key: CommandLedger.canonical(item) for key,item in value.items()}
        if isinstance(value, list):
            return [CommandLedger.canonical(item) for item in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def feature(self, methods):
        return {'version': 1, 'epoch': self.epoch, 'server_time': self.now(),
                'validity_seconds': self.validity, 'retention_seconds': self.retention,
                'capacity': self.capacity, 'durable': self.durable,
                'methods': sorted(methods), 'reconnect': 'authenticated-principal-only'}

    async def execute(self, session, command_id, method, params, dispatch, request_id):
        epoch, expires = self.parse(command_id)
        now = self.now()
        if epoch != self.epoch:
            raise RpcError('AgentEpochChanged', 'old-epoch commands cannot execute on this boot',
                           {'command_id': command_id, 'epoch': self.epoch, 'retry': 'reconcile-first'})
        if expires <= now:
            raise RpcError('CommandExpired', 'command validity expired; this ID cannot execute again',
                           {'command_id': command_id, 'retry': 'reconcile-first'})
        if expires > now+self.validity+.001:
            raise RpcError('CommandExpired', 'command expiry exceeds the advertised validity window')
        self.prune()
        principal = session.principal.identity
        key = (principal, command_id)
        fingerprint = hashlib.sha256(json.dumps([method, self.canonical(params)], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        try:
            row = self.db.execute('SELECT owner,fingerprint,state,reply FROM commands WHERE principal=? AND id=?', key).fetchone()
            if row:
                self.authorize(session, principal, row[0])
                if row[1] != fingerprint:
                    raise RpcError('CommandConflict', 'command ID is already bound to another request', {'command_id': command_id})
                if row[2] == 'pending' and key in self.pending:
                    reply = await asyncio.shield(self.pending[key][1])
                elif row[2] == 'completed':
                    reply = json.loads(row[3])
                else:
                    raise RpcError('SubmissionUnknown', 'prior dispatch outcome is unknown; automatic replay is forbidden',
                                   {'command_id': command_id, 'retry': 'operator-recovery'})
            else:
                if self.db.execute('SELECT count(*) FROM commands').fetchone()[0] >= self.capacity:
                    raise RpcError('ResourceBusy', 'command record capacity reached; wait for retention expiry')
                with self.db:
                    self.db.execute('INSERT INTO commands VALUES (?,?,?,?,?,?,?,?,?,NULL)',
                        (principal, command_id, session.reconciliation_key, epoch, method, fingerprint,
                         expires, max(expires, now+self.retention), 'pending'))
                # No await between the durable claim and publishing the shared task.
                if self.audit is not None:
                    from ringfour.enums import AuditKind
                    self.audit.record(AuditKind.COMMAND, client=session.client_id, call=method,
                        detail={'event': 'accepted', 'command_id': command_id, 'epoch': epoch,
                                'principal': principal, 'session': session.reconciliation_key})
                task = asyncio.create_task(self._run(key, dispatch))
                self.pending[key] = (session.reconciliation_key, task)
                task.add_done_callback(lambda done: self._forget(key, done))
                reply = await asyncio.shield(task)
        except sqlite3.Error as exc:
            self.storage_failure(exc)
        return json.dumps({**reply, 'id': request_id})

    def _forget(self, key, task):
        self.pending.pop(key, None)
        if not task.cancelled():
            task.exception()  # An abandoned caller must not leave an unobserved exception.

    async def _run(self, key, dispatch):
        token = current_command.set(key[1])
        try:
            reply = json.loads(await dispatch())
        finally:
            current_command.reset(token)
        reply.pop('id', None)
        if 'error' in reply:
            reply['error'].setdefault('data', {})['command_id'] = key[1]
        try:
            with self.db:
                self.db.execute("UPDATE commands SET state='completed',reply=?,retain_until=max(retain_until,?) WHERE principal=? AND id=?",
                                (json.dumps(reply, allow_nan=False), self.now()+self.retention, *key))
        except sqlite3.Error as exc:
            self.storage_failure(exc)
        return reply

    def get(self, session, command_id):
        self.parse(command_id)
        self.prune()
        principal = session.principal.identity
        row = self.db.execute('SELECT owner,epoch,method,state,reply,expires,retain_until FROM commands WHERE principal=? AND id=?', (principal, command_id)).fetchone()
        if row is None:
            raise RpcError('CommandExpired', 'command is unknown or outside advertised retention; do not infer non-execution',
                           {'command_id': command_id, 'retry': 'reconcile-first'})
        self.authorize(session, principal, row[0])
        reply = json.loads(row[4]) if row[4] else {}
        result = reply.get('result')
        operation_id = result.get('task_id') if isinstance(result, dict) else None
        return {'command_id': command_id, 'epoch': row[1], 'method': row[2], 'state': row[3],
                'result': result, 'error': reply.get('error'), 'operation_id': operation_id,
                'expires_at': row[5], 'retained_until': row[6]}

    def task_started(self, record):
        self.prune()
        if self.db.execute('SELECT count(*) FROM operations').fetchone()[0] >= self.capacity:
            raise RpcError('ResourceBusy', 'durable operation capacity reached; wait for retention expiry')
        snapshot = {'task_id': record.id, 'state': 'pending', 'sequence': 0}
        try:
            with self.db:
                self.db.execute('INSERT INTO operations VALUES (?,?,?,?,?,?,NULL)',
                    (record.id, record.session.principal.identity, record.session.reconciliation_key,
                     self.epoch, 'active', json.dumps(snapshot)))
        except sqlite3.Error as exc:
            self.storage_failure(exc)

    def task_finished(self, record, update):
        try:
            with self.db:
                self.db.execute("UPDATE operations SET state='terminal',snapshot=?,retain_until=? WHERE id=?",
                                (json.dumps(update, allow_nan=False), self.now()+self.retention, record.id))
        except sqlite3.Error as exc:
            self.storage_failure(exc)

    def operation(self, session, operation_id):
        self.prune()
        row = self.db.execute('SELECT principal,owner,epoch,state,snapshot,retain_until FROM operations WHERE id=?', (operation_id,)).fetchone()
        if row is None:
            raise RpcError('CommandExpired', 'operation is unknown or outside advertised retention', {'operation_id': operation_id})
        self.authorize(session, row[0], row[1])
        snapshot = json.loads(row[4])
        if row[5] is not None:
            snapshot['report_age'] = max(0.0, self.now()-(row[5]-self.retention))
        return {'operation_id': operation_id, 'epoch': row[2], 'state': row[3],
                'snapshot': snapshot, 'retained_until': row[5],
                'physical_outcome': 'unknown' if row[3] == 'unknown' else 'recorded'}

    async def drain(self, session=None):
        pending = [task for owner,task in self.pending.values() if session is None or owner == session.reconciliation_key]
        if pending:
            _, unfinished = await asyncio.wait(pending, timeout=15)
            if unfinished:
                raise RpcError('CleanupFailed', 'accepted command dispatch did not drain')
