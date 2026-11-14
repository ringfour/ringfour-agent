"""Optional durable motion-event journal; not connected to control paths yet.

SQLite writes may block on I/O. Never call append from a physical stop path.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path


class JournalError(ValueError):
    """The journal cannot safely store or replay the requested event."""


class JournalFull(JournalError):
    """History capacity is exhausted; no event was silently discarded."""


class JournalGap(JournalError):
    """The requested events have already been acknowledged and removed."""


_TYPES = frozenset({'motion.accepted', 'motion.refused', 'motion.completed',
                    'motion.cancelled', 'motion.failed', 'motion.stop_verified',
                    'motion.guard_tripped', 'motion.history_incomplete'})
_MAX_SEQUENCE = 9007199254740991


class MotionJournal:
    """Persist a bounded stream and its acknowledgement checkpoint in SQLite."""

    def __init__(self, path, *, capacity=4096):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('capacity must be positive')
        self.path = Path(path)
        self.capacity = capacity
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Create with private permissions before SQLite writes anything. The
        # process umask may otherwise allow a short world-readable window.
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            pass
        else:
            os.close(fd)
        os.chmod(self.path, 0o600)
        with closing(self._connect()) as conn, conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS motion_stream (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    stream_id TEXT NOT NULL,
                    last_sequence INTEGER NOT NULL CHECK(last_sequence>=0),
                    acknowledged INTEGER NOT NULL CHECK(acknowledged BETWEEN 0 AND last_sequence)
                );
                CREATE TABLE IF NOT EXISTS motion_events (
                    sequence INTEGER PRIMARY KEY,
                    event_id TEXT NOT NULL UNIQUE,
                    body TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS motion_boot (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    dirty INTEGER NOT NULL CHECK(dirty IN (0,1))
                );
                INSERT OR IGNORE INTO motion_boot VALUES (1,0);
            ''')
            conn.execute('INSERT OR IGNORE INTO motion_stream VALUES (1,?,0,0)', (str(uuid.uuid4()),))

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=5)
        conn.execute('PRAGMA synchronous=FULL')
        return conn

    def begin_boot(self):
        """Mark this producer boot dirty before it can accept observations."""
        with closing(self._connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            dirty = bool(conn.execute('SELECT dirty FROM motion_boot WHERE singleton=1').fetchone()[0])
            conn.execute('UPDATE motion_boot SET dirty=1 WHERE singleton=1')
        return dirty

    def finish_boot(self):
        """Clear only after the writer has persisted every queued event or gap."""
        with closing(self._connect()) as conn, conn:
            conn.execute('UPDATE motion_boot SET dirty=0 WHERE singleton=1')

    def append(self, event_type, *, operation_id=None, session_id=None, owner=None,
               stop_verified=None, event_id=None, occurred_at=None):
        if event_type not in _TYPES or (event_type == 'motion.stop_verified' and stop_verified is not True):
            raise JournalError('Invalid motion event or unverified stop.')
        if event_type != 'motion.stop_verified' and stop_verified is not None:
            raise JournalError('Unexpected stop verification.')
        for value in (operation_id, session_id, owner):
            if value is not None and (not isinstance(value, str) or '\x00' in value or not 1 <= len(value) <= 200):
                raise JournalError('Invalid motion attribution.')
        event_id = event_id or str(uuid.uuid4())
        try:
            if str(uuid.UUID(event_id)) != event_id:
                raise ValueError
        except (ValueError, TypeError, AttributeError):
            raise JournalError('Invalid event ID.') from None
        stamp = occurred_at or datetime.now(UTC)
        if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
            raise JournalError('Invalid event timestamp.')
        with closing(self._connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            stream, last = conn.execute('SELECT stream_id,last_sequence FROM motion_stream WHERE singleton=1').fetchone()
            if conn.execute('SELECT count(*) FROM motion_events').fetchone()[0] >= self.capacity or last >= _MAX_SEQUENCE:
                raise JournalFull('Motion event journal is full.')
            event = {'event_id': event_id, 'stream_id': stream, 'sequence': last + 1,
                     'occurred_at': stamp.astimezone(UTC).isoformat(timespec='microseconds').replace('+00:00', 'Z'),
                     'event_type': event_type, 'operation_id': operation_id, 'session_id': session_id,
                     'owner': owner, 'stop_verified': stop_verified}
            try:
                conn.execute('INSERT INTO motion_events VALUES (?,?,?)',
                             (last + 1, event_id, json.dumps(event, separators=(',', ':'), ensure_ascii=False)))
                conn.execute('UPDATE motion_stream SET last_sequence=? WHERE singleton=1', (last + 1,))
            except sqlite3.IntegrityError as exc:
                raise JournalError('Event ID already exists.') from exc
        return event

    def read(self, stream_id, after, *, limit=100):
        if type(after) is not int or after < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise JournalError('Invalid motion cursor.')
        with closing(self._connect()) as conn, conn:
            # Keep the checkpoint and event rows in one snapshot. An
            # acknowledgement from another connection may remove rows while a
            # read is in progress; mixing snapshots can report a false gap.
            conn.execute('BEGIN')
            stream, last, ack = conn.execute('SELECT stream_id,last_sequence,acknowledged FROM motion_stream WHERE singleton=1').fetchone()
            if stream_id is not None and stream_id != stream:
                raise JournalGap('Motion event stream changed.')
            if after < ack:
                raise JournalGap('Motion event history has already been acknowledged.')
            if after > last:
                raise JournalError('Motion cursor is ahead of the journal.')
            rows = conn.execute('SELECT sequence,body FROM motion_events WHERE sequence>? ORDER BY sequence LIMIT ?', (after, limit)).fetchall()
        if rows and (rows[0][0] != after + 1 or any(row[0] != after + index + 1 for index, row in enumerate(rows))):
            raise JournalGap('Motion event history has a gap.')
        if not rows and after < last:
            raise JournalGap('Motion event history has a gap.')
        return {'stream_id': stream, 'acknowledged_sequence': ack, 'last_sequence': last,
                'events': [json.loads(row[1]) for row in rows]}

    def acknowledge(self, stream_id, sequence):
        if type(sequence) is not int or sequence < 0:
            raise JournalError('Invalid acknowledgement.')
        with closing(self._connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            stream, last, ack = conn.execute('SELECT stream_id,last_sequence,acknowledged FROM motion_stream WHERE singleton=1').fetchone()
            if stream != stream_id:
                raise JournalGap('Motion event stream changed.')
            if sequence > last or sequence < ack:
                raise JournalError('Invalid acknowledgement.')
            if sequence > ack:
                count, first, final = conn.execute(
                    'SELECT count(*),min(sequence),max(sequence) FROM motion_events '
                    'WHERE sequence>? AND sequence<=?', (ack, sequence)
                ).fetchone()
                if count != sequence - ack or first != ack + 1 or final != sequence:
                    raise JournalGap('Cannot acknowledge a gap in motion event history.')
            conn.execute('DELETE FROM motion_events WHERE sequence<=?', (sequence,))
            conn.execute('UPDATE motion_stream SET acknowledged=? WHERE singleton=1', (sequence,))
