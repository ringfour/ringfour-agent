"""Optional local journal reader; never serves robot-control requests."""
from __future__ import annotations

import asyncio
import ctypes
import json
import os
import socket
import sqlite3
import stat
import struct
from pathlib import Path

from .motion_journal import JournalError, JournalGap

_MAX_REQUEST = 4096


def _peer_uid(sock):
    if hasattr(socket, 'SO_PEERCRED'):
        return struct.unpack('3i', sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))[1]
    # macOS/BSD expose getpeereid through libc, rather than Python's socket API.
    libc = ctypes.CDLL(None, use_errno=True)
    uid, gid = ctypes.c_uint(), ctypes.c_uint()
    try:
        result = libc.getpeereid(sock.fileno(), ctypes.byref(uid), ctypes.byref(gid))
    except AttributeError as exc:
        raise OSError('Unix peer credentials are unavailable') from exc
    if result:
        raise OSError(ctypes.get_errno(), 'cannot authenticate Unix peer')
    return uid.value


class MotionHandoff:
    def __init__(self, journal, path, *, reader_uid=None):
        self.journal = journal
        self.path = Path(path)
        self.reader_uid = os.getuid() if reader_uid is None else reader_uid
        self.server = None

    async def start(self):
        # A private directory prevents substitution or a connection before chmod.
        parent = self.path.parent.stat()
        if parent.st_uid != os.getuid() or parent.st_mode & 0o077:
            raise OSError('Motion handoff directory must be private and owned by the Agent UID')
        if self.path.exists():
            existing = self.path.lstat()
            if not stat.S_ISSOCK(existing.st_mode) or existing.st_uid != os.getuid():
                raise OSError('Motion handoff path already exists')
            try:
                _, writer = await asyncio.open_unix_connection(str(self.path))
            except (ConnectionRefusedError, FileNotFoundError):
                self.path.unlink(missing_ok=True)
            else:
                writer.close()
                await writer.wait_closed()
                raise OSError('Motion handoff socket is already active')
        self.server = await asyncio.start_unix_server(self._handle, path=str(self.path), limit=_MAX_REQUEST + 1)
        os.chmod(self.path, 0o600)

    async def close(self):
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

    async def _handle(self, reader, writer):
        try:
            sock = writer.get_extra_info('socket')
            if sock is None or _peer_uid(sock) != self.reader_uid:
                return
            try:
                raw = await asyncio.wait_for(reader.readline(), 5)
                if not raw.endswith(b'\n') or len(raw) > _MAX_REQUEST:
                    return
                request = json.loads(raw)
                if not isinstance(request, dict):
                    raise TypeError('invalid request')
                method = request.get('method')
                if method == 'read' and set(request) == {'method', 'stream_id', 'after', 'limit'}:
                    result = await asyncio.to_thread(self.journal.read, request['stream_id'], request['after'], limit=request['limit'])
                elif method == 'acknowledge' and set(request) == {'method', 'stream_id', 'sequence'}:
                    await asyncio.to_thread(self.journal.acknowledge, request['stream_id'], request['sequence'])
                    result = None
                else:
                    raise ValueError('invalid request')
                response = {'ok': True, 'result': result}
            except JournalGap:
                response = {'ok': False, 'error': 'history_gap'}
            except (ValueError, TypeError, JournalError):
                response = {'ok': False, 'error': 'invalid_request'}
            except (sqlite3.Error, OSError, TimeoutError):
                response = {'ok': False, 'error': 'journal_unavailable'}
            writer.write(json.dumps(response, separators=(',', ':')).encode() + b'\n')
            await writer.drain()
        except (ConnectionError, OSError, TimeoutError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
