"""Connection admission policy; never accept authority in RPC parameters.

Unix identity comes from kernel peer credentials. Explicitly trusted TCP is a
legacy/simulator deployment boundary, not authenticated or encrypted transport.
Untrusted TCP gets observation only. Remote principal authentication remains a
separate deployment contract.
"""
from dataclasses import dataclass, field
import os
import socket
import struct

PERMISSIONS = frozenset({'observe', 'operate', 'stop', 'configure', 'reset-inhibit', 'recover'})
OPERATOR = PERMISSIONS - {'recover'}


@dataclass(frozen=True)
class Principal:
    identity: str
    permissions: frozenset[str]
    authenticated: bool

    def require(self, permission):
        from .server import RpcError
        if permission not in self.permissions:
            raise RpcError('PermissionDenied', f'{permission} permission is required',
                           {'reason': 'permission denied', 'permission': permission,
                            'principal': self.identity})


@dataclass(frozen=True)
class AccessPolicy:
    local_uid: int = field(default_factory=os.getuid)
    trusted_tcp: bool = False

    def principal(self, websocket):
        transport = getattr(websocket, 'transport', None)
        peer = transport.get_extra_info('socket') if transport else None
        if peer is not None and peer.family == socket.AF_UNIX:
            try:
                _, uid, _ = struct.unpack('3i', peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
            except (OSError, AttributeError):
                return Principal('unix:unverified', frozenset({'observe'}), False)
            grants = PERMISSIONS if uid == self.local_uid else frozenset({'observe'})
            return Principal(f'unix:uid:{uid}', grants, True)
        return Principal('tcp:trusted-deployment' if self.trusted_tcp else 'tcp:anonymous',
                         OPERATOR if self.trusted_tcp else frozenset({'observe'}), False)
