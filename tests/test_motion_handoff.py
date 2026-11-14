import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest
from ringfour_agent.motion_handoff import MotionHandoff
from ringfour_agent.motion_journal import MotionJournal


async def request(path, body):
    reader, writer = await asyncio.open_unix_connection(str(path))
    writer.write(json.dumps(body).encode() + b'\n')
    await writer.drain()
    raw = await reader.readline()
    writer.close()
    await writer.wait_closed()
    return json.loads(raw) if raw else None


def test_handoff_auth_replay_ack_and_private_scope(tmp_path):
    async def exercise(path):
        journal = MotionJournal(tmp_path / 'motion.sqlite3')
        event = journal.append('motion.accepted')
        server = MotionHandoff(journal, path)
        await server.start()
        try:
            with pytest.raises(OSError, match='already active'):
                await MotionHandoff(journal, path).start()
            body = {'method': 'read', 'stream_id': None, 'after': 0, 'limit': 10}
            result = await request(path, body)
            assert result['result']['events'] == [event]
            assert await request(path, {'method': 'move'}) == {'ok': False, 'error': 'invalid_request'}
            await request(path, {'method': 'acknowledge', 'stream_id': event['stream_id'], 'sequence': 1})
            assert await request(path, body) == {'ok': False, 'error': 'history_gap'}
        finally:
            await server.close()
        denied = MotionHandoff(journal, path, reader_uid=os.getuid() + 1)
        await denied.start()
        try:
            assert await request(path, body) is None
        finally:
            await denied.close()
    with tempfile.TemporaryDirectory(dir='/tmp', prefix='r4-motion-') as socket_dir:
        asyncio.run(exercise(Path(socket_dir) / 'motion.sock'))


def test_handoff_rejects_shared_parent_directory(tmp_path):
    async def exercise(path):
        path.parent.chmod(0o755)
        with pytest.raises(OSError, match='private'):
            await MotionHandoff(MotionJournal(tmp_path / 'journal'), path).start()
    with tempfile.TemporaryDirectory(dir='/tmp', prefix='r4-motion-') as socket_dir:
        asyncio.run(exercise(Path(socket_dir) / 'motion.sock'))
