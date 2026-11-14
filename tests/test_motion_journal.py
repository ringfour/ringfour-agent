import json
import sqlite3
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
from ringfour_agent.motion_journal import (
    JournalError,
    JournalFull,
    JournalGap,
    MotionJournal,
)


def test_durable_replay_ack_and_gap(tmp_path):
    path = tmp_path / 'motion.sqlite3'
    journal = MotionJournal(path)
    first = journal.append('motion.accepted', operation_id='task-1', owner='operator')
    second = journal.append('motion.stop_verified', operation_id='task-1', stop_verified=True)
    reopened = MotionJournal(path)
    assert reopened.read(first['stream_id'], 0)['events'] == [first, second]
    reopened.acknowledge(first['stream_id'], 1)
    assert reopened.read(first['stream_id'], 1)['events'] == [second]
    with pytest.raises(JournalGap):
        reopened.read(first['stream_id'], 0)
    with pytest.raises(JournalGap):
        reopened.read(str(uuid.uuid4()), 1)
    with pytest.raises(JournalError):
        reopened.acknowledge(first['stream_id'], 3)
    reopened.acknowledge(first['stream_id'], 2)
    third = reopened.append('motion.guard_tripped')
    assert third['sequence'] == 3 and third['stream_id'] == first['stream_id']


def test_capacity_duplicates_and_stop_evidence_do_not_lose_history(tmp_path):
    path = tmp_path / 'motion.sqlite3'
    journal = MotionJournal(path, capacity=2)
    first = journal.append('motion.accepted')
    with pytest.raises(JournalError):
        journal.append('motion.stop_verified', stop_verified=False)
    with pytest.raises(JournalError):
        journal.append('motion.accepted', owner='bad\0owner')
    with pytest.raises(JournalError):
        journal.append('motion.accepted', event_id=first['event_id'])
    second = journal.append('motion.cancelled')
    assert second['sequence'] == 2
    with pytest.raises(JournalFull):
        journal.append('motion.guard_tripped')
    assert journal.read(first['stream_id'], 0)['events'] == [first, second]
    with sqlite3.connect(path) as conn:
        conn.execute('DELETE FROM motion_events WHERE sequence=1')
    with pytest.raises(JournalGap):
        journal.read(first['stream_id'], 0)
    with pytest.raises(JournalGap):
        journal.acknowledge(first['stream_id'], 2)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT acknowledged FROM motion_stream').fetchone()[0] == 0
        assert conn.execute('SELECT sequence FROM motion_events').fetchone()[0] == 2


def test_concurrent_writers_allocate_one_contiguous_stream(tmp_path):
    path = tmp_path / 'motion.sqlite3'
    journal = MotionJournal(path, capacity=80)

    def append(index):
        return MotionJournal(path, capacity=80).append('motion.accepted', operation_id=str(index))

    with ThreadPoolExecutor(max_workers=8) as pool:
        written = list(pool.map(append, range(80)))
    rows = journal.read(None, 0)['events']
    assert [row['sequence'] for row in rows] == list(range(1, 81))
    assert {row['event_id'] for row in rows} == {row['event_id'] for row in written}
    assert len({row['stream_id'] for row in written}) == 1


def test_separate_process_sees_committed_history(tmp_path):
    path = tmp_path / 'motion.sqlite3'
    journal = MotionJournal(path)
    event = journal.append('motion.completed', operation_id='task-1')
    code = (
        'import json, sys; '
        'from ringfour_agent.motion_journal import MotionJournal; '
        'print(json.dumps(MotionJournal(sys.argv[1]).read(sys.argv[2], 0)))'
    )
    completed = subprocess.run(
        [sys.executable, '-c', code, str(path), event['stream_id']],
        capture_output=True, check=True, text=True,
    )
    assert json.loads(completed.stdout)['events'] == [event]


def test_journal_files_are_private(tmp_path):
    path = tmp_path / 'motion.sqlite3'
    MotionJournal(path)
    assert path.stat().st_mode & 0o777 == 0o600
    path.chmod(0o644)
    MotionJournal(path)
    assert path.stat().st_mode & 0o777 == 0o600
