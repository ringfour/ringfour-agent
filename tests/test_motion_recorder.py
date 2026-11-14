import asyncio
import sqlite3

from ringfour_agent.motion_journal import MotionJournal
from ringfour_agent.motion_recorder import MotionRecorder


def test_clean_boot_and_unclean_boot_gap(tmp_path):
    async def exercise():
        journal = MotionJournal(tmp_path / 'journal')
        recorder = MotionRecorder(journal)
        await recorder.start()
        recorder.emit('motion.accepted', operation_id='task')
        await recorder.close()
        assert [item['event_type'] for item in journal.read(None, 0)['events']] == ['motion.accepted']
        assert not journal.begin_boot()  # previous close cleared the boot marker
        # Simulate termination after boot without giving the writer a clean close.
        recorder = MotionRecorder(journal)
        await recorder.start()
        await recorder.close()
        assert [item['event_type'] for item in journal.read(None, 0)['events']] == [
            'motion.accepted', 'motion.history_incomplete',
        ]
    asyncio.run(exercise())


def test_overflow_is_explicit_and_does_not_block_producers(tmp_path):
    async def exercise():
        journal = MotionJournal(tmp_path / 'journal')
        recorder = MotionRecorder(journal, capacity=1)
        await recorder.start()
        recorder.worker.cancel()
        await asyncio.gather(recorder.worker, return_exceptions=True)
        recorder.emit('motion.accepted', operation_id='one')
        recorder.emit('motion.accepted', operation_id='dropped')
        recorder.emit('motion.completed', operation_id='also-dropped')
        assert recorder.gap
        recorder.worker = asyncio.create_task(recorder._run())
        await recorder.close()
        rows = journal.read(None, 0)['events']
        assert [item['event_type'] for item in rows] == ['motion.accepted', 'motion.history_incomplete']
        assert rows[0]['operation_id'] == 'one'
    asyncio.run(exercise())


def test_full_journal_does_not_block_emit_and_remains_dirty(tmp_path):
    async def exercise():
        journal = MotionJournal(tmp_path / 'journal', capacity=1)
        journal.append('motion.accepted')
        recorder = MotionRecorder(journal)
        await recorder.start()
        recorder.emit('motion.completed', operation_id='one')
        await asyncio.sleep(.02)
        await recorder.close(timeout=.05)
        with sqlite3.connect(journal.path) as conn:
            assert conn.execute('SELECT dirty FROM motion_boot').fetchone()[0] == 1
        assert len(journal.read(None, 0)['events']) == 1
    asyncio.run(exercise())
