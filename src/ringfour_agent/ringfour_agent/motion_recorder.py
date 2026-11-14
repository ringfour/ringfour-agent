"""Non-blocking selective motion recording with explicit crash-gap reporting."""
import asyncio
import logging
import sqlite3
from datetime import UTC, datetime

from .motion_journal import JournalError, JournalFull

log = logging.getLogger('ringfour_agent.motion_recorder')


class MotionRecorder:
    def __init__(self, journal, *, capacity=256):
        self.journal = journal
        self.queue = asyncio.Queue(maxsize=capacity)
        self.gap = False
        self.accepting = False
        self.worker = None

    async def start(self):
        self.gap = await asyncio.to_thread(self.journal.begin_boot)
        self.accepting = True
        self.worker = asyncio.create_task(self._run())

    def emit(self, event_type, **fields):
        """Called only on the Agent loop; never await or perform I/O here."""
        if not self.accepting:
            return
        if self.gap:
            return
        try:
            fields.setdefault('occurred_at', datetime.now(UTC))
            self.queue.put_nowait((event_type, fields))
        except asyncio.QueueFull:
            self.gap = True

    async def _append(self, event_type, fields):
        while True:
            try:
                await asyncio.to_thread(self.journal.append, event_type, **fields)
                return True
            except JournalFull:
                await asyncio.sleep(1)
            except JournalError:
                await asyncio.to_thread(log.error, 'Motion event was invalid; marking history incomplete')
                return False
            except (sqlite3.Error, OSError):
                await asyncio.to_thread(log.error, 'Motion journal unavailable; retrying')
                await asyncio.sleep(1)

    async def _run(self):
        while True:
            if self.gap and self.queue.empty():
                if await self._append('motion.history_incomplete', {}):
                    self.gap = False
                continue
            try:
                event_type, fields = await asyncio.wait_for(self.queue.get(), .25)
            except TimeoutError:
                continue
            if not await self._append(event_type, fields):
                self.gap = True
            self.queue.task_done()

    async def close(self, *, timeout=5):
        self.accepting = False
        async def drain():
            await self.queue.join()
            while self.gap:
                await asyncio.sleep(.05)
            await asyncio.to_thread(self.journal.finish_boot)
        try:
            await asyncio.wait_for(drain(), timeout)
        except (TimeoutError, sqlite3.Error, OSError):
            log.error('Motion journal did not drain; next boot will report incomplete history')
        finally:
            if self.worker:
                self.worker.cancel()
                await asyncio.gather(self.worker, return_exceptions=True)
                self.worker = None
