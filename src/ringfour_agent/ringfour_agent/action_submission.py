"""Bounded ROS action handoff, retaining uncertain acceptance for cleanup.

No durable reconciliation/query API: this only prevents a late acknowledgement
from escaping the stop of an in-process adapter submission.
"""
import logging
import threading
from .server import RpcError

log = logging.getLogger(__name__)


class ActionSubmission:
    def __init__(self, future, on_accept):
        self.accepted = threading.Event()
        self.handle = self.result_future = None
        self.error = None
        self.revoked = False
        self._lock = threading.RLock()
        self._on_accept = on_accept
        future.add_done_callback(self._accepted)

    def _accepted(self, future):
        try:
            handle = future.result()
            with self._lock:
                if handle is not None and handle.accepted:
                    self.handle = handle
                    self.result_future = handle.get_result_async()
                    self._on_accept(handle, self.result_future)
                late_stop = self.revoked and self.handle is not None
        except Exception as exc:
            self.error = exc
            late_stop = self.handle is not None and self.revoked
        finally:
            self.accepted.set()
        if late_stop:
            # Never block the ROS executor waiting for its own result callback.
            threading.Thread(target=self._late_stop, daemon=True).start()

    def wait(self, timeout):
        if not self.accepted.wait(timeout):
            with self._lock: self.revoked = True
            # Acceptance may have raced the timeout before revoked was set.
            if self.accepted.is_set() and self.handle is not None:
                threading.Thread(target=self._late_stop, daemon=True).start()
            raise RpcError('CompletionUnverified', 'action acceptance is unresolved',
                           {'reason': 'submission acknowledgement timeout'})
        if self.error is not None:
            raise RpcError('CompletionUnverified', 'action acceptance failed', {'error': str(self.error)})
        if self.handle is None:
            raise RpcError('ExecutionFailed', 'controller rejected the action')
        return self.handle

    def stop(self, timeout=2.):
        with self._lock: self.revoked = True
        if not self.accepted.wait(timeout):
            raise RpcError('CleanupFailed', 'action acceptance remains unresolved during stopping')
        if self.error is not None:
            raise RpcError('CleanupFailed', 'action acceptance cannot be reconciled', {'error': str(self.error)})
        if self.handle is None: return  # rejected before acceptance
        self._stop_handle(timeout)

    def _stop_handle(self, timeout):
        handle, result = self.handle, self.result_future
        done = threading.Event()
        cancellation = handle.cancel_goal_async()
        cancellation.add_done_callback(lambda _: done.set())
        if not done.wait(timeout):
            raise RpcError('CleanupFailed', 'controller stop acknowledgement timed out')
        response = cancellation.result()
        if result is None:
            raise RpcError('CleanupFailed', 'controller terminal result is unavailable')
        if not getattr(response, 'goals_canceling', []) and not result.done():
            raise RpcError('CleanupFailed', 'controller rejected cancellation without a terminal result')
        terminal = threading.Event()
        result.add_done_callback(lambda _: terminal.set())
        if not terminal.wait(timeout):
            raise RpcError('CleanupFailed', 'controller stop has no terminal result')
        # A failed result channel is unknown, not evidence of completion.
        result.result()

    def _late_stop(self):
        try: self._stop_handle(2.)
        except Exception:
            log.exception('late accepted action stop remains unverified; quarantine must remain')
