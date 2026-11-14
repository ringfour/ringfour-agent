"""Atomic quarantine checkpoints and an unclean-boot inhibit marker.

This does not persist operation results or reconcile lost submissions.
"""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import tempfile
from threading import RLock
from .server import RpcError


class DurableQuarantine(dict):
    def __init__(self, directory, robot, revision):
        self.mutex = RLock()
        self.robot = robot
        self.path = Path(directory) / (hashlib.sha256(robot.encode()).hexdigest() + ".json")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lockfile = open(str(self.path) + '.lock', 'a')
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lockfile.close()
            raise RpcError('CleanupFailed', 'another agent owns this durable robot state') from exc
        previous = {}
        if self.path.exists():
            try:
                previous = json.loads(self.path.read_text())
                assert previous['version'] == 1 and previous['robot'] == robot
                assert isinstance(previous['quarantine'], dict) and type(previous['active_boot']) is bool
                assert all(isinstance(k,str) and isinstance(v,dict) for k,v in previous['quarantine'].items())
            except Exception as exc:
                raise RpcError("CleanupFailed", "invalid durable safety checkpoint; refusing startup") from exc
        values = previous.get('quarantine', {}).copy()
        baseline = previous.get('model_revision')
        # Older Agents could overwrite the checkpoint's model identity with
        # None before the latched description arrived. Preserve the original
        # identity retained in that quarantine rather than losing it again.
        model_record = values.get('restart-model')
        if (baseline is None and isinstance(model_record, dict)
                and model_record.get('current_revision') is None
                and isinstance(model_record.get('previous_revision'), str)):
            baseline = model_record['previous_revision']
        self.revision = revision if revision is not None else baseline
        if previous.get('active_boot'):
            values['restart'] = {'reason':'previous agent boot did not finish verified cleanup'}
        if previous and baseline != revision:
            if baseline is None or revision is None:
                values.setdefault('restart-model', {'reason':'robot model revision unavailable at checkpoint comparison',
                                                     'previous_revision':baseline, 'current_revision':revision})
            else:
                values['restart-model'] = {'reason':'robot model changed across the safety checkpoint',
                                           'previous_revision':baseline, 'current_revision':revision}
        super().__init__(values)
        self.active_boot = True
        self._persist(dict(self))  # Must succeed before any motion can be admitted.

    def _persist(self, values):
        data = {'version':1, 'robot':self.robot, 'model_revision':self.revision,
                'active_boot':self.active_boot, 'quarantine':values}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                os.chmod(temporary, 0o600)
                json.dump(data, stream, allow_nan=False)
                stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        except Exception as exc:
            raise RpcError('CleanupFailed', 'durable safety checkpoint could not be committed') from exc
        finally:
            if temporary is not None and temporary.exists(): temporary.unlink()

    def __setitem__(self, key, value):
        with self.mutex:
            # Keep new quarantine in memory even if persistence fails. The boot
            # marker already on disk inhibits an unclean restart in that case.
            super().__setitem__(key, value)
            self._persist(dict(self))

    def __delitem__(self, key):
        with self.mutex:
            proposed = dict(self); del proposed[key]
            self._persist(proposed)  # Failed persistence must not clear authority.
            super().__delitem__(key)

    def finish_boot(self):
        with self.mutex:
            self.active_boot = False
            try: self._persist(dict(self))
            except Exception:
                self.active_boot = True
                raise
            self._lockfile.close()

    def resolve_missing_model(self, expected_record, revision):
        """Atomically remove only a verified missing-model checkpoint."""
        with self.mutex:
            current = self.get('restart-model')
            if (current is not expected_record or current.get('current_revision') is not None
                    or current.get('previous_revision') != revision):
                raise RpcError('SafetyViolation', 'model checkpoint changed during recovery')
            proposed = dict(self)
            del proposed['restart-model']
            old_revision = self.revision
            self.revision = revision
            try:
                self._persist(proposed)
            except Exception:
                self.revision = old_revision
                raise
            super().__delitem__('restart-model')

    def pop(self, key, *default):
        with self.mutex:
            if key not in self:
                if default: return default[0]
                raise KeyError(key)
            value = self[key]; del self[key]; return value

    def clear(self):
        with self.mutex:
            self._persist({})
            super().clear()

    def update(self, *args, **kwargs):
        with self.mutex:
            proposed = dict(self); proposed.update(*args, **kwargs)
            # Additions must remain inhibited in memory when storage fails.
            super().update(proposed)
            self._persist(proposed)

    def setdefault(self, key, default=None):
        with self.mutex:
            if key not in self: self[key] = default
            return self[key]

    def popitem(self):
        with self.mutex:
            if not self: raise KeyError('empty quarantine')
            key = next(reversed(self)); value = self[key]
            del self[key]; return key, value

    def __ior__(self, values):
        self.update(values)
        return self
