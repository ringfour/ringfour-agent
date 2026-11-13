"""Persistent owner-scoped guards. Trip before yielding; drain before removal."""
import asyncio
import uuid
import time
from ringfour.models.measurements import SignalCondition, GuardStatus
from ringfour.enums import AuditKind, Outcome
from .measurements import measure_signal
from .server import RpcError

class GuardRegistry:
    def __init__(self, ctx):
        self.ctx = ctx
        self.records = {}

    def observe(self, condition):
        observation = measure_signal(self.ctx, condition.path, condition.max_age)
        unsafe = observation.validity != 'valid' or (
            observation.value < condition.threshold if condition.comparison == 'below'
            else observation.value > condition.threshold)
        return observation, unsafe

    async def observe_bounded(self,condition):
        try:
            return await asyncio.wait_for(asyncio.to_thread(self.observe,condition),condition.max_age)
        except asyncio.TimeoutError:
            from ringfour.models.measurements import Measurement
            observation=Measurement(value=None,robot_id=self.ctx.profile.info.name,model_revision=None,
                validity='unavailable',reason='guard observation deadline expired',frame=condition.frame)
            return observation,True

    async def install(self, owner, condition):
        condition = SignalCondition.model_validate(condition)
        if condition.path in ('pose.x','pose.y') and condition.frame!=self.ctx.profile.spec.frames.map:
            raise RpcError('SafetyViolation','pose guard requires the discovered explicit map frame')
        if condition.robot_id != self.ctx.profile.info.name:
            raise RpcError('SafetyViolation','guard belongs to another robot')
        observation,unsafe = await self.observe_bounded(condition)
        if unsafe:
            raise RpcError('SafetyViolation','guard entry requires a fresh safe observation',
                           {'reason':'guard entry','observation':observation.model_dump(mode='json')})
        identifier='g'+uuid.uuid4().hex
        self.records[identifier]={'owner':owner,'condition':condition,'observation':observation,
            'tripped':False,'reason':None,'cleanup':None,'cleanup_confirmed':None,'closing':False,'observed_at':time.monotonic()}
        self.ctx.audit.record(AuditKind.SAFETY,client=owner,call='guard.open',outcome=Outcome.SUCCESS,
                              detail={'guard_id':identifier,'condition':condition.model_dump(mode='json')})
        return self.status(identifier,owner)

    async def install_profile(self,condition):
        """Integrator-owned policy guard: ordinary sessions cannot reset/remove it."""
        return await self.install(None,condition)

    def owned(self, identifier, owner):
        record=self.records.get(identifier)
        if record is None:raise RpcError('StateUnavailable','guard does not exist')
        if record['owner'] != owner:raise RpcError('PermissionDenied','guard belongs to another owner')
        return record

    def status(self, identifier, owner):
        record=self.owned(identifier,owner)
        return GuardStatus(guard_id=identifier,condition=record['condition'],tripped=record['tripped'],
            reason=record['reason'],observation=record['observation'],cleanup_confirmed=record['cleanup_confirmed'],evaluation_age=max(0.,time.monotonic()-record['observed_at']))

    def trip(self, identifier, record, observation):
        record['tripped']=True
        record['observation']=observation
        record['observed_at']=time.monotonic()
        record['reason']=observation.reason if observation.validity!='valid' else 'condition threshold crossed'
        self.ctx.state.guard_inhibits[identifier]=record['reason']
        self.ctx.tasks.cancel_running()
        self.ctx.events.emit('guard',{'guard_id':identifier,'reason':record['reason']})
        self.ctx.audit.record(AuditKind.SAFETY,client=record['owner'],call='guard',outcome=Outcome.ERROR,
            detail={'event':'tripped','guard_id':identifier,'reason':record['reason'],'observation':observation.model_dump(mode='json')})
        async def cleanup():
            from .halt import halt_existing
            try:
                await halt_existing(self.ctx,reason='guard:'+identifier,client=record['owner'],latch=False)
                record['cleanup_confirmed']=True
            except Exception:
                record['cleanup_confirmed']=False
                raise
        record['cleanup']=asyncio.create_task(cleanup())
        self.ctx.background_halts.add(record['cleanup'])
        def done(task):
            self.ctx.background_halts.discard(task)
            if not task.cancelled():task.exception() # outcome retained in record/quarantine
        record['cleanup'].add_done_callback(done)

    async def evaluate(self):
        for identifier,record in tuple(self.records.items()):
            if record['tripped'] or record['closing']:continue
            observation,unsafe=await self.observe_bounded(record['condition'])
            if self.records.get(identifier) is not record or record['closing'] or record['tripped']:continue
            record['observation']=observation
            record['observed_at']=time.monotonic()
            if unsafe:self.trip(identifier,record,observation)

    async def reset(self, identifier, owner):
        record=self.owned(identifier,owner)
        if record['closing']:raise RpcError('ResourceBusy','guard is closing')
        if record['cleanup'] is not None:
            try:await asyncio.shield(record['cleanup'])
            except Exception:
                from .halt import halt_existing
                await halt_existing(self.ctx,reason='guard cleanup retry:'+identifier,client=owner,latch=False)
        observation,unsafe=await self.observe_bounded(record['condition'])
        if unsafe:raise RpcError('SafetyViolation','guard reset requires a fresh safe observation')
        record.update(tripped=False,reason=None,observation=observation,cleanup=None,cleanup_confirmed=None,observed_at=time.monotonic())
        self.ctx.state.guard_inhibits.pop(identifier,None)
        return self.status(identifier,owner)

    async def remove(self, identifier, owner):
        record=self.owned(identifier,owner)
        record['closing']=True
        # Inhibit first, even when a guard has not tripped. Scope exit stops/drains
        # covered work before its condition can disappear.
        self.ctx.state.guard_inhibits[identifier]='guard removal in progress'
        try:
            if record['cleanup'] is not None:
                try:await asyncio.shield(record['cleanup'])
                except Exception:pass # explicit removal retries verified cleanup below
            from .halt import halt_existing
            await halt_existing(self.ctx,reason='guard removal:'+identifier,client=owner,latch=False)
        except BaseException:
            record['closing']=False
            record['tripped']=True
            record['cleanup_confirmed']=False
            raise
        self.records.pop(identifier,None)
        self.ctx.state.guard_inhibits.pop(identifier,None)
        return {'removed':True,'cleanup_confirmed':True}

    def disconnected_after_cleanup(self, owner):
        if self.ctx.lock.quarantine:return
        for identifier,record in tuple(self.records.items()):
            if record['owner']==owner and (record['cleanup'] is None or record['cleanup'].done() and record['cleanup_confirmed']):
                self.records.pop(identifier,None)
                self.ctx.state.guard_inhibits.pop(identifier,None)
