"""Shared stop-and-cleanup path. Inhibit first; never block the monitoring loop."""
import asyncio
from ringfour.enums import AuditKind, Outcome
from .server import RpcError


async def halt_existing(ctx, *, reason, client=None, latch=True):
    if latch: ctx.state.estop_engaged = True
    ctx.tasks.cancel_running()
    failures = []
    for manager in (ctx.teleop, ctx.actions):
        if manager is not None:
            try:
                await asyncio.to_thread(manager.end_all)
            except Exception as exc:
                failures.append({'step': type(manager).__name__, 'error': str(exc)})
    clean = True
    try:
        await ctx.tasks.cancel_all()
    except Exception as exc:
        clean = False
        failures.append({'step': 'drain accepted work', 'error': str(exc)})
        ctx.lock.quarantine['halt'] = {'reason': reason, 'failures': failures}
    if clean and ctx.control is not None:
        try:
            await asyncio.to_thread(ctx.control.end_all, reason=reason)
        except Exception as exc:
            failures.append({'step': 'restore controllers', 'error': str(exc)})
    if ctx.lock.quarantine:
        failures.append({'step': 'retained quarantine', 'resources': dict(ctx.lock.quarantine)})
    report = {'reason': reason, 'inhibit_engaged': True, 'confirmed': not failures, 'failures': failures}
    ctx.audit.record(AuditKind.SAFETY, client=client, call='halt',
                     outcome=Outcome.ERROR if failures else Outcome.SUCCESS, detail=report)
    if failures:
        raise RpcError('CleanupFailed', 'halt cleanup remains unverified', {'cleanup': report})
    return report
