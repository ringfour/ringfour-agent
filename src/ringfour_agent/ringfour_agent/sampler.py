"""The watch sampler: evaluates every active watch against live state.

Runs as one asyncio task started with the server. 20 Hz in v1 — the spec
says "source rate"; sampling decouples the safety path from subscription
callbacks and 50 ms is far inside any battery/effort reaction budget.
Documented deviation, revisit if a watch ever needs to catch a
single-message spike.

A ``stop=True`` watch firing halts all motion right here, by latching an inhibit before the
tick returns, then draining verified cleanup in a separate asyncio task. The hung/disconnected owner is
exactly the case this exists for (spec section 16, discipline 6).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ringfour.enums import AuditKind, Outcome

from .watches import Watch

log = logging.getLogger("ringfour_agent.sampler")

INTERVAL = 0.05  # 20 Hz


async def run_watch_sampler(ctx: Any, paths: dict, interval: float = INTERVAL) -> None:
    pending = ctx.background_halts

    def resolve(path: str) -> float | None:
        getter = paths.get(path)
        return None if getter is None else getter()

    def on_fire(watch: Watch, value: float) -> None:
        ctx.events.emit("watch", {
            "watch_id": watch.id, "path": watch.path, "value": value,
            "stop": watch.stop,
        })
        if not watch.stop:
            return
        log.warning("stop-watch %s fired: %s = %s; halting all motion",
                    watch.id, watch.path, value)
        # Latch before yielding so another request cannot overtake stopping.
        ctx.state.estop_engaged = True
        ctx.tasks.cancel_running()
        async def cleanup():
            from .halt import halt_existing
            try:
                report = await halt_existing(ctx, reason='stop-watch:' + watch.id, client=watch.owner)
                ctx.audit.record(AuditKind.SAFETY, client=watch.owner, call='watch', outcome=Outcome.SUCCESS,
                                 detail={'reason': 'watch', 'watch_id': watch.id, 'cleanup': report})
            except Exception:
                log.exception("stop-watch cleanup remains unverified")
        task = asyncio.create_task(cleanup())
        pending.add(task)
        task.add_done_callback(pending.discard)

    while True:
        try:
            ctx.watches.evaluate(resolve, on_fire)
            if ctx.guards is not None:
                await ctx.guards.evaluate()
        except Exception:
            log.exception("watch sampler tick failed")
        await asyncio.sleep(interval)
