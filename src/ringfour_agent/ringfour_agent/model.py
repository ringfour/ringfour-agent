"""Watching `/robot_description` for a robot that changed shape underneath us.

`robot_state_publisher` can restart with a different xacro — a tool
swapped on the mod101 mount, an arm added, a link retuned. When it does,
every client holding a chain built from the old description is holding a
model of a robot that no longer exists. The agent notices and pushes
``model.changed``; the client refetches and rebuilds (spec 26.2).

**Why a poller and not the subscription callback.** `_on_description`
runs on the ROS executor thread. `EventHub.emit` fans out through
`asyncio.ensure_future`, which needs a running loop *in the calling
thread* — emitting from the ROS thread raises `RuntimeError`. Rather than
thread a loop handle into RosInterface, this compares a sha from an
asyncio task, the same shape as the watch sampler and the SLAM quality
monitor. A description changes on a process restart, so once a second is
generous; the sha is computed on receipt, not per poll, so the check is a
dict read.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger("ringfour_agent.model")

INTERVAL = 1.0


async def run_model_watcher(ctx: Any, interval: float = INTERVAL) -> None:
    """Emit ``model.changed`` whenever the description's sha moves.

    The first sha seen is the baseline and is not announced — that is the
    description the handshake already carried. An arrival from None IS
    announced: those are the clients that connected before
    robot_state_publisher came up and got ``description: null``.
    """
    ros = ctx.ros
    if ros is None or not hasattr(ros, "description_sha"):
        return
    previous = ros.description_sha()
    def revisions():
        if ctx.control is None:return {}
        return {name:ctx.control.model_revision(ctx.control.group(name)) for name in ctx.groups}
    previous_groups=revisions()
    while True:
        await asyncio.sleep(interval)
        try:
            current = ros.description_sha()
        except Exception:
            log.exception("failed to read the description sha")
            continue
        current_groups=revisions()
        if current == previous and current_groups==previous_groups:
            continue
        # Invalidate accepted work/sessions before another dispatch can use old
        # model-bound commands. Cleanup uses retained resource metadata.
        ctx.state.guard_inhibits["model-change"]="model revision changed"
        from .halt import halt_existing
        try:
            await halt_existing(ctx,reason="model revision changed",latch=False)
        except Exception:
            log.exception("model-change cleanup failed; inhibit retained")
        else:
            ctx.state.guard_inhibits.pop("model-change",None)
        log.info("robot description changed: %s -> %s",
                 (previous or "(none)")[:12], (current or "(none)")[:12])
        ctx.events.emit("model.changed", {
            "sha": current,
            "previous": previous,
            "group_shas":current_groups,
        })
        previous = current
        previous_groups=current_groups
