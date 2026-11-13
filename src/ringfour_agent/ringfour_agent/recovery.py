"""Explicit measured mobility recovery; never resume accepted motion."""
from .server import RpcError
from .stopping import require_mobility_stop, verify_mobility_stop


def recover_mobility(ctx):
    with ctx.lock.mutex:
        if "mobility" not in ctx.lock.quarantine:
            raise RpcError("SafetyViolation", "mobility is not quarantined")
        if ctx.lock.owner is not None or ctx.tasks.has_running():
            raise RpcError("ResourceBusy", "close all accepted work before recovery")
        if not ctx.state.estop_engaged:
            raise RpcError("SafetyViolation", "engage the inhibit before recovery")
        prior = ctx.lock.quarantine["mobility"]
    if ctx.ros is None or ctx.profile.spec.mobility is None:
        raise RpcError("CapabilityNotSupported", "mobility recovery requires its declared adapter")
    policy = ctx.profile.spec.safety.stop_verification
    age = ctx.profile.spec.safety.feedback_max_age
    require_mobility_stop(ctx.ros, policy, age)
    # Independent stop paths; an acknowledgement is never the stopping evidence.
    failures = []
    for name in ("nav_cancel", "publish_zero"):
        try:
            getattr(ctx.ros, name)()
        except Exception as exc:
            failures.append({"step": name, "error": str(exc)})
    if failures:
        raise RpcError("CleanupFailed", "mobility recovery stop dispatch failed", {"failures": failures})
    stopped = verify_mobility_stop(ctx.ros, policy, age)
    with ctx.lock.mutex:
        if not ctx.state.estop_engaged or ctx.lock.owner is not None or ctx.tasks.has_running():
            raise RpcError("SafetyViolation", "recovery eligibility changed")
        if ctx.lock.quarantine.get("mobility") is not prior:
            raise RpcError("SafetyViolation", "mobility quarantine changed during recovery")
        del ctx.lock.quarantine["mobility"]
    return {"resource": "mobility", "recovered": True, "inhibit_engaged": True,
            "stop_verification": stopped, "physical_stop": "uncommissioned"}


def recover_restart(ctx, resource="restart"):
    """Recommission an unclean boot only against an explicit reviewed baseline.

    Unknown vendor/tool effects and changed model checkpoints remain inhibited.
    Orphan action goals are cancelled, all declared motion is measured at rest,
    and controller states are checked twice. Old work is never resumed.
    """
    from .stopping import verify_joint_stop
    if resource not in ('restart', 'journal'):
        raise RpcError('SafetyViolation', 'unsupported checkpoint recovery resource')
    if resource == 'journal':
        ledger = getattr(ctx, 'commands', None)
        if ledger is None or ledger.failed:
            raise RpcError('CleanupFailed', 'repair storage and restart before journal recovery')
        ledger.prune()  # Commit a fresh write before any recovery effects.
    with ctx.lock.mutex:
        if resource not in ctx.lock.quarantine:
            raise RpcError('SafetyViolation','no matching checkpoint quarantine exists')
        if not ctx.state.estop_engaged or ctx.lock.owner is not None or ctx.tasks.has_running():
            raise RpcError('SafetyViolation','restart recovery requires an idle inhibited agent')
        prior = ctx.lock.quarantine[resource]
    expected = ctx.profile.spec.safety.restart_recovery_controllers
    if not expected or ctx.ros is None:
        raise RpcError('StopUnverified','restart recovery needs a reviewed controller baseline')
    if any(action.is_motion for action in (ctx.profile.spec.actions or {}).values()):
        raise RpcError('StopUnverified','opaque vendor restart recovery is not supported')
    for group in ctx.groups.values():
        if group.config.tool is not None or set(map(str,group.controllers)) != {'trajectory'}:
            raise RpcError('StopUnverified','restart recovery currently requires trajectory-only groups without tools')
        if not set(group.controllers.values()) <= set(expected):
            raise RpcError('StopUnverified','reviewed baseline omits a declared group controller')
    def states():
        actual={name:meta['state'] for name,meta in ctx.ros.loaded_controllers().items()}
        if actual != expected:
            raise RpcError('CleanupFailed','controller state differs from reviewed restart baseline',{'expected':expected,'actual':actual})
    policy = ctx.profile.spec.safety.stop_verification
    if (ctx.profile.spec.mobility is not None and policy is None) or any(
            group.config.stop_verification is None and policy is None for group in ctx.groups.values()):
        raise RpcError('StopUnverified', 'checkpoint recovery needs measured-stop policies')
    states()
    actions = [f'/{controller}/follow_joint_trajectory' for group in ctx.groups.values()
               for controller in group.controllers.values()]
    mobility = ctx.profile.spec.mobility
    if mobility is not None and mobility.nav2_navigate_to_pose:
        actions.append(mobility.nav2_navigate_to_pose)
    for name in dict.fromkeys(actions):ctx.ros.cancel_all_goals(name)
    policy=ctx.profile.spec.safety.stop_verification
    age=ctx.profile.spec.safety.feedback_max_age
    reports={}
    if mobility is not None:
        ctx.ros.publish_zero()
        reports['mobility']=verify_mobility_stop(ctx.ros,policy,age)
    for name,group in ctx.groups.items():
        reports[name]=verify_joint_stop(ctx.ros,group.joints,group.config.stop_verification or policy,group.config.feedback_max_age)
    if not reports:raise RpcError('StopUnverified','no declared motion resources can establish restart readiness')
    states()
    with ctx.lock.mutex:
        if not ctx.state.estop_engaged or ctx.lock.owner is not None or ctx.tasks.has_running() or ctx.lock.quarantine.get(resource) is not prior:
            raise RpcError('SafetyViolation','restart recovery eligibility changed')
        del ctx.lock.quarantine[resource]
    return {'resource':resource,'recovered':True,'inhibit_engaged':True,
            'criterion':'reviewed-controller-baseline-and-measured-rest','resources':reports,
            'cancelled_action_names':actions,'physical_stop':'uncommissioned'}
