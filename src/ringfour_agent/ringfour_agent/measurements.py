"""Agent-side fresh measurement acquisition. Clock conversions stay in adapters."""
import asyncio
import math
import time
from ringfour.models.measurements import Measurement
from .server import RpcError


def measure_signal(ctx, path, max_age):
    getter = (ctx.watch_paths or {}).get(path)
    if getter is None: raise RpcError('CapabilityNotSupported', f'signal {path!r} is unavailable')
    ros = ctx.ros
    provenance = None
    try:
        # ROS owns one lock for values and provenance. RLock permits reader calls.
        with ros._lock:
            if path in ('pose.x','pose.y'):
                frame=ros.resolve_frame('@map')
                sample=ros.navigation_pose_snapshot(frame)
                value=sample['position'][0 if path=='pose.x' else 1]
                provenance={**sample,'clock_domain':'ros','frame':frame}
            else:
                value = getter()
                provenance = ros.signal_provenance(path)
    except Exception as exc:
        value = None
    return envelope(ctx, value, provenance, max_age)


def envelope(ctx, value, provenance, max_age):
    source_stamp = received = source_age = receive_age = None
    clock = None
    if provenance:
        source_stamp = provenance.get('source_stamp')
        received = provenance.get('received')
        if received is not None and math.isfinite(received) and received<=time.monotonic()+.01:
            receive_age = max(0.,time.monotonic()-received)
        source_now = provenance.get('source_now')
        clock = provenance.get('clock_domain')
        if source_stamp is not None and source_now is not None and math.isfinite(source_stamp) and math.isfinite(source_now) and source_stamp > 0:
            age = source_now-source_stamp
            if age >= -.01: source_age = max(0.,age)
    reason = None
    validity = 'valid'
    if value is None:
        validity, reason = 'unavailable','no measured value'
    elif isinstance(value,(int,float)) and not math.isfinite(value):
        validity, reason = 'invalid','nonfinite measured value'
        value = None
    elif receive_age is None or source_age is None:
        validity, reason = 'unavailable','required source or receive freshness is unknown'
    elif max(receive_age,source_age)>max_age:
        validity, reason = 'stale','source or receive age exceeds budget'
    revision = ctx.ros.description_sha() if hasattr(ctx.ros,'description_sha') else None
    return Measurement(value=value,robot_id=ctx.profile.info.name,model_revision=revision,
        source_stamp=source_stamp,clock_domain=clock,received_at=received,
        source_age=source_age,receive_age=receive_age,validity=validity,reason=reason,frame=(provenance or {}).get("frame"))


async def bounded_signal(ctx,path,max_age,timeout,require_valid):
    deadline=time.monotonic()+timeout
    while True:
        try:
            result=await asyncio.wait_for(asyncio.to_thread(measure_signal,ctx,path,max_age),
                                          max(0.,deadline-time.monotonic()))
        except asyncio.TimeoutError:
            result=Measurement(value=None,robot_id=ctx.profile.info.name,model_revision=None,
                validity='unavailable',reason='measurement acquisition deadline expired')
        if result.validity=='valid' or not require_valid:return result
        if time.monotonic()>=deadline:
            kind='StateStale' if result.validity=='stale' else 'StateUnavailable'
            raise RpcError(kind,result.reason,{'observation':result.model_dump(mode='json')})
        await asyncio.sleep(min(.02,max(0.,deadline-time.monotonic())))


def measure_resource(ctx,kind,group,max_age,frame):
    ros=ctx.ros
    try:
        if kind=='pose':
            frame=ros.resolve_frame(frame or '@map')
            sample=ros.navigation_pose_snapshot(frame)
            p,q=sample['position'],sample['orientation']
            value={'frame':frame,'x':p[0],'y':p[1],'z':p[2],
                   'q':dict(zip(('x','y','z','w'),q))}
            from ringfour.models import Pose
            value=Pose.model_validate(value).model_dump(mode='json')
            provenance={**sample,'clock_domain':'ros','frame':frame}
        elif kind=='joints':
            runtime=ctx.groups.get(group)
            if runtime is None:raise RpcError('CapabilityNotSupported','unknown joint group')
            sample=ros.stop_snapshot(runtime.joints)
            # Conservative envelope uses the oldest source/receipt of every joint.
            if set(sample['positions'])!=set(runtime.joints):raise ValueError('incomplete joint coverage')
            value=[]
            for name in runtime.joints:
                unit={'revolute':'rad','continuous':'rad','prismatic':'m'}.get(ros.joint_type(name))
                if unit is None:raise ValueError('unknown joint units')
                position=sample['positions'][name]
                if not math.isfinite(position):raise ValueError('nonfinite joint feedback')
                value.append({'name':name,'position':position,'unit':unit})
            stamps=[sample['source_stamps'][name] for name in runtime.joints]
            received=[sample['received'][name] for name in runtime.joints]
            if any(stamp is None or stamp<=0 or stamp>sample['source_now']+.01 for stamp in stamps):
                raise ValueError('unknown or future joint source timestamp')
            provenance={'source_stamp':min(stamps),'received':min(received),
                        'source_now':sample['source_now'],'clock_domain':'ros'}
        else:raise RpcError('CapabilityNotSupported','unknown measurement kind')
        result=envelope(ctx,value,provenance,max_age)
        if kind=='joints' and ctx.control is not None:
            result=result.model_copy(update={'model_revision':ctx.control.model_revision(runtime)})
        return result
    except RpcError:raise
    except Exception as exc:
        return envelope(ctx,None,None,max_age).model_copy(update={'reason':str(exc)})


async def bounded_resource(ctx,kind,group,max_age,timeout,require_valid,frame):
    deadline=time.monotonic()+timeout
    while True:
        try:
            result=await asyncio.wait_for(asyncio.to_thread(measure_resource,ctx,kind,group,max_age,frame),
                                          max(0.,deadline-time.monotonic()))
        except asyncio.TimeoutError:
            result=Measurement(value=None,robot_id=ctx.profile.info.name,model_revision=None,
                validity='unavailable',reason='measurement acquisition deadline expired',frame=frame)
        if result.validity=='valid' or not require_valid:return result
        if time.monotonic()>=deadline:
            raise RpcError('StateStale' if result.validity=='stale' else 'StateUnavailable',result.reason,
                           {'observation':result.model_dump(mode='json')})
        await asyncio.sleep(min(.02,max(0.,deadline-time.monotonic())))
