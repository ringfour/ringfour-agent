"""Measured joint-stop verification with explicit integrator thresholds.

This verifies feedback against a configured criterion. It does not commission
sensor accuracy, braking performance, collision safety or hardware emergency
stopping. Cached samples and paused source clocks cannot establish the dwell.
"""
import math
import time
from .feedback import check_freshness
from .server import RpcError


def verify_joint_stop(adapter, names, policy, max_age, *, clock=time.monotonic, sleep=time.sleep, abort=None):
    names = tuple(names)
    if not names or not hasattr(adapter, 'stop_snapshot') or not hasattr(adapter, 'joint_type'):
        raise RpcError('StopUnverified', 'integration cannot supply measured joint-stop evidence',
                       {'reason': 'stop feedback unavailable', 'physical_stop': 'unverified'})
    limits, units = {}, {}
    for name in names:
        try:
            kind = adapter.joint_type(name)
        except Exception as exc:
            raise RpcError('StopUnverified', 'joint measurement units are unavailable',
                           {'reason': 'stop policy unavailable', 'joint': name, 'error': str(exc)}) from exc
        unit = {'revolute': 'rad/s', 'continuous': 'rad/s', 'prismatic': 'm/s'}.get(kind)
        limit = policy.angular_velocity_limit if unit == 'rad/s' else policy.linear_velocity_limit if unit == 'm/s' else None
        if limit is None:
            raise RpcError('StopUnverified', f'no reviewed stop threshold for {name!r}',
                           {'reason': 'stop policy unavailable', 'joint': name, 'joint_type': kind})
        limits[name], units[name] = limit, unit
    started = clock()
    deadline = started + policy.timeout
    baseline = previous = None
    samples = 0
    last_error = 'no progressive complete stationary feedback'
    while True:
        if abort is not None and abort():
            raise RpcError("SafetyViolation", "operation eligibility was revoked during verification")
        if clock() >= deadline:
            raise RpcError('StopUnverified', 'joint stop was not verified before its deadline',
                           {'reason': 'stop criterion unverified', 'last_evidence': last_error,
                            'elapsed': clock() - started, 'velocity_limits': limits,
                            'units': units, 'physical_stop': 'unverified'})
        try:
            evidence = adapter.stop_snapshot(names)
            check_freshness(names, evidence['source_stamps'], evidence['received'],
                            evidence['source_now'], evidence['receive_now'], max_age)
            for name in names:
                pos, vel = evidence['positions'][name], evidence['velocities'][name]
                if type(pos) not in (int, float) or type(vel) not in (int, float) or not math.isfinite(pos) or not math.isfinite(vel):
                    raise ValueError(f'{name}: nonfinite or missing measured feedback')
                if abs(vel) > limits[name]:
                    raise ValueError(f'{name}: measured velocity exceeds {limits[name]} {units[name]}')
            if previous is not None:
                # New complete evidence is required; rereading a cache is not
                # a second sensor sample. Backwards source time resets dwell.
                if any(evidence['source_stamps'][n] < previous['source_stamps'][n] or
                       evidence['received'][n] < previous['received'][n] for n in names):
                    baseline = previous = None
                elif not all(evidence['source_stamps'][n] > previous['source_stamps'][n] and
                             evidence['received'][n] > previous['received'][n] for n in names):
                    raise LookupError('feedback has not progressed for every joint')
            if previous is not None:
                # Check position-derived velocity too: zero-filled velocity
                # fields cannot conceal a moving measured position.
                for n in names:
                    dt = evidence['source_stamps'][n] - previous['source_stamps'][n]
                    measured = abs(evidence['positions'][n] - previous['positions'][n]) / dt
                    if measured > limits[n]:
                        raise ValueError(f'{n}: position-derived velocity exceeds stop threshold')
            samples += 1
            if baseline is None:
                baseline = evidence
            previous = evidence
            if (all(evidence['source_stamps'][n] - baseline['source_stamps'][n] >= policy.dwell and
                    evidence['received'][n] - baseline['received'][n] >= policy.dwell for n in names)):
                return {'verified': True, 'criterion': 'joint-velocity-dwell',
                        'samples': samples, 'elapsed': clock() - started,
                        'dwell': policy.dwell, 'velocities': evidence['velocities'],
                        'velocity_limits': limits, 'units': units,
                        'source_stamps': evidence['source_stamps'], 'received': evidence['received'],
                        'physical_stop': 'uncommissioned'}
        except LookupError as exc:
            last_error = str(exc)  # cached samples preserve, but cannot finish dwell
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            baseline = previous = None
            last_error = str(exc)
        except Exception as exc:
            raise RpcError('StopUnverified', 'integration failed while verifying stop',
                           {'reason': 'stop feedback unavailable', 'error': str(exc),
                            'physical_stop': 'unverified'}) from exc
        remaining = deadline - clock()
        if remaining > 0:
            sleep(min(policy.dwell / 4, .05, remaining))


def require_joint_stop(adapter, names, policy, max_age):
    """Reject before controller switching/publication without a usable contract."""
    if policy is None or not names or not hasattr(adapter, 'stop_snapshot') or not hasattr(adapter, 'joint_type'):
        raise RpcError('StopUnverified', 'motion requires a measured-stop policy and feedback adapter',
                       {'reason': 'stop contract unavailable', 'profile_field': 'safety.stop_verification'})
    try:
        evidence = adapter.stop_snapshot(names)
        check_freshness(names, evidence['source_stamps'], evidence['received'],
                        evidence['source_now'], evidence['receive_now'], max_age)
        for name in names:
            kind = adapter.joint_type(name)
            limit = policy.linear_velocity_limit if kind == 'prismatic' else policy.angular_velocity_limit if kind in ('revolute', 'continuous') else None
            if limit is None: raise ValueError(f'{name}: missing native-unit stop threshold')
            for field in ('positions', 'velocities'):
                value = evidence[field][name]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f'{name}: incomplete finite {field}')
    except Exception as exc:
        raise RpcError('StopUnverified', 'motion lacks usable measured-stop feedback',
                       {'reason': 'stop contract unavailable', 'error': str(exc)}) from exc


def _body_evidence(adapter, policy, max_age):
    if policy is None or policy.linear_velocity_limit is None or policy.angular_velocity_limit is None:
        raise ValueError('body motion requires explicit m/s and rad/s stop limits')
    evidence = adapter.mobility_stop_snapshot()
    check_freshness(('body',), {'body': evidence['source_stamp']}, {'body': evidence['received']},
                    evidence['source_now'], evidence['receive_now'], max_age)
    for field, width in (('position', 3), ('orientation', 4), ('linear', 3), ('angular', 3)):
        values = evidence[field]
        if len(values) != width or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError(f'incomplete finite body {field}')
    if abs(sum(v*v for v in evidence['orientation']) - 1) > 1e-3:
        raise ValueError('body orientation is not a unit quaternion')
    return evidence


def require_mobility_stop(adapter, policy, max_age):
    try:
        _body_evidence(adapter, policy, max_age)
    except Exception as exc:
        raise RpcError('StopUnverified', 'body motion requires a usable measured-stop contract',
                       {'reason': 'stop contract unavailable', 'profile_field': 'safety.stop_verification',
                        'error': str(exc)}) from exc


def verify_mobility_stop(adapter, policy, max_age, *, clock=time.monotonic, sleep=time.sleep):
    """Verify all six twist axes and independent 3D pose differences.

    No planar, wheeled or morphology assumptions. Quaternion signs represent
    the same orientation. Neither repeated cache reads nor paused ROS time
    count toward dwell. Physical commissioning remains external.
    """
    require_mobility_stop(adapter, policy, max_age)
    started = clock()
    previous = baseline = None
    samples = 0
    last_error = 'no progressive stationary body evidence'
    while clock() < started + policy.timeout:
        try:
            evidence = _body_evidence(adapter, policy, max_age)
            if math.dist(evidence['linear'], (0,0,0)) > policy.linear_velocity_limit or math.dist(evidence['angular'], (0,0,0)) > policy.angular_velocity_limit:
                raise ValueError('measured body velocity exceeds stop criterion')
            if previous is not None:
                ds = evidence['source_stamp'] - previous['source_stamp']
                dr = evidence['received'] - previous['received']
                if ds < 0 or dr < 0:
                    previous = baseline = None
                elif ds == 0 or dr == 0:
                    raise LookupError('body feedback has not progressed')
                else:
                    dot = abs(sum(a*b for a,b in zip(evidence['orientation'], previous['orientation'])))
                    angular = 2 * math.acos(min(1., dot)) / ds
                    linear = math.dist(evidence['position'], previous['position']) / ds
                    if angular > policy.angular_velocity_limit or linear > policy.linear_velocity_limit:
                        raise ValueError('pose-derived body velocity exceeds stop criterion')
            samples += 1
            if baseline is None: baseline = evidence
            previous = evidence
            if evidence['source_stamp'] - baseline['source_stamp'] >= policy.dwell and evidence['received'] - baseline['received'] >= policy.dwell:
                return {'verified': True, 'criterion': 'body-velocity-dwell', 'samples': samples,
                        'elapsed': clock() - started, 'dwell': policy.dwell,
                        'linear_velocity_limit': policy.linear_velocity_limit,
                        'angular_velocity_limit': policy.angular_velocity_limit,
                        'linear': evidence['linear'], 'angular': evidence['angular'],
                        'source_stamp': evidence['source_stamp'], 'received': evidence['received'],
                        'physical_stop': 'uncommissioned'}
        except LookupError as exc:
            last_error = str(exc)
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            previous = baseline = None
            last_error = str(exc)
        except Exception as exc:
            raise RpcError('StopUnverified', 'body stop integration failed', {'error': str(exc)}) from exc
        remaining = started + policy.timeout - clock()
        if remaining > 0: sleep(min(policy.dwell / 4, .05, remaining))
    raise RpcError('StopUnverified', 'body stop criterion was not verified before deadline',
                   {'last_evidence': last_error, 'physical_stop': 'unverified', 'elapsed': clock() - started})


def verify_navigation_arrival(adapter, goal, config, max_age):
    """A controller terminal success must agree with a measured framed pose."""
    try:
        evidence = adapter.navigation_pose_snapshot(goal['frame'])
        check_freshness(('body',), {'body': evidence['source_stamp']}, {'body': evidence['received']},
                        evidence['source_now'], evidence['receive_now'], max_age)
        p = tuple(goal[a] for a in ('x','y','z'))
        q = tuple(goal['q'][a] for a in ('x','y','z','w'))
        actual_p, actual_q = evidence['position'], evidence['orientation']
        if len(actual_p) != 3 or len(actual_q) != 4 or any(type(v) not in (int,float) or not math.isfinite(v) for v in (*actual_p, *actual_q)):
            raise ValueError('incomplete finite navigation pose')
        if abs(sum(v*v for v in actual_q)-1) > 1e-3:
            raise ValueError('measured navigation quaternion is not unit length')
        position_error = math.dist(p, actual_p)
        orientation_error = 2 * math.acos(min(1., abs(sum(a*b for a,b in zip(q, actual_q)))))
        if position_error > config.arrival_position_tolerance or orientation_error > config.arrival_orientation_tolerance:
            raise ValueError(f'navigation residual exceeds tolerance: {position_error} m, {orientation_error} rad')
        return {'verified': True, 'frame': goal['frame'],
                'final_pose':{'frame':goal['frame'],'x':actual_p[0],'y':actual_p[1],'z':actual_p[2],
                              'q':dict(zip(('x','y','z','w'),actual_q))},
                'position_error': position_error,
                'orientation_error': orientation_error, 'position_tolerance': config.arrival_position_tolerance,
                'orientation_tolerance': config.arrival_orientation_tolerance,
                'source_stamp': evidence['source_stamp'], 'received': evidence['received']}
    except Exception as exc:
        raise RpcError('CompletionUnverified', 'navigation controller success lacks verified arrival',
                       {'arrived': False, 'error': str(exc)}) from exc
