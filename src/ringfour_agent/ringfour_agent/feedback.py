"""Merge measured joint feedback without fabricating missing sensor values."""
import math


def check_freshness(names, stamps, received, source_now, receive_now, max_age):
    """Compare only matching clocks; require both source and receive freshness."""
    invalid = {}
    for name in names:
        source = stamps.get(name)
        reception = received.get(name)
        if source is None or reception is None or not math.isfinite(source):
            invalid[name] = "timestamp unavailable"
            continue
        source_age = source_now - source
        receive_age = receive_now - reception
        if not (0 <= source_age <= max_age and 0 <= receive_age <= max_age):
            invalid[name] = {"source_age": source_age, "receive_age": receive_age}
    if invalid:
        raise ValueError(f"fresh joint feedback unavailable: {invalid}")


def update_joint_feedback(joints, stamps, message, stamp):
    for i, name in enumerate(message.name):
        if i >= len(message.position):
            # An effort/velocity-only message supplies no new position.
            # Retain an older measured value and its age; never invent zero.
            continue
        joints[name] = (
            message.position[i],
            message.velocity[i] if i < len(message.velocity) else math.nan,
            message.effort[i] if i < len(message.effort) else math.nan,
        )
        stamps[name] = stamp


def snapshot_at(history, stamp):
    """Select an actual sample; never interpolate missing joint feedback."""
    for measured_stamp, values in reversed(history):
        if abs(measured_stamp-stamp) <= 1e-6:
            return dict(values), measured_stamp
    raise LookupError(f'No measured joint snapshot at ROS time {stamp}; TF sample may be older than the feedback history')
