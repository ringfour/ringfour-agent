"""Mandatory RPC effect declarations. Unknown registrations fail closed.

Descriptors describe the current implementation, including unverified completion.
Identity-based permission enforcement and model revision binding remain separate
v1 requirements. Dynamic vendor effects come from their validated profile.
"""
from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class ParameterSchema:
    """Strict wire types for effectful inputs, plus finite JSON for every RPC.

    Defaults are applied by handlers, never by coercing supplied wire values.
    Integration-dependent bounds/feedback are checked again at dispatch.
    """
    fields: tuple[tuple[str, str], ...] = ()
    required: tuple[str, ...] = ()
    allow_extra: bool = True

    def validate(self, params: dict[str, Any]) -> None:
        from .server import RpcError
        def reject(message):
            raise RpcError("RingfourError", message, {"reason": "invalid parameters"})
        def number(value):
            if type(value) not in (int, float): return False
            try: return math.isfinite(value)
            except OverflowError: return False
        def finite_json(value, path):
            if value is None or isinstance(value, (str, bool, int)):
                return
            if isinstance(value, float):
                if not math.isfinite(value): reject(f"{path} must be finite")
                return
            if isinstance(value, list):
                for i, item in enumerate(value): finite_json(item, f"{path}[{i}]")
                return
            if isinstance(value, dict) and all(isinstance(k, str) for k in value):
                for key, item in value.items(): finite_json(item, f"{path}.{key}")
                return
            reject(f"{path} must be a JSON value")
        if not isinstance(params, dict): reject("RPC params must be an object")
        finite_json(params, "params")
        fields = dict(self.fields)
        missing = set(self.required) - params.keys()
        if missing: reject(f"missing parameters: {sorted(missing)}")
        if not self.allow_extra:
            unknown = params.keys() - fields.keys()
            if unknown: reject(f"unknown parameters: {sorted(unknown)}")
        for name, kind in self.fields:
            if name not in params: continue
            value = params[name]
            if kind.startswith("optional-"):
                if value is None: continue
                kind = kind.removeprefix("optional-")
            if kind == "string":
                valid = isinstance(value, str) and bool(value.strip())
            elif kind in ("number", "positive", "nonnegative"):
                valid = number(value)
                if kind == "positive": valid = valid and value > 0
                if kind == "nonnegative": valid = valid and value >= 0
            elif kind == "integer": valid = type(value) is int and value >= 0
            elif kind == "pair":
                valid = (isinstance(value, list) and len(value) == 2 and
                         all(number(v) for v in value))
            elif kind == "boolean": valid = type(value) is bool
            elif kind == "object": valid = isinstance(value, dict)
            elif kind in ("pose", "goal"):
                valid = isinstance(value, dict)
                if valid:
                    if kind == "goal" and "place" in value:
                        schema(("place",), place="string").validate(value)
                    else:
                        schema(("frame",), x="number", y="number", z="number", yaw="number", q="object", frame="string").validate(value)
                        if "q" in value:
                            if "yaw" in value: reject("give yaw or q, not both")
                            schema(x="number", y="number", z="number", w="number").validate(value["q"])
                            norm = math.hypot(*(value['q'].get(k, 1. if k == 'w' else 0.) for k in ('x', 'y', 'z', 'w')))
                            if not math.isfinite(norm) or abs(norm - 1.) > 1e-6:
                                reject("goal quaternion must be unit length")
            elif kind == "goals":
                valid = isinstance(value,list) and bool(value)
                if valid:
                    for goal in value:schema(("goal",),goal="pose").validate({"goal":goal})
            elif kind == "waypoints":
                valid = (isinstance(value, list) and bool(value) and
                         all(isinstance(row, list) and bool(row) and
                             all(number(v) for v in row)
                             for row in value))
            else: raise ValueError(f"undeclared schema kind {kind!r}")
            if not valid: reject(f"{name} must be {kind}")


@dataclass(frozen=True)
class MethodDescriptor:
    effect: str
    permission: str
    resource: str
    ownership: str
    inhibit: str
    schema: ParameterSchema
    completion: str

    def resources(self, params: dict[str, Any]) -> tuple[str, ...]:
        """Logical dependencies; whole-robot arbitration remains conservative."""
        if self.resource == "group": return ("robot.motion", "group:" + params["group"])
        if self.resource == "task": return ("task:" + params["task_id"],)
        if self.resource == "vendor": return ("action:" + params["action"],)
        if self.resource == "mobility-configuration": return ("adapter:mobility", "map:active")
        return () if self.resource == "none" else (self.resource,)


def schema(required=(), **fields):
    return ParameterSchema(tuple(fields.items()), tuple(required), False)


READ = """
hello ping audit.query audit.subscribe audit.unsubscribe debug.send_payload
tf.lookup tf.frames tf.wait_for tf.transform events.subscribe events.unsubscribe
mobility.get_state slam.get_state slam.wait_for_localization slam.map_info slam.get_map
places.get places.list camera.info camera.get_frame camera.get_depth camera.get_synced
camera.get_cloud camera.deproject camera.stream_start camera.stream_stop lidar.scan
lidar.stream_start lidar.stream_stop model.description sensor.read arm.joint_states
group.tool_state range.read imu.read imu.mag arm.wrench env.read watch.poll
status.get status.diagnostics task.status
""".split()
MOTION = """
mobility.navigate_to teleop.start teleop.drive group.control_start group.follow
group.hold group.tool_set
""".split()
STOP = "task.cancel mobility.stop teleop.stop teleop.end group.stop group.control_stop".split()
CONFIGURATION = """
debug.set_estop debug.run_task slam.start slam.stop slam.save slam.load
slam.localize slam.relocalize slam.delete places.save places.delete watch.start watch.stop safety.recover
""".split()

DESCRIPTORS = {}
for effect, names in (("read", READ), ("motion", MOTION), ("stop", STOP),
                      ("configuration", CONFIGURATION), ("dynamic", ["action.invoke"])):
    for name in names:
        if name in DESCRIPTORS:
            raise RuntimeError(f"duplicate RPC declaration: {name}")
        # Read/configuration payload-specific validation remains in their handlers.
        # Configuration declares its robot-wide dependency even when not motion.
        DESCRIPTORS[name] = MethodDescriptor(
            effect, "observe" if effect == "read" else "configure" if effect == "configuration" else "stop" if effect == "stop" else "operate",
            "none" if effect == "read" else "robot.configuration" if effect == "configuration" else "robot.motion",
            "handler-owner" if effect == "stop" else "whole-robot-writer" if effect == "motion" else "none",
            "reject" if effect == "motion" else "allowed-for-stop" if effect == "stop" else "handler-policy",
            schema(), "snapshot-or-subscription" if effect == "read" else "handler-acknowledgement")


def declare(name, *, resource, ownership, completion, parameters):
    from dataclasses import replace
    DESCRIPTORS[name] = replace(DESCRIPTORS[name], resource=resource,
                               ownership=ownership, completion=completion, schema=parameters)


declare("group.control_start", resource="group", ownership="acquire-child",
        completion="controller-state-readback",
        parameters=schema(("group",), group="string", role="optional-string", raw="optional-string"))
for name in ("group.hold", "group.stop", "group.control_stop"):
    declare(name, resource="group", ownership="current-session",
            completion="cleanup-report-physical-stop-unverified" if name == "group.control_stop" else "controller-acknowledgement-physical-state-unverified",
            parameters=schema(("group", "session_id"), group="string", session_id="string"))
DESCRIPTORS['group.start_trajectory']=DESCRIPTORS['group.follow']
declare('group.start_trajectory',resource='group',ownership='current-session',
        completion='model-bound-complete-measured-joint-arrival',
        parameters=schema(('group','session_id','trajectory'),group='string',session_id='string',trajectory='object'))
declare("group.follow", resource="group", ownership="current-session",
        completion="fresh-complete-measured-joint-arrival",
        parameters=schema(("group", "session_id", "waypoints"), group="string", session_id="string", waypoints="waypoints", dt="positive"))
declare("group.tool_set", resource="group", ownership="transient-child",
        completion="dispatch-physical-completion-unverified",
        parameters=schema(("group", "position"), group="string", position="number"))
declare("teleop.start", resource="robot.motion", ownership="acquire-child",
        completion="watchdog-session-open", parameters=schema(watchdog="positive"))
declare("teleop.drive", resource="robot.motion", ownership="current-teleop-session",
        completion="dispatch-physical-completion-unverified",
        parameters=schema(linear="number", angular="number", lateral="number"))
for name in ("teleop.stop", "teleop.end", "mobility.stop"):
    declare(name, resource="robot.motion", ownership="handler-owner",
            completion="stop-acknowledgement-physical-state-unverified", parameters=schema())
declare("mobility.navigate_to", resource="robot.motion", ownership="acquire-child",
        completion="controller-result-arrival-unverified", parameters=schema(("goal",), goal="pose"))
for name in ("task.status", "task.cancel"):
    declare(name, resource="task", ownership="originating-client",
            completion="retained-task-state" if name == "task.status" else "cancellation-request",
            parameters=schema(("task_id",), task_id="string"))
declare("action.invoke", resource="vendor", ownership="profile-dependent-child",
        completion="dispatch-physical-completion-unverified",
        parameters=schema(("action",), action="string", arguments="object"))


# Payload declarations for observations and configuration. Nested typed spatial
# objects retain their model validation in handlers after finite JSON validation.
PAYLOAD_SCHEMAS = {
    "hello": schema(("protocol", "client_version"), protocol="integer", client_version="string"),
    "audit.query": schema(since="optional-integer", kind="optional-string", task_id="optional-string", limit="integer"),
    "debug.send_payload": schema(size="integer"),
    "debug.set_estop": schema(("engaged",), engaged="boolean"),
    "debug.run_task": schema(duration="positive", ticks="integer", fail="boolean"),
    "tf.lookup": schema(("parent", "child"), parent="string", child="string", timeout="nonnegative", stamp="optional-number"),
    "tf.wait_for": schema(("frame",), frame="string", timeout="nonnegative"),
    "tf.transform": schema(("kind", "entity", "to_frame"), kind="string", entity="object", to_frame="string", timeout="nonnegative"),
    "model.description": schema(group="optional-string"),
    "sensor.read": schema(("group", "signal"), group="string", signal="string"),
    "arm.joint_states": schema(arm="optional-string", include_external="boolean", stamp="optional-number"),
    "group.tool_state": schema(("group",), group="string"),
    "range.read": schema(("name",), name="string"),
    "arm.wrench": schema(("arm",), arm="string"),
    "env.read": schema(("kind",), kind="string"),
    "camera.get_cloud": schema(("camera", "voxel"), camera="string", voxel="positive"),
    "camera.deproject": schema(("frame_id", "pixel"), frame_id="string", pixel="pair", in_frame="optional-string"),
    "camera.stream_start": schema(("camera",), camera="string", fps="positive"),
    "camera.stream_stop": schema(("stream_id",), stream_id="string"),
    "lidar.stream_start": schema(hz="positive"),
    "lidar.stream_stop": schema(("stream_id",), stream_id="string"),
    "watch.start": schema(("path",), path="string", above="optional-number", below="optional-number", outside="optional-pair", stop="boolean", debounce="nonnegative", lifetime="optional-string"),
    "watch.poll": schema(("watch_id",), watch_id="string"),
    "watch.stop": schema(("watch_id",), watch_id="string"),
    "slam.start": schema(preserve_graph="boolean"),
    "slam.save": schema(("name",), name="string"),
    "slam.load": schema(("name",), name="string"),
    "slam.localize": schema(pose="optional-pose", name="optional-string"),
    "slam.wait_for_localization": schema(timeout="nonnegative"),
    "slam.map_info": schema(("name",), name="string"),
    "slam.delete": schema(("name",), name="string"),
    "places.save": schema(("name", "pose"), name="string", pose="pose"),
    "places.get": schema(("name",), name="string"),
    "places.delete": schema(("name",), name="string"),
}
for name in ("camera.info", "camera.get_frame", "camera.get_depth", "camera.get_synced"):
    PAYLOAD_SCHEMAS[name] = schema(("camera",), camera="string")
for name, parameters in PAYLOAD_SCHEMAS.items():
    from dataclasses import replace
    DESCRIPTORS[name] = replace(DESCRIPTORS[name], schema=parameters)

from dataclasses import replace
for name in ('slam.start', 'slam.stop', 'slam.load', 'slam.localize', 'slam.relocalize'):
    DESCRIPTORS[name] = replace(DESCRIPTORS[name], resource='mobility-configuration',
                                ownership='configuration-reservation')
DESCRIPTORS['safety.recover'] = replace(DESCRIPTORS['safety.recover'], permission='recover',
    resource='robot.motion', ownership='no-active-acquisitions', inhibit='must-be-engaged',
    schema=schema(('resource',), resource='string'), completion='controller-and-measured-stop-verification')


def validate_registry(registry):
    missing = set(registry) - DESCRIPTORS.keys()
    if missing:
        raise ValueError(f"Unclassified RPC methods prevent agent startup: {sorted(missing)}")
    for name in registry:
        d = DESCRIPTORS[name]
        if not all((d.effect, d.permission, d.resource, d.ownership, d.inhibit, d.completion)) or not isinstance(d.schema, ParameterSchema):
            raise ValueError(f"Incomplete admission descriptor for {name!r}")


def resolve_descriptor(method, params, profile):
    """Resolve profile-defined vendor effects before the common motion gate.

    Vendor parameter bounds and publication eligibility are rechecked by the
    worker under the motion mutex. This resolution cannot turn motion into read.
    """
    descriptor = DESCRIPTORS[method]
    if method == "debug.set_estop":
        from dataclasses import replace
        return replace(descriptor, permission="stop" if params['engaged'] else "reset-inhibit")
    if method in ("watch.start", "watch.stop"):
        from dataclasses import replace
        return replace(descriptor, permission="operate" if params.get('stop') else "observe")
    if method != "action.invoke":
        return descriptor
    from dataclasses import replace
    from .server import RpcError
    action = (profile.spec.actions or {}).get(params['action'])
    if action is None:
        raise RpcError("CapabilityNotSupported", f"undeclared action {params['action']!r}")
    effect = "motion" if action.is_motion else "configuration"
    return replace(descriptor, effect=effect,
                   permission="operate" if action.is_motion else "configure",
                   inhibit="reject" if action.is_motion else "handler-policy")

DESCRIPTORS['signal.get']=MethodDescriptor('read','observe','none','none','none',
    schema(('path',),path='string',max_age='positive',timeout='nonnegative',require_valid='boolean'),'measurement-with-provenance')
for name in ('guard.open','guard.status','guard.reset','guard.remove'):
    DESCRIPTORS[name]=MethodDescriptor('read' if name=='guard.status' else 'configuration',
        'observe' if name=='guard.status' else 'operate','robot.motion','owner','handler-policy',
        schema(('condition',),condition='object') if name=='guard.open' else schema(('guard_id',),guard_id='string'),
        'fresh-safe-entry-and-verified-removal')

DESCRIPTORS['mobility.start_route']=replace(DESCRIPTORS['mobility.navigate_to'],schema=ParameterSchema((('goals','goals'),),('goals',),False),completion='verified-arrival-at-every-waypoint')

DESCRIPTORS['group.start_tool']=replace(DESCRIPTORS['group.tool_set'],schema=schema(('group','position'),group='string',position='number',execution_timeout='positive'),completion='measured-position-residual-and-rest')

DESCRIPTORS['measurement.get']=MethodDescriptor('read','observe','none','none','none',schema(('kind',),kind='string',group='string',frame='string',max_age='positive',timeout='nonnegative',require_valid='boolean'),'measurement-with-provenance')

DESCRIPTORS.pop('group.follow',None)
DESCRIPTORS.pop('group.tool_set',None)

DESCRIPTORS["lease.renew"] = MethodDescriptor(
    "read", "observe", "none", "connection-lease", "allowed-for-stop",
    schema(("lease_id", "epoch"), lease_id="string", epoch="string"), "lease-renewed")

DESCRIPTORS["payload.set_transport"] = MethodDescriptor(
    "read", "observe", "none", "connection", "handler-policy",
    schema(("transport",), transport="string"), "transport-selected")


for name, field in [('commands.get', 'command_id'), ('operations.get', 'operation_id')]:
    DESCRIPTORS[name] = MethodDescriptor('read', 'observe', 'none', 'principal', 'allowed-for-stop',
        schema((field,), **{field: 'string'}), 'retained-authoritative-record')
