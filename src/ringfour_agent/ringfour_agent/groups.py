"""Resolving and validating the profile's ``groups:`` section (spec 8.6).

A ``GroupConfig`` says the minimum: kind, controllers, and — if IK is
wanted — a tip. Everything else is derived from the robot itself, because
the robot is the only thing that cannot be out of date:

* **joint names and order** come from the controller's claimed interfaces;
* **limits and geometry** come from ``/robot_description``;
* **whether a controller exists at all** comes from the controller_manager.

Deriving is only half the value. The other half is failing at startup: a
profile that names a controller nobody has, or a tip that is not in the
description, is a profile that will produce a ``None`` three months from
now in the middle of a mission. Every check here runs once, at boot, and
names the bad value in the message.

Two things about base101 shape this module and are easy to get wrong:

1. The trajectory controller is **declared but not loaded** (plan 6.0b).
   So "does this controller exist" cannot mean "is it in
   ``list_controllers``" — it means loaded *or* declared, and joints cannot
   be read from the trajectory controller because it has no claimed
   interfaces until someone loads it.
2. A tool joint may exist **only** inside a ``<ros2_control>`` block, with
   no ``<joint>`` and no links — a gripper can be commandable without
   being modelled. So tool joints are checked against those blocks *and*
   the kinematic tree, and being in either is enough. (base101's ``arm_6``
   happens to be in both.)

The URDF parsing itself lives in ``ringfour.description``, shared with the
client: the agent must validate against exactly the structure the client
will later be handed, and two parsers would eventually disagree about it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from ringfour.description import RobotDescription
from ringfour.enums import Controller, GroupKind
from ringfour_agent.models.profile import GroupConfig, RobotProfile

log = logging.getLogger("ringfour_agent.groups")

# Roles whose controller drives the group's own joints, best first. The
# trajectory controller is the authority on joint order when it is loaded;
# in the usual base101 state it is not, and its exclusive twin — which
# claims exactly the same joints, that being what makes them a pair — is.
_JOINT_SOURCE_ROLES = (Controller.TRAJECTORY, Controller.STREAM,
                       Controller.VELOCITY)

# How long startup waits for ros2_control's spawners before deciding the
# profile is wrong rather than the robot slow. Sized generously: on a cold
# `docker compose up` gazebo, the controller_manager and the spawners come
# up over several seconds, well after the agent's first question.
STARTUP_WAIT = 30.0


class GroupValidationError(Exception):
    """A declared group does not match the robot. Fatal at startup."""


@dataclass(frozen=True)
class GroupRuntime:
    """One group, as resolved against the live robot.

    ``joints`` is the kinematic joint list in command order, excluding the
    tool joint. ``tool_joint`` is separate because a different controller
    commands it: it must not appear in a trajectory or a stream command
    for this group, even when — as on base101 — it is a perfectly ordinary
    joint in the description.
    """

    name: str
    config: GroupConfig
    joints: tuple[str, ...]
    tool_joint: str | None
    # Where ``joints`` came from, for the startup log line: a mission that
    # gets an unexpected joint order should be able to see whether the
    # profile or the controller decided it.
    joints_from: str

    @property
    def kind(self) -> GroupKind:
        return self.config.kind

    @property
    def tip(self) -> str | None:
        return self.config.tip

    @property
    def goal_tolerance(self) -> float:
        """How far a joint may end from its final waypoint and still
        count as arrived. See GroupConfig.goal_tolerance."""
        return self.config.goal_tolerance

    @property
    def controllers(self) -> dict[Controller, str]:
        return self.config.controllers

    @property
    def all_joints(self) -> tuple[str, ...]:
        """Kinematic joints plus the tool joint, for watch paths."""
        if self.tool_joint is None:
            return self.joints
        return self.joints + (self.tool_joint,)


class _NotReady(Exception):
    """The robot has not finished coming up yet. Retried, then fatal.

    Separated from GroupValidationError because the two need opposite
    treatment at boot: a bad tip is wrong now and wrong in a minute, while
    an unspawned controller is the normal state of a stack that started
    three seconds ago.
    """


def resolve_groups(
    spec: RobotProfile, ros: Any = None, *, validate: bool = True,
    wait: float = STARTUP_WAIT, poll: float = 1.0,
) -> dict[str, GroupRuntime]:
    """Resolve every declared group against the live robot.

    ``validate=False``, or a ``ros`` that has no controller_manager methods
    at all, skips the live checks and trusts the profile — the path unit
    tests take. It is never the path a real agent takes silently: main.py
    logs where each group's joints came from.

    ``wait`` is how long to keep asking before giving up. The agent and its
    stack start together, and ros2_control spawners are asynchronous: for
    the first few seconds ``list_controllers`` legitimately answers "none",
    which is indistinguishable by content from a profile naming
    controllers that will never exist. Waiting is what tells them apart.
    Failing instantly would make the agent unable to cold-start alongside
    its own robot; not failing at all was the original bug — a group
    resolved with zero joints and boots "successfully", then raises on the
    first joint read, which is exactly the deferred failure this module
    exists to prevent.

    Raises GroupValidationError, naming the offending value.
    """
    if not spec.groups:
        return {}
    if not validate or ros is None:
        return _attempt(spec, ros, loaded=None, declared=frozenset(),
                        validate=False)

    deadline = time.monotonic() + max(wait, 0.0)
    attempt = 0
    while True:
        attempt += 1
        # None means "this object cannot answer" (a test fake); an empty
        # dict means "answered, nothing is loaded" — a real difference.
        loaded = _try(ros, "loaded_controllers", None)
        declared = _try(ros, "declared_controllers", None)
        if loaded is None and declared is None:
            # This object implements none of the controller_manager
            # surface, so there is nothing to wait for. A real
            # RosInterface always has these methods; this is a test fake.
            return _attempt(spec, ros, loaded=None, declared=frozenset(),
                            validate=False)
        try:
            return _attempt(spec, ros, loaded=loaded,
                            declared=declared or frozenset(), validate=True)
        except _NotReady as exc:
            if time.monotonic() >= deadline:
                raise GroupValidationError(
                    f"{exc} (still true after {wait:.0f}s and {attempt} "
                    f"attempts, so this is the robot's real state, not a "
                    f"slow boot)") from None
            if attempt == 1:
                log.info("waiting up to %.0fs for the robot to finish "
                         "coming up: %s", wait, exc)
            time.sleep(poll)


def _attempt(
    spec: RobotProfile, ros: Any, *,
    loaded: dict[str, dict[str, Any]] | None,
    declared: frozenset[str],
    validate: bool,
) -> dict[str, GroupRuntime]:
    """One pass over the groups. Raises _NotReady if the robot is still
    starting; GroupValidationError if the profile is simply wrong."""
    description: RobotDescription | None = None
    if validate:
        xml = _try(ros, "description", None)
        if xml:
            try:
                description = RobotDescription.parse(xml)
            except ValueError as exc:
                raise GroupValidationError(str(exc)) from exc

    known = frozenset(loaded or {}) | declared
    out: dict[str, GroupRuntime] = {}
    for name, config in spec.groups.items():
        # A group that names its own description topic is validated
        # against THAT description. Falling back to the composed one would
        # defeat the point of the override: on a namespaced launch the
        # default topic does not contain this arm's links at all, so every
        # tip would look bogus.
        group_description = description
        if config.description_topic and ros is not None:
            xml = _try_with(ros, "group_description", name, None)
            if xml:
                try:
                    group_description = RobotDescription.parse(xml)
                except ValueError as exc:
                    raise GroupValidationError(
                        f"groups.{name}.description_topic "
                        f"({config.description_topic}): {exc}") from exc
            elif validate:
                raise _NotReady(
                    f"groups.{name}.description_topic names "
                    f"{config.description_topic!r}, but nothing has been "
                    f"published there")
        if validate and config.tip and group_description is None:
            raise _NotReady(
                f"groups.{name} declares tip {config.tip!r} but no robot "
                f"description has arrived, so it cannot be checked")
        out[name] = _resolve_one(name, config, loaded, known,
                                 group_description, validate=validate)
    return out


def _resolve_one(
    name: str,
    config: GroupConfig,
    loaded: dict[str, dict[str, Any]] | None,
    known: frozenset[str],
    description: RobotDescription | None,
    *,
    validate: bool = True,
) -> GroupRuntime:
    where = f"groups.{name}"

    # 1. Every declared controller must exist, loaded or merely declared.
    #    An empty `known` while validating means the controller_manager
    #    has not come up yet - not that the profile is wrong.
    if validate and loaded is not None:
        if not known:
            raise _NotReady(
                "the controller_manager lists no controllers at all")
        for role, controller in config.controllers.items():
            if controller not in known:
                raise _NotReady(
                    f"{where}.controllers.{role}: no controller named "
                    f"{controller!r} on the controller_manager (known: "
                    f"{', '.join(sorted(known))})")

    # 2. Joints: the profile overrides, otherwise the loaded controller.
    tool_controller = config.controllers.get(Controller.TOOL)
    if config.joints:
        joints, joints_from = tuple(config.joints), "profile"
    elif loaded is None:
        # Nothing to ask. Only reachable with validation off or a fake
        # that implements no controller_manager methods.
        joints, joints_from = (), "unresolved"
    else:
        joints, joints_from = _joints_from_controllers(
            where, config, loaded, tool_controller)

    # 3. Tool joint: profile-declared, checked against ros2_control.
    unknown_tolerances = set(config.goal_tolerances) - set(joints)
    if unknown_tolerances and (joints or validate):
        raise GroupValidationError(f"{where}.goal_tolerances names unknown joints: {sorted(unknown_tolerances)}")
    tool_joint = config.tool.joint if config.tool is not None else None
    if tool_joint is not None and description is not None:
        if (tool_joint not in description.integration['hardware_joints']
                and tool_joint not in description.joints):
            raise GroupValidationError(
                f"{where}.tool.joint: {tool_joint!r} is not declared to "
                f"ros2_control and is not a joint in the description - "
                f"nothing can command it.")

    # 4. Tip: must be a real link, and downstream of every group joint.
    if config.tip is not None and description is not None:
        _validate_tip(where, config.tip, joints, description)

    return GroupRuntime(name=name, config=config, joints=joints,
                        tool_joint=tool_joint, joints_from=joints_from)


def _joints_from_controllers(
    where: str,
    config: GroupConfig,
    loaded: dict[str, dict[str, Any]],
    tool_controller: str | None,
) -> tuple[tuple[str, ...], str]:
    """Claimed interfaces of the best loaded controller for this group."""
    for role in _JOINT_SOURCE_ROLES:
        controller = config.controllers.get(role)
        if controller is None or controller == tool_controller:
            continue
        state = loaded.get(controller)
        if state and state["joints"]:
            return tuple(state["joints"]), f"{controller} ({role})"
    tried = [config.controllers[role] for role in _JOINT_SOURCE_ROLES
             if role in config.controllers]
    raise _NotReady(
        f"{where}: cannot derive joint names - none of this group's motion "
        f"controllers ({', '.join(tried) or '(none)'}) is loaded and "
        f"claiming interfaces. Either spawn one at boot or set "
        f"{where}.joints explicitly")


def _validate_tip(
    where: str, tip: str, joints: tuple[str, ...],
    description: RobotDescription,
) -> None:
    try:
        chain = description.chain_to(tip)
    except KeyError:
        raise GroupValidationError(
            f"{where}.tip: {tip!r} is not a link in the description. IK "
            f"would have no frame to solve to."
        ) from None
    # A group owns only its controller joints. Upstream lifts/torso joints
    # are valid independent groups and are frozen from measured state by IK.
    missing = [joint for joint in joints if joint not in chain]
    if missing:
        raise GroupValidationError(
            f"{where}.tip: {tip!r} is not downstream of "
            f"{', '.join(missing)} - moving "
            f"{'those joints' if len(missing) > 1 else 'that joint'} does "
            f"not move the tip, so it cannot be this group's end effector.")


def _try_with(ros: Any, method: str, argument: Any, default: Any) -> Any:
    """``_try`` for a method that takes one argument."""
    fn = getattr(ros, method, None)
    if fn is None:
        return default
    return fn(argument)


def _try(ros: Any, method: str, default: Any) -> Any:
    """Call an optional RosInterface method; ``default`` if it is absent.

    Test fakes implement the handful of methods their test needs. Missing
    ones mean "this robot cannot answer", which downgrades validation
    rather than crashing the agent — a real RosInterface always has them,
    and a real failure raises out of the method instead.
    """
    fn = getattr(ros, method, None)
    if fn is None:
        return default
    return fn()


def describe(groups: dict[str, GroupRuntime]) -> str:
    """One-line-per-group summary for the startup log."""
    if not groups:
        return "(none)"
    return "; ".join(
        f"{name} [{group.kind}] joints={len(group.joints)}"
        f" from {group.joints_from}"
        f"{'' if group.tip is None else f' tip={group.tip}'}"
        for name, group in groups.items()
    )
