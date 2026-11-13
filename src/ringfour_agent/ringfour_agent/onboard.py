"""``ros2 run ringfour_agent onboard`` — record a robot, once.

Everything a client needs but cannot work out for itself, computed on the
robot where the files and the ROS environment are, and written to a
snapshot keyed by the description's sha:

* **mesh volumes** — a client holding a description it received over a
  socket cannot open ``package://``. On base101 every mesh collision is a
  jaw, so without this the gripper is invisible to collision checking;
* **the allowed-collision matrix** — cheap, but there is no reason to
  derive it in every client process;
* **the reachability map** — minutes of sampling, and the whole point of
  doing it here is that nobody pays for it twice.

**Why this is a ROS executable and not a plain CLI** (Cristi,
2026-08-25): the two things it needs — expanding xacro conditionals, and
resolving ``package://`` — are exactly what a ROS environment provides.
A standalone CLI would have to reimplement both, badly.

**Recorded, never authored** (spec 8.7). The description is taken off the
live topic, not read from a path in a profile. That is what keeps the
provenance guarantee of 26.2 while still producing a file: the snapshot
says what the robot WAS, and its sha says which robot that was, so a
mismatch on connect is loud instead of silent.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from ringfour.collision import CollisionModel, link_poses
from ringfour.description import CollisionShape, RobotDescription
from ringfour.meshes import (
    Bounds,
    _rpy_of,
    bounds_of_stl,
    resolve_mesh_path,
)
from ringfour.snapshot import ReachMap, Snapshot

from .groups import GroupValidationError, resolve_groups
from .profile import ProfileError, load_profile
from .ros import RosInterface

log = logging.getLogger("ringfour_onboard")

DEFAULT_SAMPLES = 60_000
DEFAULT_RESOLUTION = 0.02       # metres per voxel


def _rotation(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw as a 3x3."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _share_directories() -> dict[str, str]:
    """package name -> share directory, from the ament index."""
    try:
        from ament_index_python.packages import (
            get_package_share_directory,
            get_packages_with_prefixes,
        )
    except ImportError:                       # pragma: no cover
        return {}
    out: dict[str, str] = {}
    for package in get_packages_with_prefixes():
        try:
            out[package] = get_package_share_directory(package)
        except Exception:
            continue
    return out


def measure_meshes(description: RobotDescription,
                   share: dict[str, str]) -> dict[str, tuple[CollisionShape, ...]]:
    from ringfour.onboarding import measure_meshes as measure
    return measure(description, lambda mesh: resolve_mesh_path(mesh, share))


def sample_reach(description: RobotDescription,
                 joints: tuple[str, ...], tip: str,
                 model: CollisionModel,
                 samples: int = DEFAULT_SAMPLES,
                 resolution: float = DEFAULT_RESOLUTION) -> ReachMap:
    """Sample joint space forward and bin the tip into voxels.

    Forward, not inverse: asking IK about each voxel costs most of a
    second per unreachable one, and a grid has hundreds of thousands.
    Drawing a configuration and running FK costs a fraction of a
    millisecond, and it cannot produce a configuration that breaks a joint
    limit because it draws from within them.

    Self-colliding samples are counted and discarded, which is why the map
    is only honest once the meshes have been measured — on base101 the
    jaws are the furthest-out geometry and the likeliest thing to strike
    the chassis.
    """
    limits = [description.joints[name].limits for name in joints]
    spans = [
        bound if bound is not None else (-math.pi, math.pi)
        for bound in limits
    ]
    rng = np.random.default_rng(0)      # reproducible: same robot, same map
    # The tip is usually a bare frame with no <collision>, so it is NOT in
    # model.links and asking only for those gives every sample a tip pose
    # of None. Ask for both.
    checked = tuple(dict.fromkeys(tuple(model.links) + (tip,)))

    tips: list[np.ndarray] = []
    rejected = 0
    report = max(1, samples // 10)
    for index in range(samples):
        q = {name: float(rng.uniform(lo, hi))
             for name, (lo, hi) in zip(joints, spans)}
        poses = link_poses(description, q, checked)
        if model.check(poses):
            rejected += 1
            continue
        pose = poses.get(tip)
        if pose is None:
            continue
        tips.append(pose[:3, 3].copy())
        if (index + 1) % report == 0:
            log.info("  %6d/%d sampled, %d rejected for self-collision",
                     index + 1, samples, rejected)

    if not tips:
        if rejected == samples:
            raise RuntimeError(
                f"all {samples} sampled configurations self-collided. The "
                f"collision model is too coarse, or a bounding volume is "
                f"wrong.")
        raise RuntimeError(
            f"no sample produced a pose for tip {tip!r}. Is it a link in "
            f"the description?")

    points = np.asarray(tips)
    low = points.min(axis=0) - resolution
    high = points.max(axis=0) + resolution
    shape = tuple(
        int(math.ceil((h - l) / resolution)) + 1 for l, h in zip(low, high))
    counts = np.zeros(shape, dtype=np.uint32)
    index = np.round((points - low) / resolution).astype(int)
    np.add.at(counts, tuple(index.T), 1)
    return ReachMap(origin=tuple(float(v) for v in low),
                    resolution=resolution, counts=counts,
                    samples=samples, rejected=rejected)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        prog="ringfour_agent onboard",
        description="Record a robot: meshes, collision matrix, reachability.")
    parser.add_argument("--profile", required=True,
                        help="path to the robot profile YAML")
    parser.add_argument("--group", default=None,
                        help="which group to map (default: the only arm)")
    parser.add_argument("--out", default=None,
                        help="snapshot root (default: beside the profile)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--resolution", type=float,
                        default=DEFAULT_RESOLUTION)
    parser.add_argument("--skip-reach", action="store_true",
                        help="meshes and collision matrix only (seconds)")
    parser.add_argument("--force", action="store_true",
                        help="re-record even if this sha is already stored")
    args, _ = parser.parse_known_args(argv or sys.argv[1:])

    try:
        profile = load_profile(args.profile)
    except ProfileError as exc:
        log.error("%s", exc)
        return 1

    rclpy.init(args=None)
    node = Node("ringfour_onboard")
    ros = RosInterface(node, profile)
    import threading
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    try:
        return _record(args, profile, ros)
    finally:
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


def _record(args, profile, ros) -> int:
    log.info("waiting for /robot_description ...")
    urdf = ros.description(wait=20.0)
    if not urdf:
        log.error("nothing published on /robot_description. It is latched, "
                  "so this means robot_state_publisher is not running.")
        return 1
    description = RobotDescription.parse(urdf)
    log.info("recorded %s: %d links, %d joints, sha %s",
             description.name, len(description.links),
             len(description.joints), description.sha[:12])

    # Beside the profile it belongs to, in a directory named after the
    # profile FILE — `base101_arm.yaml` -> `base101_arm/<sha>/`. The file
    # stem rather than the name inside it: the stem is what you point
    # --profile at, and it cannot drift from the thing on disk.
    stem = Path(args.profile).stem
    root = Path(args.out) if args.out else Path(args.profile).resolve().parent
    directory = root / stem / description.sha
    if (directory / "meta.json").exists() and not args.force:
        log.info("already recorded at %s (--force to redo)", directory)
        return 0

    groups = profile.spec.groups or {}
    # Default to the single arm, then to any group at all. A robot with
    # several wants --group, because a reach map is per-group.
    name = args.group
    if name is None:
        arms = profile.spec.instances.get("arms", ())
        name = arms[0] if arms else next(iter(groups), None)
    if name is None or name not in groups:
        log.error("no group to record (profile declares %s)", sorted(groups))
        return 1
    group = groups[name]

    log.info("resolving meshes ...")
    hulls = measure_meshes(description, _share_directories())
    if not hulls:
        log.info("  (none — every collision is already a primitive)")

    merged = dict(description.collisions)
    for link, shapes in hulls.items():
        merged[link] = tuple(
            s for s in merged.get(link, ()) if s.modelled) + shapes
    resolved = RobotDescription(
        urdf=description.urdf, sha=description.sha, name=description.name,
        links=description.links, joints=description.joints,
        hardware_joints=description.hardware_joints, collisions=merged,
        _parent_of=description._parent_of)

    # The group's joints come off the controller, exactly as they do at
    # agent startup (6.1) — the profile deliberately does not list them.
    # Falling back to "every moving joint" would sample the wheels, which
    # cost time and cannot move the tip.
    try:
        runtime = resolve_groups(profile.spec, ros, wait=15.0)
        joints = runtime[name].joints
    except GroupValidationError as exc:
        log.error("cannot resolve %s's joints: %s", name, exc)
        return 1
    if not joints:
        log.error("group %r resolved to no joints", name)
        return 1
    log.info("building the collision matrix over the robot joint model ...")
    started = time.monotonic()
    model = CollisionModel.build(resolved)
    log.info("  %d links, %d pairs to test, %d unmodelled, %.2fs",
             len(model.links), len(model.pairs), len(model.unmodelled),
             time.monotonic() - started)
    if model.unmodelled:
        log.warning("  STILL UNMODELLED: %s", ", ".join(model.unmodelled))

    reach = None
    if not args.skip_reach:
        if not group.tip:
            log.warning("group %r declares no tip; skipping the reach map",
                        name)
        elif any(resolved.joints[n].moves and n not in joints
                 for n in resolved.chain_to(group.tip)):
            log.warning("group %r has independently controlled joints in its "
                        "tip chain; skipping the reach map because its format "
                        "does not record external joint positions", name)
        else:
            log.info("sampling %d configurations for the reach map ...",
                     args.samples)
            started = time.monotonic()
            reach = sample_reach(resolved, joints, group.tip, model,
                                 args.samples, args.resolution)
            log.info("  %d voxels reachable at %.0f mm, %d rejected, %.1fs",
                     reach.reachable_voxels, args.resolution * 1000,
                     reach.rejected, time.monotonic() - started)

    snapshot = Snapshot(
        robot=profile.info.name, sha=description.sha, description=description,
        hulls=hulls, reach=reach,
        meta={
            "recorded": datetime.now(timezone.utc).isoformat(),
            "group": name,
            "tip": group.tip,
            "joints": list(joints),
            "links": len(description.links),
            "pairs": len(model.pairs),
            "unmodelled": list(model.unmodelled),
            "samples": args.samples if reach is not None else 0,
            "resolution": args.resolution,
        },
    )
    written = snapshot.write(directory)
    log.info("wrote %s", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
