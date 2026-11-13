"""SLAM orchestration (spec section 11): the mode_manager's successor.

slam_toolbox stays up for the whole session (base101_slam launch);
everything here is service calls and bookkeeping, never process control.
Modes (Cristi's ruling, 2026-06-12): start() resumes mapping, stop()
pauses it (idle), load()/localize() switch to localization via
deserialize. Userland lifecycle control comes later; for now the nodes
are assumed up.

Map storage layout (one directory per map under the profile's
slam.map_dir, default ~/.ringfour/maps):

    <map_dir>/<name>/<name>.posegraph + .data   serialized pose graph
    <map_dir>/<name>/grid.png + grid.yaml       2D occupancy export
    <map_dir>/<name>/map_info.yaml              MapInfo metadata

Localization quality is a HEURISTIC (slam_toolbox exposes no metric):
a 2 Hz monitor tracks the map->odom correction; smooth drift is healthy,
jumps mean scan-matching is fighting. quality = exp(-jumps); is_lost =
no TF at all or quality below 0.2. Honest enough for watches and
wait_for_localization; revisit if a real metric appears.
"""

from __future__ import annotations

import asyncio
import logging
import math
import shutil
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ringfour.enums import SlamMode

from .server import RpcError

log = logging.getLogger("ringfour_agent.slam")

# Map name for places saved before the first save()/load(): they belong
# to the session's scratch map and are migrated into the named directory
# on save().
SCRATCH_MAP = "_unsaved"

QUALITY_LOST_THRESHOLD = 0.2

# map_server's trinary palette and thresholds (nav2 map_io.cpp).
#
# The two free thresholds are NOT a typo, and map_saver_cli writes exactly
# this pair. They answer different questions:
#
#   WRITING, cell -> pixel: a cell of 25 or less is free. That is the 0.25
#   in SaveParameters.
#
#   READING, pixel -> cell: map_server inverts the pixel, occ =
#   (255 - pixel) / 255. Grey is 205, so unknown comes back as 0.19608 -
#   and a free threshold of 0.25 would swallow it and turn every unknown
#   cell into free space. 0.196 sits just under it, which is the whole
#   reason for that odd-looking number.
_PX_FREE = 254
_PX_OCCUPIED = 0
_PX_UNKNOWN = 205
_CELL_FREE_MAX = 25            # writing: cell <= this is free
_CELL_OCCUPIED_MIN = 65        # writing: cell >= this is occupied
_YAML_FREE_THRESH = 0.196      # reading: keeps grey unknown, not free
_YAML_OCCUPIED_THRESH = 0.65

# Correction jumps (meters per sample at 2 Hz) that count as "fighting".
_JUMP_METERS = 0.05
_JUMP_DECAY = 0.85  # per sample; ~3 s to forgive one jump


class SlamManager:
    """Mode state machine + map files for one robot."""

    def __init__(self, ros: Any, map_dir: str | None) -> None:
        self._ros = ros
        self.map_dir = Path(map_dir or "~/.ringfour/maps").expanduser()
        self.map_dir.mkdir(parents=True, exist_ok=True)
        self.mode = SlamMode.MAPPING
        self.active_map = SCRATCH_MAP
        # quality monitor state (read by watches via ctx.watch_paths)
        self.quality: float = 0.0
        self._last_correction: tuple[float, float, float] | None = None
        self._jump_level = 1.0  # 1.0 = no recent jumps -> quality 1.0

    # -- mode switching ---------------------------------------------------------

    def start(self, preserve_graph: bool = False) -> SlamMode:
        """Ensure mapping mode. From idle: unpause. From localization:
        reset the graph and map fresh (the loaded map stays on disk).

        slam_toolbox's Pause service is a TOGGLE; the bridge is the only
        mode owner (doctrine), so we track paused-ness here rather than
        trusting the response shape.
        """
        if self.mode is SlamMode.IDLE:
            self._ros.slam_pause_toggle()   # paused -> running
        elif self.mode is SlamMode.LOCALIZATION:
            if not preserve_graph:
                self._ros.slam_reset(pause=False)
            # Keep places when resuming the loaded graph without a reset.
            source = self.map_dir / self.active_map / "places.yaml"
            if preserve_graph and source.exists():
                target = self.map_dir / SCRATCH_MAP / "places.yaml"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            self.active_map = SCRATCH_MAP
        self.mode = SlamMode.MAPPING
        return self.mode

    def stop(self) -> SlamMode:
        """Pause map building (idle). The map and TF stay published."""
        if self.mode is SlamMode.MAPPING:
            self._ros.slam_pause_toggle()   # running -> paused
            self.mode = SlamMode.IDLE
        return self.mode

    # -- save / load ---------------------------------------------------------------

    def save(self, name: str) -> dict[str, Any]:
        """Serialize pose graph + export the 2D grid + write metadata.
        Places saved on the scratch map migrate to the named map."""
        self._check_name(name)
        target = self.map_dir / name
        target.mkdir(parents=True, exist_ok=True)
        self._ros.slam_serialize(str(target / name))
        grid_info = self._export_grid(target)
        info = {
            "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "resolution": grid_info.get("resolution", 0.05),
            "dimensions": grid_info.get("dimensions", [0.0, 0.0]),
            "backend": "slam_toolbox",
            "is_3d": False,
        }
        (target / "map_info.yaml").write_text(yaml.safe_dump(info))
        source_places = self.map_dir / self.active_map / "places.yaml"
        if source_places.exists() and source_places != target / "places.yaml":
            shutil.copyfile(source_places, target / "places.yaml")
        self.active_map = name
        return info

    def load(self, name: str) -> None:
        """Load a saved map and continue MAPPING on top of it."""
        posegraph = self._posegraph_path(name)
        self._ros.slam_deserialize(str(posegraph), match_type=1)
        self.active_map = name
        self.mode = SlamMode.MAPPING

    def localize(self, pose: dict[str, Any] | None = None,
                 name: str | None = None) -> None:
        """Switch to localization mode against a saved map.

        ``name`` defaults to the active map; ``pose`` (wire Pose dict)
        defaults to the robot's current pose estimate — good enough when
        the robot hasn't been kidnapped since mapping.
        """
        name = name or self.active_map
        if name == SCRATCH_MAP:
            raise RpcError(
                "RingfourError",
                "no saved map to localize against — slam.save() first "
                "(or slam.load() a saved one)",
            )
        posegraph = self._posegraph_path(name)
        x, y, theta = self._initial_pose(pose)
        self._ros.slam_deserialize(str(posegraph), match_type=3,
                                   x=x, y=y, theta=theta)
        self.active_map = name
        self.mode = SlamMode.LOCALIZATION

    def relocalize(self) -> None:
        """Re-run localization at the current pose estimate (the
        is_lost recovery path)."""
        self.localize()

    def _initial_pose(self, pose: dict[str, Any] | None) -> tuple:
        if pose is not None:
            yaw = _yaw_of(pose)
            return float(pose.get("x", 0.0)), float(pose.get("y", 0.0)), yaw
        try:
            state = self._ros.get_state()
            current = state["pose"]
            return current["x"], current["y"], _yaw_of(current)
        except Exception:
            return 0.0, 0.0, 0.0  # no odom yet: best effort from origin

    # -- map registry ---------------------------------------------------------------

    def saved_maps(self) -> list[str]:
        return sorted(
            p.parent.name for p in self.map_dir.glob("*/*.posegraph"))

    def map_info(self, name: str) -> dict[str, Any]:
        path = self.map_dir / name / "map_info.yaml"
        if not path.exists():
            raise RpcError("RingfourError", f"no saved map {name!r}")
        return yaml.safe_load(path.read_text())

    def delete(self, name: str) -> None:
        self._check_name(name)
        target = self.map_dir / name
        if not target.is_dir():
            raise RpcError("RingfourError", f"no saved map {name!r}")
        if name == self.active_map:
            raise RpcError("RingfourError",
                           f"map {name!r} is active; switch maps first")
        shutil.rmtree(target)

    def _posegraph_path(self, name: str) -> Path:
        self._check_name(name)
        path = self.map_dir / name / f"{name}.posegraph"
        if not path.exists():
            raise RpcError(
                "RingfourError",
                f"no saved map {name!r} (have: {self.saved_maps()})",
            )
        return path.with_suffix("")  # slam_toolbox wants the stem

    @staticmethod
    def _check_name(name: str) -> None:
        if (not name or name.startswith((".", "_"))
                or any(c in name for c in "/\\")):
            raise RpcError("RingfourError",
                           f"bad map name {name!r} (no paths, no leading "
                           "./_; '_unsaved' is reserved)")

    def _export_grid(self, target: Path) -> dict[str, Any]:
        """Write the nav2 map_server pair, grid.png + grid.yaml.

        This used to shell out to `ros2 run nav2_map_server map_saver_cli`,
        which meant 274 MB of Nav2 in the agent's image to re-subscribe to
        /map and render a message this process is already holding. The
        rules below are map_io.cpp's, so the output is the same file the
        CLI wrote and map_server still loads it.

        Best effort, as before: if there is no map yet, the pose graph is
        still saved and the grid is skipped.
        """
        try:
            grid = self._ros.map_grid(wait=1.0)
        except Exception as exc:
            log.warning("no grid on /map (%s); pose graph saved, grid "
                        "export skipped", exc)
            return {}

        try:
            resolution = float(grid["resolution"])
            origin = grid["origin"]
            _write_png_gray(target / "grid.png", _trinary(grid["data"]))
            (target / "grid.yaml").write_text(yaml.safe_dump({
                "image": "grid.png",
                "mode": "trinary",
                "resolution": resolution,
                "origin": [float(origin["x"]), float(origin["y"]),
                           _yaw_of(origin)],
                "negate": 0,
                "occupied_thresh": _YAML_OCCUPIED_THRESH,
                "free_thresh": _YAML_FREE_THRESH,
            }, sort_keys=False))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            log.warning("grid export failed (%s); pose graph saved", exc)
            return {}

        return {
            "resolution": resolution,
            "dimensions": [grid["width"] * resolution,
                           grid["height"] * resolution],
        }

    # -- localization quality (heuristic) ---------------------------------------------

    def sample_quality(self) -> None:
        """One monitor tick: update self.quality from map->odom motion."""
        correction = self._ros.map_odom_correction()
        if correction is None:
            self.quality = 0.0
            self._last_correction = None
            return
        if self._last_correction is not None:
            dx = correction[0] - self._last_correction[0]
            dy = correction[1] - self._last_correction[1]
            if math.hypot(dx, dy) > _JUMP_METERS:
                # Bound memory of old jumps: repeated multiplication can
                # underflow to zero, which can never recover by division.
                self._jump_level = max(.01, self._jump_level * .3)
            else:
                self._jump_level = min(
                    1.0, self._jump_level / _JUMP_DECAY)  # recover slowly
        self._last_correction = correction
        self.quality = max(0.0, min(1.0, self._jump_level))

    @property
    def is_lost(self) -> bool:
        return self.quality < QUALITY_LOST_THRESHOLD

    def state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "backend": "slam_toolbox",
            "active_map": (None if self.active_map == SCRATCH_MAP
                           else self.active_map),
            "localization_quality": self.quality,
            "is_lost": self.is_lost,
            "saved_maps": self.saved_maps(),
            "supports_preserve_graph": True,
        }


async def run_quality_monitor(slam: SlamManager,
                              interval: float = 0.5) -> None:
    """2 Hz background sampler updating slam.quality (feeds the
    slam.localization_quality watch path and wait_for_localization)."""
    while True:
        try:
            await asyncio.to_thread(slam.sample_quality)
        except Exception:
            log.exception("slam quality sample failed")
        await asyncio.sleep(interval)


def _trinary(data: np.ndarray) -> np.ndarray:
    """OccupancyGrid cells -> map_server greyscale, trinary mode.

    nav2_map_server/src/map_io.cpp: <=25 is free, >=65 is occupied, and
    everything else - including the -1 that means unknown - is grey. The
    row flip is not cosmetic: an OccupancyGrid's first row is its bottom
    edge (origin sits at the lower-left corner), an image's first row is
    its top edge, so a map written without it comes out mirrored.
    """
    known = (data >= 0) & (data <= 100)
    px = np.full(data.shape, _PX_UNKNOWN, dtype=np.uint8)
    px[known & (data <= _CELL_FREE_MAX)] = _PX_FREE
    px[known & (data >= _CELL_OCCUPIED_MIN)] = _PX_OCCUPIED
    return px[::-1]


def _write_png_gray(path: Path, pixels: np.ndarray) -> None:
    """Write an 8-bit greyscale PNG. No Pillow: the agent's dependency
    list is three packages long and this is the whole format - a signature,
    IHDR, one zlib-compressed IDAT whose scanlines each carry a leading
    filter byte, IEND."""
    height, width = pixels.shape

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload
                + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    scanlines = b"".join(b"\x00" + row.tobytes()
                         for row in np.ascontiguousarray(pixels))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines, 9))
        + chunk(b"IEND", b""))


def _yaw_of(pose: dict[str, Any]) -> float:
    q = pose.get("q") or {}
    x, y = float(q.get("x", 0.0)), float(q.get("y", 0.0))
    z, w = float(q.get("z", 0.0)), float(q.get("w", 1.0))
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
