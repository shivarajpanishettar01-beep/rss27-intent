"""Episode schema and LeRobot v3.0 loaders (Open-H + own data). SPEC section 4, INTERFACES.

In-memory unit: :class:`Episode`. All arrays are indexed by step k at time ``time[k]`` (s, one
clock). Frames: ``base`` = robot base, ``tool`` = probe TCP (z = probe axis into tissue).

Dataset fields (LeRobot v3.0, docs/INTERFACES.md):
- ``observation.bmode``   uint8 B-mode, stored losslessly as PNG (grey replicated to 3 channels,
                          because LeRobot 0.6.1's image writer rejects single-channel uint8).
- ``observation.state``   (12,) TCP pose in base [x, y, z (m), rx, ry, rz (rotation vector, rad)]
                          followed by 6 joint positions (rad).
- ``observation.wrench``  (6,) tool frame [Fx, Fy, Fz (N), Tx, Ty, Tz (N m)].
- ``action``              (6,) TCP displacement over one step, tool frame at step k:
                          [dx, dy, dz (m), rx, ry, rz (rad)] taking pose k to pose k+1 (last row 0).
                          The policy's 5-D target [x, y, rx, ry, rz] is a projection of the composed
                          chunk (usintent.data.chunks); keeping dz makes chunk composition exact.
- ``action.contact_cmd``  logged operator mode (switches at the key press).
- ``mode``                scheduler mode m_t (switches at the cue).
- ``observation.camera``  uint8 RGB probe-camera frame (only if the rig has a probe camera).
- ``station_index``, ``hold_id``, ``episode_time``, ``session_id``, ``day``, ``operator_id``,
  ``phantom_id`` (which phantom/target: e.g. vascular vs central-line neck phantom).
- ``gt.*``                synthetic ground truth (hold-world only; absent in real data).
Per-hold sidecar: ``meta/holds.parquet`` (columns in :data:`HOLD_COLUMNS`).
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

MODES = ("HOLD", "SWEEP", "LIFT", "RECONTACT")
EXIT_TYPES = ("SWEEP", "LIFT")
HOLD_COLUMNS = [
    "hold_id", "episode_index", "station_index", "start_time", "cue_time", "key_time", "D",
    "reaction_time", "exit_type", "backtrack", "sampled_D", "distribution", "block_id", "censored",
    "sweep_speed_mps",
]
STATE_DIM, ACTION_DIM, WRENCH_DIM = 12, 6, 6


@dataclass
class Episode:
    """One recorded (or synthetic) episode. See module docstring for units and frames."""

    episode_index: int
    fps: float
    time: NDArray[np.float64]            # (N,) s
    state: NDArray[np.float64]           # (N, 12)
    action: NDArray[np.float64]          # (N, 6) tool frame
    wrench: NDArray[np.float64]          # (N, 6) tool frame
    mode: NDArray[np.str_]               # (N,) scheduler mode
    contact_cmd: NDArray[np.str_]        # (N,) operator mode
    station_index: NDArray[np.int64]     # (N,) -1 between stations
    hold_id: NDArray[np.int64]           # (N,) -1 outside holds (hold = start .. key press)
    session_id: str
    day: str
    operator_id: str
    phantom_id: str                      # phantom / target id (analyses stratify by it)
    holds: pd.DataFrame                  # sidecar rows of this episode (HOLD_COLUMNS)
    bmode: NDArray[np.uint8] | None = None   # (N, rows, cols)
    camera: NDArray[np.uint8] | None = None  # (N, rows, cols, 3) if a probe camera exists
    extras: dict[str, NDArray[Any]] = field(default_factory=dict)  # "gt.*" synthetic truth

    def __len__(self) -> int:
        return int(self.time.shape[0])

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    def validate(self) -> None:
        """Shape, dtype and clock checks; raises ValueError."""
        n = len(self)
        shapes = {"state": (n, STATE_DIM), "action": (n, ACTION_DIM), "wrench": (n, WRENCH_DIM)}
        for name, shp in shapes.items():
            if getattr(self, name).shape != shp:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shp}")
        for name in ("mode", "contact_cmd", "station_index", "hold_id"):
            if getattr(self, name).shape != (n,):
                raise ValueError(f"{name} must have shape ({n},)")
        if self.bmode is not None and self.bmode.shape[0] != n:
            raise ValueError("bmode length differs from time")
        if self.camera is not None and (self.camera.shape[0] != n or self.camera.ndim != 4):
            raise ValueError("camera must be (N, rows, cols, 3)")
        if not self.phantom_id:
            raise ValueError("phantom_id must be set")
        if n > 1 and not np.all(np.diff(self.time) > 0):
            raise ValueError("timestamps are not strictly monotonic")
        bad = set(np.unique(self.mode)) | set(np.unique(self.contact_cmd))
        if not bad <= set(MODES):
            raise ValueError(f"unknown modes {bad - set(MODES)}")
        if not set(HOLD_COLUMNS) <= set(self.holds.columns):
            raise ValueError(f"holds table lacks {set(HOLD_COLUMNS) - set(self.holds.columns)}")


def holds_table(episodes: list[Episode]) -> pd.DataFrame:
    """Concatenate the per-hold sidecar rows of several episodes."""
    if not episodes:
        return pd.DataFrame(columns=HOLD_COLUMNS)
    return pd.concat([e.holds for e in episodes], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# LeRobot v3.0 writer / reader. API confirmed from the installed lerobot==0.6.1 source:
#   LeRobotDataset.create(repo_id, fps, features, root, robot_type, use_videos, ...)
#   ds.add_frame(frame: dict)   (frame must include "task")
#   ds.save_episode(); ds.finalize()
#   LeRobotDataset(repo_id, root=...) ; ds.hf_dataset (HF datasets.Dataset) ; ds.meta.fps
# ---------------------------------------------------------------------------------------------

_TASK = "cued-hold vascular scan"


def _features(ep: Episode) -> dict[str, dict[str, Any]]:
    feats: dict[str, dict[str, Any]] = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": None},
        "observation.wrench": {"dtype": "float32", "shape": (WRENCH_DIM,), "names": None},
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": None},
        "action.contact_cmd": {"dtype": "string", "shape": (1,), "names": None},
        "mode": {"dtype": "string", "shape": (1,), "names": None},
        "station_index": {"dtype": "int64", "shape": (1,), "names": None},
        "hold_id": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_time": {"dtype": "float64", "shape": (1,), "names": None},
        "session_id": {"dtype": "string", "shape": (1,), "names": None},
        "day": {"dtype": "string", "shape": (1,), "names": None},
        "operator_id": {"dtype": "string", "shape": (1,), "names": None},
        "phantom_id": {"dtype": "string", "shape": (1,), "names": None},
    }
    if ep.bmode is not None:
        r, c = ep.bmode.shape[1:]
        feats["observation.bmode"] = {"dtype": "image", "shape": (r, c, 3),
                                      "names": ["height", "width", "channel"]}
    if ep.camera is not None:
        feats["observation.camera"] = {"dtype": "image", "shape": tuple(ep.camera.shape[1:]),
                                       "names": ["height", "width", "channel"]}
    for k, v in ep.extras.items():
        feats[k] = {"dtype": "float64", "shape": (1,) if v.ndim == 1 else v.shape[1:], "names": None}
    return feats


def write_lerobot(episodes: list[Episode], root: Path, repo_id: str = "local/usintent",
                  overwrite: bool = False) -> Path:
    """Write episodes as a LeRobot v3.0 dataset at ``root`` plus ``meta/holds.parquet``.

    float64 arrays are stored as float32 except ``episode_time`` and ``gt.*`` (float64).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not episodes:
        raise ValueError("no episodes")
    fps = episodes[0].fps
    if any(e.fps != fps for e in episodes) or abs(fps - round(fps)) > 1e-9:
        raise ValueError("LeRobot needs one integer fps for all episodes")
    root = Path(root)
    if root.exists():
        if not overwrite:
            raise FileExistsError(root)
        shutil.rmtree(root)
    feats = _features(episodes[0])
    ds = LeRobotDataset.create(repo_id, fps=round(fps), features=feats, root=root,
                               robot_type="ur7e", use_videos=False)
    for ep in episodes:
        ep.validate()
        if _features(ep) != feats:
            raise ValueError("all episodes must share one feature set")
        for k in range(len(ep)):
            frame: dict[str, Any] = {
                "task": _TASK,
                "observation.state": ep.state[k].astype(np.float32),
                "observation.wrench": ep.wrench[k].astype(np.float32),
                "action": ep.action[k].astype(np.float32),
                "action.contact_cmd": str(ep.contact_cmd[k]),
                "mode": str(ep.mode[k]),
                "station_index": np.array([ep.station_index[k]], dtype=np.int64),
                "hold_id": np.array([ep.hold_id[k]], dtype=np.int64),
                "episode_time": np.array([ep.time[k]], dtype=np.float64),
                "session_id": ep.session_id,
                "day": ep.day,
                "operator_id": ep.operator_id,
                "phantom_id": ep.phantom_id,
            }
            if ep.camera is not None:
                frame["observation.camera"] = ep.camera[k]
            if ep.bmode is not None:
                frame["observation.bmode"] = np.repeat(ep.bmode[k][:, :, None], 3, axis=2)
            for name, v in ep.extras.items():
                frame[name] = np.atleast_1d(np.asarray(v[k], dtype=np.float64))
            ds.add_frame(frame)
        ds.save_episode()
    ds.finalize()
    holds = holds_table(episodes)
    holds.to_parquet(root / "meta" / "holds.parquet", index=False)
    (root / "meta" / "usintent.json").write_text(json.dumps(
        {"schema": "usintent-v1", "fps": fps, "n_episodes": len(episodes),
         "synthetic": any(k.startswith("gt.") for k in feats)}, indent=2))
    return root


def _col(table: pd.DataFrame, name: str) -> NDArray[Any]:
    vals = table[name].to_numpy()
    if vals.dtype == object and len(vals) and isinstance(vals[0], (list, np.ndarray)):
        return np.stack([np.asarray(v) for v in vals])
    return np.asarray(vals)


def read_lerobot(root: Path, repo_id: str = "local/usintent") -> list[Episode]:
    """Read a dataset written by :func:`write_lerobot` (or our converter) back into Episodes."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = Path(root)
    ds = LeRobotDataset(repo_id, root=root)
    hf = ds.hf_dataset
    images = ("observation.bmode", "observation.camera")
    keep = [c for c in hf.column_names if c not in images]
    table = hf.select_columns(keep).to_pandas()
    holds_path = root / "meta" / "holds.parquet"
    holds = pd.read_parquet(holds_path) if holds_path.is_file() else pd.DataFrame(columns=HOLD_COLUMNS)
    has_bmode = "observation.bmode" in hf.column_names
    episodes = []
    for ep_idx, rows in table.groupby("episode_index", sort=True):
        idx = rows.index.to_numpy()
        bmode = camera = None
        if "observation.camera" in hf.column_names:
            camera = np.asarray(hf.select(idx.tolist()).with_format("numpy")["observation.camera"]
                                ).astype(np.uint8)
        if has_bmode:
            imgs = hf.select(idx.tolist()).with_format("numpy")["observation.bmode"]
            arr = np.asarray(imgs)
            # HF image decode gives (N, rows, cols, 3) uint8; grey is replicated across channels
            bmode = arr[..., 0].astype(np.uint8) if arr.ndim == 4 else arr.astype(np.uint8)

        def scalar(name: str, rows: pd.DataFrame = rows) -> NDArray[Any]:
            return np.asarray(_col(rows, name)).reshape(len(rows), -1)[:, 0]

        extras = {c: np.asarray(_col(rows, c), dtype=np.float64).reshape(len(rows), -1).squeeze(-1)
                  if np.asarray(_col(rows, c)).reshape(len(rows), -1).shape[1] == 1
                  else np.asarray(_col(rows, c), dtype=np.float64)
                  for c in rows.columns if c.startswith("gt.")}
        episodes.append(Episode(
            episode_index=int(ep_idx),
            fps=float(ds.meta.fps),
            time=scalar("episode_time").astype(np.float64),
            state=np.asarray(_col(rows, "observation.state"), dtype=np.float64),
            action=np.asarray(_col(rows, "action"), dtype=np.float64),
            wrench=np.asarray(_col(rows, "observation.wrench"), dtype=np.float64),
            mode=scalar("mode").astype(str),
            contact_cmd=scalar("action.contact_cmd").astype(str),
            station_index=scalar("station_index").astype(np.int64),
            hold_id=scalar("hold_id").astype(np.int64),
            session_id=str(scalar("session_id")[0]),
            day=str(scalar("day")[0]),
            operator_id=str(scalar("operator_id")[0]),
            phantom_id=str(scalar("phantom_id")[0]),
            holds=holds[holds["episode_index"] == ep_idx].reset_index(drop=True),
            bmode=bmode, camera=camera,
            extras=extras,
        ))
    return episodes


# ---------------------------------------------------------------------------------------------
# Dataset inventory and Open-H mapping
# ---------------------------------------------------------------------------------------------


def dataset_info(root: Path) -> dict[str, Any]:
    """LeRobot v3.0 ``meta/info.json`` summary: codebase_version, fps, totals, feature names."""
    info = json.loads((Path(root) / "meta" / "info.json").read_text())
    return {
        "codebase_version": info.get("codebase_version"),
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "total_frames": info.get("total_frames"),
        "robot_type": info.get("robot_type"),
        "features": sorted(info.get("features", {})),
    }


def load_dataset(root: Path, repo_id: str = "local/usintent") -> list[Episode]:
    """Load one of our datasets (synthetic, cued, or converted)."""
    return read_lerobot(Path(root), repo_id)


def _rot_to_rotvec(rot: NDArray[np.float64], fmt: str) -> NDArray[np.float64]:
    from scipy.spatial.transform import Rotation

    if fmt == "rotvec":
        return rot
    if fmt == "quat_xyzw":
        return np.asarray(Rotation.from_quat(rot).as_rotvec(), dtype=np.float64)
    if fmt == "quat_wxyz":
        return np.asarray(Rotation.from_quat(rot[:, [1, 2, 3, 0]]).as_rotvec(), dtype=np.float64)
    if fmt.startswith("euler_"):
        return np.asarray(Rotation.from_euler(fmt[6:], rot).as_rotvec(), dtype=np.float64)
    raise ValueError(f"unknown rotation format {fmt!r}")


def load_openh(root: Path, mapping: dict[str, Any], repo_id: str = "openh/local") -> list[Episode]:
    """Map an Open-H LeRobot dataset onto :class:`Episode`.

    ``mapping`` is ``configs/data.yaml: openh`` (every key must be filled after inspecting the
    downloaded dataset's ``meta/info.json``; nulls raise MissingRigFact):
      pose_key, pos_slice [i0, i1], rot_slice [i0, i1], rot_format (rotvec | quat_xyzw |
      quat_wxyz | euler_<axes>), bmode_key, wrench_key (or "none"), session_key (or "episode"),
      day_key (or "none").
    Per-step actions are derived from the mapped TCP poses (same rule as our recorder converter),
    so the passthrough check exercises our decode path on real kinematics. Modes are unknown
    (Open-H has no cue/key logs): mode = contact_cmd = "SWEEP", hold_id = -1; use
    threshold segmentation.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from usintent.config import require
    from usintent.data.chunks import actions_from_poses

    m = {k: require(mapping, k) for k in ("pose_key", "pos_slice", "rot_slice", "rot_format",
                                          "bmode_key", "wrench_key", "session_key", "day_key",
                                          "phantom_id")}
    ds = LeRobotDataset(repo_id, root=Path(root))
    hf = ds.hf_dataset
    cols = [c for c in hf.column_names if c != m["bmode_key"]]
    table = hf.select_columns(cols).to_pandas()
    episodes = []
    for ep_idx, rows in table.groupby("episode_index", sort=True):
        raw = np.asarray(_col(rows, m["pose_key"]), dtype=np.float64)
        pose = np.zeros((len(rows), 6))
        pose[:, :3] = raw[:, slice(*m["pos_slice"])]
        pose[:, 3:] = _rot_to_rotvec(raw[:, slice(*m["rot_slice"])], m["rot_format"])
        state = np.zeros((len(rows), STATE_DIM))
        state[:, :6] = pose
        wrench = (np.zeros((len(rows), 6)) if m["wrench_key"] == "none"
                  else np.asarray(_col(rows, m["wrench_key"]), dtype=np.float64))
        imgs = np.asarray(hf.select(rows.index.tolist()).with_format("numpy")[m["bmode_key"]])
        bmode = imgs.mean(axis=-1).round().astype(np.uint8) if imgs.ndim == 4 else imgs
        session = (f"openh-ep{ep_idx}" if m["session_key"] == "episode"
                   else str(np.asarray(_col(rows, m["session_key"])).ravel()[0]))
        day = ("unknown" if m["day_key"] == "none"
               else str(np.asarray(_col(rows, m["day_key"])).ravel()[0]))
        t = rows["timestamp"].to_numpy(dtype=np.float64)
        n = len(rows)
        episodes.append(Episode(
            episode_index=int(ep_idx), fps=float(ds.meta.fps), time=t, state=state,
            action=actions_from_poses(pose), wrench=wrench,
            mode=np.full(n, "SWEEP"), contact_cmd=np.full(n, "SWEEP"),
            station_index=np.full(n, -1, dtype=np.int64), hold_id=np.full(n, -1, dtype=np.int64),
            session_id=session, day=day, operator_id="openh", phantom_id=str(m["phantom_id"]),
            holds=pd.DataFrame(columns=HOLD_COLUMNS), bmode=bmode,
        ))
    return episodes
