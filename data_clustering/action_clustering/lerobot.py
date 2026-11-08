"""Read numeric EE columns without decoding or copying the embedded images."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def read_columns(path, extra=()):
    table = pq.ParquetFile(path).read(columns=["observation.state", "action", *extra])
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 16 or action.shape != state.shape or len(state) == 0:
        raise ValueError(f"{path}: expected nonempty [T,16] state/action")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError(f"{path}: non-finite EE values")
    if not np.allclose(action[:-1], state[1:], atol=1e-6, rtol=0):
        raise ValueError(f"{path}: actions do not match the next recorded state")
    for values in (state, action):
        for start in (3, 11):
            if not np.allclose(np.linalg.norm(values[:, start:start + 4], axis=1), 1, atol=1e-5):
                raise ValueError(f"{path}: quaternion must be unit-length wxyz")
        if np.any(values[:, [7, 15]] < 0) or np.any(values[:, [7, 15]] > 1):
            raise ValueError(f"{path}: grippers must be in [0,1]")
    return state, action, table


@lru_cache(maxsize=128)
def dataset_metadata(root):
    root = Path(root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "meta/robotwin_ee_conversion.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("quaternion_order") != "wxyz":
        raise ValueError(f"{root}: expected a complete RoboTwin EE conversion with wxyz quaternions")
    if manifest.get("representation") != "absolute_world_ee_pose":
        raise ValueError(f"{root}: unsupported pose convention")
    return info, manifest


def episode_metadata(path, source):
    _, manifest = dataset_metadata(source["root"])
    episode_index = int(Path(path).stem.split("_")[-1])
    item = manifest["sources"][episode_index]
    if item["episode_index"] != episode_index:
        raise ValueError("Manifest episode order mismatch")
    return item


def load_episode(path, source):
    info, _ = dataset_metadata(source["root"])
    if not np.isclose(info["fps"], source["fps"], atol=1e-9, rtol=0):
        raise ValueError("Configured nominal fps differs from LeRobot metadata")
    state, action, _ = read_columns(path)
    if len(state) != episode_metadata(path, source)["frames"]:
        raise ValueError("Parquet length differs from conversion manifest")
    # The last action is a recorded next state, so recover T+1 measured states.
    # This is valid specifically for this converter, not for arbitrary policies.
    full = np.concatenate([state, action[-1:]], axis=0)
    left = full[:, [0, 1, 2, 4, 5, 6, 3]]
    right = full[:, [8, 9, 10, 12, 13, 14, 11]]
    return {"actions": action, "gripper": full[:, [7, 15]],
            "ee_pose": np.stack([left, right], axis=1)}
