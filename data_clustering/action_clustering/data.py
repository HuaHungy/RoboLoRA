"""Episode adapters and bounded-memory, episode-separated window preparation."""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .config import fresh_dir, write_json
from .progress import Progress


def finite(name, array):
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def load_episode(path, source):
    """NPZ: T actions, T+1 measured states. HDF5: never invent a terminal state."""
    if source["format"] == "npz":
        with np.load(path, allow_pickle=False) as f:
            data = {k: finite(k, f[k]) for k in
                    ("actions", "qpos", "gripper", "ee_pose", "object_features") if k in f}
        if "actions" not in data or "gripper" not in data:
            raise ValueError(f"{path}: NPZ requires actions and gripper")
        count = len(data["actions"])
        for key, value in data.items():
            if key != "actions" and len(value) != count + 1:
                raise ValueError(f"{path}: {key} must have T+1 measured states")
    elif source["format"] == "lerobot_ee":
        from .lerobot import load_episode as load_lerobot_episode

        data = load_lerobot_episode(path, source)
    elif source["format"] == "aloha_hdf5":
        with h5py.File(path, "r") as f:
            qpos = finite("qpos", f["observations/qpos"][:])
            actions = finite("actions", f["action"][:])
        if len(qpos) not in {len(actions), len(actions) + 1}:
            raise ValueError(f"{path}: expected T or T+1 qpos states")
        count = min(len(actions), len(qpos) - 1)
        indices = source["gripper_indices"]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= qpos.shape[1] for i in indices):
            raise ValueError("Invalid gripper_indices")
        data = {"actions": actions[:count], "qpos": qpos[:count + 1],
                "gripper": qpos[:count + 1, indices]}
    else:
        raise ValueError(f"Unknown format: {source['format']}")
    if data["actions"].ndim != 2 or data["actions"].shape[1] == 0:
        raise ValueError("actions must be [T, action_dim]")
    if data["gripper"].ndim != 2 or data["gripper"].shape[1] == 0:
        raise ValueError("gripper must be [T+1, number_of_arms]")
    for key in ("qpos", "object_features"):
        if key in data and data[key].ndim != 2:
            raise ValueError(f"{key} must be a matrix")
    return data


def window_features(ep, start, length, mode, source):
    end = start + length
    grip = ep["gripper"][start:end + 1]
    if mode == "joint":
        if "qpos" not in ep or source.get("action_semantics") != "absolute_joint_target":
            raise ValueError("joint mode needs qpos and action_semantics=absolute_joint_target")
        q = ep["qpos"][start:end + 1]
        a = ep["actions"][start:end]
        if a.shape[1] != q.shape[1]:
            raise ValueError("Joint action/state dimensions differ")
        indices = source["gripper_indices"]
        if len(indices) != grip.shape[1] or len(set(indices)) != len(indices) or any(i < 0 or i >= q.shape[1] for i in indices):
            raise ValueError("gripper_indices must match gripper channels")
        arms = np.ones(q.shape[1], dtype=bool)
        arms[indices] = False
        # Absolute gripper values remain meaningful; arm motion is relative.
        dynamics = np.concatenate([np.diff(q[:, arms], axis=0), a[:, arms] - q[:-1, arms],
                                   a[:, indices], grip[:-1], np.diff(grip, axis=0)], axis=-1)
        state = np.concatenate([q[-1, arms] - q[0, arms], grip[-1] - grip[0]])
    else:
        if "ee_pose" not in ep:
            raise ValueError("ee mode requires measured ee_pose; joint angles are not a substitute")
        pose = ep["ee_pose"][start:end + 1]
        if pose.shape != (length + 1, grip.shape[1], 7):
            raise ValueError("ee_pose must be [T+1, arms, 7], xyz + quaternion xyzw")
        relative = []
        for arm in range(pose.shape[1]):
            rotation = Rotation.from_quat(pose[:, arm, 3:])
            origin = rotation[0].inv()
            position = origin.apply(pose[:, arm, :3] - pose[0, arm, :3])
            orientation = (origin * rotation).as_rotvec()
            relative.append(np.concatenate([position, orientation], axis=-1))
        relative = np.concatenate(relative, axis=-1).astype(np.float32)
        dynamics = np.concatenate([relative[1:], grip[:-1], np.diff(grip, axis=0)], axis=-1)
        state = np.concatenate([relative[-1], grip[-1] - grip[0]])
    if "object_features" in ep:
        state = np.concatenate([state, ep["object_features"][end] - ep["object_features"][start]])
    return finite("dynamics", dynamics), finite("state_delta", state)


def discover(cfg):
    episodes, seen = [], set()
    ids = [s["id"] for s in cfg["datasets"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Each dataset id must be unique")
    fps = [float(s["fps"]) for s in cfg["datasets"]]
    if not all(np.isfinite(x) and x > 0 for x in fps) or len(set(fps)) != 1:
        raise ValueError("Use one known positive sampling rate; resample mixed-rate sources first")
    if cfg["representation"]["mode"] == "joint":
        if len({s["robot_id"] for s in cfg["datasets"]}) != 1:
            raise ValueError("joint mode forbids mixing robot embodiments; use common ee features")
        if len({tuple(s["gripper_indices"]) for s in cfg["datasets"]}) != 1:
            raise ValueError("joint mode requires identical joint ordering and gripper indices")
    for source in cfg["datasets"]:
        paths = sorted(Path(source["root"]).glob(source["glob"]))
        if not paths:
            raise ValueError(f"No input files for dataset {source['id']}")
        for path in paths:
            if path.resolve() in seen:
                raise ValueError(f"Duplicate input episode: {path}")
            seen.add(path.resolve())
            relative = path.relative_to(source["root"])
            episode_id = source["id"] + "/" + relative.as_posix()
            episodes.append((episode_id, path, source, relative))
    return episodes


def load_visual(source, relative, required_steps, enabled):
    if not enabled:
        return None
    if not source.get("visual_cache"):
        raise ValueError("visual_weight > 0 requires visual_cache for EVERY dataset")
    path = Path(source["visual_cache"]) / relative.with_suffix(".npy")
    visual = np.load(path, mmap_mode="r", allow_pickle=False)
    if visual.ndim != 2 or len(visual) < required_steps or visual.shape[1] == 0:
        raise ValueError(f"{path}: expected per-frame [T, visual_dim] features")
    return visual


def prepare(cfg):
    episodes = discover(cfg)
    length, stride = cfg["window"]["length"], cfg["window"]["stride"]
    enabled = cfg["representation"]["visual_weight"] > 0
    eligible, skipped, shape, total = [], [], None, 0
    # Two passes avoid holding every episode/window/image in memory.
    scan_progress = Progress("准备 1/2：扫描轨迹", len(episodes), "条")
    for scanned, (episode_id, path, source, relative) in enumerate(episodes, 1):
        ep = load_episode(path, source)
        scan_progress.update(scanned)
        n = len(ep["actions"])
        count = max(0, (n - length) // stride + 1)
        if count == 0:
            skipped.append({"episode_id": episode_id, "reason": "shorter_than_window", "steps": n})
            continue
        dynamics, state = window_features(ep, 0, length, cfg["representation"]["mode"], source)
        visual = load_visual(source, relative, n, enabled)
        dims = (dynamics.shape[1], state.size, visual.shape[1] if visual is not None else 0)
        if shape is not None and dims != shape:
            raise ValueError("Feature schemas differ: align arm roles, object slots and visual dimensions first")
        shape = dims
        eligible.append((episode_id, path, source, relative, n, count))
        total += count
    if len(eligible) < 2:
        raise ValueError("At least two eligible episodes are needed for a leakage-free validation split")
    rng = np.random.default_rng(cfg["seed"])
    groups = {}
    for idx, entry in enumerate(eligible):
        key = entry[2]["id"] if cfg["training"]["stratify_by_dataset"] else "all"
        groups.setdefault(key, []).append(idx)
    validation = set()
    for key, group in groups.items():
        if len(group) < 2:
            raise ValueError(f"Split group {key} needs at least two eligible episodes")
        order = rng.permutation(group)
        nval = min(len(group) - 1, max(1, round(len(group) * cfg["training"]["validation_fraction"])))
        validation.update(order[:nval].tolist())
    out = fresh_dir(Path(cfg["output_dir"]) / "windows")
    (out / "episodes").mkdir()
    arrays = {
        "action": np.lib.format.open_memmap(out / "action.npy", mode="w+", dtype="float32", shape=(total, length, shape[0])),
        "state": np.lib.format.open_memmap(out / "state.npy", mode="w+", dtype="float32", shape=(total, shape[1])),
    }
    if shape[2]:
        arrays["visual"] = np.lib.format.open_memmap(out / "visual.npy", mode="w+", dtype="float32", shape=(total, shape[2]))
    rows, idx = [], 0
    window_progress = Progress("准备 2/2：生成窗口", total, "窗口")
    for e, (episode_id, path, source, relative, n, count) in enumerate(eligible):
        ep = load_episode(path, source)
        visual = load_visual(source, relative, n, enabled)
        cache_name = hashlib.sha256(episode_id.encode()).hexdigest()[:20] + ".npz"
        np.savez_compressed(out / "episodes" / cache_name, **ep)
        instructions = source.get("instruction", "")
        provenance = {}
        if source["format"] == "lerobot_ee":
            from .lerobot import episode_metadata

            item = episode_metadata(path, source)
            instructions = item["instruction"]
            provenance = {"native_source_path": item["source_path"],
                          "lerobot_episode_index": item["episode_index"],
                          "timestamp_policy": "nominal_frame_index_not_physical_time"}
        sidecar = path.parent / "instructions.json"
        if sidecar.exists():
            instructions = json.loads(sidecar.read_text(encoding="utf-8"))
        for start in range(0, n - length + 1, stride):
            action, state = window_features(ep, start, length, cfg["representation"]["mode"], source)
            arrays["action"][idx], arrays["state"][idx] = action, state
            if visual is not None:
                arrays["visual"][idx] = finite("visual", visual[start])
            rows.append({**provenance, "window_id": idx, "episode_id": episode_id, "source_path": str(path.resolve()),
                         "episode_cache": "episodes/" + cache_name, "robot_id": source["robot_id"],
                         "dataset_id": source["id"], "instruction": instructions,
                         "start": start, "end": start + length, "fps": source["fps"],
                         "start_seconds": start / source["fps"], "end_seconds": (start + length) / source["fps"],
                         "image_key": source.get("image_key"),
                         "split": "validation" if e in validation else "train"})
            idx += 1
        window_progress.update(idx)
    for array in arrays.values():
        array.flush()
    index = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (out / "index.jsonl").write_text(index, encoding="utf-8")
    meta = {"window_count": total, "episode_count": len(eligible), "skipped_episodes": skipped,
            "action_dim": shape[0], "state_dim": shape[1], "visual_dim": shape[2],
            "window_length": length, "index_sha256": hashlib.sha256(index.encode()).hexdigest(),
            "config": cfg, "tail_policy": "drop; no invented endpoint or padding"}
    write_json(out / "metadata.json", meta)
    print(f"Prepared {total} windows from {len(eligible)} episodes -> {out}")
    return meta


def read_index(directory):
    return [json.loads(line) for line in (Path(directory) / "index.jsonl").read_text(encoding="utf-8").splitlines()]
