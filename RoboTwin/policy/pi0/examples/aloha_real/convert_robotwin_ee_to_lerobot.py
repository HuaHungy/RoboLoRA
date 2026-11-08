"""Convert native RoboTwin HDF5 directly to 16D end-effector LeRobot data.

Use the version of LeRobot pinned by this pi0 checkout. See README_robotwin_ee.md.
"""

import json
from pathlib import Path
import re
from typing import Literal

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tqdm
import tyro

from openpi.shared.robotwin_ee import CAMERA_MAP
from openpi.shared.robotwin_ee import EE_NAMES
from openpi.shared.robotwin_ee import pack_state


def read_ee_states(ep: h5py.File) -> np.ndarray:
    keys = [f"endpose/{arm}_{field}" for arm in ("left", "right") for field in ("endpose", "gripper")]
    missing = [key for key in keys if key not in ep]
    if missing:
        raise ValueError(f"{ep.filename}: missing {missing}. Use native RoboTwin data, not joint-only processed HDF5.")
    state = pack_state(*(ep[key][:] for key in keys))
    if len(state) < 2:
        raise ValueError(f"{ep.filename}: at least two EE frames are required")
    return state


def read_image(ep, native_camera, frame_index):
    value = ep[f"observation/{native_camera}/rgb"][frame_index]
    if isinstance(value, np.ndarray) and value.ndim == 3:
        image = value
    else:
        encoded = (
            np.frombuffer(value, np.uint8) if isinstance(value, (bytes, np.bytes_)) else np.asarray(value, np.uint8)
        )
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{ep.filename}: invalid RGB frame {native_camera}/{frame_index}")
    # Native pkl2hdf5.py feeds RGB directly to cv2.imencode. imdecode returns the
    # original RGB array here; adding BGR->RGB would swap channels a second time.
    return image


def episode_instructions(path: Path, fallback: str | None, split: str):
    candidates = [path.parent.parent / "instructions" / f"{path.stem}.json", path.parent / "instructions.json"]
    for candidate in candidates:
        if candidate.is_file():
            content = json.loads(candidate.read_text(encoding="utf-8"))
            instructions = content.get(split, content.get("instructions"))
            if (
                not isinstance(instructions, list)
                or not instructions
                or not all(isinstance(s, str) and s.strip() for s in instructions)
            ):
                raise ValueError(f"{candidate}: expected nonempty instruction list for {split!r}")
            return instructions
    if fallback and fallback.strip():
        return [fallback]
    raise ValueError(f"{path}: no instructions found; supply --task with a meaningful instruction")


def convert(
    raw_dir: Path,
    repo_id: str,
    *,
    root: Path | None = None,
    episodes: list[int] | None = None,
    max_episodes: int | None = None,
    fps: float = 250 / 15,
    mode: Literal["image", "video"] = "image",
    task: str | None = None,
    instruction_split: str = "seen",
    seed: int = 42,
    image_writer_threads: int = 4,
    push_to_hub: bool = False,
):
    """Convert episodes; fps defines nominal frame-index time, not resampling.

    episodes: Zero-based positions in the naturally sorted source file list.
    root: New dataset directory, default HF_LEROBOT_HOME/repo_id. Never overwritten.
    """
    raw_dir = raw_dir.expanduser().resolve()
    if not raw_dir.is_dir():
        raise ValueError(f"Missing raw directory: {raw_dir}")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if image_writer_threads < 0:
        raise ValueError("image_writer_threads must be nonnegative")
    if mode not in {"image", "video"}:
        raise ValueError("mode must be image or video")
    paths = sorted(
        raw_dir.rglob("*.hdf5"), key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(p))]
    )
    if episodes is not None:
        if not episodes or len(set(episodes)) != len(episodes) or any(i < 0 or i >= len(paths) for i in episodes):
            raise ValueError("episodes must contain unique valid indices into the naturally sorted file list")
        paths = [paths[i] for i in episodes]
    if max_episodes is not None:
        if max_episodes < 1:
            raise ValueError("max_episodes must be positive")
        paths = paths[:max_episodes]
    if not paths:
        raise ValueError(f"No HDF5 episodes found under {raw_dir}")
    destination = (root if root is not None else HF_LEROBOT_HOME / repo_id).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Dataset already exists; choose a new --root or --repo-id: {destination}")

    # Validate every numeric trajectory, instruction source and camera length
    # before writing any dataset. Decode only one frame per camera in this pass.
    sources, image_shapes = [], None
    rng = np.random.default_rng(seed)
    for path in paths:
        with h5py.File(path, "r") as ep:
            state = read_ee_states(ep)
            shapes = {}
            for camera, native in CAMERA_MAP.items():
                key = f"observation/{native}/rgb"
                if key not in ep or len(ep[key]) != len(state):
                    raise ValueError(f"{path}: camera {native} length must equal EE state length")
                shapes[camera] = read_image(ep, native, 0).shape
            if image_shapes is not None and shapes != image_shapes:
                raise ValueError("Camera sizes differ across episodes; align source image sizes first")
            image_shapes = shapes
            instructions = episode_instructions(path, task, instruction_split)
            sources.append(
                {
                    "episode_index": len(sources),
                    "source_path": str(path),
                    "source_frames": len(state),
                    "frames": len(state) - 1,
                    "instruction": str(rng.choice(instructions)),
                }
            )
    features = {
        "observation.state": {"dtype": "float32", "shape": (16,), "names": [EE_NAMES]},
        "action": {"dtype": "float32", "shape": (16,), "names": [EE_NAMES]},
    }
    for camera, (height, width, channels) in image_shapes.items():
        features[f"observation.images.{camera}"] = {
            "dtype": mode,
            "shape": (channels, height, width),
            "names": ["channels", "height", "width"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=destination,
        fps=fps,
        robot_type="robotwin_dual_arm_ee",
        features=features,
        use_videos=mode == "video",
        image_writer_processes=0,
        image_writer_threads=image_writer_threads,
    )
    manifest = {
        "status": "in_progress",
        "representation": "absolute_world_ee_pose",
        "state_action_names": EE_NAMES,
        "quaternion_order": "wxyz",
        "quaternion_sign": "dominant-component-positive at episode start, then temporal continuity",
        "reference_point": "RoboTwin get_left/right_ee_pose, NOT get_left/right_tcp_pose",
        "state": "EE(t), gripper(t)",
        "action": "EE(t+1), gripper(t+1)",
        "action_source": "next recorded demonstration state, not low-level controller command",
        "frame_alignment": "source images[t], state[t], action[t+1]; final source frame is target only",
        "fps": fps,
        "timestamp_policy": "nominal frame_index/fps; no physical-time resampling",
        "timestamp_note": "Native HDF5 lacks per-frame time; segment boundaries may have extra samples.",
        "image_channels": "RGB; native RoboTwin cv2 encode/decode round trip without channel swap",
        "seed": seed,
        "sources": sources,
    }
    manifest_path = dataset.root / "meta/robotwin_ee_conversion.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        for source in tqdm.tqdm(sources, desc="RoboTwin EE -> LeRobot"):
            with h5py.File(source["source_path"], "r") as ep:
                state = read_ee_states(ep)
                for t in range(len(state) - 1):
                    frame = {"observation.state": state[t], "action": state[t + 1], "task": source["instruction"]}
                    for camera, native in CAMERA_MAP.items():
                        frame[f"observation.images.{camera}"] = read_image(ep, native, t)
                    dataset.add_frame(frame)
            dataset.save_episode()
        manifest["status"] = "complete"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        dataset.stop_image_writer()
    if push_to_hub:
        dataset.push_to_hub()
    print(f"Saved {len(sources)} episodes / {dataset.meta.total_frames} frames to {dataset.root}")
    return dataset


if __name__ == "__main__":
    tyro.cli(convert)
