"""Full numeric/metadata audit, plus sampled image checks against native HDF5."""

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from .config import write_json
from .lerobot import dataset_metadata, read_columns


CAMERAS = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}


def check_native(full, raw):
    for arm, offset in (("left", 0), ("right", 8)):
        pose = raw[f"endpose/{arm}_endpose"][:]
        grip = raw[f"endpose/{arm}_gripper"][:].reshape(-1)
        if pose.shape != (len(full), 7):
            raise ValueError("Native state length differs from reconstructed T+1 states")
        if not np.allclose(full[:, offset:offset + 3], pose[:, :3], atol=1e-6, rtol=0):
            raise ValueError("Position differs from native source")
        if not np.allclose(full[:, offset + 7], grip, atol=1e-6, rtol=0):
            raise ValueError("Gripper differs from native source")
        q = pose[:, 3:] / np.linalg.norm(pose[:, 3:], axis=1, keepdims=True)
        dots = np.sum(q * full[:, offset + 3:offset + 7], axis=1)
        if not np.allclose(np.abs(dots), 1, atol=1e-5):
            raise ValueError("Quaternion orientation differs from native source")


def check_images(path, raw, length, info):
    keys = ["observation.images." + name for name in CAMERAS]
    table = pq.ParquetFile(path).read(columns=keys)
    checked = 0
    for index in sorted({0, length // 2, length - 1}):
        for camera, native in CAMERAS.items():
            key = "observation.images." + camera
            cell = table[key][index].as_py()
            with Image.open(io.BytesIO(cell["bytes"])) as im:
                image = np.asarray(im.convert("RGB"))
            encoded = raw[f"observation/{native}/rgb"][index]
            original = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            if not np.array_equal(image, original):
                raise ValueError(f"Image content/channel mismatch: {key}, frame {index}")
            if [image.shape[2], *image.shape[:2]] != info["features"][key]["shape"]:
                raise ValueError("Image shape differs from feature metadata")
            checked += 1
    return checked


def audit(root, raw_root, report_path):
    root, raw_root = Path(root).resolve(), Path(raw_root).resolve()
    report = {"status": "running", "root": str(root), "tasks": [], "errors": [],
              "image_scope": "first episode of each task, first/middle/last frames, all 3 cameras",
              "numeric_scope": "all rows in all episodes, compared to original HDF5",
              "warnings": ["Timestamps are nominal frame-index time, not exact simulation time.",
                           "No object state or extracted visual embedding is present in numeric features."]}
    hashes = defaultdict(list)
    expected_tasks = sorted(p.parent.parent.name for p in raw_root.glob("*/demo_aloha/data")
                            if any(p.glob("*.hdf5")))
    for task in expected_tasks:
        folder = root / (task + "_pi0_ee")
        row = {"task": task, "root": str(folder), "episodes": 0, "frames": 0,
               "windows_50_stride25": 0, "sampled_images": 0, "bytes": 0, "errors": []}
        try:
            info, manifest = dataset_metadata(str(folder))
            sources = manifest["sources"]
            paths = sorted(folder.glob("data/chunk-*/*.parquet"))
            native_paths = {str(p.resolve()) for p in (raw_root / task / "demo_aloha/data").glob("*.hdf5")}
            if native_paths != {str(Path(s["source_path"]).resolve()) for s in sources}:
                raise ValueError("Manifest does not contain exactly the current native episodes")
            if not (len(paths) == len(sources) == info["total_episodes"]):
                raise ValueError("Episode counts differ across manifest, files and info")
            for feature in ("observation.state", "action"):
                if info["features"][feature]["shape"] != [16]:
                    raise ValueError("Expected 16D EE features")
            tasks = {int(t["task_index"]): t["task"] for t in
                     map(json.loads, (folder / "meta/tasks.jsonl").read_text().splitlines())}
            episodes = list(map(json.loads, (folder / "meta/episodes.jsonl").read_text().splitlines()))
            if len(episodes) != len(sources):
                raise ValueError("episodes.jsonl length mismatch")
            for e, path in enumerate(paths):
                try:
                    source = sources[e]
                    state, action, table = read_columns(path, ("timestamp", "frame_index", "episode_index", "task_index"))
                    n = len(state)
                    if n != source["frames"] or n != episodes[e]["length"] or source["source_frames"] != n + 1:
                        raise ValueError("Episode frame counts differ")
                    if not np.array_equal(table["frame_index"].to_numpy(), np.arange(n)):
                        raise ValueError("Frame indices are not contiguous")
                    if not np.all(table["episode_index"].to_numpy() == e):
                        raise ValueError("Episode index mismatch")
                    if not np.allclose(table["timestamp"].to_numpy(), np.arange(n) / info["fps"], atol=1e-4, rtol=0):
                        raise ValueError("Nominal timestamps do not match fps")
                    if any(tasks[int(i)] != source["instruction"] for i in np.unique(table["task_index"].to_numpy())):
                        raise ValueError("Task instruction mapping mismatch")
                    full = np.concatenate([state, action[-1:]], axis=0)
                    for offset in (3, 11):
                        if np.any(np.sum(full[:-1, offset:offset + 4] * full[1:, offset:offset + 4], axis=1) < -1e-6):
                            raise ValueError("Quaternion sign discontinuity")
                    with h5py.File(source["source_path"], "r") as raw:
                        check_native(full, raw)
                        for native in CAMERAS.values():
                            if len(raw[f"observation/{native}/rgb"]) != n + 1:
                                raise ValueError("Native camera/pose lengths differ")
                        if e == 0:
                            row["sampled_images"] += check_images(path, raw, n, info)
                    digest = hashlib.sha256(full.tobytes()).hexdigest()
                    hashes[digest].append({"task": task, "episode": e, "path": str(path)})
                    row["episodes"] += 1
                    row["frames"] += n
                    row["windows_50_stride25"] += max(0, (n - 50) // 25 + 1)
                    row["bytes"] += path.stat().st_size
                except Exception as exc:
                    row["errors"].append({"file": str(path), "error": str(exc)})
            if row["frames"] != info["total_frames"]:
                row["errors"].append({"error": "Valid frame total differs from info.json"})
        except Exception as exc:
            row["errors"].append({"error": str(exc)})
        report["tasks"].append(row)
        report["errors"].extend({"task": task, **error} for error in row["errors"])
        print(f"{task}: {row['episodes']} episodes, {row['frames']} frames, errors={len(row['errors'])}", flush=True)
    report["duplicate_motion_groups"] = [group for group in hashes.values() if len(group) > 1]
    report["summary"] = {key: sum(row[key] for row in report["tasks"]) for key in
                         ("episodes", "frames", "windows_50_stride25", "sampled_images", "bytes")}
    report["summary"]["tasks"] = len(report["tasks"])
    report["status"] = "passed" if report["tasks"] and not report["errors"] else "failed"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], **report["summary"],
                      "duplicate_motion_groups": len(report["duplicate_motion_groups"])}, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/lerobot"))
    parser.add_argument("--raw-root", type=Path, default=Path("../RoboTwin/data"))
    parser.add_argument("--report", type=Path, default=Path("reports/lerobot_ee_audit.json"))
    args = parser.parse_args()
    result = audit(args.root, args.raw_root, args.report)
    raise SystemExit(0 if result["status"] == "passed" else 1)
