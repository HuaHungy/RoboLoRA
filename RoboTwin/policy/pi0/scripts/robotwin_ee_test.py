"""Regression tests for native EE conversion and pi0 action semantics."""

import importlib.util
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from openpi.policies.robotwin_ee_policy import RoboTwinEEInputs
from openpi.policies.robotwin_ee_policy import RoboTwinEEOutputs
from openpi.shared.robotwin_ee import CAMERA_MAP
from openpi.shared.robotwin_ee import normalize_pose_sequence
from openpi.training.lerobot_compat import LeRobotDataset

_PATH = Path(__file__).resolve().parents[1] / "examples/aloha_real/convert_robotwin_ee_to_lerobot.py"
_SPEC = importlib.util.spec_from_file_location("convert_robotwin_ee", _PATH)
converter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(converter)


def native_episode(folder, name="episode0", offset=0):
    (folder / "data").mkdir(parents=True, exist_ok=True)
    (folder / "instructions").mkdir(exist_ok=True)
    path = folder / "data" / f"{name}.hdf5"
    pose = np.zeros((6, 7), dtype=np.float64)
    pose[:, 0] = np.arange(6) / 10 + offset
    pose[:, 3] = 1
    pose[1::2, 3] = -2  # same orientation with alternating sign and norm
    image = np.full((16, 20, 3), [240, 10, 20], dtype=np.uint8)
    success, jpeg = cv2.imencode(".jpg", image)
    assert success
    with h5py.File(path, "w") as f:
        for arm in ("left", "right"):
            f[f"endpose/{arm}_endpose"] = pose
            f[f"endpose/{arm}_gripper"] = np.linspace(0, 1, len(pose))
        for native in CAMERA_MAP.values():
            f[f"observation/{native}/rgb"] = np.array([jpeg.tobytes()] * len(pose), dtype=f"S{len(jpeg)}")
    (folder / "instructions" / f"{name}.json").write_text(json.dumps({"seen": ["move the cup"]}))
    return path


def test_quaternion_continuity_and_invalid_pose():
    pose = np.array([[0, 0, 0, 2, 0, 0, 0], [1, 2, 3, -3, 0, 0, 0]], dtype=float)
    result = normalize_pose_sequence(pose)
    np.testing.assert_array_equal(result[:, 3], [1, 1])
    np.testing.assert_array_equal(result[:, :3], pose[:, :3])
    pose[0, 3:] = 0
    with pytest.raises(ValueError, match="quaternion"):
        normalize_pose_sequence(pose)


def test_real_lerobot_round_trip_and_horizon(tmp_path):
    raw = tmp_path / "native"
    path = native_episode(raw)
    native_episode(raw, "episode10", offset=10)
    output = tmp_path / "lerobot"
    converter.convert(raw, "test/ee", root=output, image_writer_threads=0)
    fps = 250 / 15
    dataset = LeRobotDataset("test/ee", root=output, delta_timestamps={"action": [i / fps for i in range(50)]})
    assert len(dataset) == 10
    first = dataset[0]
    assert first["observation.state"].shape == (16,)
    assert first["action"].shape == (50, 16)
    np.testing.assert_allclose(first["action"][:5, 0], np.arange(1, 6) / 10, atol=1e-6)
    np.testing.assert_allclose(first["action"][5:, 0], 0.5, atol=1e-6)  # no crossing into episode10
    assert first["action_is_pad"][5:].all()
    with h5py.File(path) as f:
        rgb = converter.read_image(f, "head_camera", 0)
    actual = first["observation.images.cam_high"].numpy().transpose(1, 2, 0) * 255
    np.testing.assert_allclose(actual, rgb, atol=1e-4)
    manifest = json.loads((output / "meta/robotwin_ee_conversion.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["sources"][0]["source_frames"] == 6
    assert manifest["quaternion_order"] == "wxyz"
    # Dataset destruction must never be an implicit side effect of conversion.
    with pytest.raises(FileExistsError):
        converter.convert(raw, "test/ee", root=output)
    assert (output / "meta/info.json").exists()


def test_preflight_rejects_joint_only_and_misaligned_images(tmp_path):
    raw = tmp_path / "native"
    path = native_episode(raw)
    with h5py.File(path, "a") as f:
        del f["observation/head_camera/rgb"]
        f["observation/head_camera/rgb"] = np.zeros((2, 16, 20, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="length"):
        converter.convert(raw, "test/ee", root=tmp_path / "output")
    assert not (tmp_path / "output").exists()
    with h5py.File(path, "a") as f:
        del f["endpose"]
    with pytest.raises(ValueError, match="native RoboTwin"):
        converter.convert(raw, "test/ee", root=tmp_path / "output")


def test_ee_policy_preserves_absolute_targets_and_all_16_dims():
    state = np.arange(16, dtype=np.float32)
    actions = np.tile(state + 10, (50, 1))
    data = {
        "state": state.copy(),
        "actions": actions.copy(),
        "prompt": "move",
        "images": {"cam_high": np.zeros((3, 16, 20), dtype=np.uint8)},
    }
    result = RoboTwinEEInputs(action_dim=32)(data)
    np.testing.assert_array_equal(result["state"][:16], state)
    np.testing.assert_array_equal(result["actions"][:, :16], actions)
    assert result["actions"].shape == (50, 32)
    assert not result["actions"][:, 16:].any()
    assert result["image"]["base_0_rgb"].shape == (16, 20, 3)
    with pytest.raises(ValueError, match="16"):
        RoboTwinEEInputs(action_dim=32)({**data, "state": np.zeros(14)})
    prediction = np.zeros((2, 32), dtype=np.float32)
    prediction[:, [3, 11]] = 2
    prediction[:, [7, 15]] = [-0.1, 1.1]
    out = RoboTwinEEOutputs()({"actions": prediction})["actions"]
    assert out.shape == (2, 16)
    np.testing.assert_array_equal(out[:, [3, 11]], 1)
    np.testing.assert_array_equal(out[:, [7, 15]], [[0, 1], [0, 1]])


def test_normalization_uses_full_batch_and_tail():
    spec = importlib.util.spec_from_file_location(
        "compute_norm_stats", Path(__file__).with_name("compute_norm_stats.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Nine frames exercises both an 8-element batch and a 1-element tail.
    dataset = [{"state": np.array([float(i)]), "actions": np.full((50, 1), float(i))} for i in range(9)]
    stats = module.compute_statistics(dataset)
    for key in ("state", "actions"):
        np.testing.assert_allclose(stats[key].mean, [4.0])
        np.testing.assert_allclose(stats[key].std, [np.std(np.arange(9))])
    limited = module.compute_statistics(dataset, max_frames=3, seed=1)
    ids = np.random.default_rng(1).choice(9, 3, replace=False)
    np.testing.assert_allclose(limited["state"].mean, [ids.mean()])
