"""RoboTwin world-frame EE pose convention, shared by conversion and policy IO."""

import numpy as np

EE_DIM = 16
EE_NAMES = [
    f"{arm}_{field}" for arm in ("left", "right") for field in ("x", "y", "z", "qw", "qx", "qy", "qz", "gripper")
]
CAMERA_MAP = {"cam_high": "head_camera", "cam_left_wrist": "left_camera", "cam_right_wrist": "right_camera"}


def normalize_pose_sequence(pose: np.ndarray) -> np.ndarray:
    """Normalize wxyz quaternions and pick a continuous sign within an episode."""
    pose = np.asarray(pose, dtype=np.float64).copy()
    if pose.ndim != 2 or pose.shape[1] != 7 or not np.isfinite(pose).all():
        raise ValueError("EE poses must be finite [T, 7] arrays: xyz + quaternion wxyz")
    norms = np.linalg.norm(pose[:, 3:7], axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Invalid zero-length EE quaternion")
    pose[:, 3:7] /= norms
    if len(pose):
        # Deterministic first sign, including rotations close to 180 degrees.
        first = pose[0, 3:7]
        if first[np.argmax(np.abs(first))] < 0:
            pose[0, 3:7] *= -1
    for i in range(1, len(pose)):
        if np.dot(pose[i - 1, 3:7], pose[i, 3:7]) < 0:
            pose[i, 3:7] *= -1
    return pose.astype(np.float32)


def pack_state(left_pose, left_gripper, right_pose, right_gripper):
    """Pack a whole episode. Grippers are normalized values from RoboTwin."""
    left = normalize_pose_sequence(left_pose)
    right = normalize_pose_sequence(right_pose)
    grips = []
    for gripper in (left_gripper, right_gripper):
        value = np.asarray(gripper, dtype=np.float32)
        if value.shape not in {(len(left),), (len(left), 1)}:
            raise ValueError("Gripper must have one scalar per EE frame")
        value = value.reshape(-1, 1)
        if not np.isfinite(value).all() or np.any(value < -1e-5) or np.any(value > 1 + 1e-5):
            raise ValueError("RoboTwin gripper values must be finite and in [0, 1]")
        grips.append(np.clip(value, 0, 1))
    if len(left) != len(right):
        raise ValueError("Left/right EE trajectory lengths differ")
    return np.concatenate([left, grips[0], right, grips[1]], axis=1)
