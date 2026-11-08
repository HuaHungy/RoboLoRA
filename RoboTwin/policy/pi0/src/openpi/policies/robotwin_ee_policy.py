"""π0 IO for 16D absolute RoboTwin end-effector targets; no joint transforms."""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.policies import aloha_policy
from openpi.shared.robotwin_ee import EE_DIM


@dataclasses.dataclass(frozen=True)
class RoboTwinEEInputs(transforms.DataTransformFn):
    action_dim: int

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape != (EE_DIM,) or self.action_dim < EE_DIM or not np.isfinite(state).all():
            raise ValueError("RoboTwin EE state must be finite [16] and model action_dim >= 16")
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.ndim != 2 or actions.shape[-1] != EE_DIM or not np.isfinite(actions).all():
                raise ValueError("RoboTwin EE actions must be finite [horizon, 16]")
        # Reuse camera mapping and padding only. adapt_to_pi=False disables all
        # Aloha joint flips and gripper conversions; no DeltaActions is applied.
        return aloha_policy.AlohaInputs(action_dim=self.action_dim, adapt_to_pi=False)(dict(data))


@dataclasses.dataclass(frozen=True)
class RoboTwinEEOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[-1] < EE_DIM:
            raise ValueError("Expected [horizon, >=16] EE model output")
        actions = actions[:, :EE_DIM].copy()
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite EE prediction")
        for start in (3, 11):
            quaternion = actions[:, start : start + 4]
            norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
            if np.any(norms < 1e-8):
                raise ValueError("Model predicted a zero-length EE quaternion")
            actions[:, start : start + 4] = quaternion / norms
        actions[:, [7, 15]] = np.clip(actions[:, [7, 15]], 0, 1)
        return {"actions": actions}
