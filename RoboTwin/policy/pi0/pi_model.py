#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import os
import json
import sys
import jax
import numpy as np
from openpi.models import model as _model
from openpi.policies import aloha_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

import cv2
from PIL import Image

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

parent_directory = os.path.dirname(os.path.abspath(__file__))


class PI0:

    def __init__(self, train_config_name, model_name, checkpoint_id, pi0_step):
        self.train_config_name = train_config_name
        self.model_name = model_name
        self.checkpoint_id = checkpoint_id

        config = _config.get_config(self.train_config_name)

        # Direct LoRAHub feature backbone + ACT/DP action decoder.  This path
        # replaces Pi0's action expert; it is not a Side residual adapter.
        self.decoder_policy = None
        self.side_policy = None
        decoder_config = os.environ.get("LORAHUB_DECODER_CONFIG")
        if decoder_config:
            lorahub_root = os.environ.get(
                "LORAHUB_JAX_ROOT",
                os.path.abspath(os.path.join(parent_directory, "../../../LoRAHub-JAX")),
            )
            if lorahub_root not in sys.path:
                sys.path.insert(0, lorahub_root)
            from action_decoders.predict import LoRAHubActionDecoderPolicy
            self.decoder_policy = LoRAHubActionDecoderPolicy(decoder_config)
            print(f"loading direct LoRAHub action decoder success: {decoder_config}")
            self.img_size = (224, 224)
            self.observation_window = None
            self.pi0_step = self.decoder_policy.action_steps
            return

        # Optional legacy Side-Tuning path.  Kept only for the independent
        # pooling/Q-Former experiments; ACT/DP no longer use this path.
        side_config = os.environ.get("PI0_SIDE_CONFIG")
        if side_config:
            lorahub_root = os.environ.get(
                "LORAHUB_JAX_ROOT",
                os.path.abspath(os.path.join(parent_directory, "../../../LoRAHub-JAX")),
            )
            if lorahub_root not in sys.path:
                sys.path.insert(0, lorahub_root)
            from side.predict import KVSidePi0
            self.side_policy = KVSidePi0(side_config)
            print(f"loading KV Side-Tuning policy success: {side_config}")
            self.img_size = (224, 224)
            self.observation_window = None
            self.pi0_step = pi0_step
            return

        specified_path = f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}/assets/"
        entries = os.listdir(specified_path)
        assets_id = entries[0]

        self.policy = _policy_config.create_trained_policy(
            config,
            f"policy/pi0/checkpoints/{self.train_config_name}/{self.model_name}/{self.checkpoint_id}",
            robotwin_repo_id=assets_id)
        print("loading model success!")
        self.img_size = (224, 224)
        self.observation_window = None
        self.pi0_step = pi0_step

    # set img_size
    def set_img_size(self, img_size):
        self.img_size = img_size

    # set language randomly
    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction:{instruction}")

    # Update the observation window buffer
    def update_observation_window(self, img_arr, state):
        img_front, img_right, img_left, puppet_arm = (
            img_arr[0],
            img_arr[1],
            img_arr[2],
            state,
        )
        img_front = np.transpose(img_front, (2, 0, 1))
        img_right = np.transpose(img_right, (2, 0, 1))
        img_left = np.transpose(img_left, (2, 0, 1))

        self.observation_window = {
            "state": state,
            "images": {
                "cam_high": img_front,
                "cam_left_wrist": img_left,
                "cam_right_wrist": img_right,
            },
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        if self.decoder_policy is not None:
            images = self.observation_window["images"]
            inputs = [{
                "images": images,
                "state": self.observation_window["state"],
                "instruction": self.observation_window["prompt"],
            }]
            delta_actions = self.decoder_policy.predict(inputs)[0]
            state = np.asarray(self.observation_window["state"], dtype=np.float32)
            delta_mask = np.ones(14, dtype=bool)
            delta_mask[[6, 13]] = False
            absolute_actions = np.asarray(delta_actions, dtype=np.float32).copy()
            absolute_actions[:, delta_mask] += state[delta_mask]
            if self.decoder_policy.temporal_aggregation:
                absolute_actions = self.decoder_policy.aggregate_absolute_actions(
                    absolute_actions)
            return absolute_actions
        if self.side_policy is not None:
            images = self.observation_window["images"]
            inputs = [{
                "images": images,
                "state": self.observation_window["state"],
                "instruction": self.observation_window["prompt"],
            }]
            diffusion_steps = int(os.environ.get("PI0_SIDE_DIFFUSION_STEPS", "10"))
            # LoRAHub/Side training is performed in Pi0 DeltaActions space.
            # RoboTwin's environment consumes absolute joint actions, so apply
            # the current state to joint dimensions here; grippers remain
            # absolute, matching the DeltaActions mask used during training.
            delta_actions = self.side_policy.predict(
                inputs, diffusion_steps=diffusion_steps)[0]
            state = np.asarray(self.observation_window["state"], dtype=np.float32)
            delta_mask = np.ones(14, dtype=bool)
            delta_mask[[6, 13]] = False
            absolute_actions = np.asarray(delta_actions, dtype=np.float32).copy()
            absolute_actions[:, delta_mask] += state[delta_mask]
            return absolute_actions
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        if self.decoder_policy is not None:
            self.decoder_policy.reset()
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")
