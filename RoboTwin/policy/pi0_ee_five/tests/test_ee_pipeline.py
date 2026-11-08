from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import HERE as ROOT, JOINT, JOINT_REPO, PI0, TASKS, build_config, config_name, latest_checkpoint, load_module, read_settings, setup
from prepare import action_weights
from pi0_ee_five.deploy_policy import encode_observation, eval as evaluate_policy, reset_model


def observation():
    return {'endpose': {'left_endpose': [1, 2, 3, -2, 0, 0, 0], 'left_gripper': .2,
                        'right_endpose': [4, 5, 6, 0, -3, 0, 0], 'right_gripper': .8},
            'observation': {name: {'rgb': np.full((12, 18, 3), i, np.uint8)} for i, name in
                            enumerate(['head_camera', 'left_camera', 'right_camera'])},
            'joint_action': {'vector': np.full(14, 999)}}


class EETests(unittest.TestCase):
    def test_pooled_normalization_includes_between_task_variance(self):
        setup(read_settings())
        from joint_data import pooled_stats
        parts = [np.random.default_rng(n).normal(loc=n, size=(n, 32)) for n in (3, 11, 27)]
        result = pooled_stats([(len(a), a.mean(0), a.std(0)) for a in parts])
        combined = np.concatenate(parts)
        np.testing.assert_allclose(result.mean, combined.mean(0), atol=1e-12)
        np.testing.assert_allclose(result.std, combined.std(0), atol=1e-12)

    def test_joint_pipeline_trains_once_then_evaluates_five_tasks(self):
        import pipeline
        settings = read_settings()
        calls = []
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'HERE', Path(tmp)), \
             patch.object(pipeline, 'read_settings', return_value=settings), \
             patch.object(pipeline, 'invoke', side_effect=lambda script, args, *a, **kw: calls.append((script, args))), \
             patch.object(pipeline, 'latest_checkpoint', return_value=Path(tmp) / '100000') as ckpt, \
             patch.object(sys, 'argv', ['pipeline.py', 'all']):
            pipeline.main()
        training = [args for script, args in calls if script == 'train_task.py']
        self.assertEqual(len(training), 1)
        self.assertEqual(training[0][1], JOINT)
        evaluation = [args for script, args in calls if script == 'evaluate.py']
        self.assertEqual([args[1] for args in evaluation], list(TASKS))
        self.assertTrue(all(call.args[1] == JOINT for call in ckpt.call_args_list))
        self.assertLess([s for s, _ in calls].index('train_task.py'), [s for s, _ in calls].index('evaluate.py'))

    def test_all_eval_tasks_load_joint_model_and_joint_stats(self):
        setup(read_settings())
        from pi0_ee_five.deploy_policy import EEPolicy
        with patch('openpi.policies.policy_config.create_trained_policy') as create:
            for task in TASKS:
                EEPolicy(read_settings(), task, ROOT / 'runs/_test', '/tmp/example-checkpoint')
        self.assertEqual(create.call_count, 5)
        for call in create.call_args_list:
            self.assertEqual(call.args[0].data.repo_id, JOINT_REPO)
            self.assertEqual(call.kwargs['robotwin_repo_id'], JOINT_REPO)

    def test_upstream_optimizer_initialization_shape_with_compat(self):
        setup(read_settings())
        from train_compat import install_train_state_compat
        install_train_state_compat()
        import jax
        from openpi.training import sharding
        trainer = load_module('ee_test_upstream_train', PI0 / 'scripts/train.py')
        cfg = build_config(JOINT, read_settings(), ROOT / 'runs/_test')
        # Executes the exact eval_shape path that failed, without allocating base weights.
        state, _ = trainer.init_train_state(cfg, jax.random.key(42), sharding.make_mesh(1), resume=True)
        self.assertIsNone(state.ema_params)
        self.assertGreater(len(jax.tree.leaves(state.opt_state)), 0)

    def test_exact_normalization_weights_match_real_repeat_last_chunks(self):
        for length in (1, 3, 50, 87):
            for horizon in (1, 4, 50):
                action = np.random.default_rng(length).normal(size=(length, 16))
                chunks = np.stack([action[np.minimum(np.arange(t, t + horizon), length - 1)] for t in range(length)])
                weights = action_weights(length, horizon)
                self.assertEqual(weights.sum(), length * horizon)
                np.testing.assert_allclose(np.average(action, axis=0, weights=weights), chunks.mean(axis=(0, 1)), atol=1e-12)
                np.testing.assert_allclose(np.average(action ** 2, axis=0, weights=weights), (chunks ** 2).mean(axis=(0, 1)), atol=1e-12)

    def test_observation_uses_ee_not_joints_and_maps_cameras_without_color_swap(self):
        value = encode_observation(observation())
        np.testing.assert_allclose(value['state'], [1, 2, 3, 1, 0, 0, 0, .2, 4, 5, 6, 0, 1, 0, 0, .8])
        self.assertEqual(value['images']['cam_high'].shape, (3, 12, 18))
        self.assertTrue((value['images']['cam_left_wrist'] == 1).all())
        self.assertTrue((value['images']['cam_right_wrist'] == 2).all())
        previous = value['state'].copy(); previous[3:7] *= -1; previous[11:15] *= -1
        current = encode_observation(observation(), previous)['state']
        np.testing.assert_array_equal(previous, current)

    def test_native_control_uses_ee_and_stops_immediately_on_success(self):
        calls = []
        env = SimpleNamespace(eval_success=False, take_action_cnt=0, step_lim=100,
                              get_instruction=lambda: 'adjust the bottle', get_obs=observation)
        def take_action(action, action_type):
            calls.append((action.copy(), action_type))
            env.take_action_cnt += 1
            env.eval_success = True
        env.take_action = take_action
        actions = np.tile(np.arange(16, dtype=np.float32), (50, 1))
        model = SimpleNamespace(previous_state=None, execute_steps=10,
                                policy=SimpleNamespace(infer=lambda inputs: {'actions': actions}))
        evaluate_policy(env, model, observation())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], 'ee')
        np.testing.assert_array_equal(calls[0][0], actions[0])
        reset_model(model)
        self.assertIsNone(model.previous_state)

    def test_lora_config_freezes_vision_and_keeps_adapters_and_control_projections_trainable(self):
        settings = read_settings(); setup(settings)
        import flax.nnx as nnx
        from openpi.transforms import DeltaActions
        cfg = build_config('adjust_bottle', settings, HERE / 'runs/_test_only')
        predicate = nnx.filterlib.to_predicate(cfg.freeze_filter)
        value = nnx.Param(np.zeros((1,)))
        self.assertTrue(predicate(('PaliGemma', 'img', 'embedding', 'kernel'), value))
        self.assertTrue(predicate(('PaliGemma', 'llm', 'layers', 'attn', 'kernel'), value))
        self.assertFalse(predicate(('PaliGemma', 'llm', 'layers', 'attn', 'lora_a'), value))
        self.assertFalse(predicate(('action_out_proj', 'kernel'), value))
        self.assertIsNone(cfg.ema_decay)
        self.assertEqual(cfg.batch_size, 1)
        self.assertEqual(cfg.fsdp_devices, 1)
        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        self.assertFalse(any(isinstance(t, DeltaActions) for t in data.data_transforms.inputs))
        self.assertTrue(data.prompt_from_task)
        self.assertEqual(data.repo_id, 'adjust_bottle_pi0_ee')

    def test_checkpoint_selection_rejects_incomplete_and_foreign_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'checkpoints' / config_name('adjust_bottle') / 'model'
            for step in ('100', '200'):
                path = root / step
                (path / 'params').mkdir(parents=True)
                (path / 'train_state').mkdir()
                (path / '_CHECKPOINT_METADATA').write_text('{}')
            stats = root / '100/assets/adjust_bottle_pi0_ee/norm_stats.json'
            stats.parent.mkdir(parents=True); stats.write_text('{}')
            self.assertEqual(latest_checkpoint(tmp, 'adjust_bottle').name, '100')
            with self.assertRaises(FileNotFoundError):
                latest_checkpoint(tmp, 'adjust_bottle', 200)


if __name__ == '__main__':
    unittest.main()
