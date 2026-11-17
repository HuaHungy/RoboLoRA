from pathlib import Path
import pickle
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
PI0 = HERE.parent / 'pi0'
sys.path[:0] = [str(HERE), str(HERE.parent), str(PI0 / 'src')]

from common import CLUSTERS, build_config, read_settings, repo_id, setup


class FakeReader:
    episodes = [0, 1]

    def __getitem__(self, index):
        episode = index // 50
        return {'episode_index': np.int64(episode), 'frame_index': np.int64(index % 50),
                'action': np.zeros((50, 16), np.float32),
                'action_is_pad': np.zeros(50, bool), 'task': f'task-{episode}'}


class ClusterPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = read_settings()
        setup(cls.settings)

    def test_default_parameters_are_cluster_specific(self):
        self.assertEqual(tuple(self.settings['clusters']), CLUSTERS)
        self.assertEqual(self.settings['batch_size'], 16)
        self.assertEqual(self.settings['peak_lr'], 1e-5)
        self.assertEqual(self.settings['warmup_steps'], 100)
        self.assertEqual(self.settings['save_interval'], 500)
        self.assertIsNone(self.settings['keep_period'])
        self.assertEqual(self.settings['cluster_train_steps'], {
            '8': 2300, '24': 1200, '3': 3200, '6': 1300,
            '9': 1300, '31': 1100, '23': 2500})

    def test_build_config_uses_independent_repo_and_checkpoint(self):
        import flax.nnx as nnx
        cfg = build_config(8, self.settings, HERE / 'runs/_test')
        self.assertEqual(cfg.data.repo_id, 'cluster_08_pi0_ee')
        self.assertEqual(cfg.name, 'cluster_08_pi0_ee_lora')
        self.assertEqual((cfg.batch_size, cfg.num_train_steps, cfg.save_interval), (16, 2300, 500))
        self.assertEqual(cfg.model.action_horizon, 50)
        self.assertEqual(cfg.project_name, 'robotwin_pi0_cluster_lora')
        predicate = nnx.filterlib.to_predicate(cfg.freeze_filter)
        value = nnx.Param(np.zeros((1,)))
        self.assertFalse(predicate(('PaliGemma', 'llm', 'layers', 'attn', 'lora_a'), value))
        self.assertFalse(predicate(('PaliGemma', 'llm', 'layers', 'mlp_1',
                                    'linear_lora_b'), value))
        self.assertTrue(predicate(('PaliGemma', 'llm', 'layers', 'attn', 'w'), value))
        self.assertTrue(predicate(('PaliGemma', 'img', 'embedding', 'kernel'), value))
        # Control projections retain the base float32 dtype, but the separate
        # trainable filter below still excludes them from optimization.
        self.assertFalse(predicate(('state_proj', 'kernel'), value))
        trainable = nnx.filterlib.to_predicate(cfg.trainable_filter)
        self.assertTrue(trainable(('PaliGemma', 'llm', 'layers', 'attn', 'lora_a'), value))
        self.assertFalse(trainable(('state_proj', 'kernel'), value))
        self.assertFalse(trainable(('action_in_proj', 'kernel'), value))
        self.assertFalse(trainable(('action_out_proj', 'kernel'), value))
        self.assertEqual(cfg.policy_metadata['trainable_scope'], 'lora_only_20_leaves')

    def test_trainable_tree_matches_lorahub_twenty_leaf_contract(self):
        import flax.nnx as nnx
        import jax
        cfg = build_config(3, self.settings, HERE / 'runs/_test')
        model = nnx.eval_shape(cfg.model.create, jax.random.key(0))
        trainable = nnx.state(model).filter(cfg.trainable_filter)
        leaves = jax.tree_util.tree_flatten_with_path(trainable)[0]
        self.assertEqual(len(leaves), 20)
        self.assertTrue(all('lora' in '/'.join(str(part) for part in path)
                            for path, _ in leaves))
        self.assertEqual(sum(getattr(leaf, 'value', leaf).size for _, leaf in leaves), 49_987_584)

    def test_start_dataset_exposes_one_unpadded_chunk_per_episode(self):
        from cluster_data import ClusterStartDataset
        starts = [{'episode_index': 0, 'split': 'train'}, {'episode_index': 1, 'split': 'train'}]
        dataset = ClusterStartDataset(FakeReader(), starts, 'train')
        self.assertEqual(len(dataset), 2)
        self.assertEqual(int(dataset[1]['episode_index']), 1)
        self.assertEqual(dataset[1]['action'].shape, (50, 16))
        pickle.loads(pickle.dumps(dataset))

    def test_all_normalization_assets_are_32d_training_only(self):
        from openpi.shared import normalize
        for cluster in CLUSTERS:
            cfg = build_config(cluster, self.settings, HERE / 'runs/_test')
            root = cfg.assets_dirs / repo_id(cluster)
            stats = normalize.load(root)
            provenance = __import__('json').loads((root / 'provenance.json').read_text())
            self.assertEqual(stats['state'].mean.shape, (32,))
            self.assertEqual(stats['actions'].mean.shape, (32,))
            np.testing.assert_array_equal(stats['state'].mean[16:], 0)
            np.testing.assert_array_equal(stats['state'].std[16:], 0)
            np.testing.assert_array_equal(stats['actions'].mean[16:], 0)
            np.testing.assert_array_equal(stats['actions'].std[16:], 0)
            self.assertEqual(provenance['state_count'], provenance['train_windows'])
            self.assertEqual(provenance['action_count'], provenance['train_windows'] * 50)
            self.assertIn('training windows only', provenance['normalization'])

    def test_actual_transformed_batch_with_spawn_workers(self):
        from cluster_data import install_cluster_dataset
        from openpi.training import data_loader
        install_cluster_dataset('train')
        cfg = build_config(24, self.settings, HERE / 'runs/_test')
        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        dataset = data_loader.create_dataset(data_config, cfg.model)
        self.assertEqual(len(dataset), 366)
        transformed = data_loader.transform_dataset(dataset, data_config)
        loader = data_loader.TorchDataLoader(transformed, local_batch_size=2, num_batches=1,
                                             num_workers=2, shuffle=True, seed=42)
        batch = next(iter(loader))
        self.assertEqual(batch['state'].shape, (2, 32))
        self.assertEqual(batch['actions'].shape, (2, 50, 32))
        self.assertEqual(len(batch['image']), 3)
        self.assertTrue(np.isfinite(batch['actions']).all())


if __name__ == '__main__':
    unittest.main()
