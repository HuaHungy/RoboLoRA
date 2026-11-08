import itertools
import logging
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import JOINT, build_config, read_settings, setup


class ServerTests(unittest.TestCase):
    def test_server_config_portable_paths_global_batch_and_retention(self):
        settings = read_settings(HERE / 'config.server.json')
        settings['fsdp_devices'] = 4
        cfg = build_config(JOINT, settings, HERE / 'runs/_server_test')
        self.assertEqual((cfg.batch_size, cfg.num_train_steps, cfg.save_interval, cfg.keep_period, cfg.fsdp_devices),
                         (32, 20000, 5000, 5000, 4))
        self.assertEqual(Path(settings['dataset_root']), HERE.parents[2] / 'data_clustering/data/lerobot')
        self.assertEqual(Path(settings['pretrained_params']),
                         HERE.parents[2] / 'transfer/openpi/openpi-assets/checkpoints/pi0_base/params')

    def test_checkpoint_schedule_and_loop_save_completed_updates(self):
        setup(read_settings())
        import train_loop
        self.assertEqual([n for n in range(1, 20001) if train_loop.should_save(n, 20000, 5000)],
                         [5000, 10000, 15000, 20000])
        cfg = SimpleNamespace(batch_size=1, seed=42, fsdp_devices=1, keep_period=5, overwrite=False,
                              resume=False, wandb_enabled=False, num_workers=0, num_train_steps=12,
                              log_interval=5, save_interval=5, checkpoint_dir=Path('/tmp/test/run/config/model'))
        trainer = SimpleNamespace(init_logging=lambda: None, init_wandb=lambda *a, **kw: None,
                                  training_utils=SimpleNamespace(array_tree_to_info=lambda b: 'tiny batch'),
                                  init_train_state=lambda *a, **kw: (SimpleNamespace(step=0), None),
                                  train_step=lambda c, r, s, b: (SimpleNamespace(step=s.step + 1), {'loss': np.float32(1)}))
        manager = SimpleNamespace(wait_until_finished=lambda: None, close=lambda: None)
        saved = []
        with patch.object(train_loop.checkpoints, 'initialize_checkpoint_dir', return_value=(manager, False)), \
             patch.object(train_loop.checkpoints, 'save_state', side_effect=lambda m,s,l,n: saved.append((n,s.step))), \
             patch.object(train_loop.data_loader, 'create_data_loader', return_value=itertools.repeat(np.zeros(1))), \
             patch.object(train_loop.jax, 'jit', side_effect=lambda f, **kw: f), \
             patch.object(train_loop.jax, 'block_until_ready', side_effect=lambda s: s), \
             patch.object(train_loop.wandb, 'log'):
            train_loop.main(trainer, cfg)
        self.assertEqual(saved, [(5, 5), (10, 10), (12, 12)])

    def test_orbax_retains_all_four_server_checkpoints(self):
        setup(read_settings())
        from openpi.training import checkpoints
        logging.getLogger().setLevel(logging.ERROR)
        with tempfile.TemporaryDirectory() as tmp:
            manager, _ = checkpoints.initialize_checkpoint_dir(Path(tmp) / 'checkpoints', keep_period=5000,
                                                               overwrite=False, resume=False)
            try:
                for step in (5000, 10000, 15000, 20000):
                    manager.save(step, {'params': {'value': np.array([step])}, 'train_state': {'step': np.array(step)},
                                        'assets': lambda path: path.mkdir(parents=True, exist_ok=True)})
                    manager.wait_until_finished()
                self.assertEqual(sorted(manager.all_steps()), [5000, 10000, 15000, 20000])
            finally:
                manager.close()


if __name__ == '__main__':
    unittest.main()
