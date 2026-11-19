"""Local loop using upstream pi0 math, with checkpoints at completed optimizer steps.

Based on policy/pi0/scripts/train.py (Apache-2.0). The upstream loop labels the
5001st update as checkpoint 5000; this extension saves at exactly 5000 updates.
"""

import functools
import logging
import os
from pathlib import Path

import jax
import jax.numpy as jnp
from flax.training import common_utils
import tqdm_loggable.auto as tqdm
import wandb

from openpi.training import checkpoints, data_loader, sharding


def should_save(completed_steps, total_steps, interval):
    return completed_steps > 0 and (completed_steps % interval == 0 or completed_steps == total_steps)


def main(trainer, config):
    trainer.init_logging()
    if config.batch_size % jax.device_count():
        raise ValueError('Global batch size must be divisible by visible GPU count')
    cache = Path(os.environ.get('JAX_COMPILATION_CACHE_DIR', Path(config.checkpoint_dir).parents[2] / '.jax_cache'))
    jax.config.update('jax_compilation_cache_dir', str(cache))
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume)
    try:
        trainer.init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        loader = data_loader.create_data_loader(config, sharding=data_sharding,
                                               num_workers=config.num_workers, shuffle=True)
        data_iter = iter(loader)
        batch = next(data_iter)
        logging.info('Initialized mixed-task batch: %s', trainer.training_utils.array_tree_to_info(batch))
        state, state_sharding = trainer.init_train_state(config, init_rng, mesh, resume=resuming)
        jax.block_until_ready(state)
        if resuming:
            state = checkpoints.restore_state(manager, state, loader)
        step_fn = jax.jit(functools.partial(trainer.train_step, config),
                          in_shardings=(replicated, state_sharding, data_sharding),
                          out_shardings=(state_sharding, replicated), donate_argnums=(1,))
        accumulation_steps = getattr(config, 'policy_metadata', {}).get('gradient_accumulation_steps', 1)
        micro_step_start = int(state.step)
        if micro_step_start % accumulation_steps:
            raise RuntimeError('Checkpoint was saved in the middle of a gradient accumulation cycle')
        start = micro_step_start // accumulation_steps
        total_micro_steps = config.num_train_steps * accumulation_steps
        progress = tqdm.tqdm(initial=start, total=config.num_train_steps, dynamic_ncols=True)
        infos = []
        for micro_step in range(micro_step_start, total_micro_steps):
            with sharding.set_mesh(mesh):
                state, info = step_fn(train_rng, state, batch)
            infos.append(info)
            if (micro_step + 1) % accumulation_steps:
                batch = next(data_iter)
                continue
            completed = (micro_step + 1) // accumulation_steps
            progress.update(1)
            if completed == start + 1 or completed % config.log_interval == 0 or completed == config.num_train_steps:
                metrics = jax.device_get(jax.tree.map(jnp.mean, common_utils.stack_forest(infos)))
                progress.write(f"Step {completed}: " + ', '.join(f'{k}={v:.4f}' for k, v in metrics.items()))
                wandb.log(metrics, step=completed)
                infos = []
            if should_save(completed, config.num_train_steps, config.save_interval):
                if int(state.step) != completed * accumulation_steps:
                    raise RuntimeError('Micro-step and checkpoint label disagree')
                checkpoints.save_state(manager, state, loader, completed)
            if micro_step + 1 < total_micro_steps:
                batch = next(data_iter)
        manager.wait_until_finished()
        progress.close()
    finally:
        manager.close()
