"""Local configuration extension; never modifies the upstream config registry."""

import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROBOTWIN = HERE.parents[1]
PI0 = HERE.parent / 'pi0'
TASKS = ('adjust_bottle', 'click_alarmclock', 'place_can_basket', 'place_container_plate', 'rotate_qrcode')
JOINT = 'joint'
JOINT_REPO = 'robotwin_five_pi0_ee'


def model_key(settings, task):
    return JOINT if settings.get('training_mode') == 'joint' else task


def repo_id(task):
    return JOINT_REPO if task == JOINT else f'{task}_pi0_ee'


def read_settings(path=None):
    path = Path(path or HERE / 'config.json').resolve()
    settings = json.loads(path.read_text())
    for key in ('dataset_root', 'pretrained_params'):
        settings[key] = str((path.parent / Path(settings[key]).expanduser()).resolve())
    if tuple(settings['tasks']) != TASKS:
        raise ValueError('config.json must contain the five requested tasks in their original order')
    for key in ('batch_size', 'num_train_steps', 'save_interval', 'action_horizon', 'execute_steps', 'diffusion_steps', 'eval_episodes'):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if settings['num_workers'] < 0 or settings['execute_steps'] > settings['action_horizon']:
        raise ValueError('Invalid workers or execute_steps')
    return settings


def setup(settings):
    sys.path.insert(0, str(PI0 / 'src'))
    sys.path.insert(0, str(HERE.parent))
    os.environ['HF_LEROBOT_HOME'] = settings['dataset_root']
    os.environ.setdefault('HF_DATASETS_CACHE', str(HERE / '.cache/hf_datasets'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def config_name(task):
    if task not in (*TASKS, JOINT):
        raise ValueError(f'Unsupported task: {task}')
    return f'{task}_pi0_ee_five_lora'


def build_config(task, settings, run_dir, resume=False):
    setup(settings)
    import flax.nnx as nnx
    from openpi.models.pi0 import Pi0Config
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training.config import DataConfig, LeRobotRoboTwinEEDataConfig, TrainConfig
    from openpi.training.optimizer import CosineDecaySchedule
    from openpi.training.weight_loaders import CheckpointWeightLoader

    model = Pi0Config(paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora',
                      action_horizon=settings['action_horizon'])
    # Upstream LoRA freezes LLM base weights, but not the large image backbone.
    # Freeze it explicitly to keep the single-4090 configuration conservative.
    freeze = nnx.Any(model.get_freeze_filter(), PathRegex('.*PaliGemma/img/.*'))
    return TrainConfig(
        name=config_name(task), exp_name='model', project_name='robotwin_pi0_ee_five', model=model,
        data=LeRobotRoboTwinEEDataConfig(repo_id=repo_id(task), base_config=DataConfig(local_files_only=True)),
        freeze_filter=freeze, ema_decay=None, batch_size=settings['batch_size'],
        num_workers=settings['num_workers'], num_train_steps=settings['num_train_steps'],
        weight_loader=CheckpointWeightLoader(settings['pretrained_params']),
        lr_schedule=CosineDecaySchedule(warmup_steps=min(settings['warmup_steps'], settings['num_train_steps'] - 1),
                                       peak_lr=settings['peak_lr'], decay_steps=settings['num_train_steps']),
        assets_base_dir=str(HERE / 'assets'), checkpoint_base_dir=str(Path(run_dir) / 'checkpoints'),
        save_interval=settings['save_interval'], keep_period=settings.get('keep_period'), log_interval=50,
        seed=settings['seed'], fsdp_devices=settings.get('fsdp_devices', 1), wandb_enabled=False, overwrite=False, resume=resume,
        policy_metadata={'action_representation': 'absolute_world_ee_xyz_wxyz_gripper', 'task': task,
                         'training_tasks': list(TASKS) if task == JOINT else [task]},
    )


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def latest_checkpoint(run_dir, task, step=None):
    root = Path(run_dir) / 'checkpoints' / config_name(task) / 'model'
    candidates = [root / str(step)] if step is not None else sorted(
        (p for p in root.glob('*') if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name), reverse=True)
    for path in candidates:
        if ((path / '_CHECKPOINT_METADATA').is_file() and (path / 'params').is_dir()
                and (path / 'train_state').is_dir()
                and (path / 'assets' / repo_id(task) / 'norm_stats.json').is_file()):
            return path
    raise FileNotFoundError(f'No committed checkpoint with EE normalization assets under {root}')
