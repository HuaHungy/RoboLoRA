"""Configuration and paths for independent cluster LoRA experts."""

import importlib.util
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROBOTWIN = HERE.parents[1]
PI0 = HERE.parent / 'pi0'
CLUSTERS = (8, 24, 3, 6, 9, 31, 23)


def repo_id(cluster):
    cluster = int(cluster)
    if cluster not in CLUSTERS:
        raise ValueError(f'Unsupported cluster: {cluster}')
    return f'cluster_{cluster:02d}_pi0_ee'


def config_name(cluster):
    return f'{repo_id(cluster)}_lora'


def read_settings(path=None):
    path = Path(path or HERE / 'config.json').resolve()
    settings = json.loads(path.read_text())
    for key in ('dataset_root', 'pretrained_params', 'assets_root'):
        settings[key] = str((path.parent / Path(settings[key]).expanduser()).resolve())
    if tuple(settings.get('clusters', ())) != CLUSTERS:
        raise ValueError(f'clusters must be exactly {CLUSTERS} in relevance order')
    steps = settings.get('cluster_train_steps')
    if not isinstance(steps, dict) or set(steps) != {str(k) for k in CLUSTERS}:
        raise ValueError('cluster_train_steps must define every configured cluster exactly once')
    for key in ('batch_size', 'save_interval', 'action_horizon', 'log_interval'):
        if not isinstance(settings.get(key), int) or settings[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for cluster, value in steps.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f'cluster_train_steps[{cluster}] must be positive')
    if settings['action_horizon'] != 50:
        raise ValueError('Cluster exports contain exactly 50 actions; action_horizon must be 50')
    if not isinstance(settings.get('warmup_steps'), int) or settings['warmup_steps'] < 0:
        raise ValueError('warmup_steps must be a nonnegative integer')
    if not isinstance(settings.get('num_workers'), int) or settings['num_workers'] < 0:
        raise ValueError('num_workers must be a nonnegative integer')
    if not isinstance(settings.get('fsdp_devices', 1), int) or settings['fsdp_devices'] <= 0:
        raise ValueError('fsdp_devices must be positive')
    if settings.get('keep_period') is not None and (
            not isinstance(settings['keep_period'], int) or settings['keep_period'] <= 0):
        raise ValueError('keep_period must be null or positive')
    if not isinstance(settings.get('peak_lr'), (int, float)) or settings['peak_lr'] <= 0:
        raise ValueError('peak_lr must be positive')
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


def build_config(cluster, settings, run_dir, resume=False):
    cluster = int(cluster)
    setup(settings)
    import flax.nnx as nnx
    from openpi.models.pi0 import Pi0Config
    from openpi.shared.nnx_utils import PathRegex
    from openpi.training.config import DataConfig, LeRobotRoboTwinEEDataConfig, TrainConfig
    from openpi.training.optimizer import CosineDecaySchedule
    from openpi.training.weight_loaders import CheckpointWeightLoader

    steps = settings['cluster_train_steps'][str(cluster)]
    model = Pi0Config(paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora',
                      action_horizon=settings['action_horizon'])

    class LoRAOnlyTrainConfig(TrainConfig):
        @property
        def trainable_filter(self):
            # LoRAHub-JAX extracts exactly these 20 leaves.  Keep this separate
            # from freeze_filter: OpenPI also uses freeze_filter to cast frozen
            # parameters to bfloat16, while the base control projections must
            # retain their original float32 values for lossless recomposition.
            return nnx.All(nnx.Param, PathRegex('.*lora.*'))

    freeze = nnx.Any(model.get_freeze_filter(), PathRegex('.*PaliGemma/img/.*'))
    return LoRAOnlyTrainConfig(
        name=config_name(cluster), exp_name='model', project_name='robotwin_pi0_cluster_lora', model=model,
        data=LeRobotRoboTwinEEDataConfig(repo_id=repo_id(cluster), base_config=DataConfig(local_files_only=True)),
        freeze_filter=freeze, ema_decay=None, batch_size=settings['batch_size'],
        num_workers=settings['num_workers'], num_train_steps=steps,
        weight_loader=CheckpointWeightLoader(settings['pretrained_params']),
        lr_schedule=CosineDecaySchedule(warmup_steps=min(settings['warmup_steps'], steps - 1),
                                       peak_lr=settings['peak_lr'], decay_steps=steps),
        assets_base_dir=settings['assets_root'], checkpoint_base_dir=str(Path(run_dir) / 'checkpoints'),
        save_interval=settings['save_interval'], keep_period=settings.get('keep_period'),
        log_interval=settings['log_interval'], seed=settings['seed'],
        fsdp_devices=settings.get('fsdp_devices', 1), wandb_enabled=False, overwrite=False, resume=resume,
        policy_metadata={'action_representation': 'absolute_world_ee_xyz_wxyz_gripper',
                         'cluster_id': cluster, 'cluster_variant': 'lambda0_03__calibrated5',
                         'cluster_k': 32, 'sample_unit': 'one_start_per_50_step_window',
                         'trainable_scope': 'lora_only_20_leaves'},
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


def latest_checkpoint(run_dir, cluster, step=None):
    root = Path(run_dir) / 'checkpoints' / config_name(cluster) / 'model'
    candidates = [root / str(step)] if step is not None else sorted(
        (p for p in root.glob('*') if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name), reverse=True)
    for path in candidates:
        if ((path / '_CHECKPOINT_METADATA').is_file() and (path / 'params').is_dir()
                and (path / 'train_state').is_dir()
                and (path / 'assets' / repo_id(cluster) / 'norm_stats.json').is_file()):
            return path
    raise FileNotFoundError(f'No committed cluster checkpoint under {root}')


def parse_cluster_steps(values, parser):
    result = {}
    for value in values or ():
        if not re.fullmatch(r'\d+=\d+', value):
            parser.error('--cluster-step must use CLUSTER=STEPS, for example 8=2300')
        cluster, steps = map(int, value.split('='))
        if cluster not in CLUSTERS or steps <= 0 or cluster in result:
            parser.error(f'Invalid or duplicate --cluster-step: {value}')
        result[cluster] = steps
    return result
