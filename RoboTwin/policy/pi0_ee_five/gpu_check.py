"""Read-only device/dependency check. No model creation or training."""

import importlib.util
import argparse
import json
import subprocess

import jax
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation', action='store_true')
args = parser.parse_args()
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version',
                               '--format=csv,noheader'], text=True).strip()
devices = jax.devices()
if not any(d.platform == 'gpu' for d in devices) or not torch.cuda.is_available():
    raise RuntimeError('NVIDIA GPU must be visible to both JAX and PyTorch')
required = ['pyarrow', 'lerobot'] + (['sapien', 'mplib', 'curobo'] if args.evaluation else [])
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise RuntimeError(f'Missing dependencies: {missing}')
print(json.dumps({'nvidia_smi': gpu, 'jax_devices': [str(d) for d in devices],
                  'torch_cuda': torch.cuda.is_available(), 'training_started': False}, indent=2), flush=True)
