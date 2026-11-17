#!/usr/bin/env python
"""真实 Pi0 上验证 full/partial LoRA 更新的数值、状态和显存表现。

用法：
  # 旧式完整状态更新基线（A -> B）
  python tests/validate_pi0_partial_update.py --mode full --iterations 2

  # 新式部分更新，执行 A -> B -> A 并继续到 25 次
  python tests/validate_pi0_partial_update.py --mode partial --iterations 25 \
      --baseline outputs/partial_update_validation/full_predictions_b.npz
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
ROBOTWIN_SRC = ROOT.parent / "RoboTwin" / "policy" / "pi0" / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROBOTWIN_SRC))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("full", "partial"), required=True)
parser.add_argument("--iterations", type=int, required=True)
parser.add_argument(
    "--config", default="config/place_empty_cup_10_4090_v2_smoke.json")
parser.add_argument("--baseline")
parser.add_argument(
    "--output-dir", default="outputs/partial_update_validation")
args = parser.parse_args()

if args.mode == "partial" and args.iterations < 3:
    parser.error("partial 模式至少需要 3 次迭代以验证 A→B→A")

os.environ["PI0_LORA_UPDATE_MODE"] = args.mode
os.environ["PI0_VERIFY_PARTIAL_UPDATE"] = "1" if args.mode == "partial" else "0"
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import jax

_JAX_DEVICES = jax.devices()
_GPU_BACKEND_OK = any(device.platform == "gpu" for device in _JAX_DEVICES)
if not _GPU_BACKEND_OK:
    raise RuntimeError(
        f"真实 Pi0 显存验证要求 JAX GPU 后端，当前设备为: {_JAX_DEVICES}")

from script.common import data_loader
from script.common.pi0_forward import (
    get_pi0_non_lora_value_ids,
    get_pi0_runtime_state_summary,
    get_pi0_runtime_stats,
    pi0_forward,
)
from vla_lorahub import algorithm
from vla_lorahub.constant import BASE_CHECKPOINT_DIR


class GpuMemoryMonitor:
    def __init__(self, interval=0.2):
        self.interval = interval
        self.pid = os.getpid()
        self.samples = []
        self.sources = []
        # 某些容器能读取 GPU 总显存，却会隐藏宿主机/容器进程 PID，导致
        # --query-compute-apps 永远为空。CUDA_VISIBLE_DEVICES 使用数字索引时，
        # 退回到对应物理 GPU 的 memory.used 仍可验证候选间是否持续增长。
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
        self.physical_gpu_index = visible if visible.isdigit() else "0"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _sample(self):
        output = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ], text=True, stderr=subprocess.DEVNULL)
        used = None
        source = "process"
        for line in output.splitlines():
            fields = [part.strip() for part in line.split(",")]
            if len(fields) == 2 and int(fields[0]) == self.pid:
                used = int(fields[1])
                break
        if used is None:
            # 容器隔离场景：进程表不可见，但按物理 GPU 查询仍然可用。
            fallback = subprocess.check_output([
                "nvidia-smi", "--id", self.physical_gpu_index,
                "--query-gpu=memory.used", "--format=csv,noheader,nounits",
            ], text=True, stderr=subprocess.DEVNULL)
            used = int(fallback.splitlines()[0].strip())
            source = f"gpu_total:{self.physical_gpu_index}"
        self.samples.append((time.monotonic(), used))
        self.sources.append(source)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def peak_since(self, index):
        values = [value for _, value in self.samples[index:]]
        return max(values, default=0)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


with open(resolve(args.config)) as file:
    config = json.load(file)

algorithm_cfg = config["algorithm"]
norm_stats_path = resolve(config["norm_stats_path"])
if not norm_stats_path.exists():
    reference_path = resolve(config["reference_norm_stats_path"])
    if not reference_path.exists():
        raise FileNotFoundError(
            f"norm_stats 与 reference_norm_stats 均不存在: "
            f"{norm_stats_path}, {reference_path}")
    norm_stats_path = reference_path

if algorithm_cfg.get("sampling_mode") == "pi0_random":
    frame_dataset = data_loader.Pi0FrameDataset(
        resolve(config["data_dir"]), norm_stats_path,
        action_horizon=algorithm_cfg["action_horizon"],
        episode_cache_size=algorithm_cfg.get("episode_cache_size", 2),
    )
    sampler = data_loader.Pi0RandomBatchSampler(
        frame_dataset,
        batch_size=algorithm_cfg["candidate_batch_size"],
        reuse_candidates=max(1, args.iterations),
        seed=algorithm_cfg.get("seed", 42),
    )
    inputs, targets, _ = sampler.get_batch(0)
else:
    inputs, targets = data_loader.load_few_shot_data(
        resolve(config["data_dir"]), norm_stats_path,
        frames_per_ep=algorithm_cfg["frames_per_ep"],
        action_horizon=algorithm_cfg["action_horizon"],
        keyframe_strategy=algorithm_cfg["keyframe_strategy"],
    )
norm_stats = data_loader.load_norm_stats(norm_stats_path)
delta_stats = data_loader.compute_delta_loss_stats(
    targets, std_floor=algorithm_cfg["loss"]["std_floor"])

module_paths = [str(resolve(path)) for path in config["lora_modules"]]
base_params = algorithm.load_base_params_from_checkpoint(BASE_CHECKPOINT_DIR)
lora_cache, module_names = algorithm.load_lora_modules(module_paths)
n_modules = len(module_names)

weights_a = np.full((4, n_modules), 1.0 / n_modules, dtype=np.float32)
weights_b = weights_a.copy()
logits = np.linspace(-0.8, 0.8, n_modules, dtype=np.float32)
v2_mode = algorithm_cfg.get("v2_mode", "action_focused")
search_group_ids = set(algorithm.get_v2_search_group_ids(v2_mode))
fixed_group_ids = set(range(4)) - search_group_ids
for offset, group_id in enumerate(sorted(search_group_ids)):
    group_logits = logits if offset % 2 == 0 else logits[::-1]
    weights_b[group_id] = algorithm.softmax_normalize(group_logits, 1.0)

fixed_main = algorithm.compose_lora_weights_per_group(
    lora_cache, weights_a, module_names,
    composition_mode=algorithm_cfg["composition_mode"],
    include_group_ids=fixed_group_ids)


def make_params(weights):
    dynamic = algorithm.compose_lora_weights_per_group(
        lora_cache, weights, module_names,
        composition_mode=algorithm_cfg["composition_mode"],
        include_group_ids=search_group_ids)
    composed = {**fixed_main, **dynamic}
    return algorithm.build_composed_model_params(
        base_params, composed, update_lora_keys=set(dynamic))


params_a = make_params(weights_a)
params_b = make_params(weights_b)


def predict(params):
    return np.stack(pi0_forward(
        params, inputs, config["train_config_name"], norm_stats,
        diffusion_steps=config["diffusion_steps"],
        max_batch=algorithm_cfg["max_batch"],
    ))


def loss_of(predictions):
    return algorithm.delta_grouped_vla_loss(
        {}, inputs, targets, forward_fn=lambda _params, _inputs: list(predictions),
        delta_stats=delta_stats,
        joint_weight=algorithm_cfg["loss"]["joint_weight"],
        gripper_weight=algorithm_cfg["loss"]["gripper_weight"],
        huber_delta=algorithm_cfg["loss"]["huber_delta"],
    )


monitor = GpuMemoryMonitor()
monitor.start()
records = []
predictions = []
non_lora_reference = None
non_lora_changes = []
try:
    for index in range(args.iterations):
        candidate = "A" if index % 2 == 0 else "B"
        sample_start = len(monitor.samples)
        started = time.perf_counter()
        output = predict(params_a if candidate == "A" else params_b)
        elapsed = time.perf_counter() - started
        predictions.append(output if index < 3 else None)
        current_ids = get_pi0_non_lora_value_ids()
        if non_lora_reference is None:
            non_lora_reference = current_ids
        changed = sum(
            non_lora_reference.get(key) != value
            for key, value in current_ids.items())
        non_lora_changes.append(changed)
        record = {
            "iteration": index + 1,
            "candidate": candidate,
            "seconds": elapsed,
            "loss": loss_of(output),
            "gpu_peak_mib": monitor.peak_since(sample_start),
            "non_lora_changed": changed,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
finally:
    monitor.stop()

output_dir = resolve(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
result = {
    "mode": args.mode,
    "iterations": args.iterations,
    "jax_devices": [f"{device.platform}:{device.id}" for device in _JAX_DEVICES],
    "gpu_backend_ok": _GPU_BACKEND_OK,
    "records": records,
    "runtime_stats": get_pi0_runtime_stats(),
    "state_summary": get_pi0_runtime_state_summary(),
    "global_gpu_peak_mib": max((value for _, value in monitor.samples), default=0),
    "gpu_memory_sources": sorted(set(monitor.sources)),
    "gpu_memory_monitor_ok": any(value > 0 for _, value in monitor.samples),
    "non_lora_identity_ok": max(non_lora_changes, default=0) == 0,
}

if args.mode == "full":
    np.savez_compressed(output_dir / "full_predictions_b.npz", predictions=predictions[1])
else:
    result["aba_max_abs"] = float(np.max(np.abs(predictions[0] - predictions[2])))
    result["aba_loss_abs"] = float(abs(records[0]["loss"] - records[2]["loss"]))
    if args.baseline:
        baseline = np.load(resolve(args.baseline))["predictions"]
        result["full_vs_partial_b_max_abs"] = float(
            np.max(np.abs(baseline - predictions[1])))
    steady_peaks = [record["gpu_peak_mib"] for record in records[3:]]
    result["steady_gpu_range_mib"] = (
        max(steady_peaks) - min(steady_peaks) if steady_peaks else 0)
    # A→B→A 的 3 次探针只验证数值和监控链路；显存平台期必须覆盖
    # 至少 20 个连续候选，避免空 steady 区间被误判为 0 波动。
    result["memory_plateau_sample_ok"] = bool(
        args.iterations >= 20 and len(steady_peaks) >= 17)
    result["checks_passed"] = bool(
        result["aba_max_abs"] <= 1e-5
        and result["aba_loss_abs"] <= 1e-8
        and result.get("full_vs_partial_b_max_abs", 0.0) <= 1e-5
        and result["gpu_backend_ok"]
        and result["non_lora_identity_ok"]
        and result["gpu_memory_monitor_ok"]
        and result["memory_plateau_sample_ok"]
        and result["steady_gpu_range_mib"] <= 256
    )

result_path = output_dir / f"{args.mode}_result.json"
with open(result_path, "w") as file:
    json.dump(result, file, indent=2)
print(json.dumps(result, indent=2), flush=True)
if args.mode == "partial" and not result["checks_passed"]:
    raise SystemExit(1)
