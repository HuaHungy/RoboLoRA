"""
Pi0 真实推理: 将 "基座 + 组合 LoRA" 的 PyTree 加载到 Pi0 模型并做扩散采样。

设计要点:
  - 模型只创建一次 (nnx.split 缓存 graphdef/state), 后续仅原地更新参数,
    避免反复 JIT 编译导致的显存/内存持续增长。
  - 基座 numpy 数组 (约 12GB) 永不复制, 只浅拷贝结构 + 替换 LoRA 叶子。
  - 每次推理固定 RNG 种子, 使 loss 对权重完全确定。
"""

import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import os
import time
import dataclasses

from script.common.data_loader import unnormalize_vector

# ---- 图像预处理 (与 openpi 训练一致) ----
def resize_with_pad(img: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """openpi `resize_with_pad` 的 PIL 等价实现: 等比缩放 + 居中黑边填充 (无变形)。

    与 RoboTwin/src/openpi/shared/image_tools.py 的 resize_with_pad 行为对齐:
      ratio = max(cur_w/width, cur_h/height) -> 等比缩放 (int 截断)
      uint8 域线性插值, pad 余数按 divmod 分配 (pad_h0 前 / pad_h1 后)
    返回 uint8 (height, width, c), 与 openpi 在 uint8 域 pad 0 一致。
    """
    from PIL import Image
    h, w = img.shape[:2]
    ratio = max(w / width, h / height)
    resized_h = int(h / ratio)
    resized_w = int(w / ratio)
    resized = np.array(Image.fromarray(img).resize((resized_w, resized_h), Image.BILINEAR))
    out = np.zeros((height, width) + img.shape[2:], dtype=img.dtype)
    pad_h0, rem_h = divmod(height - resized_h, 2)
    pad_w0, rem_w = divmod(width - resized_w, 2)
    out[pad_h0:pad_h0 + resized_h, pad_w0:pad_w0 + resized_w] = resized
    return out


# ---- 模型运行时缓存 (进程内只初始化一次) ----
_model_cache = None
_tokenizer = None
_graphdef = None
_state = None
_rng = None
_base_pytree = None  # 基座 PyTree 缓存 (numpy, 永不改变)
_observation_cache = None
_observation_cache_key = None
_forward_calls = 0
_lora_specs = None
_non_lora_value_ids = None
_sample_actions_jit = None
_preprocess_images_jit = None
_normalize_states_jit = None
_unnormalize_actions_jit = None
_prompt_cache = {}
_runtime_stats = {
    "initial_device_bytes": 0,
    "initial_device_seconds": 0.0,
    "full_updates": 0,
    "partial_updates": 0,
    "updated_leaves": 0,
    "updated_bytes": 0,
    "update_seconds": 0.0,
    "non_lora_identity_checks": 0,
    "cache_clears": 0,
}
# LoRA state is now a dynamic input to one JIT executable, so candidates no
# longer create executables containing different LoRA constants.  Cache clearing
# is disabled by default; it remains configurable for conservative deployments.
_CACHE_CLEAR_INTERVAL = int(os.environ.get("PI0_CACHE_CLEAR_INTERVAL", "0"))


def _get_data_jit_functions():
    """Create the exact OpenPI JAX preprocessing functions once."""
    global _preprocess_images_jit, _normalize_states_jit, _unnormalize_actions_jit
    if _preprocess_images_jit is None:
        from openpi.shared import image_tools

        @jax.jit
        def preprocess_images(images):
            resized = image_tools.resize_with_pad(images, 224, 224)
            return resized.astype(jnp.float32) / 127.5 - 1.0

        @jax.jit
        def normalize_states(states, mean, std):
            return (states - mean) / (std + 1e-6)

        @jax.jit
        def unnormalize_actions(actions, mean, std):
            return actions * (std + 1e-6) + mean

        _preprocess_images_jit = preprocess_images
        _normalize_states_jit = normalize_states
        _unnormalize_actions_jit = unnormalize_actions
    return (_preprocess_images_jit, _normalize_states_jit,
            _unnormalize_actions_jit)


def _make_sample_actions_jit(graphdef):
    """Compile sampling with model state as data, never as frozen constants."""
    def sample_actions_with_state(state, rng, observation, *, num_steps):
        runtime_model = nnx.merge(graphdef, state)
        return runtime_model.sample_actions(
            rng, observation, num_steps=num_steps)

    return jax.jit(
        sample_actions_with_state, static_argnames=("num_steps",))


def _flat_to_pytree(flat_params: dict, strip_value_suffix: bool = True) -> dict:
    """将扁平参数 {path: np.ndarray} → JAX 嵌套 PyTree。

    路径格式: "params/PaliGemma/llm/.../q_einsum/w/value" 或 ".../q_einsum/w"
    转换后:   {"PaliGemma": {"llm": {...: {"q_einsum": {"w": array}}}}}
    """
    nested = {}
    for key, value in flat_params.items():
        if key.startswith("__"):
            continue
        parts = key.split("/")
        if parts[0] == "params":
            parts = parts[1:]
        if strip_value_suffix and parts[-1] == "value":
            parts = parts[:-1]
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested


def _build_lora_pytree(base_p: dict, composed_lora: dict) -> dict:
    """增量更新 PyTree: 基座只建一次, LoRA 叶子每轮原地替换。

    关键: 永不对基座 numpy 数组做深拷贝 (12GB), 只浅拷贝结构 + 替换 LoRA 叶子。
    """
    global _base_pytree

    if _base_pytree is None:
        _base_pytree = _flat_to_pytree(base_p)

    # 浅拷贝结构, 基座叶子 (12GB numpy) 共享引用, 不复制
    def shallow_copy(src):
        if isinstance(src, dict):
            return {k: shallow_copy(v) for k, v in src.items()}
        return src  # numpy array: 共享引用, 不复制
    pytree = shallow_copy(_base_pytree)

    # 只替换 LoRA 叶子 (共约 190MB)
    for lora_key, lora_val in composed_lora.items():
        parts = lora_key.split("/")
        if parts[0] == "params":
            parts = parts[1:]
        if parts[-1] == "value":
            parts = parts[:-1]
        d = pytree
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = lora_val

    return pytree


def _build_lora_update_pytree(composed_lora: dict, update_keys: set[str]) -> dict:
    """构建只包含本候选发生变化的 LoRA 叶子的部分 PyTree。"""
    unknown = set(update_keys) - set(composed_lora)
    if unknown:
        raise ValueError(f"待更新 LoRA key 不存在: {sorted(unknown)}")
    return _flat_to_pytree({key: composed_lora[key] for key in update_keys})


def _flatten_mapping(tree, prefix=()):
    """将 dict/NNX State 展平为 path tuple -> leaf，供运行时校验。"""
    if hasattr(tree, "items"):
        result = {}
        for key, value in tree.items():
            result.update(_flatten_mapping(value, prefix + (key,)))
        return result
    return {prefix: tree}


def _leaf_value(leaf):
    return leaf.value if hasattr(leaf, "value") else leaf


def _is_lora_path(path) -> bool:
    return "lora_a" in "/".join(map(str, path)) or "lora_b" in "/".join(map(str, path))


def _capture_non_lora_value_ids(state) -> dict:
    return {
        path: id(_leaf_value(leaf))
        for path, leaf in _flatten_mapping(state).items()
        if not _is_lora_path(path)
    }


def _block_until_ready(tree) -> None:
    for leaf in jax.tree.leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def get_pi0_runtime_stats() -> dict:
    """返回部分更新诊断统计的副本，供 smoke/benchmark 验证。"""
    return dict(_runtime_stats)


def get_pi0_non_lora_value_ids() -> dict:
    """返回当前非 LoRA 状态叶子的对象标识，仅用于一致性测试。"""
    if _state is None:
        return {}
    return _capture_non_lora_value_ids(_state)


def get_pi0_runtime_state_summary() -> dict:
    """统计当前状态的设备/主机叶子数，供集成验证使用。"""
    summary = {
        "non_lora_leaves": 0,
        "non_lora_device_leaves": 0,
        "lora_leaves": 0,
        "lora_device_leaves": 0,
    }
    if _state is None:
        return summary
    for path, leaf in _flatten_mapping(_state).items():
        value = _leaf_value(leaf)
        is_device = isinstance(value, jax.Array)
        prefix = "lora" if _is_lora_path(path) else "non_lora"
        summary[f"{prefix}_leaves"] += 1
        summary[f"{prefix}_device_leaves"] += int(is_device)
    return summary


def _validate_lora_update_specs(composed_lora: dict, update_keys: set[str]) -> None:
    global _lora_specs
    current_specs = {
        key: (tuple(np.shape(value)), np.dtype(value.dtype))
        for key, value in composed_lora.items()
    }
    if _lora_specs is None:
        _lora_specs = current_specs
    elif set(current_specs) != set(_lora_specs):
        missing = set(_lora_specs) - set(current_specs)
        extra = set(current_specs) - set(_lora_specs)
        raise ValueError(f"LoRA key 集合发生变化: missing={missing}, extra={extra}")
    for key in update_keys:
        if current_specs[key] != _lora_specs[key]:
            raise ValueError(
                f"LoRA 参数规格发生变化 {key}: "
                f"expected={_lora_specs[key]}, got={current_specs[key]}")


def _get_pi0_runtime(pytree: dict, lora_update_pytree: dict,
                     train_config_name: str):
    """首次完整加载模型；之后只替换发生变化的 LoRA 状态叶子。"""
    global _model_cache, _graphdef, _state, _rng, _sample_actions_jit
    global _non_lora_value_ids
    if _model_cache is not None:
        # 测试模式 full 精确复现旧实现，生产默认 partial。
        update_mode = os.environ.get("PI0_LORA_UPDATE_MODE", "partial")
        started = time.perf_counter()
        if update_mode == "full":
            _state.replace_by_pure_dict(pytree)
            updated_tree = pytree
            _runtime_stats["full_updates"] += 1
        elif update_mode == "partial":
            # 仅把变化的 LoRA 放到设备；V2 action_focused 下即 10 个 Expert
            # 叶子。基座和固定主干保持原对象，避免全量 device_put 带来的
            # 约 4GiB 额外峰值。下方的周期清理负责回收 Flax 控制流捕获的
            # 候选 executable/常量。
            device_updates = jax.tree.map(jax.device_put, lora_update_pytree)
            _block_until_ready(device_updates)
            _state.replace_by_pure_dict(device_updates)
            updated_tree = device_updates
            _runtime_stats["partial_updates"] += 1
        else:
            raise ValueError(f"未知 PI0_LORA_UPDATE_MODE={update_mode!r}")

        flat_updates = _flatten_mapping(updated_tree)
        _runtime_stats["updated_leaves"] += len(flat_updates)
        _runtime_stats["updated_bytes"] += sum(
            int(getattr(_leaf_value(value), "nbytes", 0))
            for value in flat_updates.values())
        _runtime_stats["update_seconds"] += time.perf_counter() - started

        if update_mode == "partial" and os.environ.get(
                "PI0_VERIFY_PARTIAL_UPDATE", "0") == "1":
            current_ids = _capture_non_lora_value_ids(_state)
            if current_ids != _non_lora_value_ids:
                changed = [
                    "/".join(map(str, path)) for path, value_id in current_ids.items()
                    if _non_lora_value_ids.get(path) != value_id
                ]
                raise AssertionError(
                    f"部分 LoRA 更新改变了非 LoRA 状态对象: {changed[:5]}")
            _runtime_stats["non_lora_identity_checks"] += 1
        return nnx.merge(_graphdef, _state)

    from openpi.training import config as _config
    from openpi.models import gemma as _gemma
    train_config = _config.get_config(train_config_name)
    _rng = jax.random.key(0)

    # Exact delta composition concatenates every source adapter's rank blocks.
    # OpenPI variants normally construct rank-16/rank-32 leaves, so infer the
    # actual ranks from the composed PyTree and temporarily construct matching
    # LoRA leaves.  alpha follows rank to preserve OpenPI's scaling value 1.0.
    flat_pytree = _flatten_mapping(pytree)
    inferred_ranks = {"main": None, "expert": None}
    for path, leaf in flat_pytree.items():
        joined = "/".join(map(str, path))
        if "lora_a" not in joined:
            continue
        is_expert = ("q_einsum_1" in joined or "kv_einsum_1" in joined or
                     "attn_vec_einsum_1" in joined or "/mlp_1/" in joined)
        inferred_ranks["expert" if is_expert else "main"] = int(
            np.shape(_leaf_value(leaf))[-1])

    original_get_config = _gemma.get_config

    def get_config_with_composed_rank(variant):
        config = original_get_config(variant)
        if not config.lora_configs:
            return config
        family = "expert" if "300m" in variant else "main"
        rank = inferred_ranks[family]
        if rank is None:
            return config
        lora_configs = {
            name: dataclasses.replace(value, rank=rank, alpha=float(rank))
            for name, value in config.lora_configs.items()
        }
        return dataclasses.replace(config, lora_configs=lora_configs)

    _gemma.get_config = get_config_with_composed_rank
    try:
        model = train_config.model.load(pytree, remove_extra_params=False)
    finally:
        _gemma.get_config = original_get_config
    _graphdef, _state = nnx.split(model)

    # Unlike module_jit, state is an explicit dynamic argument.  This compiles
    # one executable for fixed tensor shapes while observing every candidate's
    # updated LoRA leaves.
    _sample_actions_jit = _make_sample_actions_jit(_graphdef)
    if os.environ.get("PI0_LORA_UPDATE_MODE", "partial") == "partial":
        started = time.perf_counter()
        initial_lora_state = jax.tree.map(jax.device_put, lora_update_pytree)
        _block_until_ready(initial_lora_state)
        _state.replace_by_pure_dict(initial_lora_state)
        _runtime_stats["initial_device_bytes"] = sum(
            int(getattr(value, "nbytes", 0))
            for value in jax.tree.leaves(initial_lora_state))
        _runtime_stats["initial_device_seconds"] = time.perf_counter() - started

    _non_lora_value_ids = _capture_non_lora_value_ids(_state)
    _model_cache = True
    return nnx.merge(_graphdef, _state)


def pi0_forward(params: dict, inputs: list, train_config_name: str,
                norm_stats: dict, diffusion_steps: int = 10,
                max_batch: int = 2, diffusion_seed: int = 42) -> list:
    """Pi0 模型前向推理。

    Args:
        params: 模型参数字典 (含 "__lora_keys__" 标记基座/LoRA 边界)
        inputs: [{"images": {...}, "state", "instruction"}, ...]
        train_config_name: RoboTwin 训练配置名 (如 place_empty_cup_pi0_lora)
        norm_stats: norm_stats 字典 (含 "actions" 等)
        diffusion_steps: 扩散采样步数 (默认 10, 搜索阶段可调 4 提速)
        max_batch: 单次扩散推理的 batch 上限，用于控制 50 步动作窗口的显存峰值
        diffusion_seed: 固定扩散噪声种子；候选间共享以保证公平比较

    Returns:
        [predicted_delta_seq (50, 14), ...]
        模型输出为 normalized delta 动作 (与 openpi 训练 DeltaActions 一致):
        unnormalize 后即为原始 delta 空间, 直接与 data_loader 的 delta GT 对齐计算 loss。
    """
    global _forward_calls
    _forward_calls += 1

    from openpi.models.model import Observation
    from openpi.models.tokenizer import PaligemmaTokenizer

    # --- 1. 分离基座 + composed_lora ---
    lora_keys = params.get("__lora_keys__", set())
    update_lora_keys = params.get("__lora_update_keys__", lora_keys)
    base_p = {k: v for k, v in params.items()
              if k not in lora_keys and not k.startswith("__")}
    composed = {k: params[k] for k in lora_keys}
    _validate_lora_update_specs(composed, update_lora_keys)

    # --- 2. 构建包含 w + lora_a/b 的完整 PyTree ---
    pytree = _build_lora_pytree(base_p, composed)
    lora_update_pytree = _build_lora_update_pytree(composed, update_lora_keys)

    # --- 3. 加载模型 ---
    model = _get_pi0_runtime(pytree, lora_update_pytree, train_config_name)
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = PaligemmaTokenizer(max_len=model.max_token_len)

    # --- 4. OpenPI JAX-JIT 批量预处理 ---
    global _observation_cache, _observation_cache_key
    batch_size = len(inputs)
    sample_ids = tuple(
        int(inp.get("dataset_index", inp.get("frame_index", index)))
        for index, inp in enumerate(inputs))
    cache_key = (sample_ids, batch_size, train_config_name)
    if _observation_cache is None or _observation_cache_key != cache_key:
        img_high_batch, img_left_batch, img_right_batch = [], [], []
        state_batch, prompt_batch, prompt_mask_batch = [], [], []
        for inp in inputs:
            img_high_batch.append(inp["images"]["cam_high"])
            img_left_batch.append(inp["images"]["cam_left_wrist"])
            img_right_batch.append(inp["images"]["cam_right_wrist"])
            state_batch.append(np.array(inp["state"], dtype=np.float32))
            instruction = inp["instruction"]
            cached_prompt = _prompt_cache.get(instruction)
            if cached_prompt is None:
                cached_prompt = _tokenizer.tokenize(instruction)
                _prompt_cache[instruction] = cached_prompt
            prompt_tokens, prompt_mask = cached_prompt
            prompt_batch.append(prompt_tokens.astype(np.int32))
            prompt_mask_batch.append(prompt_mask.astype(np.bool_))

        preprocess_images, normalize_states, _ = _get_data_jit_functions()
        high_images = preprocess_images(jnp.asarray(np.stack(img_high_batch)))
        left_images = preprocess_images(jnp.asarray(np.stack(img_left_batch)))
        right_images = preprocess_images(jnp.asarray(np.stack(img_right_batch)))
        states = jnp.asarray(np.stack(state_batch), dtype=jnp.float32)
        normalization_flags = {
            bool(inp.get("state_is_normalized", True)) for inp in inputs}
        if len(normalization_flags) != 1:
            raise ValueError("同一 batch 中 state 归一化状态不一致")
        if not next(iter(normalization_flags)):
            state_stats = norm_stats["state"]
            states = normalize_states(
                states,
                jnp.asarray(state_stats["mean"], dtype=jnp.float32),
                jnp.asarray(state_stats["std"], dtype=jnp.float32),
            )
        prompts = jnp.asarray(np.stack(prompt_batch), dtype=jnp.int32)
        prompt_masks = jnp.asarray(np.stack(prompt_mask_batch), dtype=jnp.bool_)
        mask = jnp.ones(batch_size, dtype=jnp.bool_)
        _observation_cache = Observation(
            images={
                "base_0_rgb": high_images,
                "left_wrist_0_rgb": left_images,
                "right_wrist_0_rgb": right_images,
            },
            image_masks={"base_0_rgb": mask, "left_wrist_0_rgb": mask,
                         "right_wrist_0_rgb": mask},
            state=states,
            tokenized_prompt=prompts,
            tokenized_prompt_mask=prompt_masks,
        )
        _observation_cache_key = cache_key
        print(f"  [Pi0/JIT] 已缓存 {batch_size} 条预处理 Observation", flush=True)
    obs = _observation_cache

    # --- 6. 推理 (固定 RNG, 使 loss 对权重确定) ---
    # 模型输出 (chunk, action_horizon=50, action_dim=32) 的 normalized delta 动作;
    # unnormalize 后取前 14 维 (delta 空间), 与 few-shot GT 的 50 步 delta 轨迹对齐;
    # 大 batch 分块推理 (chunk<=8), 控制峰值显存 (24GB)。
    predicted = []
    for start in range(0, batch_size, max_batch):
        end = min(start + max_batch, batch_size)
        chunk_obs = jax.tree.map(lambda t: t[start:end], obs)
        # Common random numbers across candidates, but a different deterministic
        # stream for each chunk/sample group.  batch=1 no longer repeats exactly
        # the same diffusion noise for every few-shot sample.
        sample_rng = jax.random.fold_in(
            jax.random.key(int(diffusion_seed)), start)
        if _sample_actions_jit is None:
            raise RuntimeError("Pi0 JIT runtime 尚未初始化")
        acts = _sample_actions_jit(
            _state, sample_rng, chunk_obs, num_steps=diffusion_steps)
        # acts: (chunk, action_horizon, action_dim)
        _, _, unnormalize_actions = _get_data_jit_functions()
        action_stats = norm_stats["actions"]
        acts = unnormalize_actions(
            acts,
            jnp.asarray(action_stats["mean"], dtype=jnp.float32),
            jnp.asarray(action_stats["std"], dtype=jnp.float32),
        )[..., :14]
        acts_host = np.asarray(acts, dtype=np.float32)
        predicted.extend(acts_host)
        del acts, chunk_obs

    # 显式释放 JAX 中间结果，并在新的 LoRA executable 累积到峰值前
    # 周期性释放编译缓存。
    del obs
    import gc; gc.collect()
    if _CACHE_CLEAR_INTERVAL > 0 and _forward_calls % _CACHE_CLEAR_INTERVAL == 0:
        print(f"  [JAX] forward={_forward_calls}, 清理编译缓存")
        jax.clear_caches()
        _runtime_stats["cache_clears"] += 1
    return predicted
