"""
Few-shot 数据加载: 从 RoboTwin 标准 parquet 数据集读取输入/输出样本。

数据格式 (RoboTwin / lerobot v2.1):
  <data_dir>/data/chunk-000/episode_*.parquet
  <data_dir>/meta/episodes.jsonl   (任务描述)
"""

import io
import json
import glob
import os
from bisect import bisect_right
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image


def load_norm_stats(path: str) -> dict:
    """加载 norm_stats.json, 返回 {"state": {...}, "actions": {...}, ...}"""
    with open(path) as f:
        payload = json.load(f)
    return payload["norm_stats"]


def assert_norm_stats_match(current_path: str, reference_path: str,
                            rtol: float = 1e-6, atol: float = 1e-6) -> None:
    """检查当前任务统计量是否与目标 checkpoint 的统计量一致。

    LoRAHub 的 forward 和最终 RoboTwin 评测必须使用同一套统计量；否则
    模型输出的 unnormalize 结果与训练/评测语义不一致。只比较两边都存在
    的 state/actions 统计字段，并在不一致时立即终止运行。
    """
    current = load_norm_stats(current_path)
    reference = load_norm_stats(reference_path)
    max_abs_diff = 0.0
    checked = []
    for group in ("state", "actions"):
        if group not in current or group not in reference:
            raise ValueError(f"norm_stats 缺少分组: {group}")
        for field in ("mean", "std", "q01", "q99"):
            if field not in current[group] or field not in reference[group]:
                continue
            lhs = np.asarray(current[group][field], dtype=np.float32)
            rhs = np.asarray(reference[group][field], dtype=np.float32)
            if lhs.shape != rhs.shape:
                raise ValueError(
                    f"norm_stats shape 不一致: {group}.{field}: "
                    f"{lhs.shape} != {rhs.shape}")
            max_abs_diff = max(max_abs_diff, float(np.max(np.abs(lhs - rhs))))
            try:
                np.testing.assert_allclose(lhs, rhs, rtol=rtol, atol=atol)
            except AssertionError as exc:
                raise ValueError(
                    f"norm_stats 与目标 checkpoint 不一致: {group}.{field}; "
                    f"max_abs_diff={max_abs_diff:.3e}\n{exc}") from exc
            checked.append(f"{group}.{field}")
    print(f"  norm_stats 校验通过: {current_path} == {reference_path} "
          f"(fields={len(checked)}, max_abs_diff={max_abs_diff:.3e})")


def compute_norm_stats(data_dir: str, state_dim: int = 32, action_dim: int = 32,
                       action_horizon: int = 50) -> dict:
    """按 openpi RunningStats 方式从数据集全量帧重算归一化统计。

    openpi 计算方式 (src/openpi/shared/normalize.py):
      mean = E[x];  std = sqrt(E[x^2] - E[x]^2)  (ddof=0, 与 openpi 一致);
      同时给出 q01/q99 分位 (openpi 落盘格式必需字段)。
    数据来源: 数据集全部 episode 的全部帧 (非仅中间帧), state/actions 分开统计。

    Returns:
        {"state": {"mean", "std", "q01", "q99"}, "actions": {...}}  (均补齐到 32 维)
    """
    ep_dir = os.path.join(data_dir, "data", "chunk-000")
    ep_paths = sorted(glob.glob(os.path.join(ep_dir, "episode_*.parquet")))
    if not ep_paths:
        raise FileNotFoundError(f"未找到数据: {ep_dir}")

    episode_states_list, episode_actions_list = [], []
    for p in ep_paths:
        df = pd.read_parquet(p)
        # parquet 中 state/action 为嵌套列 (每行一个数组), 需逐行 stack
        episode_states_list.append(
            np.stack(df["observation.state"].to_numpy()).astype(np.float32))
        episode_actions_list.append(
            np.stack(df["action"].to_numpy()).astype(np.float32))
    # OpenPI computes action statistics after AlohaInputs (pad to model dim) and
    # DeltaActions.  Each dataset item contains a complete future action chunk,
    # so statistics must cover all frame×horizon delta targets rather than the
    # raw per-frame absolute action column.
    action_windows = []
    for episode_states, episode_actions in zip(
            episode_states_list, episode_actions_list):
        action_windows.append(make_delta_action_windows(
            episode_states, episode_actions,
            action_horizon=action_horizon,
            action_dim=min(action_dim, episode_actions.shape[1]),
        ))
    actions = np.concatenate(action_windows, axis=0).reshape(-1, action_windows[0].shape[-1])
    states = np.concatenate(episode_states_list, axis=0)

    def _stats(x: np.ndarray) -> dict:
        mean = x.mean(axis=0).astype(np.float32)
        var = x.var(axis=0)  # ddof=0, 与 openpi RunningStats 一致
        std = np.sqrt(np.maximum(0.0, var)).astype(np.float32)
        q01 = np.percentile(x, 1, axis=0).astype(np.float32)
        q99 = np.percentile(x, 99, axis=0).astype(np.float32)
        return {"mean": mean, "std": std, "q01": q01, "q99": q99}

    def _pad(stats: dict, dim: int) -> dict:
        out = {}
        for k, v in stats.items():
            padded = np.zeros(dim, dtype=np.float32)
            padded[:len(v)] = v
            out[k] = padded
        return out

    print(f"  统计来源: {len(ep_paths)} 条 episode, {states.shape[0]} 帧, "
          f"state {states.shape[1]}D / delta action "
          f"{actions.shape[1]}D×{action_horizon} horizon")
    return {"state": _pad(_stats(states), state_dim),
            "actions": _pad(_stats(actions), action_dim)}


def save_norm_stats(path: str, norm_stats: dict) -> None:
    """按 openpi 落盘格式保存: {"norm_stats": {"state": {...}, "actions": {...}}}."""
    def _to_list(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _to_list(v) for k, v in obj.items()}
        return obj

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"norm_stats": _to_list(norm_stats)}, f, indent=2)
    print(f"  已保存 norm_stats → {path}")


def load_episode_tasks(episodes_path: str) -> list:
    """从 episodes.jsonl 读取每条 episode 的任务描述 (取第一条 task)."""
    tasks = []
    with open(episodes_path) as f:
        for line in f:
            record = json.loads(line)
            tasks.append(record["tasks"][0])
    return tasks


def decode_image_bytes(image_field) -> np.ndarray:
    """解码 parquet 中的图像 bytes (兼容 dict 包裹格式)."""
    if isinstance(image_field, dict):
        image_field = image_field["bytes"]
    return np.array(Image.open(io.BytesIO(image_field)))


def normalize_vector(x: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return (x - mean) / (std + 1e-6)


def unnormalize_vector(x: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return x * (std + 1e-6) + mean


def make_delta_mask(action_dim: int) -> np.ndarray:
    """Return the RoboTwin/Pi0 delta mask for a dual-arm action vector."""
    mask = np.ones(action_dim, dtype=bool)
    if action_dim >= 14:
        mask[[6, 13]] = False
    return mask


def make_padded_action_indices(frame_index: int, episode_length: int,
                               action_horizon: int) -> np.ndarray:
    """OpenPI/LeRobot future indices, clipped to the current episode end."""
    if episode_length <= 0 or action_horizon <= 0:
        raise ValueError("episode_length 和 action_horizon 必须 > 0")
    if frame_index < 0 or frame_index >= episode_length:
        raise IndexError(f"frame_index 越界: {frame_index}/{episode_length}")
    return np.minimum(
        frame_index + np.arange(action_horizon, dtype=np.int64),
        episode_length - 1,
    )


def make_delta_action_windows(states: np.ndarray, actions: np.ndarray, *,
                              action_horizon: int = 50,
                              action_dim: int = 14) -> np.ndarray:
    """Build OpenPI/LeRobot frame×horizon targets after DeltaActions.

    Returns shape ``(num_frames, action_horizon, action_dim)``.  Future queries
    are clipped to the episode boundary and every joint delta is anchored to
    the state at the current observation frame, exactly like OpenPI's
    ``DeltaActions`` transform on a LeRobot action chunk.
    """
    states = np.asarray(states, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    if states.ndim != 2 or actions.ndim != 2 or len(states) != len(actions):
        raise ValueError("states/actions 必须是等长二维数组")
    dim = min(int(action_dim), states.shape[1], actions.shape[1])
    offsets = np.arange(int(action_horizon), dtype=np.int64)[None, :]
    query = np.minimum(
        np.arange(len(actions), dtype=np.int64)[:, None] + offsets,
        len(actions) - 1,
    )
    chunks = actions[query, :dim].copy()
    delta_mask = make_delta_mask(dim)
    chunks[..., delta_mask] -= states[:, None, :dim][..., delta_mask]
    return chunks


def select_uniform_frame_starts(episode_length: int, action_horizon: int,
                                frames_per_ep: int) -> np.ndarray:
    """Evenly sample valid starts whose complete action chunk fits in an episode."""
    episode_length = int(episode_length)
    action_horizon = int(action_horizon)
    frames_per_ep = int(frames_per_ep)
    if frames_per_ep <= 0:
        raise ValueError(f"frames_per_ep 必须 > 0, got {frames_per_ep}")
    if action_horizon <= 0:
        raise ValueError(f"action_horizon 必须 > 0, got {action_horizon}")
    if episode_length < action_horizon:
        raise ValueError(
            f"episode 帧数 {episode_length} < action_horizon {action_horizon}")
    max_start = episode_length - action_horizon
    count = min(frames_per_ep, max_start + 1)
    return np.unique(
        np.rint(np.linspace(0, max_start, count)).astype(np.int32))


def select_keyframe_starts(states: np.ndarray, actions: np.ndarray,
                           action_horizon: int, frames_per_ep: int,
                           strategy: str = "motion_stratified") -> np.ndarray:
    """Select complete Pi0 action windows from one episode.

    ``motion_stratified`` splits the valid episode range into temporal phases and
    picks the most active window in each phase.  This keeps phase coverage while
    avoiding the old ``frames_per_ep=1 -> frame 0`` behaviour.
    """
    if frames_per_ep <= 0:
        raise ValueError(f"frames_per_ep 必须 > 0, got {frames_per_ep}")
    if len(actions) < action_horizon:
        raise ValueError(
            f"episode 帧数 {len(actions)} < action_horizon {action_horizon}")

    max_start = len(actions) - action_horizon
    valid = np.arange(max_start + 1, dtype=np.int32)
    count = min(frames_per_ep, len(valid))
    if strategy == "uniform":
        return select_uniform_frame_starts(
            len(actions), action_horizon, frames_per_ep)
    if strategy != "motion_stratified":
        raise ValueError(f"未知 keyframe_strategy: {strategy}")

    action_dim = min(actions.shape[1], states.shape[1])
    delta_mask = make_delta_mask(action_dim)
    scores = np.zeros(len(valid), dtype=np.float32)
    activity_horizon = min(10, action_horizon)
    for start in valid:
        # Measure local movement rather than absolute pose displacement.  The
        # first difference is relative to the current observed state, exactly
        # like the Pi0 DeltaActions transform.
        chunk = actions[start:start + activity_horizon, :action_dim]
        anchor = states[start, :action_dim][None, :]
        steps = np.diff(np.concatenate([anchor, chunk], axis=0), axis=0)
        joint_activity = np.linalg.norm(steps[:, delta_mask], axis=1).mean()
        gripper_activity = (np.abs(steps[:, ~delta_mask]).mean()
                            if np.any(~delta_mask) else 0.0)
        scores[start] = joint_activity + 0.25 * gripper_activity

    starts = []
    for phase in np.array_split(valid, count):
        # Ignore the outer 20% of each phase when possible.  Otherwise two
        # neighbouring phases can select adjacent boundary frames for the same
        # future motion event.
        margin = int(len(phase) * 0.2)
        candidates = phase[margin:len(phase) - margin] if margin else phase
        if len(candidates) == 0:
            candidates = phase
        # argmax is deterministic and chooses the earlier frame on exact ties.
        starts.append(int(candidates[np.argmax(scores[candidates])]))
    return np.asarray(starts, dtype=np.int32)


def compute_delta_loss_stats(outputs: list[np.ndarray],
                             std_floor: float = 1e-3) -> dict:
    """Compute objective-only statistics from the actual delta GT windows.

    These values must not replace checkpoint ``norm_stats``.  They only balance
    dimensions in the black-box objective after model output is unnormalized.
    """
    if not outputs:
        raise ValueError("无法从空 outputs 计算 delta loss stats")
    values = np.concatenate([np.asarray(x, dtype=np.float32) for x in outputs], axis=0)
    mean = values.mean(axis=0).astype(np.float32)
    std = values.std(axis=0).astype(np.float32)
    scale = np.maximum(std, float(std_floor)).astype(np.float32)
    active = (std >= float(std_floor)).astype(bool)
    return {
        "mean": mean,
        "std": std,
        "scale": scale,
        "active": active,
        "std_floor": float(std_floor),
        "num_values": int(values.shape[0]),
    }


def save_delta_loss_stats(path: str, stats: dict) -> None:
    """Save objective-only delta statistics as JSON."""
    payload = {}
    for key, value in stats.items():
        payload[key] = value.tolist() if isinstance(value, np.ndarray) else value
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  已保存 delta loss stats → {path}")


def load_few_shot_data(data_dir: str, norm_stats_path: str,
                       state_dim: int = 32, action_dim: int = 14,
                       frames_per_ep: int = 1, action_horizon: int = 50,
                       keyframe_strategy: str = "motion_stratified") -> tuple:
    """加载 few-shot 样本: 每条 episode 取多个关键帧, 每个样本带完整动作轨迹 GT。

    Args:
        data_dir: 数据集根目录 (含 data/ 与 meta/ 子目录)
        norm_stats_path: norm_stats.json 路径
        state_dim: 模型输入 state 维数 (不足补零)
        action_dim: 模型输出 action 维数 (截断)
        frames_per_ep: 每条 episode 采样的起始帧数
        action_horizon: GT 动作序列长度 (与 Pi0 模型采样轨迹对齐, 默认 50)
        keyframe_strategy: ``motion_stratified`` 或兼容旧行为的 ``uniform``

    Returns:
        (inputs, outputs)
        inputs: [{"images": {...}, "state", "instruction"}, ...]
        outputs: [delta_action_seq (action_horizon, action_dim), ...]

    注意: 输出为 delta 动作空间 (与 openpi 训练 DeltaActions 一致):
      - 关节维 (索引 0-5, 7-12) 转 delta = action - state
      - gripper 维 (索引 6, 13) 保持绝对
      模型 sample_actions 输出同样为 delta 空间, 两者直接对齐计算 loss。
    """
    ep_dir = os.path.join(data_dir, "data", "chunk-000")
    ep_paths = sorted(glob.glob(os.path.join(ep_dir, "episode_*.parquet")))
    n_eps = len(ep_paths)
    print(f"  数据目录: {data_dir}, 共 {n_eps} 条, "
          f"每条约取 {frames_per_ep} 帧, GT 轨迹 {action_horizon} 步")

    norm_stats = load_norm_stats(norm_stats_path)
    episode_tasks = load_episode_tasks(os.path.join(data_dir, "meta", "episodes.jsonl"))
    assert len(episode_tasks) >= n_eps, \
        f"episodes.jsonl 任务数 {len(episode_tasks)} < 数据条数 {n_eps}"

    inputs, outputs = [], []
    for ep in range(n_eps):
        df = pd.read_parquet(ep_paths[ep])
        n_frames = len(df)
        actions_all = np.stack(df["action"].to_numpy()).astype(np.float32)
        states_all = np.stack(df["observation.state"].to_numpy()).astype(np.float32)

        starts = select_keyframe_starts(
            states_all, actions_all, action_horizon, frames_per_ep,
            strategy=keyframe_strategy)

        for i in starts:
            row = df.iloc[i]
            img_high = decode_image_bytes(row["observation.images.cam_high"])
            img_left = decode_image_bytes(row["observation.images.cam_left_wrist"])
            img_right = decode_image_bytes(row["observation.images.cam_right_wrist"])

            state = np.array(row["observation.state"], dtype=np.float32)
            state_pad = np.zeros(state_dim, dtype=np.float32)
            state_pad[:len(state)] = state
            state_pad = normalize_vector(state_pad, norm_stats["state"])

            # GT 动作: 与 openpi 训练一致转为 delta 空间。
            # DeltaActions mask 与训练 make_bool_mask(6, -1, 6, -1) 一致:
            #   关节维 (0-5, 7-12) 用未归一化 state 转 delta, gripper (6, 13) 保持绝对。
            gt_seq = actions_all[i:i + action_horizon, :action_dim].copy()
            delta_mask = make_delta_mask(action_dim)
            gt_seq[:, delta_mask] -= state[:action_dim][delta_mask]

            inputs.append({
                "images": {
                    "cam_high": img_high,
                    "cam_left_wrist": img_left,
                    "cam_right_wrist": img_right,
                },
                "state": state_pad,
                "state_is_normalized": True,
                "instruction": episode_tasks[ep],
                "episode_index": ep,
                "frame_index": int(i),
            })
            outputs.append(gt_seq)

            print(
                f"  Episode {ep} frame {i:3d}: "
                f"cam_high={img_high.shape}, cam_left={img_left.shape}, cam_right={img_right.shape}, "
                f"state={state.shape}->{state_pad.shape}, GT={gt_seq.shape}, prompt={episode_tasks[ep]}"
            )
    return inputs, outputs


class Pi0FrameDataset:
    """Lazy frame-level dataset matching OpenPI/LeRobot sampling semantics.

    Every frame in every episode is a valid current observation.  Future action
    indices are clipped to the final frame of the same episode, matching
    LeRobotDataset's episode-boundary padding.  Images are decoded only for the
    sampled mini-batch; episode parquet tables are held in a small LRU cache.
    """

    def __init__(self, data_dir: str, norm_stats_path: str, *,
                 state_dim: int = 32, action_dim: int = 14,
                 action_horizon: int = 50, episode_cache_size: int = 10):
        self.data_dir = os.fspath(data_dir)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.episode_cache_size = max(1, int(episode_cache_size))
        self.norm_stats = load_norm_stats(norm_stats_path)

        ep_dir = os.path.join(self.data_dir, "data", "chunk-000")
        self.episode_paths = sorted(
            glob.glob(os.path.join(ep_dir, "episode_*.parquet")))
        if not self.episode_paths:
            raise FileNotFoundError(f"未找到数据: {ep_dir}")
        self.episode_tasks = load_episode_tasks(
            os.path.join(self.data_dir, "meta", "episodes.jsonl"))
        if len(self.episode_tasks) < len(self.episode_paths):
            raise ValueError(
                f"episodes.jsonl 任务数 {len(self.episode_tasks)} < "
                f"数据条数 {len(self.episode_paths)}")

        # Reading parquet metadata avoids decoding all images during startup.
        import pyarrow.parquet as pq
        self.episode_lengths = [
            int(pq.ParquetFile(path).metadata.num_rows)
            for path in self.episode_paths
        ]
        self.cumulative_lengths = np.cumsum(
            self.episode_lengths, dtype=np.int64)
        self._episode_cache = OrderedDict()

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1])

    def resolve_index(self, dataset_index: int) -> tuple[int, int]:
        index = int(dataset_index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(f"dataset index 越界: {dataset_index}")
        episode_index = bisect_right(self.cumulative_lengths, index)
        previous = (0 if episode_index == 0 else
                    int(self.cumulative_lengths[episode_index - 1]))
        return episode_index, index - previous

    def _load_episode(self, episode_index: int) -> pd.DataFrame:
        cached = self._episode_cache.pop(episode_index, None)
        if cached is None:
            cached = pd.read_parquet(self.episode_paths[episode_index])
        self._episode_cache[episode_index] = cached
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return cached

    def _build_sample(self, dataset_index: int) -> tuple[dict, np.ndarray]:
        item, target, _ = self._build_sample_with_mask(dataset_index)
        return item, target

    def _build_sample_with_mask(
            self, dataset_index: int) -> tuple[dict, np.ndarray, np.ndarray]:
        episode_index, frame_index = self.resolve_index(dataset_index)
        df = self._load_episode(episode_index)
        row = df.iloc[frame_index]

        state = np.asarray(row["observation.state"], dtype=np.float32)
        state_pad = np.zeros(self.state_dim, dtype=np.float32)
        state_pad[:min(len(state), self.state_dim)] = state[:self.state_dim]

        # LeRobot uses the final valid episode index for out-of-bound queries.
        query_indices = make_padded_action_indices(
            frame_index, len(df), self.action_horizon)
        valid_steps = np.arange(self.action_horizon) < (len(df) - frame_index)
        gt_seq = np.stack(
            df["action"].iloc[query_indices].to_numpy()).astype(np.float32)
        gt_seq = gt_seq[:, :self.action_dim].copy()
        delta_mask = make_delta_mask(self.action_dim)
        gt_seq[:, delta_mask] -= state[:self.action_dim][delta_mask]

        item = {
            "images": {
                "cam_high": decode_image_bytes(
                    row["observation.images.cam_high"]),
                "cam_left_wrist": decode_image_bytes(
                    row["observation.images.cam_left_wrist"]),
                "cam_right_wrist": decode_image_bytes(
                    row["observation.images.cam_right_wrist"]),
            },
            # Keep the padded state raw.  pi0_forward applies the exact OpenPI
            # z-score transform to the complete batch in JAX.
            "state": state_pad,
            "state_is_normalized": False,
            "instruction": self.episode_tasks[episode_index],
            "episode_index": episode_index,
            "frame_index": frame_index,
            "dataset_index": int(dataset_index),
        }
        return item, gt_seq, valid_steps.astype(np.float32)

    def get_batch(self, dataset_indices, *, return_action_mask: bool = False):
        """Load a mini-batch, optionally returning valid future-step masks.

        The mask is one for real future actions and zero for positions that were
        clipped to the final episode action to keep the horizon shape fixed.
        Existing callers retain the historical two-value return signature.
        """
        inputs, outputs, action_masks = [], [], []
        for index in np.asarray(dataset_indices, dtype=np.int64).reshape(-1):
            item, target, action_mask = self._build_sample_with_mask(int(index))
            inputs.append(item)
            outputs.append(target)
            action_masks.append(action_mask)
        if return_action_mask:
            return inputs, outputs, np.stack(action_masks).astype(np.float32)
        return inputs, outputs

    def build_frame_pool(self, *, gripper_delta_threshold: float = 0.005,
                         static_stride: int = 2) -> np.ndarray:
        """按 gripper 活跃度构造评估帧池。

        抓取/释放阶段的帧 (gripper 逐帧变化超过 ``gripper_delta_threshold``)
        全部保留; gripper 准静止段的帧其未来动作高度冗余, 按 ``static_stride``
        间隔抽取。每条 episode 的首末帧始终保留, 避免丢失段边界。
        """
        gripper_indices = [i for i in (6, 13) if i < self.action_dim]
        if not gripper_indices:
            return np.arange(len(self), dtype=np.int64)
        stride = max(1, int(static_stride))
        pool: list[int] = []
        base = 0
        for episode_index, length in enumerate(self.episode_lengths):
            df = self._load_episode(episode_index)
            actions = np.stack(df["action"].to_numpy()).astype(np.float32)
            keep = np.zeros(length, dtype=bool)
            if length > 1:
                gripper = actions[:, gripper_indices]
                diff = np.abs(np.diff(gripper, axis=0)).max(axis=1)
                activity = np.zeros(length, dtype=np.float32)
                activity[1:] = diff
                activity[:-1] = np.maximum(activity[:-1], diff)
                active = activity > float(gripper_delta_threshold)
            else:
                active = np.ones(length, dtype=bool)
            keep[active] = True
            static = np.flatnonzero(~active)
            keep[static[::stride] if stride > 1 else static] = True
            keep[0] = keep[-1] = True
            pool.extend((base + np.flatnonzero(keep)).tolist())
            base += length
        return np.asarray(pool, dtype=np.int64)

    def all_delta_outputs(self) -> list[np.ndarray]:
        """Build objective statistics over every Pi0 frame without decoding images."""
        outputs = []
        for episode_index, length in enumerate(self.episode_lengths):
            df = self._load_episode(episode_index)
            actions = np.stack(df["action"].to_numpy()).astype(np.float32)
            states = np.stack(
                df["observation.state"].to_numpy()).astype(np.float32)
            chunks = make_delta_action_windows(
                states, actions,
                action_horizon=self.action_horizon,
                action_dim=self.action_dim,
            )
            outputs.extend(chunks)
        return outputs

    def manifest(self) -> list[dict]:
        result = []
        global_index = 0
        for episode_index, length in enumerate(self.episode_lengths):
            for frame_index in range(length):
                result.append({
                    "dataset_index": global_index,
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "instruction": self.episode_tasks[episode_index],
                })
                global_index += 1
        return result


class Pi0RandomBatchSampler:
    """Deterministic mini-batches with common batches per CMA generation.

    Sharing one batch across a complete CMA population gives every candidate the
    same objective.  Two batch orders are supported:
      - ``random``: an independent random subset per generation (legacy).
      - ``sequential``: walk the frame pool in index order, so every frame is
        evaluated exactly once per epoch.  The trailing batch is padded with the
        last pool frame, keeping the batch shape constant (no JAX recompilation)
        and making every epoch an identical frame set.

    ``frame_pool`` optionally restricts the sequential walk to a precomputed
    subset of dataset indices (e.g. gripper-activity weighted frames).
    """

    def __init__(self, dataset: Pi0FrameDataset, *, batch_size: int,
                 batches_per_evaluation: int = 1, reuse_candidates: int = 1,
                 final_evaluation_batches: int = 2, seed: int = 42,
                 order: str = "random", frame_pool=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.batches_per_evaluation = int(batches_per_evaluation)
        self.reuse_candidates = max(1, int(reuse_candidates))
        self.final_evaluation_batches = max(1, int(final_evaluation_batches))
        self.seed = int(seed)
        self.order = str(order)
        if self.order not in ("random", "sequential"):
            raise ValueError(f"未知 batch_sampling_order: {self.order}")
        if self.batch_size <= 0 or self.batches_per_evaluation <= 0:
            raise ValueError("batch_size 和 batches_per_evaluation 必须 > 0")
        self.frame_pool = (None if frame_pool is None else
                           np.asarray(frame_pool, dtype=np.int64).reshape(-1))
        if self.frame_pool is not None:
            if self.order != "sequential":
                raise ValueError("frame_pool 仅支持 sequential 顺序采样")
            if len(self.frame_pool) == 0:
                raise ValueError("frame_pool 不能为空")
        self._generation = None
        self._cached_batch = None
        self._cached_info = None
        self._final_batch = None
        self._final_info = None

    @property
    def samples_per_evaluation(self) -> int:
        return self.batch_size * self.batches_per_evaluation

    @property
    def pool_size(self) -> int:
        """顺序模式实际遍历的帧数 (启用帧池时为池大小)。"""
        if self.frame_pool is None:
            return len(self.dataset)
        return int(len(self.frame_pool))

    @property
    def batches_per_epoch(self) -> int:
        return -(-self.pool_size // self.samples_per_evaluation)

    def _draw(self, generation: int, count: int, *, final: bool = False):
        if self.order == "sequential":
            # 顺序模式: 每 count 帧组成一批, 每 epoch 从池首帧开始按序推进。
            # 尾部不足一批时用池末帧补齐, 不跨 epoch 回绕, 因此每个 epoch 的
            # 帧集完全一致(跨 epoch 同 batch 位置的 loss 可直接比较)。
            batch_index = 0 if final else int(generation)
            per_epoch = self.batches_per_epoch
            epoch = batch_index // per_epoch
            start = (batch_index % per_epoch) * count
            pool = (np.arange(len(self.dataset), dtype=np.int64)
                    if self.frame_pool is None else self.frame_pool)
            tail = min(start + count, self.pool_size)
            window = pool[start:tail]
            indices = np.concatenate([
                window,
                np.full(count - len(window), pool[-1], dtype=np.int64),
            ])
        else:
            batch_index = int(generation)
            epoch = None
            start = None
            salt = 10_000_019 if final else 0
            rng = np.random.default_rng(self.seed + salt + int(generation))
            replace = count > len(self.dataset)
            indices = rng.choice(len(self.dataset), size=count, replace=replace)
        inputs, outputs = self.dataset.get_batch(indices)
        info = {
            "generation": int(generation),
            "order": self.order,
            "batch_index": int(batch_index),
            "epoch": (None if epoch is None else int(epoch)),
            "dataset_indices": [int(x) for x in indices],
            "num_samples": int(count),
            "final": bool(final),
            "pool_index": None if start is None else int(start),
            "pool_size": self.pool_size if start is not None else None,
        }
        return inputs, outputs, info

    def _describe(self, info: dict) -> str:
        if self.order == "sequential":
            start = info["pool_index"]
            end = min(start + info["num_samples"], info["pool_size"])
            pad = start + info["num_samples"] - end
            position = info["batch_index"] % self.batches_per_epoch
            pad_text = f"+池末帧x{pad}" if pad else ""
            return (f"epoch={info['epoch']} batch={position}/"
                    f"{self.batches_per_epoch} 帧池[{start}..{end}){pad_text} "
                    f"首帧号={info['dataset_indices'][0]}")
        return f"随机样本={info['num_samples']}, indices={info['dataset_indices']}"

    def get_batch(self, evaluation_index: int):
        generation = int(evaluation_index) // self.reuse_candidates
        if generation != self._generation:
            count = self.samples_per_evaluation
            self._cached_batch = self._draw(generation, count)
            self._cached_info = self._cached_batch[2]
            self._generation = generation
            print(
                f"  [Data] generation={generation} {self._describe(self._cached_info)}",
                flush=True)
        return self._cached_batch

    def get_final_batch(self):
        if self._final_batch is None:
            count = self.batch_size * self.final_evaluation_batches
            self._final_batch = self._draw(0, count, final=True)
            self._final_info = self._final_batch[2]
            print(
                f"  [Data] final validation {self._describe(self._final_info)}",
                flush=True)
        return self._final_batch
