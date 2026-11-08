"""Train-only normalization, validation checkpoint selection and batch embedding."""

import hashlib
import json
import random
from time import monotonic
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import fresh_dir, write_json
from .data import read_index
from .model import ActionRepresentation
from .progress import Progress, duration


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metadata(cfg):
    folder = Path(cfg["output_dir"]) / "windows"
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    for key in ("seed", "window", "representation", "datasets"):
        if meta["config"][key] != cfg[key]:
            raise ValueError(f"Prepared data/config mismatch: {key}; use a new output_dir")
    for key in ("validation_fraction", "stratify_by_dataset"):
        if meta["config"]["training"].get(key, False) != cfg["training"][key]:
            raise ValueError(f"{key} changed after preparation")
    digest = hashlib.sha256((folder / "index.jsonl").read_bytes()).hexdigest()
    if digest != meta["index_sha256"]:
        raise ValueError("Window index has changed since preparation")
    return folder, meta


def fit_statistics(folder, ids):
    stats = {}
    for name in ("action", "state", "visual"):
        if not (folder / f"{name}.npy").exists():
            continue
        array = np.load(folder / f"{name}.npy", mmap_mode="r")
        count = 0
        mean = np.zeros(array.shape[-1], dtype=np.float64)
        m2 = np.zeros_like(mean)
        # Parallel variance merge is stable even for near-constant channels.
        for start in range(0, len(ids), 256):
            batch = np.asarray(array[ids[start:start + 256]], dtype=np.float64).reshape(-1, array.shape[-1])
            n = len(batch)
            batch_mean = batch.mean(axis=0)
            delta = batch_mean - mean
            new_count = count + n
            m2 += np.square(batch - batch_mean).sum(axis=0) + delta**2 * count * n / new_count
            mean += delta * n / new_count
            count = new_count
        stats[name + "_mean"] = mean.astype(np.float32)
        stats[name + "_std"] = np.maximum(np.sqrt(m2 / count), 1e-4).astype(np.float32)
    return stats


class WindowDataset(Dataset):
    def __init__(self, folder, ids, stats):
        self.ids, self.stats = np.asarray(ids), stats
        self.arrays = {k: np.load(folder / f"{k}.npy", mmap_mode="r")
                       for k in ("action", "state", "visual") if (folder / f"{k}.npy").exists()}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        row = int(self.ids[idx])
        return tuple(torch.from_numpy(np.asarray(
            (self.arrays[k][row] - self.stats[k + "_mean"]) / self.stats[k + "_std"], dtype=np.float32))
            if k in self.arrays else torch.empty(0) for k in ("action", "state", "visual"))


def fit(cfg):
    folder, meta = metadata(cfg)
    rows = read_index(folder)
    ids = {split: np.array([r["window_id"] for r in rows if r["split"] == split])
           for split in ("train", "validation")}
    if any(len(x) == 0 for x in ids.values()):
        raise ValueError("Both episode splits must contain windows")
    tr = cfg["training"]
    seed_all(cfg["seed"], tr["threads"])
    stats = fit_statistics(folder, ids["train"])
    device = torch.device(tr["device"])
    args = {"length": meta["window_length"], "action_dim": meta["action_dim"],
            "state_dim": meta["state_dim"], "visual_dim": meta["visual_dim"],
            "hidden_dim": tr["hidden_dim"], "latent_dim": tr["latent_dim"],
            "visual_weight": cfg["representation"]["visual_weight"]}
    model = ActionRepresentation(**args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"])
    loaders = {split: DataLoader(WindowDataset(folder, split_ids, stats),
                                batch_size=tr["batch_size"], shuffle=split == "train")
               for split, split_ids in ids.items()}
    out = fresh_dir(Path(cfg["output_dir"]) / "training")
    np.savez(out / "normalization.npz", **stats)
    write_json(out / "config.json", cfg)
    best, history = float("inf"), []
    progress = Progress("训练（含验证）", tr["epochs"] * sum(len(loader) for loader in loaders.values()), "批次")
    batches = 0
    for epoch in range(1, tr["epochs"] + 1):
        epoch_started = monotonic()
        record = {"epoch": epoch}
        for split, loader in loaders.items():
            training = split == "train"
            model.train(training)
            sums = np.zeros(3)
            for batch in loader:
                action, state, visual = [x.to(device) for x in batch]
                with torch.set_grad_enabled(training):
                    _, pred_action, pred_state = model(action, state, visual)
                    action_loss = torch.mean((pred_action - action)**2)
                    state_loss = torch.mean((pred_state - state)**2)
                    loss = tr["action_loss_weight"] * action_loss + tr["state_loss_weight"] * state_loss
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite loss; inspect units/normalization and learning rate")
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                sums += np.array([loss.item(), action_loss.item(), state_loss.item()]) * len(action)
                batches += 1
                progress.update(batches)
            for name, value in zip(("loss", "action_loss", "state_loss"), sums / len(loader.dataset)):
                record[split + "_" + name] = float(value)
        history.append(record)
        if record["validation_loss"] < best:
            best = record["validation_loss"]
            torch.save({"model": model.state_dict(), "model_args": args,
                        "index_sha256": meta["index_sha256"], "epoch": epoch,
                        "validation_loss": best}, out / "best.pt")
        write_json(out / "history.json", history)
        print(f"Epoch {epoch:03d}/{tr['epochs']:03d}: train={record['train_loss']:.5f} "
              f"validation={record['validation_loss']:.5f} | 本轮耗时 {duration(monotonic() - epoch_started)}", flush=True)
        progress.update(batches, force=True)
    return history


def embed(cfg):
    folder, meta = metadata(cfg)
    seed_all(cfg["seed"], cfg["training"]["threads"])
    training = Path(cfg["output_dir"]) / "training"
    device = torch.device(cfg["training"]["device"])
    checkpoint = torch.load(training / "best.pt", map_location=device, weights_only=True)
    if checkpoint["index_sha256"] != meta["index_sha256"]:
        raise ValueError("Checkpoint does not match the prepared window index")
    with np.load(training / "normalization.npz", allow_pickle=False) as f:
        stats = dict(f)
    model = ActionRepresentation(**checkpoint["model_args"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    loader = DataLoader(WindowDataset(folder, np.arange(meta["window_count"]), stats),
                        batch_size=cfg["training"]["batch_size"], shuffle=False)
    out = fresh_dir(Path(cfg["output_dir"]) / "embeddings")
    z = np.lib.format.open_memmap(out / "z.npy", mode="w+", dtype="float32",
                                shape=(meta["window_count"], checkpoint["model_args"]["latent_dim"]))
    start = 0
    progress = Progress("提取潜在表示", meta["window_count"], "窗口")
    with torch.inference_mode():
        for batch in loader:
            encoded = model.encode(*[x.to(device) for x in batch]).cpu().numpy()
            if not np.isfinite(encoded).all() or not np.allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-5):
                raise ValueError("Latent vectors must be finite and unit-normalized")
            z[start:start + len(encoded)] = encoded
            start += len(encoded)
            progress.update(start)
    z.flush()
    write_json(out / "metadata.json", {"index_sha256": meta["index_sha256"],
                                      "checkpoint_epoch": checkpoint["epoch"], "shape": list(z.shape)})
    print(f"Encoded {len(z)} normalized latent vectors -> {out}")
