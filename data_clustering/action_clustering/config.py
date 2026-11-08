"""Small JSON configuration with paths relative to the config file."""

import json
from pathlib import Path


DEFAULTS = {
    "seed": 42,
    "window": {"length": 50, "stride": 25},
    "representation": {"mode": "joint", "visual_weight": 0.0},
    "training": {"epochs": 30, "batch_size": 128, "lr": 0.001,
                 "hidden_dim": 256, "latent_dim": 128, "validation_fraction": 0.2,
                 "action_loss_weight": 1.0, "state_loss_weight": 1.0,
                 "stratify_by_dataset": False,
                 "device": "cpu", "threads": 4},
    "clustering": {"k_values": [16, 32, 48], "hdbscan": False,
                   "min_cluster_size": 20, "representatives": 5,
                   "silhouette_sample_size": 2000},
}


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        supplied = json.load(f)
    unknown = set(supplied) - set(DEFAULTS) - {"datasets", "output_dir"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = {}
    for key, default in DEFAULTS.items():
        value = supplied.get(key, {}) if isinstance(default, dict) else supplied.get(key, default)
        if isinstance(default, dict):
            if set(value) - set(default):
                raise ValueError(f"Unknown {key} keys: {set(value) - set(default)}")
            value = {**default, **value}
        cfg[key] = value
    cfg["output_dir"] = str((path.parent / supplied["output_dir"]).resolve())
    cfg["datasets"] = []
    for source in supplied["datasets"]:
        source = dict(source)
        for key in ("root", "visual_cache"):
            if source.get(key):
                source[key] = str((path.parent / source[key]).resolve())
        cfg["datasets"].append(source)
    if not cfg["datasets"]:
        raise ValueError("datasets must not be empty")
    if cfg["representation"]["mode"] not in {"joint", "ee"}:
        raise ValueError("representation.mode must be joint or ee")
    if cfg["representation"]["visual_weight"] < 0:
        raise ValueError("visual_weight must be nonnegative")
    for value in cfg["window"].values():
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Window length and stride must be positive integers")
    tr = cfg["training"]
    if not 0 < tr["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    for key in ("epochs", "batch_size", "hidden_dim", "latent_dim", "threads"):
        if not isinstance(tr[key], int) or tr[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if tr["lr"] <= 0 or min(tr["action_loss_weight"], tr["state_loss_weight"]) < 0:
        raise ValueError("Invalid learning rate or loss weights")
    if tr["action_loss_weight"] + tr["state_loss_weight"] == 0:
        raise ValueError("At least one loss weight must be positive")
    cl = cfg["clustering"]
    if any(not isinstance(k, int) or k < 2 for k in cl["k_values"]):
        raise ValueError("Each K must be an integer >= 2")
    if cl["min_cluster_size"] < 2 or cl["representatives"] < 1 or cl["silhouette_sample_size"] < 3:
        raise ValueError("Invalid clustering settings")
    return cfg


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def fresh_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path
