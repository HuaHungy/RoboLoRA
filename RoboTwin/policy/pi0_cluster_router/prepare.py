"""Extract seven trained cluster LoRAs, build the routed bank, and stage assets."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from common import CLUSTERS, CONFIG_NAME, HERE, JOINT_REPO, build_config, read_settings, setup, write_json


def bank_key(source_key):
    key = source_key[7:] if source_key.startswith("params/") else source_key
    if key.endswith("/value"):
        key = key[:-6]
    if "lora_a" in key:
        return key.replace("lora_a", "lora_bank_a")
    if "lora_b" in key:
        return key.replace("lora_b", "lora_bank_b")
    raise ValueError(f"Not a LoRA A/B key: {source_key}")


def pair_count(keys):
    normalized = {key[7:] if key.startswith("params/") else key for key in keys}
    a_keys = {key for key in normalized if "lora_a" in key}
    for key in a_keys:
        if key.replace("lora_a", "lora_b") not in normalized:
            raise ValueError(f"Missing B partner for {key}")
    if len(normalized) != 2 * len(a_keys):
        raise ValueError("LoRA module contains unpaired leaves")
    return len(a_keys)


def extract_modules(settings):
    repo_root = HERE.parents[2]
    sys.path.insert(0, str(repo_root / "LoRAHub-JAX"))
    from vla_lorahub.checkpoint_utils import extract_lora_from_checkpoint

    summary_path = Path(settings["cluster_training_summary"])
    summary = json.loads(summary_path.read_text())
    if not summary.get("all_configured_clusters_complete"):
        raise ValueError(f"Cluster training is not complete: {summary_path}")
    checkpoints = summary.get("completed_checkpoints", {})
    if set(map(int, checkpoints)) != set(CLUSTERS):
        raise ValueError("Training summary does not contain exactly the seven configured clusters")

    module_dir = repo_root / "LoRAHub-JAX/lora_modules/clusters"
    module_dir.mkdir(parents=True, exist_ok=True)
    module_paths = []
    for cluster in CLUSTERS:
        checkpoint = Path(checkpoints[str(cluster)])
        if not (checkpoint / "_CHECKPOINT_METADATA").is_file():
            raise FileNotFoundError(f"Incomplete cluster checkpoint: {checkpoint}")
        output = module_dir / f"cluster_{cluster:02d}_pi0_ee_lora.npz"
        extracted = extract_lora_from_checkpoint(
            str(checkpoint), str(output), task_name=f"cluster_{cluster:02d}")
        if len(extracted) != 20 or pair_count(extracted) != 10:
            raise ValueError(f"Cluster {cluster:02d} is not a 20-leaf/10-pair LoRA")
        module_paths.append(output)
        del extracted
        gc.collect()
    return module_paths, checkpoints


def build_bank(module_paths, settings, checkpoints):
    modules = [np.load(path, allow_pickle=True) for path in module_paths]
    try:
        key_sets = [{key for key in module.files if key != "__metadata__"} for module in modules]
        if any(keys != key_sets[0] for keys in key_sets[1:]):
            raise ValueError("The seven LoRA modules do not share identical parameter keys")
        if len(key_sets[0]) != 20 or pair_count(key_sets[0]) != 10:
            raise ValueError("Expert modules must contain exactly 10 A/B pairs")
        bank = {}
        source_shapes = {}
        for key in sorted(key_sets[0]):
            values = [np.asarray(module[key]) for module in modules]
            if any(value.shape != values[0].shape for value in values[1:]):
                raise ValueError(f"Expert shape mismatch: {key}")
            if values[0].shape[0] != 18:
                raise ValueError(f"Expected the leading 18 Transformer blocks: {key}")
            # Source [block,...] -> bank [block,expert,...], matching Linen scan.
            bank[bank_key(key)] = np.stack(values, axis=1)
            source_shapes[key] = list(values[0].shape)

        bank_path = Path(settings["bank_path"])
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = bank_path.with_suffix(".tmp.npz")
        np.savez(temporary, **bank)
        temporary.replace(bank_path)
        digest = hashlib.sha256()
        with bank_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        metadata = {
            "format": "pi0_topk_routed_lora_bank_v1",
            "cluster_order": list(CLUSTERS),
            "num_experts": len(CLUSTERS),
            "source_leaves_per_expert": 20,
            "source_ab_pairs_per_expert": 10,
            "transformer_blocks_per_leaf": 18,
            "router_groups": 10,
            "top_k": settings["router_top_k"],
            "composition": "sum_k w_k A_k B_k; exactly equivalent to sqrt(w) rank concatenation",
            "module_paths": [str(path) for path in module_paths],
            "checkpoints": checkpoints,
            "source_shapes": source_shapes,
            "bank_shapes": {key: list(value.shape) for key, value in bank.items()},
            "sha256": digest.hexdigest(),
            "bytes": bank_path.stat().st_size,
        }
        write_json(settings["bank_metadata"], metadata)
        return metadata
    finally:
        for module in modules:
            module.close()


def stage_assets(settings):
    source = Path(settings["source_joint_assets"])
    destination = Path(settings["assets_root"]) / CONFIG_NAME / JOINT_REPO
    if not (source / "norm_stats.json").is_file():
        raise FileNotFoundError(f"Missing five-task normalization assets: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("norm_stats.json", "provenance.json"):
        if (source / name).is_file():
            shutil.copy2(source / name, destination / name)
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--reuse-bank", action="store_true")
    args = parser.parse_args()
    settings = read_settings(args.settings)
    setup(settings)
    if args.reuse_bank:
        if not Path(settings["bank_path"]).is_file() or not Path(settings["bank_metadata"]).is_file():
            parser.error("--reuse-bank requested but bank artifacts are missing")
        metadata = json.loads(Path(settings["bank_metadata"]).read_text())
    else:
        module_paths, checkpoints = extract_modules(settings)
        metadata = build_bank(module_paths, settings, checkpoints)
    assets = stage_assets(settings)

    # Ensure configured shapes and frozen/trainable scopes are valid before GPU use.
    cfg = build_config(settings, HERE / "runs/_prepare_only")
    if cfg.assets_dirs / JOINT_REPO != assets:
        raise RuntimeError("Normalization assets were staged under the wrong config directory")
    report = {
        "status": "ready",
        "bank_path": settings["bank_path"],
        "bank_sha256": metadata["sha256"],
        "bank_bytes": metadata["bytes"],
        "cluster_order": list(CLUSTERS),
        "top_k": settings["router_top_k"],
        "router_groups": 10,
        "joint_assets": str(assets),
        "training_started": False,
    }
    write_json(HERE / "reports/preparation.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
