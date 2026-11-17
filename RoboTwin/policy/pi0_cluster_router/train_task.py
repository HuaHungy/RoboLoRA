"""Fine-tune the Top-3 router and residual LoRA on the joint five-task data."""

import argparse
import logging
from pathlib import Path

from common import HERE, PI0, build_config, latest_checkpoint, load_module, read_settings, setup, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = read_settings(args.run_dir / "run_config.json")
    setup(settings)
    from joint_data import install_joint_dataset
    from pi0_ee_five.train_compat import install_train_state_compat
    install_joint_dataset()
    install_train_state_compat()

    import jax
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("JAX GPU is unavailable; refusing routed pi0 training on CPU")
    logging.basicConfig(level=logging.INFO)
    cfg = build_config(settings, args.run_dir, args.resume)
    if not (cfg.assets_dirs / "robotwin_five_pi0_ee/norm_stats.json").is_file():
        raise FileNotFoundError("Missing joint five-task normalization; run prepare first")
    print(
        f"Top-{settings['router_top_k']} router: experts=7, groups=10, "
        f"batch={cfg.batch_size}, steps={cfg.num_train_steps}, "
        "trainable=router+residual-LoRA, frozen=base+historical-bank",
        flush=True,
    )
    trainer = load_module("pi0_cluster_router_upstream_train", PI0 / "scripts/train.py")
    from pi0_ee_five.train_loop import main as train
    train(trainer, cfg)
    checkpoint = latest_checkpoint(args.run_dir, cfg.num_train_steps)
    write_json(args.run_dir / "status/trained.json", {
        "checkpoint": str(checkpoint),
        "steps": cfg.num_train_steps,
        "router_top_k": settings["router_top_k"],
        "clusters": settings["clusters"],
        "tasks": settings["tasks"],
    })


if __name__ == "__main__":
    main()
