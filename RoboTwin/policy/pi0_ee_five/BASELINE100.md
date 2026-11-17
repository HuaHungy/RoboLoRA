# Standard Pi0 LoRA baseline on the five-task 100-episode subset

This baseline uses `robotwin_five_pi0_ee_subset20_seed42`: 20 trajectories from
each of the five tasks, 100 trajectories and 15,789 frames in total. It starts
from `pi0_base` and trains the standard Pi0 LoRA adapters without the cluster
expert bank, MLP router, or routed residual branch.

The training settings match the routed run: global batch 4, 5,000 optimizer
steps, seed 42, peak learning rate `1e-5`, 500 warmup steps, cosine decay,
50-step action horizon, and checkpoints every 1,000 steps. Evaluation uses the
same five tasks, unseen instructions, seed 0, 20 expert-solvable episodes per
task, 10 executed actions per inference, and 10 diffusion steps. All 100
evaluation episodes save MP4 videos.

Run normalization, training, evaluation, and comparison in one command:

```bash
cd /home/huahungy/RoboLoRAHub
GPU_IDS=0 bash RoboTwin/policy/pi0_ee_five/run_baseline100.sh
```

Resume the same experiment with `--resume`. The preparation report is written
to `reports/baseline100_preparation.json`; training and evaluation outputs are
under `runs/pi0_baseline100_b4_s5000`; the final comparison is written as both
`comparison_vs_router.json` and `comparison_vs_router.md`.

For the server experiment with global batch 32, select the GPUs through
`GPU_IDS`. By default all selected GPUs form one FSDP group; override
`FSDP_DEVICES` when the server topology requires a smaller group:

```bash
cd /path/to/RoboLoRA
GPU_IDS=0,1,2,3 FSDP_DEVICES=4 \
  bash RoboTwin/policy/pi0_ee_five/run_baseline100_b32.sh
```

This writes to `runs/pi0_baseline100_b32_s5000`, so it cannot overwrite the
batch-4 experiment. The global batch must be divisible by the number of GPUs,
and `FSDP_DEVICES` must divide the number of GPUs.

The existing routed checkpoint used all five 50-episode sources (250 episodes),
whereas this requested baseline uses 100 episodes. The generated comparison
records this difference and should not be presented as a strict data-matched
ablation.
