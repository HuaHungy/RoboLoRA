# Frozen LoRAHub backbone + direct ACT/DP decoders

This suite does **not** use a Side network and does not add a residual to Pi0.
The composed LoRAHub checkpoint is frozen at the observation encoder boundary.
Its final prefix hidden tokens condition one of two mutually exclusive action
heads:

- `ACT_direct`: 50 learned action queries, query self-attention, cross-attention
  to LoRAHub tokens, and direct `50x14` action output.
- `DP_direct`: noisy `50x14` action tokens, diffusion timestep conditioning,
  self/cross-attention decoder, and direct epsilon prediction.

Both heads use normalized Pi0 DeltaActions (joint deltas and absolute grippers),
hold episodes 8 and 9 out for validation, save `best.pkl`, and execute five
actions before replanning in RoboTwin.

## Validate without training

```bash
cd /home/huahungy/RoboLoRAHub/LoRAHub-JAX
bash decoder_experiments/place_empty_cup_random20000/check_feasibility.sh --full
```

`--full` loads the real merged checkpoint and performs one optimizer update for
each decoder without writing artifacts.

## Run ACT and DP training + RoboTwin evaluation

```bash
cd /home/huahungy/RoboLoRAHub/LoRAHub-JAX
bash decoder_experiments/place_empty_cup_random20000/run_all.sh --run
```

The runner waits 60 seconds between training and evaluation jobs.  Run only one
decoder with `--only ACT` or `--only DP`.  Training resumes from `latest.pkl`.
Training artifacts live under `artifacts/`, suite logs under `logs/`, and native
RoboTwin results under `RoboTwin/eval_result/place_empty_cup/pi0/demo_aloha/`.
