# LoRAHub + direct ACT ablation suite

This directory compares six ACT changes against one fixed baseline.  All
train-time experiments use the same frozen `random_20000` LoRAHub checkpoint,
episode split (train 0-7, validation 8-9), seed 42, batch size 2, 30 epochs,
optimizer, decoder width and action horizon.  This is an isolated ablation, not
a cumulative ladder.

| ID | Only change relative to E0 | Training |
|---|---|---|
| E0 | baseline learned query content + weighted MSE | yes |
| E1 | learned `query_pos` separated from zero `tgt` | yes |
| E2 | mask padded future action steps in action/smooth/jerk loss | yes |
| E3 | add supervised `is_pad_head`, weight 0.1 | yes |
| E4 | temporal aggregation, replan every step | **reuses E0 checkpoint** |
| E5 | 32-D CVAE latent, 2-layer encoder, KL=0.01 with 10-epoch warmup | yes |
| E6 | joint Huber (`delta=1`) + gripper L1 reconstruction | yes |

E3 deliberately does not enable E2's mask in the action loss, so it measures
the auxiliary head rather than a two-change combination.  E4 is inference-only
and must reuse E0; retraining it would introduce an unnecessary source of
variation.  For E5, the posterior is used during training/validation and the
zero prior centre is used at deployment, matching native ACT's deterministic
inference convention.

The action-valid mask is `[B, 50]`.  At an episode's final frame it is
`[1, 0, ..., 0]`; repeated terminal actions keep static JAX shapes but are
excluded only in E2.  Batch padding remains covered by the independent
`sample_mask` in every experiment.

## Preflight

CPU schema/model/backward tests:

```bash
cd /home/huahungy/RoboLoRAHub/LoRAHub-JAX
bash decoder_experiments/place_empty_cup_random20000/act_ablations/check_feasibility.sh
```

The full runner additionally performs one real-checkpoint GPU update for every
trainable architecture before starting a long run.  Each smoke process has a
900-second timeout.

## Train and evaluate all experiments

```bash
cd /home/huahungy/RoboLoRAHub/LoRAHub-JAX

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
unset JAX_PLATFORMS

bash decoder_experiments/place_empty_cup_random20000/act_ablations/run_all.sh --run
```

Run selected experiments:

```bash
bash decoder_experiments/place_empty_cup_random20000/act_ablations/run_all.sh \
  --run --only E0_baseline,E2_action_mask,E6_huber_gripper_l1
```

`WAIT_SECONDS` defaults to 60 between adjacent training/evaluation jobs.  Every
trainable experiment resumes from `latest.pkl`.  E4 automatically trains E0 if
the baseline checkpoint is absent.

## Outputs

- Configs: `configs/*.json`
- Decoder checkpoints/loss CSV/curves: `artifacts/<experiment>/`
- Orchestration logs: `logs/`
- Job/result locations: `manifest.tsv`
- Comparison table: `summary.csv` and `summary.md`
- RoboTwin videos/results:
  `RoboTwin/eval_result/place_empty_cup/pi0/demo_aloha/<experiment>/`

Use RoboTwin success rate as the primary cross-experiment metric.  Raw
validation losses are not numerically comparable for E3, E5 and E6 because
their objectives include pad BCE, KL, or a different reconstruction scale.
Compare their validation curves only within the same experiment, then compare
success rate, visible jitter, arm-start timing and task phase completion across
the saved RoboTwin videos.
