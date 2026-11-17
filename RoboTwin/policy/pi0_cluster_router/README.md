# Seven-cluster Top-3 LoRA Router

This extension composes the seven independently trained atomic-action LoRAs and
fine-tunes the composition on the joint five-task EE dataset. It leaves the
original datasets and independent cluster checkpoints untouched.

## Parameter structure

Each cluster module has 20 array leaves: 10 A/B pairs. Every leaf has a leading
dimension of 18 because Linen scan stacks the corresponding projection from all
18 Transformer blocks into one array. The ten routed groups are:

1. main LLM Q
2. action expert Q
3. main LLM KV
4. action expert KV
5. main LLM attention output
6. action expert attention output
7. main LLM MLP gating
8. main LLM MLP output
9. action expert MLP gating
10. action expert MLP output

The MLP router pools the frozen Pi0 image and language prefix representation and
outputs `[batch, 10, 7]` logits. Each group retains exactly Top-3 experts. The
forward pass is hard Top-3; a straight-through softmax supplies gradients to all
seven logits.

For a selected set, the history branch computes

`sum_k w_k (x A_k) B_k`.

This is exactly the same update as concatenating `sqrt(w_k) A_k` and
`sqrt(w_k) B_k`; computing the sum directly avoids allocating expanded rank-48
and rank-96 tensors. Main-LLM source rank is 16 and action-expert source rank is
32, rather than rank 8.

A separate trainable residual branch is initialized with `A ~ N(0, 0.01^2)` and
`B = 0`. Its per-site scale eta is trainable. The Pi0 backbone and all seven
historical experts remain frozen. Only the router MLP, residual A/B, and eta are
optimized by the five-task flow-matching loss, with a small router load-balance
term.

## Artifacts

Preparation creates:

- `LoRAHub-JAX/lora_modules/clusters/cluster_XX_pi0_ee_lora.npz`: seven portable
  20-leaf source modules.
- `expert_bank/cluster7_lora_bank.npz`: 20 bank arrays with shape
  `[18, 7, ...]`.
- `expert_bank/cluster7_lora_bank.json`: order, shapes, checkpoint provenance,
  checksum, and composition contract.
- `assets/cluster7_top3_router_pi0_ee/robotwin_five_pi0_ee/`: the shared 32D EE
  normalization used by the five-task dataset.

## Commands

Run from the repository root. Bank preparation is CPU-only and may temporarily
use substantial host RAM because Orbax checkpoints contain the full Pi0 model.

```bash
bash RoboTwin/policy/pi0_cluster_router/run.sh prepare
```

Local RTX 4090 training should begin with batch size 1 because the forward pass
holds seven frozen adapters and evaluates three routed branches per group:

```bash
nohup bash RoboTwin/policy/pi0_cluster_router/run.sh train \
  --gpu 0 \
  --batch-size 1 \
  --train-steps 20000 \
  --save-interval 5000 \
  --run-name cluster7_top3_router_v1 \
  --reuse-bank \
  > cluster7_top3_router_v1.log 2>&1 &
```

Resume with the identical arguments and `--resume`. The default learning rate is
`1e-5`, with 500 warmup steps and cosine decay.

Evaluate the same routed checkpoint separately on all five tasks. Start with a
small no-video smoke evaluation, then use more episodes with videos for the
reportable result:

```bash
bash RoboTwin/policy/pi0_cluster_router/run.sh eval \
  --gpu 0 \
  --run-name cluster7_top3_router_b4_s5000 \
  --checkpoint-step 5000 \
  --eval-episodes 3 \
  --no-video

bash RoboTwin/policy/pi0_cluster_router/run.sh eval \
  --gpu 0 \
  --run-name cluster7_top3_router_b4_s5000 \
  --checkpoint-step 5000 \
  --eval-episodes 20 \
  --eval-seed 0
```

Each task writes `result.json`, `episodes.json`, and optional MP4 files under
`runs/<run-name>/evaluations/<task>/`. The combined per-task and overall success
rates are written to `runs/<run-name>/evaluation_summary.json`.

Generate the loss and gradient curve without loading the model:

```bash
RoboTwin/policy/pi0/.venv/bin/python \
  RoboTwin/policy/pi0_cluster_router/analyze_training.py \
  --run-dir RoboTwin/policy/pi0_cluster_router/runs/cluster7_top3_router_b4_s5000
```
