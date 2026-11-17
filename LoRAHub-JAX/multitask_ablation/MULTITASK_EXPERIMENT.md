# Five-task Action Expert experiment

This pipeline compares two initializations on the same 50-episode dataset:

1. native Pi0 base;
2. the Pi0 base after two-epoch V3 sequential LoRAHub optimization.

Both arms use the same data, normalization statistics, seed, batch size, optimizer,
and strict Action Expert-only freeze mask. Each arm trains for 3000 steps and keeps
checkpoints 1000, 2000, and 3000. Every checkpoint is evaluated on all five tasks,
for 30 standard RoboTwin evaluation runs in total.

## Commands

Run the complete pipeline:

```bash
cd /home/huahungy/RoboLoRAHub/LoRAHub-JAX
bash multitask_ablation/run_multitask_experiment.sh all
```

Run or resume individual stages:

```bash
bash multitask_ablation/run_multitask_experiment.sh search
bash multitask_ablation/run_multitask_experiment.sh compose
bash multitask_ablation/run_multitask_experiment.sh train
bash multitask_ablation/run_multitask_experiment.sh eval
```

All terminal output is copied to
`outputs/place_multitask_right_50_v3_pi0_seq/logs/`. Successful evaluations get
restart markers in `outputs/place_multitask_right_50_v3_pi0_seq/eval_state/`, so an
interrupted `eval` command resumes at the first unfinished task. Set `FORCE_EVAL=1`
to rerun completed evaluations.
