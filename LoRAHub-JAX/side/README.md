# KV Side Residual

This stage trains a tiny residual on top of a frozen LoRAHub-composed Pi0.
The backbone produces prefix K/V features and action-expert hidden states;
only `side/kv_side_network.py` parameters are optimized.

The default configuration uses:

- all frames from all 10 episodes in deterministic sequential order;
- batch size 32, with a masked final batch;
- prefix K/V mean pooling from layers 6, 12, and 18;
- a 145,358-parameter residual MLP;
- flow-matching residual injection at every Pi0 ODE step;
- atomic `checkpoints/latest.pkl` and numbered side checkpoints;
- a live tqdm progress bar with step rate and ETA;
- `logs/loss_curve.png` generated from `training.csv`.

Run:

```bash
cd /home/huahungy/RoboLoRAHub/LoRAHub-JAX
PYTHON=/home/huahungy/RoboLoRAHub/RoboTwin/policy/pi0/.venv/bin/python \
  bash run_side.sh config/place_empty_cup_10_v3_pi0_side.json
```

The merged Pi0 checkpoint must already exist at the path configured by
`model.merged_checkpoint`.  Side checkpoints are saved under the configured
`output_dir/checkpoints/` directory.

For RoboTwin evaluation, set the side config before calling the existing Pi0
evaluation script.  The Pi0 adapter converts the predicted DeltaActions back
to the absolute joint actions expected by RoboTwin:

```bash
cd /home/huahungy/RoboLoRAHub/RoboTwin/policy/pi0
export LORAHUB_JAX_ROOT=/home/huahungy/RoboLoRAHub/LoRAHub-JAX
export PI0_SIDE_CONFIG=$LORAHUB_JAX_ROOT/config/place_empty_cup_10_v3_pi0_side.json
export PI0_SIDE_DIFFUSION_STEPS=10
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

bash eval.sh place_empty_cup demo_aloha \
  place_empty_cup_pi0_lora_10ep \
  place_empty_cup_10_v3_pi0_aligned_random_20000 0 0
```
