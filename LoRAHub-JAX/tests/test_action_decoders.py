import jax
import jax.numpy as jnp
import numpy as np
import optax

from action_decoders.diffusion import cosine_schedule
from action_decoders.models import ACTDecoder, DecoderConfig, DiffusionDecoder
from action_decoders.predict import LoRAHubActionDecoderPolicy
from action_decoders.train import _action_objective, _make_act_steps


def test_direct_act_and_dp_shapes():
    cfg = DecoderConfig(
        action_dim=14, action_horizon=50, state_dim=32,
        model_dim=64, heads=4, layers=2, ff_dim=128)
    memory = jnp.zeros((2, 11, 96), jnp.float32)
    memory_mask = jnp.ones((2, 11), bool)
    state = jnp.zeros((2, 32), jnp.float32)

    act = ACTDecoder(config=cfg)
    act_variables = act.init(jax.random.key(0), memory, memory_mask, state)
    act_output = act.apply(act_variables, memory, memory_mask, state)

    dp = DiffusionDecoder(config=cfg)
    noisy = jnp.zeros((2, 50, 14), jnp.float32)
    timestep = jnp.asarray([0, 99], jnp.int32)
    dp_variables = dp.init(
        jax.random.key(1), memory, memory_mask, state, noisy, timestep)
    dp_noise = dp.apply(
        dp_variables, memory, memory_mask, state, noisy, timestep)

    assert act_output["actions"].shape == (2, 50, 14)
    assert act_output["is_pad_logits"].shape == (2, 50)
    assert dp_noise.shape == (2, 50, 14)


def test_act_ablation_branches():
    cfg = DecoderConfig(
        action_dim=14, action_horizon=50, state_dim=32,
        model_dim=64, heads=4, layers=2, ff_dim=128,
        separate_query_pos=True, predict_is_pad=True,
        use_cvae=True, latent_dim=8, cvae_encoder_layers=1)
    memory = jnp.zeros((2, 11, 96), jnp.float32)
    memory_mask = jnp.ones((2, 11), bool)
    state = jnp.zeros((2, 32), jnp.float32)
    target = jnp.zeros((2, 50, 14), jnp.float32)
    valid = jnp.ones((2, 50), jnp.float32)
    noise = jnp.zeros((2, 8), jnp.float32)
    model = ACTDecoder(config=cfg)
    variables = model.init(
        jax.random.key(2), memory, memory_mask, state, target, valid, noise)
    train_output = model.apply(
        variables, memory, memory_mask, state, target, valid, noise)
    infer_output = model.apply(variables, memory, memory_mask, state)
    assert train_output["actions"].shape == infer_output["actions"].shape == (2, 50, 14)
    assert train_output["mu"].shape == (2, 8)
    assert "action_query_pos" in variables["params"]
    assert "action_queries" not in variables["params"]


def test_action_mask_and_hybrid_loss_are_finite():
    prediction = jnp.zeros((2, 4, 14), jnp.float32)
    target = jnp.ones_like(prediction)
    sample_mask = jnp.asarray([1.0, 0.0])
    action_mask = jnp.asarray([[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    values = _action_objective(
        prediction, target, sample_mask, action_mask, 14, 2.0, 0.05, 0.01,
        "joint_huber_gripper_l1", 1.0)
    assert all(np.isfinite(float(value)) for value in values)
    assert float(values[1]) > 0


def test_temporal_aggregation_fuses_overlapping_chunks():
    policy = object.__new__(LoRAHubActionDecoderPolicy)
    policy.temporal_aggregation = True
    policy.temporal_agg_decay = 0.01
    policy.temporal_agg_recent_weighted = True
    policy._aggregation_step = 0
    policy._action_history = []
    policy.decoder_cfg = DecoderConfig(action_dim=14, action_horizon=3)
    first = np.stack([np.zeros(14), np.ones(14), np.full(14, 2.0)])
    second = np.stack([np.full(14, 3.0), np.full(14, 4.0), np.full(14, 5.0)])
    out0 = policy.aggregate_absolute_actions(first)
    out1 = policy.aggregate_absolute_actions(second)
    assert out0.shape == out1.shape == (1, 14)
    assert np.allclose(out0, 0.0)
    assert np.all((out1 > 1.0) & (out1 < 3.0))


def test_each_act_training_ablation_runs_one_update():
    variants = [
        ({}, {}),
        ({"separate_query_pos": True}, {}),
        ({}, {"use_action_valid_mask": True}),
        ({"predict_is_pad": True}, {"pad_loss_weight": 0.1}),
        ({"use_cvae": True, "latent_dim": 8, "cvae_encoder_layers": 1},
         {"kl_weight": 0.01}),
        ({}, {"reconstruction_loss": "joint_huber_gripper_l1"}),
    ]
    memory = jnp.zeros((2, 5, 24), jnp.float32)
    memory_mask = jnp.ones((2, 5), bool)
    state = jnp.zeros((2, 8), jnp.float32)
    target = jnp.ones((2, 4, 4), jnp.float32)
    sample_mask = jnp.ones((2,), jnp.float32)
    action_mask = jnp.asarray([[1, 1, 1, 1], [1, 1, 0, 0]], jnp.float32)
    for model_overrides, training_overrides in variants:
        cfg = DecoderConfig(
            action_dim=4, action_horizon=4, state_dim=8,
            model_dim=32, heads=4, layers=1, ff_dim=64,
            **model_overrides)
        model = ACTDecoder(config=cfg)
        if cfg.use_cvae:
            variables = model.init(
                jax.random.key(3), memory, memory_mask, state, target,
                action_mask, jnp.zeros((2, cfg.latent_dim), jnp.float32))
        else:
            variables = model.init(jax.random.key(3), memory, memory_mask, state)
        params = variables["params"]
        optimizer = optax.adam(1e-3)
        train_cfg = {
            "gripper_weight": 1.0, "smooth_weight": 0.0,
            "jerk_weight": 0.0, "ema_decay": 0.9,
            **training_overrides,
        }
        train_step, _ = _make_act_steps(model, optimizer, cfg, train_cfg)
        result = train_step(
            params, params, optimizer.init(params), memory, memory_mask, state,
            target, sample_mask, action_mask, jax.random.key(4), jnp.asarray(1.0))
        assert np.isfinite(float(result[3]))


def test_cosine_schedule_is_valid():
    betas, alphas, alpha_bars = cosine_schedule(100)
    assert betas.shape == alphas.shape == alpha_bars.shape == (100,)
    assert np.all((betas > 0) & (betas < 1))
    assert np.all(np.diff(alpha_bars) < 0)
    assert 0 < alpha_bars[-1] < alpha_bars[0] < 1
