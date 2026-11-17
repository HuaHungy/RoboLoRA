import jax
import jax.numpy as jnp

from side.kv_side_network import (
    KVSideConfig,
    KVSideQFormerConfig,
    KVSideQFormerResidual,
    KVSideResidual,
    parameter_count,
)
from side.train_side import _batch_indices


def test_kv_side_shape_zero_initialization_and_size():
    cfg = KVSideConfig()
    model = KVSideResidual(config=cfg)
    variables = model.init(
        jax.random.key(0),
        jnp.zeros((2, cfg.context_dim)),
        jnp.zeros((2, 50, cfg.action_hidden_dim)),
        jnp.zeros((2,)),
    )
    output = model.apply(
        variables,
        jnp.ones((2, cfg.context_dim)),
        jnp.ones((2, 50, cfg.action_hidden_dim)),
        jnp.ones((2,)),
    )
    assert output.shape == (2, 50, cfg.output_dim)
    assert float(jnp.max(jnp.abs(output))) == 0.0
    assert parameter_count(variables["params"]) < 200_000


def test_sequential_batch_covers_tail_without_dropping_frames():
    indices, mask = _batch_indices(8, total=10, batch_size=4)
    assert indices.tolist() == [8, 9, 9, 9]
    assert mask.tolist() == [1.0, 1.0, 0.0, 0.0]


def test_qformer_side_shape_and_zero_initialization():
    cfg = KVSideQFormerConfig()
    model = KVSideQFormerResidual(config=cfg)
    variables = model.init(
        jax.random.key(1),
        jnp.zeros((2, 7, cfg.kv_token_dim)),
        jnp.zeros((2, 50, cfg.action_hidden_dim)),
        jnp.zeros((2,)),
    )
    output = model.apply(
        variables,
        jnp.ones((2, 7, cfg.kv_token_dim)),
        jnp.ones((2, 50, cfg.action_hidden_dim)),
        jnp.ones((2,)),
    )
    assert output.shape == (2, 50, cfg.output_dim)
    assert float(jnp.max(jnp.abs(output))) == 0.0
    assert parameter_count(variables["params"]) < 2_000_000
