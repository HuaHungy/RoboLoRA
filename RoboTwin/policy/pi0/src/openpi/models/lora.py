import math
import re

import flax.linen as nn
import flax.struct as struct
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    """Configuration for LoRA."""

    # LoRA rank.
    rank: int
    # LoRA scaling factor.
    alpha: float = 1.0
    # Initialization function for LoRA parameters.
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # Optional separate initializer for B.  Routed residual LoRA uses the
    # conventional A~N(0,sigma^2), B=0 initialization while legacy configs keep
    # the historical OpenPI behavior by leaving this unset.
    init_b_fn: nn.initializers.Initializer | None = None
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # Axes in the weight to apply LoRA to. Should typically be the last two axes.
    axes: tuple[int, int] = (-2, -1)
    # Axis label which is used by LoRA in einsum equations. Must not be present in the original equation.
    label: str = "L"
    # Number of frozen historical LoRA experts.  When nonzero, every LoRA site
    # contains an expert bank in addition to the ordinary trainable A/B pair.
    expert_bank_size: int = 0
    # Number of bank experts evaluated in the forward pass.  Zero means all.
    expert_top_k: int = 0
    # Initial scale of the trainable residual branch.
    residual_scale: float = 1.0

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """Einsum with LoRA support. Can be used as a drop-in replacement for the Gemma Einsum."""

    # Shape of the weight.
    shape: tuple[int, ...]
    # Initialization function for the weight.
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # Setup LoRA parameters.
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_b_fn or config.init_fn, shape_b)
            if config.expert_bank_size:
                self.bank_a = self.param(
                    "lora_bank_a", nn.initializers.zeros,
                    (config.expert_bank_size, *shape_a),
                )
                self.bank_b = self.param(
                    "lora_bank_b", nn.initializers.zeros,
                    (config.expert_bank_size, *shape_b),
                )
                self.residual_scale = self.param(
                    "lora_residual_scale",
                    nn.initializers.constant(config.residual_scale),
                    (),
                )

    @nn.compact
    def __call__(self, eqn: str, x, routing_weights=None):
        dtype = x.dtype  # original dtype, could be half-precision
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            residual_scale = self.residual_scale if config.expert_bank_size else 1.0
            result = result + lora * config.scaling_value * jnp.asarray(residual_scale, dtype=dtype)
            if config.expert_bank_size:
                weights = _routing_weights(
                    routing_weights, x.shape[0], config.expert_bank_size, dtype)
                output_labels = eqn.split("->", maxsplit=1)[1]
                if output_labels.count("B") != 1:
                    raise ValueError(
                        "Routed LoRA einsum output must contain exactly one batch axis B, "
                        f"got {eqn}")
                output_batch_axis = output_labels.index("B")

                def expert_forward(one_x, a, b):
                    value = jnp.einsum(eqn_a, one_x, a.astype(dtype))
                    return jnp.einsum(eqn_b, value, b.astype(dtype))

                result = result + _apply_expert_bank(
                    x, self.bank_a, self.bank_b, weights,
                    config.expert_top_k, expert_forward,
                    output_batch_axis=output_batch_axis)

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """Feed forward module."""

    features: int
    hidden_dim: int
    # If not None, apply LoRA to the weight.
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, )),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # Setup LoRA parameters.
            # TODO: follow up with a simplified init_fn api.
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param("gating_einsum_lora_b", self.lora_config.init_b_fn or self.lora_config.init_fn,
                           (2, self.lora_config.rank, self.hidden_dim)),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_b_fn or self.lora_config.init_fn,
                           (self.lora_config.rank, self.features)),
            )
            if self.lora_config.expert_bank_size:
                n = self.lora_config.expert_bank_size
                r = self.lora_config.rank
                self.w_gating_bank = (
                    self.param("gating_einsum_lora_bank_a", nn.initializers.zeros,
                               (n, 2, self.features, r)),
                    self.param("gating_einsum_lora_bank_b", nn.initializers.zeros,
                               (n, 2, r, self.hidden_dim)),
                )
                self.w_linear_bank = (
                    self.param("linear_lora_bank_a", nn.initializers.zeros,
                               (n, self.hidden_dim, r)),
                    self.param("linear_lora_bank_b", nn.initializers.zeros,
                               (n, r, self.features)),
                )
                self.gating_residual_scale = self.param(
                    "gating_einsum_lora_residual_scale",
                    nn.initializers.constant(self.lora_config.residual_scale), (),
                )
                self.linear_residual_scale = self.param(
                    "linear_lora_residual_scale",
                    nn.initializers.constant(self.lora_config.residual_scale), (),
                )

    @nn.compact
    def __call__(self, x, routing_weights=None):
        dtype = x.dtype  # original dtype, could be half-precision
        gating_route, linear_route = ((None, None) if routing_weights is None else routing_weights)
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
            None if not self.lora_config or not self.lora_config.expert_bank_size
            else (self.w_gating_bank[0][:, 0], self.w_gating_bank[1][:, 0]),
            gating_route,
            None if not self.lora_config or not self.lora_config.expert_bank_size
            else self.gating_residual_scale,
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
            None if not self.lora_config or not self.lora_config.expert_bank_size
            else (self.w_gating_bank[0][:, 1], self.w_gating_bank[1][:, 1]),
            gating_route,
            None if not self.lora_config or not self.lora_config.expert_bank_size
            else self.gating_residual_scale,
        )
        activations = gate_value * ff1

        outputs = self._dot(
            activations, self.w_linear, self.w_linear_lora,
            None if not self.lora_config or not self.lora_config.expert_bank_size else self.w_linear_bank,
            linear_route,
            None if not self.lora_config or not self.lora_config.expert_bank_size
            else self.linear_residual_scale,
        )
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array,
             lora_weights: tuple[at.Array, at.Array] | None,
             bank_weights: tuple[at.Array, at.Array] | None = None,
             routing_weights=None, residual_scale=None) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        residual = jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))
        if residual_scale is not None:
            residual = residual * residual_scale.astype(x.dtype)
        result = base + residual
        if bank_weights is not None:
            weights = _routing_weights(
                routing_weights, x.shape[0], bank_weights[0].shape[0], x.dtype)

            def expert_forward(one_x, a, b):
                return jnp.dot(
                    jnp.dot(one_x, a.astype(one_x.dtype)), b.astype(one_x.dtype))

            result = result + _apply_expert_bank(
                x, bank_weights[0], bank_weights[1], weights,
                self.lora_config.expert_top_k, expert_forward)
        return result


def _routing_weights(weights, batch_size: int, expert_count: int, dtype):
    if weights is None:
        return jnp.full((batch_size, expert_count), 1.0 / expert_count, dtype=dtype)
    weights = jnp.asarray(weights, dtype=dtype)
    if weights.shape != (batch_size, expert_count):
        raise ValueError(
            f"routing weights must have shape {(batch_size, expert_count)}, got {weights.shape}")
    return weights


def _apply_expert_bank(
    x, bank_a, bank_b, weights, top_k, expert_forward, *, output_batch_axis=0
):
    """Evaluate only selected experts and mix their exact low-rank deltas."""
    top_k = top_k or bank_a.shape[0]
    selected_weights, indices = jax.lax.top_k(weights, top_k)
    selected_a = bank_a[indices]
    selected_b = bank_b[indices]

    def sample_forward(one_x, experts_a, experts_b, one_weights):
        # Restore a singleton B axis because the original einsum equations use B.
        values = jax.vmap(
            lambda a, b: jnp.squeeze(
                expert_forward(one_x[None], a, b), axis=output_batch_axis)
        )(experts_a, experts_b)
        expand = (slice(None),) + (None,) * (values.ndim - 1)
        return jnp.sum(values * one_weights[expand], axis=0)

    mixed = jax.vmap(sample_forward)(x, selected_a, selected_b, selected_weights)
    # vmap introduces the batch dimension at axis 0.  Einsum outputs such as
    # 2BSKH place B elsewhere, so restore the equation's original layout.
    return jnp.moveaxis(mixed, 0, output_batch_axis)
