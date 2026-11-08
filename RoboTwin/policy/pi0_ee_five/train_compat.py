"""Local fix for Optax ArrayTree forward references in the installed type checker."""


def install_train_state_compat():
    from typing import Any
    from flax import nnx, struct
    import optax
    from openpi.models import model
    from openpi.shared import array_typing as at
    from openpi.training import utils

    if getattr(utils.TrainState, '_ee_compatible', False):
        return

    # Preserve the upstream PyTree layout/field names and all other annotations.
    # Optax's recursive OptState alias cannot be resolved by this beartype version
    # in openpi.training.utils. Only that field's runtime annotation is relaxed.
    @at.typecheck
    @struct.dataclass
    class TrainState:
        step: at.Int[at.ArrayLike, '']
        params: nnx.State
        model_def: nnx.GraphDef[model.BaseModel]
        opt_state: Any
        tx: optax.GradientTransformation = struct.field(pytree_node=False)
        ema_decay: float | None = struct.field(pytree_node=False)
        ema_params: nnx.State | None = None

    TrainState._ee_compatible = True
    utils.TrainState = TrainState
