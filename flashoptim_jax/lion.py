from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from . import utils
from .compression import no_ecc_leaf, reconstruct_leaf, split_leaf
from .momentum_kernel import MomentumLeafStepResult, fused_momentum_leaf_impl


class FlashLionState(NamedTuple):
    """Optimizer state for Lion."""
    count: jax.Array
    mu: Any
    ecc: Any


def _flash_lion_leaf_unfused_impl(
    grad: jax.Array,
    param: jax.Array,
    ecc: jax.Array,
    mu: Any,
    lr: jax.Array,
    b1: float,
    b2: float,
    weight_decay: float,
    group_size: int,
    quantize: bool,
    master_weight_bits: int | None,
) -> MomentumLeafStepResult:
    """Apply an unfused Lion update to one parameter leaf.
    
    Functionally identical to the kernel. Kept mostly for comparison and readability.
    """
    # Reconstruct the effective fp32 parameter value.
    grad_f32 = jnp.asarray(grad, dtype=jnp.float32)
    use_ecc = utils._use_ecc_leaf(param, master_weight_bits)
    if use_ecc:
        param_f32 = reconstruct_leaf(param, ecc)
    else:
        param_f32 = jnp.asarray(param, dtype=jnp.float32)

    # Materialize optimizer state from either quantized or full storage.
    mu_f32 = utils._materialize_momentum(mu, group_size)

    # Lion update in fp32.
    if weight_decay:
        param_f32 = param_f32 * (1.0 - lr * weight_decay)

    update = jnp.sign(mu_f32 + (1.0 - b1) * (grad_f32 - mu_f32))
    new_param_f32 = param_f32 - lr * update
    mu_f32 = mu_f32 + (1.0 - b2) * (grad_f32 - mu_f32)

    # Store the updated parameter back in the configured param/ECC format.
    if use_ecc:
        new_param, new_ecc = split_leaf(
            new_param_f32,
            narrow_dtype=param.dtype,
            master_weight_bits=master_weight_bits,
        )
    else:
        new_param = new_param_f32.astype(param.dtype)
        new_ecc = no_ecc_leaf()

    # Re-store momentum in either quantized or full form.
    return MomentumLeafStepResult(
        param=new_param.astype(param.dtype),
        ecc=new_ecc,
        mu=utils._store_momentum(mu_f32, quantize, group_size),
    )


def _flash_lion_leaf_impl(
    grad: jax.Array,
    param: jax.Array,
    ecc: jax.Array,
    mu: Any,
    lr: jax.Array,
    b1: float,
    b2: float,
    weight_decay: float,
    group_size: int,
    quantize: bool,
    master_weight_bits: int | None,
    fused: bool,
) -> MomentumLeafStepResult:
    """Dispatch a leaf update to the fused or unfused Lion path."""
    use_ecc = utils._use_ecc_leaf(param, master_weight_bits)
    ecc_dtype = jnp.int16 if master_weight_bits == 32 else jnp.int8
    if fused:
        return fused_momentum_leaf_impl(
            grad, param, ecc, mu,
            lr=lr,
            mom_coef=b2,
            update_coef=b1,
            weight_decay=weight_decay,
            do_lion=True,
            nesterov=False,
            decoupled_weight_decay=True,
            quantize=quantize,
            group_size=group_size,
            use_ecc=use_ecc,
            ecc_dtype=ecc_dtype,
        )
    return _flash_lion_leaf_unfused_impl(
        grad, param, ecc, mu,
        lr=lr, b1=b1, b2=b2,
        weight_decay=weight_decay,
        group_size=group_size,
        quantize=quantize,
        master_weight_bits=master_weight_bits,
    )


def _init_lion_state(
    params: Any,
    defaults: dict[str, Any],
    param_groups: list[utils.ParamGroupSpec] | None,
    name: str,
) -> FlashLionState:
    """Initialize Lion state trees for a parameter pytree."""
    utils._validate_fused_param_dtypes(params, defaults, param_groups, name)
    utils._validate_meaningful_master_weight_bits(params, defaults, param_groups, name)
    paths, leaves, treedef = utils._tree_leaves_with_paths(params)
    leaf_states = []
    for path, leaf in zip(paths, leaves, strict=True):
        config = utils._group_config_for_path(path, leaf, defaults=defaults, param_groups=param_groups)
        utils._validate_master_weight_bits(config["master_weight_bits"])
        utils._validate_fused_group_size(
            config["group_size"], quantize=config["quantize"], fused=config["fused"],
        )
        leaf_states.append(
            utils._init_momentum_leaf_state(
                leaf, config["group_size"],
                quantize=config["quantize"],
                master_weight_bits=config["master_weight_bits"],
                use_momentum=True,
            )
        )
    return FlashLionState(
        count=jnp.zeros((), dtype=jnp.int32),
        mu=treedef.unflatten([leaf_state[0] for leaf_state in leaf_states]),
        ecc=treedef.unflatten([leaf_state[1] for leaf_state in leaf_states]),
    )


def _compute_lion_step(
    params: Any,
    grads: Any,
    state: FlashLionState,
    defaults: dict[str, Any],
    param_groups: list[utils.ParamGroupSpec] | None,
    name: str,
) -> tuple[jax.Array, Any, list[Any]]:
    """Compute per-leaf Lion results across a pytree."""

    count = state.count + jnp.asarray(1, dtype=jnp.int32)
    grad_paths, grad_leaves, grad_treedef = utils._tree_leaves_with_paths(grads)
    paths, leaves, treedef = utils._tree_leaves_with_paths(params)
    ecc_leaves, ecc_treedef = jax.tree_util.tree_flatten(state.ecc)
    mu_leaves, mu_treedef = jax.tree_util.tree_flatten(state.mu, is_leaf=utils._is_quantized_leaf)
    if not (treedef == grad_treedef == ecc_treedef == mu_treedef):
        raise ValueError(f"{name} state trees must match parameter structure")
    if paths != grad_paths:
        raise ValueError(f"{name} gradients must match parameter paths")

    leaf_results = []
    for path, grad_leaf, param_leaf, ecc_leaf, mu_leaf in zip(
        paths, grad_leaves, leaves, ecc_leaves, mu_leaves, strict=True
    ):
        config = utils._group_config_for_path(path, param_leaf, defaults=defaults, param_groups=param_groups)
        lr = utils._resolve_learning_rate(config["learning_rate"], state.count)
        leaf_results.append(
            _flash_lion_leaf_impl(
                jnp.asarray(grad_leaf, dtype=jnp.float32),
                param_leaf,
                ecc_leaf,
                mu_leaf,
                lr=lr,
                b1=config["b1"],
                b2=config["b2"],
                weight_decay=config["weight_decay"],
                group_size=config["group_size"],
                quantize=config["quantize"],
                master_weight_bits=config["master_weight_bits"],
                fused=config["fused"],
            )
        )
    return count, treedef, leaf_results


def flash_lion(
    learning_rate: utils.ScalarOrSchedule = 1e-4,
    b1: float = 0.9,
    b2: float = 0.99,
    weight_decay: float = 0.0,
    quantize: bool = True,
    master_weight_bits: int | None = 24,
    group_size: int = 32,
    fused: bool = True,
    param_groups: list[utils.ParamGroupSpec] | None = None,
) -> utils.FlashOptimizer:
    """Create a flashoptimized Lion optimizer."""

    name = "flash_lion"
    utils._validate_master_weight_bits(master_weight_bits)
    utils._validate_fused_group_size(group_size, quantize=quantize, fused=fused)
    defaults = {
        "learning_rate": learning_rate,
        "b1": b1, "b2": b2,
        "weight_decay": weight_decay,
        "quantize": quantize,
        "master_weight_bits": master_weight_bits,
        "group_size": group_size,
        "fused": fused,
    }

    def init_fn(params: Any) -> FlashLionState:
        return _init_lion_state(params, defaults, param_groups, name)

    def step_fn(params: Any, state: FlashLionState, grads: Any) -> tuple[Any, FlashLionState]:
        count, treedef, leaf_results = _compute_lion_step(
            params, grads, state, defaults, param_groups, name,
        )
        new_params = treedef.unflatten([result.param for result in leaf_results])
        new_ecc = treedef.unflatten([result.ecc for result in leaf_results])
        new_mu = treedef.unflatten([result.mu for result in leaf_results])
        return new_params, FlashLionState(count=count, mu=new_mu, ecc=new_ecc)

    return utils.FlashOptimizer(init=init_fn, step=step_fn)

# Checkpointing utils

def flash_lion_state_dict(state: FlashLionState) -> dict[str, Any]:
    """Serialize Lion optimizer state into checkpointable trees."""
    return {
        "count": jnp.asarray(state.count),
        "mu": utils._tree_state_dict(state.mu),
        "ecc": utils._tree_state_dict(state.ecc),
    }


def load_flash_lion_state_dict(state_dict: dict[str, Any]) -> FlashLionState:
    """Restore Lion optimizer state from a serialized dict."""
    return FlashLionState(
        count=jnp.asarray(state_dict["count"], dtype=jnp.int32),
        mu=utils._tree_from_state_dict(state_dict["mu"]),
        ecc=utils._tree_from_state_dict(state_dict["ecc"]),
    )
