from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from . import utils
from .compression import no_ecc_leaf, reconstruct_leaf, split_leaf
from .momentum_kernel import MomentumLeafStepResult, fused_momentum_leaf_impl


class FlashSGDState(NamedTuple):
    """Optimizer state for SGD-family transforms."""
    count: jax.Array
    mu: Any
    ecc: Any


def _flash_sgd_leaf_unfused_impl(
    grad: jax.Array,
    param: jax.Array,
    ecc: jax.Array,
    mu: Any,
    lr: jax.Array,
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
    decoupled_weight_decay: bool,
    group_size: int,
    quantize: bool,
    master_weight_bits: int | None,
) -> MomentumLeafStepResult:
    """Apply the unfused SGD path to one parameter leaf.
    
    Functionally identical to the kernel. Kept mostly for comparison and readability.
    """
    # Reconstruct the effective fp32 parameter value.
    grad_f32 = jnp.asarray(grad, dtype=jnp.float32)
    use_ecc = utils._use_ecc_leaf(param, master_weight_bits)
    if use_ecc:
        param_f32 = reconstruct_leaf(param, ecc)
    else:
        param_f32 = jnp.asarray(param, dtype=jnp.float32)

    # Apply weight decay and materialize optional momentum state.
    if not decoupled_weight_decay and weight_decay:
        grad_f32 = grad_f32 + weight_decay * param_f32

    mu_f32 = utils._materialize_optional_momentum(mu, group_size)
    if momentum > 0.0:
        if mu_f32 is None:
            mu_f32 = grad_f32
        else:
            mu_f32 = momentum * mu_f32 + (1.0 - dampening) * grad_f32
        step_direction = grad_f32 + momentum * mu_f32 if nesterov else mu_f32
    else:
        step_direction = grad_f32

    # SGD / SGDW update in fp32.
    if decoupled_weight_decay and weight_decay:
        param_f32 = param_f32 * (1.0 - lr * weight_decay)

    new_param_f32 = param_f32 - lr * step_direction

    # Store the updated parameter back in the configured param/ECC format.
    if use_ecc:
        new_param, new_ecc = split_leaf(
            new_param_f32, narrow_dtype=param.dtype, master_weight_bits=master_weight_bits,
        )
    else:
        new_param = new_param_f32.astype(param.dtype)
        new_ecc = no_ecc_leaf()

    # Re-store momentum and return the materialized parameter.
    mu_out = utils._store_optional_momentum(mu_f32, quantize, group_size)
    new_param = new_param.astype(param.dtype)
    return MomentumLeafStepResult(param=new_param, ecc=new_ecc, mu=mu_out)


def _flash_sgd_leaf_impl(
    grad: jax.Array,
    param: jax.Array,
    ecc: jax.Array,
    mu: Any,
    lr: jax.Array,
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
    decoupled_weight_decay: bool,
    group_size: int,
    quantize: bool,
    master_weight_bits: int | None,
    fused: bool,
) -> MomentumLeafStepResult:
    """Dispatch a leaf update to the fused or unfused SGD path."""
    use_ecc = utils._use_ecc_leaf(param, master_weight_bits)
    ecc_dtype = jnp.int16 if master_weight_bits == 32 else jnp.int8
    # TODO: Make kernel for momentum == 0 that does not allocate mu buffer,
    # So can use fused SGD.
    can_fuse = fused and momentum > 0.0
    if can_fuse:
        return fused_momentum_leaf_impl(
            grad, param, ecc, mu,
            lr=lr,
            mom_coef=momentum,
            update_coef=1.0 - dampening,
            weight_decay=weight_decay,
            do_lion=False,
            nesterov=nesterov,
            decoupled_weight_decay=decoupled_weight_decay,
            quantize=quantize,
            group_size=group_size,
            use_ecc=use_ecc,
            ecc_dtype=ecc_dtype,
        )
    return _flash_sgd_leaf_unfused_impl(
        grad, param, ecc, mu, lr=lr,
        momentum=momentum,
        dampening=dampening,
        weight_decay=weight_decay,
        nesterov=nesterov,
        decoupled_weight_decay=decoupled_weight_decay,
        group_size=group_size,
        quantize=quantize,
        master_weight_bits=master_weight_bits,
    )


def _validate_sgd_hparams(momentum: float, dampening: float, nesterov: bool) -> None:
    """Validate SGD-specific hyperparameter combinations."""
    if nesterov and (momentum <= 0.0 or dampening != 0.0):
        raise ValueError("Nesterov momentum requires positive momentum and zero dampening")


def _init_sgd_state(
    params: Any,
    defaults: dict[str, Any],
    param_groups: list[utils.ParamGroupSpec] | None,
    name: str,
) -> FlashSGDState:
    """Initialize SGD state trees for a parameter pytree."""
    utils._validate_fused_param_dtypes(params, defaults, param_groups, name)
    utils._validate_meaningful_master_weight_bits(params, defaults, param_groups, name)
    paths, leaves, treedef = utils._tree_leaves_with_paths(params)
    leaf_states = []
    for path, leaf in zip(paths, leaves, strict=True):
        config = utils._group_config_for_path(path, leaf, defaults=defaults, param_groups=param_groups)
        utils._validate_master_weight_bits(config["master_weight_bits"])
        utils._validate_fused_group_size(
            config["group_size"],
            quantize=config["quantize"],
            fused=config["fused"],
        )
        _validate_sgd_hparams(config["momentum"], config["dampening"], config["nesterov"])
        leaf_states.append(
            utils._init_momentum_leaf_state(
                leaf,
                config["group_size"],
                quantize=config["quantize"],
                master_weight_bits=config["master_weight_bits"],
                use_momentum=config["momentum"] > 0.0,
            )
        )
    return FlashSGDState(
        count=jnp.zeros((), dtype=jnp.int32),
        mu=treedef.unflatten([leaf_state[0] for leaf_state in leaf_states]),
        ecc=treedef.unflatten([leaf_state[1] for leaf_state in leaf_states]),
    )


def _sgd_leaf_results(
    params: Any,
    grads: Any,
    state: FlashSGDState,
    defaults: dict[str, Any],
    param_groups: list[utils.ParamGroupSpec] | None,
    decoupled_weight_decay: bool,
    name: str,
) -> tuple[jax.Array, Any, list[Any]]:
    """Compute per-leaf SGD step results across a pytree."""
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
        config = utils._group_config_for_path(
            path, param_leaf, defaults=defaults, param_groups=param_groups
        )
        lr = utils._resolve_learning_rate(config["learning_rate"], state.count)
        leaf_results.append(
            _flash_sgd_leaf_impl(
                jnp.asarray(grad_leaf, dtype=jnp.float32),
                param_leaf,
                ecc_leaf,
                mu_leaf,
                lr=lr,
                momentum=config["momentum"],
                dampening=config["dampening"],
                weight_decay=config["weight_decay"],
                nesterov=config["nesterov"],
                decoupled_weight_decay=decoupled_weight_decay,
                group_size=config["group_size"],
                quantize=config["quantize"],
                master_weight_bits=config["master_weight_bits"],
                fused=config["fused"],
            )
        )
    return count, treedef, leaf_results


def _make_sgd_transform(
    learning_rate: utils.ScalarOrSchedule,
    momentum: float,
    dampening: float,
    weight_decay: float,
    nesterov: bool,
    quantize: bool,
    master_weight_bits: int | None,
    group_size: int,
    fused: bool,
    decoupled_weight_decay: bool,
    param_groups: list[utils.ParamGroupSpec] | None,
    name: str,
) -> utils.FlashOptimizer:
    """Construct an SGD-style flash optimizer transform."""
    utils._validate_master_weight_bits(master_weight_bits)
    utils._validate_fused_group_size(group_size, quantize=quantize, fused=fused)
    _validate_sgd_hparams(momentum, dampening, nesterov)
    defaults = {
        "learning_rate": learning_rate,
        "momentum": momentum,
        "dampening": dampening,
        "weight_decay": weight_decay,
        "nesterov": nesterov,
        "quantize": quantize,
        "master_weight_bits": master_weight_bits,
        "group_size": group_size,
        "fused": fused,
    }

    def init_fn(params: Any) -> FlashSGDState:
        return _init_sgd_state(params, defaults, param_groups, name)

    def step_fn(params: Any, state: FlashSGDState, grads: Any) -> tuple[Any, FlashSGDState]:
        count, treedef, leaf_results = _sgd_leaf_results(
            params, grads, state, defaults, param_groups,
            decoupled_weight_decay, name,
        )
        new_params = treedef.unflatten([result.param for result in leaf_results])
        new_ecc = treedef.unflatten([result.ecc for result in leaf_results])
        new_mu = treedef.unflatten([result.mu for result in leaf_results])
        return new_params, FlashSGDState(count=count, mu=new_mu, ecc=new_ecc)

    return utils.FlashOptimizer(init=init_fn, step=step_fn)


def flash_sgd(
    learning_rate: utils.ScalarOrSchedule = 1e-3,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    quantize: bool = True,
    master_weight_bits: int | None = 24,
    group_size: int = 32,
    fused: bool = True,
    param_groups: list[utils.ParamGroupSpec] | None = None,
) -> utils.FlashOptimizer:
    """Build a flash SGD optimizer."""
    return _make_sgd_transform(
        learning_rate, momentum=momentum, dampening=dampening,
        weight_decay=weight_decay, nesterov=nesterov,
        quantize=quantize, master_weight_bits=master_weight_bits,
        group_size=group_size, fused=fused, param_groups=param_groups,
        decoupled_weight_decay=False,
        name="flash_sgd",
    )


def flash_sgdw(
    learning_rate: utils.ScalarOrSchedule = 1e-3,
    momentum: float = 0.0,
    dampening: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    quantize: bool = True,
    master_weight_bits: int | None = 24,
    group_size: int = 32,
    fused: bool = True,
    param_groups: list[utils.ParamGroupSpec] | None = None,
) -> utils.FlashOptimizer:
    """Build a decoupled-weight-decay flash SGD optimizer."""
    return _make_sgd_transform(
        learning_rate, momentum=momentum, dampening=dampening,
        weight_decay=weight_decay, nesterov=nesterov,
        quantize=quantize, master_weight_bits=master_weight_bits,
        group_size=group_size, fused=fused, param_groups=param_groups,
        decoupled_weight_decay=True,
        name="flash_sgdw",
    )

# Checkpointing utils

def flash_sgd_state_dict(state: FlashSGDState) -> dict[str, Any]:
    """Serialize SGD optimizer state into checkpointable trees."""
    return {
        "count": jnp.asarray(state.count),
        "mu": utils._tree_state_dict(state.mu),
        "ecc": utils._tree_state_dict(state.ecc),
    }


def load_flash_sgd_state_dict(state_dict: dict[str, Any]) -> FlashSGDState:
    """Restore SGD optimizer state from a serialized dict."""
    return FlashSGDState(
        count=jnp.asarray(state_dict["count"], dtype=jnp.int32),
        mu=utils._tree_from_state_dict(state_dict["mu"]),
        ecc=utils._tree_from_state_dict(state_dict["ecc"]),
    )
