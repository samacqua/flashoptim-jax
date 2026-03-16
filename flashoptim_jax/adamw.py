import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from . import utils
from .compression import no_ecc_leaf, reconstruct_leaf, split_leaf
from .quantization import QuantizedArray


class FlashAdamState(NamedTuple):
    count: jax.Array
    mu: Any
    nu: Any
    ecc: Any


class AdamWLeafStepResult(NamedTuple):
    param: jax.Array
    ecc: Any
    mu: Any
    nu: Any


def _flash_adamw_leaf_quantized_kernel(
    grad_ref, param_ref, ecc_ref, mu_values_ref, nu_values_ref, mu_scales_ref,
    nu_scales_ref, lr_ref, b1_ref, b2_ref, eps_ref, weight_decay_ref,
    bias_correction1_ref, bias_correction2_ref, update_ref, ecc_out_ref,
    mu_values_out_ref, mu_scales_out_ref, nu_values_out_ref, nu_scales_out_ref,
    block_size: int,
    group_size: int,
    param_dtype: jnp.dtype,
    decoupled_weight_decay: bool,
    use_ecc: bool,
    ecc_dtype: jnp.dtype,
    ecc_max: float,
    mantissa_bits: int,
    min_normal_exponent: int,
):
    """Apply one fused AdamW step for a quantized leaf block."""

    # Grid indexing
    groups_per_block = block_size // group_size
    pid = pl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + jnp.arange(block_size)
    mask = offsets < param_ref.shape[0]
    scales_start = pid * groups_per_block
    scales_offsets = scales_start + jnp.arange(groups_per_block)
    scales_mask = scales_offsets < mu_scales_ref.shape[0]

    # Load everything
    lr = plgpu.load(lr_ref).astype(jnp.float32)
    b1 = plgpu.load(b1_ref).astype(jnp.float32)
    b2 = plgpu.load(b2_ref).astype(jnp.float32)
    eps = plgpu.load(eps_ref).astype(jnp.float32)
    weight_decay = plgpu.load(weight_decay_ref).astype(jnp.float32)
    bias_correction1 = plgpu.load(bias_correction1_ref).astype(jnp.float32)
    bias_correction2 = plgpu.load(bias_correction2_ref).astype(jnp.float32)

    grad = plgpu.load(grad_ref.at[offsets], mask=mask, other=0.0).astype(jnp.float32)
    param_lp = plgpu.load(param_ref.at[offsets], mask=mask, other=0.0)
    ecc = plgpu.load(ecc_ref.at[offsets], mask=mask, other=0)
    mu_vals = plgpu.load(mu_values_ref.at[offsets], mask=mask, other=0).astype(jnp.float32)
    nu_vals = plgpu.load(nu_values_ref.at[offsets], mask=mask, other=0).astype(jnp.float32)
    mu_sc = plgpu.load(mu_scales_ref.at[scales_offsets], mask=scales_mask, other=0.0).astype(
        jnp.float32
    )
    nu_sc = plgpu.load(nu_scales_ref.at[scales_offsets], mask=scales_mask, other=0.0).astype(
        jnp.float32
    )

    # Reconstruct fp32 master weight
    if use_ecc:
        param_f32 = utils._reconstruct_from_split(
            param_lp, ecc, ecc_max,
            mantissa_bits=mantissa_bits,
            min_normal_exponent=min_normal_exponent,
        )
    else:
        param_f32 = param_lp.astype(jnp.float32)

    # Dequantize momentum and variance
    mu_groups = mu_vals.reshape((groups_per_block, group_size)) / 127.0     # [-1, 1]
    mu_normalized = mu_groups / (2.0 - jnp.abs(mu_groups))  # inverse soft-sign
    mu = (mu_normalized * mu_sc[:, None]).reshape((block_size,))    # restore og scale

    nu_groups = nu_vals.reshape((groups_per_block, group_size)) / 255.0     # [0, 1]
    nu_sqrt = (nu_groups * nu_sc[:, None]).reshape((block_size,))   # restore og scale (of sqrt)
    nu = nu_sqrt * nu_sqrt  # square to get variance

    # Standard adam update
    if not decoupled_weight_decay:
        grad = grad + weight_decay * param_f32

    mu = b1 * mu + (1.0 - b1) * grad
    nu = b2 * nu + (1.0 - b2) * jnp.square(grad)

    if decoupled_weight_decay:
        param_f32 = param_f32 * (1.0 - lr * weight_decay)
    param_f32 = param_f32 - (lr / bias_correction1) * mu / (
        jnp.sqrt(nu) / jnp.sqrt(bias_correction2) + eps
    )

    # Split updated param back to low-precision + ECC
    if use_ecc:
        new_param_lp, new_ecc = utils._split_to_low_precision_ecc(
            param_f32, param_dtype=param_dtype, ecc_dtype=ecc_dtype,
            ecc_max=ecc_max, mantissa_bits=mantissa_bits,
            min_normal_exponent=min_normal_exponent,
        )
    else:
        new_param_lp = param_f32.astype(param_dtype)
        new_ecc = jnp.zeros_like(ecc)

    # Re-quantize momentum and variance
    mu_groups = mu.reshape((groups_per_block, group_size))                # group fp32 momentum
    mu_absmaxs = jnp.maximum(jnp.max(jnp.abs(mu_groups), axis=1), 1e-12)  # per-group scale
    mu_norm = mu_groups / mu_absmaxs[:, None]                             # [-1, 1]
    mu_transformed = 2.0 * mu_norm / (1.0 + jnp.abs(mu_norm))             # softsign
    mu_out = jnp.floor(jnp.clip(                                          # round to int8
        mu_transformed * 127.0, -127.0, 127.0) + 0.5).reshape((block_size,)).astype(jnp.int8)                                                    

    nu_sqrt_groups = jnp.sqrt(nu).reshape((groups_per_block, group_size))  # sqrt pre-scale
    nu_absmaxs = jnp.maximum(jnp.max(nu_sqrt_groups, axis=1), 1e-12)       # per-group scale
    nu_norm = nu_sqrt_groups / nu_absmaxs[:, None]                         # [0, 1]
    nu_out = jnp.floor(jnp.clip(                                           # round uint8
        nu_norm * 255.0, 0.0, 255.0) + 0.5).reshape((block_size,)).astype(jnp.uint8)

    # Store results
    plgpu.store(update_ref.at[offsets], new_param_lp, mask=mask)
    plgpu.store(ecc_out_ref.at[offsets], new_ecc, mask=mask)
    plgpu.store(mu_values_out_ref.at[offsets], mu_out, mask=mask)
    plgpu.store(nu_values_out_ref.at[offsets], nu_out, mask=mask)
    plgpu.store(
        mu_scales_out_ref.at[scales_offsets], mu_absmaxs.astype(jnp.float16),
        mask=scales_mask,
    )
    plgpu.store(
        nu_scales_out_ref.at[scales_offsets], nu_absmaxs.astype(jnp.float16),
        mask=scales_mask,
    )


def _flash_adamw_leaf_full_kernel(
    grad_ref, param_ref, ecc_ref, mu_ref, nu_ref, lr_ref, b1_ref, b2_ref,
    eps_ref, weight_decay_ref, bias_correction1_ref, bias_correction2_ref,
    update_ref, ecc_out_ref, mu_out_ref, nu_out_ref,
    block_size: int,
    param_dtype: jnp.dtype,
    decoupled_weight_decay: bool,
    use_ecc: bool,
    ecc_dtype: jnp.dtype,
    ecc_max: float,
    mantissa_bits: int,
    min_normal_exponent: int,
):
    """Apply one fused AdamW step for a full-precision leaf block."""

    # Grid indexing
    pid = pl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + jnp.arange(block_size)
    mask = offsets < param_ref.shape[0]

    # Load everything
    lr = plgpu.load(lr_ref).astype(jnp.float32)
    b1 = plgpu.load(b1_ref).astype(jnp.float32)
    b2 = plgpu.load(b2_ref).astype(jnp.float32)
    eps = plgpu.load(eps_ref).astype(jnp.float32)
    weight_decay = plgpu.load(weight_decay_ref).astype(jnp.float32)
    bias_correction1 = plgpu.load(bias_correction1_ref).astype(jnp.float32)
    bias_correction2 = plgpu.load(bias_correction2_ref).astype(jnp.float32)

    grad = plgpu.load(grad_ref.at[offsets], mask=mask, other=0.0).astype(jnp.float32)
    param_lp = plgpu.load(param_ref.at[offsets], mask=mask, other=0.0)
    ecc = plgpu.load(ecc_ref.at[offsets], mask=mask, other=0)
    mu = plgpu.load(mu_ref.at[offsets], mask=mask, other=0.0).astype(jnp.float32)
    nu = plgpu.load(nu_ref.at[offsets], mask=mask, other=0.0).astype(jnp.float32)

    # Reconstruct fp32 master weight
    if use_ecc:
        param_f32 = utils._reconstruct_from_split(
            param_lp, ecc, ecc_max,
            mantissa_bits=mantissa_bits,
            min_normal_exponent=min_normal_exponent,
        )
    else:
        param_f32 = param_lp.astype(jnp.float32)

    # Standard Adam update
    if not decoupled_weight_decay:
        grad = grad + weight_decay * param_f32

    mu = b1 * mu + (1.0 - b1) * grad
    nu = b2 * nu + (1.0 - b2) * jnp.square(grad)

    if decoupled_weight_decay:
        param_f32 = param_f32 * (1.0 - lr * weight_decay)
    param_f32 = param_f32 - (lr / bias_correction1) * mu / (
        jnp.sqrt(nu) / jnp.sqrt(bias_correction2) + eps
    )

    # Split updated param back to low-precision + ECC
    if use_ecc:
        new_param_lp, new_ecc = utils._split_to_low_precision_ecc(
            param_f32, param_dtype=param_dtype, ecc_dtype=ecc_dtype,
            ecc_max=ecc_max, mantissa_bits=mantissa_bits,
            min_normal_exponent=min_normal_exponent,
        )
    else:
        new_param_lp = param_f32.astype(param_dtype)
        new_ecc = jnp.zeros_like(ecc)

    # Store results
    plgpu.store(update_ref.at[offsets], new_param_lp, mask=mask)
    plgpu.store(ecc_out_ref.at[offsets], new_ecc, mask=mask)
    plgpu.store(mu_out_ref.at[offsets], mu.astype(jnp.float32), mask=mask)
    plgpu.store(nu_out_ref.at[offsets], nu.astype(jnp.float32), mask=mask)


def _fused_flash_adamw_leaf_impl(
    grad: jax.Array,
    param: jax.Array,
    ecc: jax.Array,
    mu: Any,
    nu: Any,
    lr: jax.Array,
    b1: float,
    b2: float,
    eps: float,
    weight_decay: float,
    bias_correction1: jax.Array,
    bias_correction2: jax.Array,
    quantize: bool,
    group_size: int,
    decoupled_weight_decay: bool,
    use_ecc: bool,
    ecc_dtype: jnp.dtype,
) -> AdamWLeafStepResult:
    """Run the fused AdamW implementation for one leaf."""
    param_dtype = jnp.dtype(param.dtype)
    ecc_dtype = jnp.dtype(ecc_dtype)
    ecc_max, mantissa_bits, min_normal_exponent = utils._fused_ecc_constants(
        param_dtype, ecc_dtype, use_ecc,
    )

    # Setup the kernel + args to kernel.
    if quantize:
        layout, state = utils.pack_leaf_state_quantized(
            grad, param, ecc, mu, nu, group_size, ecc_dtype, param_dtype,
        )
        kernel_fn = _flash_adamw_leaf_quantized_kernel
        out_shape = [
            jax.ShapeDtypeStruct((layout.size,), param_dtype),
            jax.ShapeDtypeStruct((layout.size,), ecc_dtype),
            jax.ShapeDtypeStruct((layout.size,), jnp.int8),
            jax.ShapeDtypeStruct((layout.num_groups,), jnp.float16),
            jax.ShapeDtypeStruct((layout.size,), jnp.uint8),
            jax.ShapeDtypeStruct((layout.num_groups,), jnp.float16),
        ]
        in_specs = [pl.no_block_spec] * 14
        out_specs = [pl.no_block_spec] * 6
        name = "flash_adamw_leaf_step"
        input_output_aliases = {1: 0, 2: 1, 3: 2, 5: 3, 4: 4, 6: 5}
        kernel_args = (
            state.grad, state.param, state.ecc, state.mu_values, state.nu_values,
            state.mu_scales, state.nu_scales,
            jnp.asarray(lr, dtype=jnp.float32),
            jnp.asarray(b1, dtype=jnp.float32),
            jnp.asarray(b2, dtype=jnp.float32),
            jnp.asarray(eps, dtype=jnp.float32),
            jnp.asarray(weight_decay, dtype=jnp.float32),
            jnp.asarray(bias_correction1, dtype=jnp.float32),
            jnp.asarray(bias_correction2, dtype=jnp.float32),
        )
        kernel_kwargs = {"group_size": group_size}
    else:
        layout, state = utils.pack_leaf_state_full(grad, param, ecc, mu, nu, ecc_dtype, param_dtype)
        kernel_fn = _flash_adamw_leaf_full_kernel
        out_shape = [
            jax.ShapeDtypeStruct((layout.size,), param_dtype),
            jax.ShapeDtypeStruct((layout.size,), ecc_dtype),
            jax.ShapeDtypeStruct((layout.size,), jnp.float32),
            jax.ShapeDtypeStruct((layout.size,), jnp.float32),
        ]
        in_specs = [pl.no_block_spec] * 12
        out_specs = [pl.no_block_spec] * 4
        name = "flash_adamw_leaf_full_step"
        input_output_aliases = {1: 0, 2: 1, 3: 2, 4: 3}
        kernel_args = (
            state.grad, state.param, state.ecc, state.mu, state.nu,
            jnp.asarray(lr, dtype=jnp.float32),
            jnp.asarray(b1, dtype=jnp.float32),
            jnp.asarray(b2, dtype=jnp.float32),
            jnp.asarray(eps, dtype=jnp.float32),
            jnp.asarray(weight_decay, dtype=jnp.float32),
            jnp.asarray(bias_correction1, dtype=jnp.float32),
            jnp.asarray(bias_correction2, dtype=jnp.float32),
        )
        kernel_kwargs = {}

    # Format args + call kernel.
    num_blocks = (layout.size + utils.BLOCK_SIZE - 1) // utils.BLOCK_SIZE
    kernel = pl.pallas_call(
        functools.partial(
            kernel_fn,
            block_size=utils.BLOCK_SIZE,
            **kernel_kwargs,
            param_dtype=param_dtype,
            decoupled_weight_decay=decoupled_weight_decay,
            use_ecc=use_ecc,
            ecc_dtype=ecc_dtype,
            ecc_max=ecc_max,
            mantissa_bits=mantissa_bits,
            min_normal_exponent=min_normal_exponent,
        ),
        out_shape=out_shape,
        grid=(num_blocks,),
        in_specs=in_specs,
        out_specs=out_specs,
        compiler_params=plgpu.CompilerParams(num_warps=8, num_stages=2),
        debug=False,
        name=name,
        input_output_aliases=input_output_aliases,
    )
    outputs = kernel(*kernel_args)

    # Format kernel output.
    if quantize:
        param_out, ecc_out, mu_values_out, mu_scales_out, nu_values_out, nu_scales_out = outputs
        mu_out = QuantizedArray(
            values=mu_values_out.reshape(layout.shape).astype(jnp.int8),
            scales=mu_scales_out.astype(jnp.float16),
        )
        nu_out = QuantizedArray(
            values=nu_values_out.reshape(layout.shape).astype(jnp.uint8),
            scales=nu_scales_out.astype(jnp.float16),
        )
    else:
        param_out, ecc_out, mu_out, nu_out = outputs
        mu_out = mu_out.reshape(layout.shape).astype(jnp.float32)
        nu_out = nu_out.reshape(layout.shape).astype(jnp.float32)

    return AdamWLeafStepResult(
        param=param_out.reshape(layout.shape).astype(param_dtype),
        ecc=ecc_out.reshape(layout.shape).astype(ecc_dtype) if use_ecc else no_ecc_leaf(),
        mu=mu_out,
        nu=nu_out,
    )


def _flash_adamw_leaf_unfused(
    grad: jax.Array,
    param: jax.Array,
    ecc: jax.Array,
    mu: Any,
    nu: Any,
    lr: jax.Array,
    b1: float,
    b2: float,
    eps: float,
    weight_decay: float,
    decoupled_weight_decay: bool,
    bias_correction1: jax.Array,
    bias_correction2: jax.Array,
    quantize: bool,
    group_size: int,
    master_weight_bits: int | None,
) -> utils.LeafStepResult:
    """Apply an unfused AdamW update to one parameter leaf.
    
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
    nu_f32 = utils._materialize_variance(nu, group_size)

    # Adam / AdamW update in fp32.
    if not decoupled_weight_decay and weight_decay:
        grad_f32 = grad_f32 + weight_decay * param_f32

    mu_f32 = b1 * mu_f32 + (1.0 - b1) * grad_f32
    nu_f32 = b2 * nu_f32 + (1.0 - b2) * (grad_f32 * grad_f32)

    if decoupled_weight_decay and weight_decay:
        param_f32 = param_f32 * (1.0 - lr * weight_decay)

    new_param_f32 = param_f32 - (lr / bias_correction1) * mu_f32 / (
        jnp.sqrt(nu_f32) / jnp.sqrt(bias_correction2) + eps
    )

    # Store the updated parameter back in the configured param/ECC format.
    if use_ecc:
        new_param, new_ecc = split_leaf(
            new_param_f32,
            narrow_dtype=param.dtype,
            master_weight_bits=master_weight_bits,
        )
        update = (new_param.astype(param.dtype) - param).astype(param.dtype)
    else:
        new_param = new_param_f32.astype(param.dtype)
        new_ecc = no_ecc_leaf()
        update = (new_param - param).astype(param.dtype)

    # Re-store moments in either quantized or full form.
    return utils.LeafStepResult(
        update=update,
        ecc=new_ecc,
        mu=utils._store_momentum(mu_f32, quantize, group_size),
        nu=utils._store_variance(nu_f32, quantize, group_size),
    )


def _flash_adamw_leaf_impl(
    grad: jax.Array,
    param: jax.Array,
    ecc: jax.Array,
    mu: Any,
    nu: Any,
    lr: jax.Array,
    b1: float,
    b2: float,
    eps: float,
    weight_decay: float,
    decoupled_weight_decay: bool,
    bias_correction1: jax.Array,
    bias_correction2: jax.Array,
    quantize: bool,
    group_size: int,
    master_weight_bits: int | None,
    fused: bool,
) -> AdamWLeafStepResult:
    """Dispatch a leaf update to the fused or unfused AdamW path."""
    use_ecc = utils._use_ecc_leaf(param, master_weight_bits)
    ecc_dtype = jnp.int16 if master_weight_bits == 32 else jnp.int8
    if fused:
        return _fused_flash_adamw_leaf_impl(
            grad, param, ecc, mu, nu, lr=lr, b1=b1, b2=b2, eps=eps,
            weight_decay=weight_decay, decoupled_weight_decay=decoupled_weight_decay,
            bias_correction1=bias_correction1, bias_correction2=bias_correction2,
            quantize=quantize, group_size=group_size, use_ecc=use_ecc,
            ecc_dtype=ecc_dtype,
        )

    result = _flash_adamw_leaf_unfused(
        grad, param, ecc, mu, nu, lr=lr, b1=b1, b2=b2, eps=eps,
        weight_decay=weight_decay, decoupled_weight_decay=decoupled_weight_decay,
        bias_correction1=bias_correction1, bias_correction2=bias_correction2,
        quantize=quantize, group_size=group_size, master_weight_bits=master_weight_bits,
    )
    new_param = param + result.update
    return AdamWLeafStepResult(param=new_param, ecc=result.ecc, mu=result.mu, nu=result.nu)


def _init_adam_state(
    params: Any,
    defaults: dict[str, Any],
    param_groups: list[utils.ParamGroupSpec] | None,
    name: str,
) -> FlashAdamState:
    """Initialize Adam state trees for a parameter pytree."""
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
        leaf_states.append(
            utils._init_leaf_state(
                leaf, config["group_size"], quantize=config["quantize"],
                master_weight_bits=config["master_weight_bits"],
            )
        )
    return FlashAdamState(
        count=jnp.zeros((), dtype=jnp.int32),
        mu=treedef.unflatten([leaf_state[0] for leaf_state in leaf_states]),
        nu=treedef.unflatten([leaf_state[1] for leaf_state in leaf_states]),
        ecc=treedef.unflatten([leaf_state[2] for leaf_state in leaf_states]),
    )


def _compute_adam_step(
    params: Any,
    grads: Any,
    state: FlashAdamState | FlashAdamState,
    defaults: dict[str, Any],
    param_groups: list[utils.ParamGroupSpec] | None,
    decoupled_weight_decay: bool,
    name: str,
) -> tuple[jax.Array, Any, list[Any]]:
    """Compute per-leaf AdamW step results for a pytree."""
    count = state.count + jnp.asarray(1, dtype=jnp.int32)
    grad_paths, grad_leaves, grad_treedef = utils._tree_leaves_with_paths(grads)
    paths, leaves, treedef = utils._tree_leaves_with_paths(params)
    ecc_leaves, ecc_treedef = jax.tree_util.tree_flatten(state.ecc)
    mu_leaves, mu_treedef = jax.tree_util.tree_flatten(state.mu, is_leaf=utils._is_quantized_leaf)
    nu_leaves, nu_treedef = jax.tree_util.tree_flatten(state.nu, is_leaf=utils._is_quantized_leaf)
    if not (treedef == grad_treedef == ecc_treedef == mu_treedef == nu_treedef):
        raise ValueError(f"{name} state trees must match parameter structure")
    if paths != grad_paths:
        raise ValueError(f"{name} gradients must match parameter paths")

    leaf_results = []
    for path, grad_leaf, param_leaf, ecc_leaf, mu_leaf, nu_leaf in zip(
        paths, grad_leaves, leaves, ecc_leaves, mu_leaves, nu_leaves, strict=True,
    ):
        config = utils._group_config_for_path(path, param_leaf, defaults=defaults, param_groups=param_groups)
        lr = utils._resolve_learning_rate(config["learning_rate"], state.count)
        bias_correction1 = 1.0 - config["b1"] ** count.astype(jnp.float32)
        bias_correction2 = 1.0 - config["b2"] ** count.astype(jnp.float32)
        leaf_results.append(
            _flash_adamw_leaf_impl(
                jnp.asarray(grad_leaf, dtype=jnp.float32),
                param_leaf, ecc_leaf, mu_leaf, nu_leaf, lr=lr,
                b1=config["b1"], b2=config["b2"], eps=config["eps"],
                weight_decay=config["weight_decay"],
                decoupled_weight_decay=decoupled_weight_decay, bias_correction1=bias_correction1,
                bias_correction2=bias_correction2, quantize=config["quantize"],
                group_size=config["group_size"], master_weight_bits=config["master_weight_bits"],
                fused=config["fused"],
            )
        )
    return count, treedef, leaf_results


def _make_adam_transform(
    learning_rate: utils.ScalarOrSchedule,
    b1: float,
    b2: float,
    eps: float,
    weight_decay: float,
    quantize: bool,
    master_weight_bits: int | None,
    group_size: int,
    fused: bool,
    decoupled_weight_decay: bool,
    param_groups: list[utils.ParamGroupSpec] | None,
    name: str,
) -> utils.FlashOptimizer:
    """Create a FlashOptimizer wrapper for Adam-style updates."""

    utils._validate_master_weight_bits(master_weight_bits)
    utils._validate_fused_group_size(group_size, quantize=quantize, fused=fused)
    defaults = {    # Potentially overrided by param_groups.
        "learning_rate": learning_rate,
        "b1": b1, "b2": b2, "eps": eps,
        "weight_decay": weight_decay,
        "quantize": quantize,
        "master_weight_bits": master_weight_bits,
        "group_size": group_size,
        "fused": fused,
    }

    def init_fn(params: Any) -> FlashAdamState:
        """Initialize optimizer state for a parameter pytree."""
        return _init_adam_state(params, defaults, param_groups, name)

    def step_fn(
        params: Any,
        state: FlashAdamState,
        grads: Any,
    ) -> tuple[Any, FlashAdamState]:
        """Apply one optimizer step to parameters and state."""
        count, treedef, leaf_results = _compute_adam_step(
            params, grads, state, defaults, param_groups,
            decoupled_weight_decay, name,
        )
        new_params = treedef.unflatten([result.param for result in leaf_results])
        new_ecc = treedef.unflatten([result.ecc for result in leaf_results])
        new_mu = treedef.unflatten([result.mu for result in leaf_results])
        new_nu = treedef.unflatten([result.nu for result in leaf_results])
        return new_params, FlashAdamState(count=count, mu=new_mu, nu=new_nu, ecc=new_ecc)

    return utils.FlashOptimizer(init=init_fn, step=step_fn)


def flash_adam(
    learning_rate: utils.ScalarOrSchedule = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    quantize: bool = True,
    master_weight_bits: int | None = 24,
    group_size: int = 32,
    fused: bool = True,
    param_groups: list[utils.ParamGroupSpec] | None = None,
) -> utils.FlashOptimizer:
    """Construct the flash Adam optimizer."""
    return _make_adam_transform(
        learning_rate, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay, quantize=quantize,
        master_weight_bits=master_weight_bits, group_size=group_size, fused=fused,
        decoupled_weight_decay=False, param_groups=param_groups,
        name="flash_adam",
    )


def flash_adamw(
    learning_rate: utils.ScalarOrSchedule = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    quantize: bool = True,
    master_weight_bits: int | None = 24,
    group_size: int = 32,
    fused: bool = True,
    param_groups: list[utils.ParamGroupSpec] | None = None,
) -> utils.FlashOptimizer:
    """Construct the flash AdamW optimizer."""
    return _make_adam_transform(
        learning_rate, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay, quantize=quantize,
        master_weight_bits=master_weight_bits, group_size=group_size, fused=fused,
        decoupled_weight_decay=True, param_groups=param_groups,
        name="flash_adamw",
    )


# Checkpointing utils

def flash_adamw_state_dict(state: FlashAdamState) -> dict[str, Any]:
    """Serialize AdamW optimizer state into checkpointable trees."""
    return {
        "count": jnp.asarray(state.count), "mu": utils._tree_state_dict(state.mu),
        "nu": utils._tree_state_dict(state.nu), "ecc": utils._tree_state_dict(state.ecc),
    }


def load_flash_adamw_state_dict(state_dict: dict[str, Any]) -> FlashAdamState:
    """Restore AdamW optimizer state from a serialized dict."""
    return FlashAdamState(
        count=jnp.asarray(state_dict["count"], dtype=jnp.int32),
        mu=utils._tree_from_state_dict(state_dict["mu"]),
        nu=utils._tree_from_state_dict(state_dict["nu"]),
        ecc=utils._tree_from_state_dict(state_dict["ecc"]),
    )
