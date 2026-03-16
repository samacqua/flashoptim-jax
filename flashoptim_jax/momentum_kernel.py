import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as plgpu

from . import utils
from .compression import no_ecc_leaf
from .quantization import QuantizedArray


class MomentumLeafStepResult(NamedTuple):
    """Per-leaf step result for momentum-based optimizers (Lion, SGD)."""
    param: jax.Array
    ecc: Any
    mu: Any


def _momentum_leaf_quantized_kernel(
    grad_ref, param_ref, ecc_ref, mu_values_ref, mu_scales_ref,
    lr_ref, mom_coef_ref, update_coef_ref, weight_decay_ref,
    update_ref, ecc_out_ref, mu_values_out_ref, mu_scales_out_ref,
    block_size: int,
    group_size: int,
    param_dtype: jnp.dtype,
    do_lion: bool,
    nesterov: bool,
    decoupled_weight_decay: bool,
    use_ecc: bool,
    ecc_dtype: jnp.dtype,
    ecc_max: float,
    mantissa_bits: int,
    min_normal_exponent: int,
):
    """Fused quantized momentum kernel for one parameter block (Lion or SGD)."""
    groups_per_block = block_size // group_size

    pid = pl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + jnp.arange(block_size)
    mask = offsets < param_ref.shape[0]

    scales_start = pid * groups_per_block
    scales_offsets = scales_start + jnp.arange(groups_per_block)
    scales_mask = scales_offsets < mu_scales_ref.shape[0]

    lr = plgpu.load(lr_ref).astype(jnp.float32)
    mom_coef = plgpu.load(mom_coef_ref).astype(jnp.float32)
    update_coef = plgpu.load(update_coef_ref).astype(jnp.float32)
    weight_decay = plgpu.load(weight_decay_ref).astype(jnp.float32)

    grad = plgpu.load(grad_ref.at[offsets], mask=mask, other=0.0).astype(jnp.float32)
    param_lp = plgpu.load(param_ref.at[offsets], mask=mask, other=0.0)
    ecc = plgpu.load(ecc_ref.at[offsets], mask=mask, other=0)
    mu_vals = plgpu.load(mu_values_ref.at[offsets], mask=mask, other=0).astype(jnp.float32)
    mu_sc = plgpu.load(mu_scales_ref.at[scales_offsets], mask=scales_mask, other=0.0).astype(
        jnp.float32
    )

    if use_ecc:
        param_f32 = utils._reconstruct_from_split(
            param_lp, ecc, ecc_max, mantissa_bits=mantissa_bits,
            min_normal_exponent=min_normal_exponent,
        )
    else:
        param_f32 = param_lp.astype(jnp.float32)

    mu_groups = mu_vals.reshape((groups_per_block, group_size)) / 127.0
    mu_normalized = mu_groups / (2.0 - jnp.abs(mu_groups))
    mu = (mu_normalized * mu_sc[:, None]).reshape((block_size,))

    if not do_lion and not decoupled_weight_decay:
        grad = grad + weight_decay * param_f32

    if do_lion:
        step = jnp.sign(update_coef * mu + (1.0 - update_coef) * grad)
        mu = mom_coef * mu + (1.0 - mom_coef) * grad
    else:
        mu = mom_coef * mu + update_coef * grad
        step = grad + mom_coef * mu if nesterov else mu

    if do_lion or decoupled_weight_decay:
        param_f32 = param_f32 * (1.0 - lr * weight_decay)
    param_f32 = param_f32 - lr * step

    if use_ecc:
        new_param_lp, new_ecc = utils._split_to_low_precision_ecc(
            param_f32, param_dtype=param_dtype, ecc_dtype=ecc_dtype, ecc_max=ecc_max,
            mantissa_bits=mantissa_bits, min_normal_exponent=min_normal_exponent,
        )
    else:
        new_param_lp = param_f32.astype(param_dtype)
        new_ecc = jnp.zeros_like(ecc)

    mu_groups = mu.reshape((groups_per_block, group_size))
    mu_absmaxs = jnp.maximum(jnp.max(jnp.abs(mu_groups), axis=1), 1e-12)
    mu_norm = mu_groups / mu_absmaxs[:, None]
    mu_transformed = 2.0 * mu_norm / (1.0 + jnp.abs(mu_norm))
    mu_out = jnp.floor(jnp.clip(mu_transformed * 127.0, -127.0, 127.0) + 0.5).reshape(
        (block_size,)
    ).astype(jnp.int8)

    plgpu.store(update_ref.at[offsets], new_param_lp, mask=mask)
    plgpu.store(ecc_out_ref.at[offsets], new_ecc, mask=mask)
    plgpu.store(mu_values_out_ref.at[offsets], mu_out, mask=mask)
    plgpu.store(
        mu_scales_out_ref.at[scales_offsets],
        mu_absmaxs.astype(jnp.float16),
        mask=scales_mask,
    )


def _momentum_leaf_full_kernel(
    grad_ref, param_ref, ecc_ref, mu_ref,
    lr_ref, mom_coef_ref, update_coef_ref, weight_decay_ref,
    update_ref, ecc_out_ref, mu_out_ref,
    block_size: int,
    param_dtype: jnp.dtype,
    do_lion: bool,
    nesterov: bool,
    decoupled_weight_decay: bool,
    use_ecc: bool,
    ecc_dtype: jnp.dtype,
    ecc_max: float,
    mantissa_bits: int,
    min_normal_exponent: int,
):
    """Fused full-precision momentum kernel for one parameter block (Lion or SGD)."""
    pid = pl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + jnp.arange(block_size)
    mask = offsets < param_ref.shape[0]

    lr = plgpu.load(lr_ref).astype(jnp.float32)
    mom_coef = plgpu.load(mom_coef_ref).astype(jnp.float32)
    update_coef = plgpu.load(update_coef_ref).astype(jnp.float32)
    weight_decay = plgpu.load(weight_decay_ref).astype(jnp.float32)

    grad = plgpu.load(grad_ref.at[offsets], mask=mask, other=0.0).astype(jnp.float32)
    param_lp = plgpu.load(param_ref.at[offsets], mask=mask, other=0.0)
    ecc = plgpu.load(ecc_ref.at[offsets], mask=mask, other=0)
    mu = plgpu.load(mu_ref.at[offsets], mask=mask, other=0.0).astype(jnp.float32)

    if use_ecc:
        param_f32 = utils._reconstruct_from_split(
            param_lp, ecc, ecc_max, mantissa_bits=mantissa_bits,
            min_normal_exponent=min_normal_exponent,
        )
    else:
        param_f32 = param_lp.astype(jnp.float32)

    if not do_lion and not decoupled_weight_decay:
        grad = grad + weight_decay * param_f32

    if do_lion:
        step = jnp.sign(update_coef * mu + (1.0 - update_coef) * grad)
        mu = mom_coef * mu + (1.0 - mom_coef) * grad
    else:
        mu = mom_coef * mu + update_coef * grad
        step = grad + mom_coef * mu if nesterov else mu

    if do_lion or decoupled_weight_decay:
        param_f32 = param_f32 * (1.0 - lr * weight_decay)
    param_f32 = param_f32 - lr * step

    if use_ecc:
        new_param_lp, new_ecc = utils._split_to_low_precision_ecc(
            param_f32, param_dtype=param_dtype, ecc_dtype=ecc_dtype,
            ecc_max=ecc_max, mantissa_bits=mantissa_bits,
            min_normal_exponent=min_normal_exponent,
        )
    else:
        new_param_lp = param_f32.astype(param_dtype)
        new_ecc = jnp.zeros_like(ecc)

    plgpu.store(update_ref.at[offsets], new_param_lp, mask=mask)
    plgpu.store(ecc_out_ref.at[offsets], new_ecc, mask=mask)
    plgpu.store(mu_out_ref.at[offsets], mu.astype(jnp.float32), mask=mask)


def fused_momentum_leaf_impl(
    grad: jax.Array,
    param: jax.Array,
    ecc: jax.Array,
    mu: Any,
    lr: jax.Array,
    mom_coef: float,
    update_coef: float,
    weight_decay: float,
    do_lion: bool,
    nesterov: bool,
    decoupled_weight_decay: bool,
    quantize: bool,
    group_size: int,
    use_ecc: bool,
    ecc_dtype: jnp.dtype = jnp.int8,
) -> MomentumLeafStepResult:
    """Dispatch one leaf through the fused momentum kernel (Lion or SGD)."""
    param_dtype = jnp.dtype(param.dtype)
    ecc_dtype = jnp.dtype(ecc_dtype)
    ecc_max, mantissa_bits, min_normal_exponent = utils._fused_ecc_constants(
        param_dtype, ecc_dtype, use_ecc,
    )

    ecc_array = jnp.zeros_like(param, dtype=ecc_dtype)
    if jnp.asarray(ecc).shape != ():
        ecc_array = jnp.asarray(ecc, dtype=ecc_dtype)

    if quantize:
        layout = utils.make_leaf_layout(param, group_size)
        kernel_fn = _momentum_leaf_quantized_kernel
        out_shape = [
            jax.ShapeDtypeStruct((layout.size,), param_dtype),
            jax.ShapeDtypeStruct((layout.size,), ecc_dtype),
            jax.ShapeDtypeStruct((layout.size,), jnp.int8),
            jax.ShapeDtypeStruct((layout.num_groups,), jnp.float16),
        ]
        in_specs = [pl.no_block_spec] * 9
        out_specs = [pl.no_block_spec] * 4
        input_output_aliases = {1: 0, 2: 1, 3: 2, 4: 3}
        name = "flash_momentum_leaf_step"
        kernel_args = (
            jnp.ravel(jnp.asarray(grad, dtype=jnp.float32)),
            jnp.ravel(jnp.asarray(param, dtype=param_dtype)),
            jnp.ravel(ecc_array),
            jnp.ravel(jnp.asarray(mu.values, dtype=jnp.int8)),
            jnp.asarray(mu.scales, dtype=jnp.float16).reshape((layout.num_groups,)),
            jnp.asarray(lr, dtype=jnp.float32),
            jnp.asarray(mom_coef, dtype=jnp.float32),
            jnp.asarray(update_coef, dtype=jnp.float32),
            jnp.asarray(weight_decay, dtype=jnp.float32),
        )
        kernel_kwargs = {"group_size": group_size}
    else:
        layout = utils.make_leaf_layout(param, utils.GROUP_SIZE)
        kernel_fn = _momentum_leaf_full_kernel
        out_shape = [
            jax.ShapeDtypeStruct((layout.size,), param_dtype),
            jax.ShapeDtypeStruct((layout.size,), ecc_dtype),
            jax.ShapeDtypeStruct((layout.size,), jnp.float32),
        ]
        in_specs = [pl.no_block_spec] * 8
        out_specs = [pl.no_block_spec] * 3
        input_output_aliases = {1: 0, 2: 1, 3: 2}
        name = "flash_momentum_leaf_full_step"
        kernel_args = (
            jnp.ravel(jnp.asarray(grad, dtype=jnp.float32)),
            jnp.ravel(jnp.asarray(param, dtype=param_dtype)),
            jnp.ravel(ecc_array),
            jnp.ravel(jnp.asarray(mu, dtype=jnp.float32)),
            jnp.asarray(lr, dtype=jnp.float32),
            jnp.asarray(mom_coef, dtype=jnp.float32),
            jnp.asarray(update_coef, dtype=jnp.float32),
            jnp.asarray(weight_decay, dtype=jnp.float32),
        )
        kernel_kwargs = {}

    num_blocks = (layout.size + utils.BLOCK_SIZE - 1) // utils.BLOCK_SIZE
    kernel = pl.pallas_call(
        functools.partial(
            kernel_fn,
            block_size=utils.BLOCK_SIZE,
            **kernel_kwargs,
            param_dtype=param_dtype,
            do_lion=do_lion,
            nesterov=nesterov,
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
        input_output_aliases=input_output_aliases,
        compiler_params=plgpu.CompilerParams(num_warps=8, num_stages=2),
        interpret=False,
        debug=False,
        name=name,
    )
    outputs = kernel(*kernel_args)

    if quantize:
        param_out, ecc_out, mu_values_out, mu_scales_out = outputs
        mu_out = QuantizedArray(
            values=mu_values_out.reshape(layout.shape).astype(jnp.int8),
            scales=mu_scales_out.astype(jnp.float16),
        )
    else:
        param_out, ecc_out, mu_out = outputs
        mu_out = mu_out.reshape(layout.shape).astype(jnp.float32)

    return MomentumLeafStepResult(
        param=param_out.reshape(layout.shape).astype(param_dtype),
        ecc=ecc_out.reshape(layout.shape).astype(ecc_dtype) if use_ecc else no_ecc_leaf(),
        mu=mu_out,
    )
