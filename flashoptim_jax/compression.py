"""Parameter compression helpers for low-precision weights plus ECC buffers."""

from typing import Any

import jax
import jax.numpy as jnp

from .utils import (
    _reconstruct_from_split,
    _split_to_low_precision_ecc,
    _validate_master_weight_bits,
    no_ecc_leaf,
)


_ECC_MAX_BY_DTYPE = {
    jnp.dtype(jnp.int8): 127.0,
    jnp.dtype(jnp.int16): 32767.0,
}
_LOW_PRECISION_DTYPES = (jnp.dtype(jnp.bfloat16), jnp.dtype(jnp.float16))
_LOW_PRECISION_CONSTANTS = {
    jnp.dtype(jnp.bfloat16): {
        "mantissa_bits": 7,
        "min_normal_exponent": -126,
        "exponent_bias": 127,
    },
    jnp.dtype(jnp.float16): {
        "mantissa_bits": 10,
        "min_normal_exponent": -14,
        "exponent_bias": 15,
    },
}


def cast_tree_bf16(tree: Any) -> Any:
    """Cast every leaf in a pytree to bfloat16."""
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.bfloat16), tree)

def _ecc_dtype_for(narrow_dtype: jnp.dtype, master_weight_bits: int | None) -> jnp.dtype | None:
    """Choose the ECC dtype implied by a stored dtype and master width."""
    _validate_master_weight_bits(master_weight_bits)
    narrow_dtype = jnp.dtype(narrow_dtype)
    if master_weight_bits is None or narrow_dtype not in _LOW_PRECISION_DTYPES:
        return None
    num_err_bytes = master_weight_bits // 8 - narrow_dtype.itemsize
    if num_err_bytes <= 0:
        return None
    if num_err_bytes == 1:
        return jnp.dtype(jnp.int8)
    if num_err_bytes == 2:
        return jnp.dtype(jnp.int16)
    raise ValueError(
        f"Unsupported ECC width for dtype {narrow_dtype} and master_weight_bits={master_weight_bits}"
    )


def _log_half_ulp(x_narrow: jax.Array) -> jax.Array:
    """Return the base-2 log of each value's half-ULP."""
    x_narrow = jnp.asarray(x_narrow)
    x_dtype = jnp.dtype(x_narrow.dtype)
    if x_dtype not in _LOW_PRECISION_CONSTANTS:
        raise ValueError(f"Unsupported low-precision dtype: {x_dtype}")
    constants = _LOW_PRECISION_CONSTANTS[x_dtype]
    mantissa_bits = constants["mantissa_bits"]
    min_normal_exponent = constants["min_normal_exponent"]
    exponent_bias = constants["exponent_bias"]

    x_bits = jnp.abs(x_narrow).view(jnp.uint16).astype(jnp.int32)
    exponent_bits = x_bits >> mantissa_bits
    unbiased_exponent = jnp.where(
        exponent_bits == 0,
        jnp.full_like(exponent_bits, min_normal_exponent),
        exponent_bits - exponent_bias,
    )
    return unbiased_exponent - mantissa_bits - 1

def has_ecc(theta_lp: jax.Array, rho: jax.Array) -> bool:
    """Return whether a stored parameter leaf has a usable ECC buffer."""
    theta_dtype = jnp.dtype(jnp.asarray(theta_lp).dtype)
    rho_dtype = jnp.dtype(jnp.asarray(rho).dtype)
    return theta_dtype in _LOW_PRECISION_DTYPES and rho_dtype in _ECC_MAX_BY_DTYPE


def split_leaf(
    theta: jax.Array,
    narrow_dtype: jnp.dtype = jnp.bfloat16,
    master_weight_bits: int | None = 24,
) -> tuple[jax.Array, jax.Array]:
    """Split one fp32 leaf into low-precision params and optional ECC."""
    theta_f32 = jnp.asarray(theta, dtype=jnp.float32)
    theta_lp = theta_f32.astype(narrow_dtype)
    ecc_dtype = _ecc_dtype_for(theta_lp.dtype, master_weight_bits)
    if ecc_dtype is None:
        return theta_lp, no_ecc_leaf()

    constants = _LOW_PRECISION_CONSTANTS[jnp.dtype(theta_lp.dtype)]
    return _split_to_low_precision_ecc(
        theta_f32,
        param_dtype=jnp.dtype(theta_lp.dtype),
        ecc_dtype=ecc_dtype,
        ecc_max=_ECC_MAX_BY_DTYPE[ecc_dtype],
        mantissa_bits=constants["mantissa_bits"],
        min_normal_exponent=constants["min_normal_exponent"],
    )


def reconstruct_leaf(theta_lp: jax.Array, rho: jax.Array) -> jax.Array:
    """Reconstruct one fp32 leaf from low-precision params and ECC."""
    theta_lp = jnp.asarray(theta_lp)
    rho = jnp.asarray(rho)
    if not has_ecc(theta_lp, rho):
        return theta_lp.astype(jnp.float32)

    constants = _LOW_PRECISION_CONSTANTS[jnp.dtype(theta_lp.dtype)]
    return _reconstruct_from_split(
        theta_lp,
        rho,
        _ECC_MAX_BY_DTYPE[jnp.dtype(rho.dtype)],
        mantissa_bits=constants["mantissa_bits"],
        min_normal_exponent=constants["min_normal_exponent"],
    )


def reconstruct_weights(params_lp: Any, errors: Any) -> Any:
    """Reconstruct fp32 weights across a parameter pytree."""
    return jax.tree_util.tree_map(reconstruct_leaf, params_lp, errors)


def set_fp32_params(
    fp32_params: Any,
    param_template: Any,
    master_weight_bits: int | None = 24,
) -> tuple[Any, Any]:
    """Convert fp32 weights into stored params and ECC trees."""
    param_dtypes = jax.tree_util.tree_map(lambda x: jnp.asarray(x).dtype, param_template)
    leaves, treedef = jax.tree_util.tree_flatten(fp32_params)
    dtype_leaves, dtype_treedef = jax.tree_util.tree_flatten(param_dtypes)
    if treedef != dtype_treedef:
        raise ValueError("fp32_params and param_template must have matching tree structure")
    split_leaves = [
        split_leaf(leaf, narrow_dtype=dtype_leaf, master_weight_bits=master_weight_bits)
        for leaf, dtype_leaf in zip(leaves, dtype_leaves, strict=True)
    ]
    params = treedef.unflatten([param for param, _ in split_leaves])
    ecc = treedef.unflatten([ecc_leaf for _, ecc_leaf in split_leaves])
    return params, ecc
