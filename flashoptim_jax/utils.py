"""Shared utilities for optimizer state, validation, and fused kernels."""

from collections.abc import Iterable
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from .quantization import (
    QuantizedArray,
    dequantize_momentum,
    dequantize_variance,
    quantize_momentum,
    quantize_variance,
)


ECC_MAX_INT8 = 127.0
ECC_MAX_INT16 = 32767.0
BLOCK_SIZE = 1024
GROUP_SIZE = 32

ScalarOrSchedule = float | Callable[[jax.Array], jax.Array]
ParamGroupSpec = dict[str, Any]
_NO_STATE_DTYPE = jnp.int32

_DTYPE_CONSTANTS = {
    jnp.dtype(jnp.bfloat16): {"mantissa_bits": 7, "min_normal_exponent": -126},
    jnp.dtype(jnp.float16): {"mantissa_bits": 10, "min_normal_exponent": -14},
}
_LOW_PRECISION_PARAM_DTYPES = tuple(_DTYPE_CONSTANTS)
_FUSED_PARAM_DTYPES = (jnp.dtype(jnp.float32),) + _LOW_PRECISION_PARAM_DTYPES


class LeafKernelLayout(NamedTuple):
    """Metadata describing how one parameter leaf is packed for kernels."""
    shape: tuple[int, ...]
    size: int
    num_groups: int
    group_size: int


class LeafKernelQuantizedState(NamedTuple):
    """Flattened quantized leaf state passed into fused kernels."""
    grad: jax.Array
    param: jax.Array
    ecc: jax.Array
    mu_values: jax.Array
    mu_scales: jax.Array
    nu_values: jax.Array
    nu_scales: jax.Array


class LeafKernelFullState(NamedTuple):
    """Flattened full-precision leaf state passed into fused kernels."""
    grad: jax.Array
    param: jax.Array
    ecc: jax.Array
    mu: jax.Array
    nu: jax.Array


class LeafStepResult(NamedTuple):
    """Per-leaf optimizer outputs from the unfused update path."""
    update: jax.Array
    ecc: Any
    mu: Any
    nu: Any


class FlashOptimizer(NamedTuple):
    """Minimal optimizer interface with `init` and `step` callables."""
    init: Callable[[Any], Any]
    step: Callable[[Any, Any, Any], tuple[Any, Any]]


def _is_quantized_leaf(x: Any) -> bool:
    """Return whether a leaf stores quantized state."""
    return isinstance(x, QuantizedArray)


def _path_entry_value(entry: Any) -> Any:
    """Extract a stable path component from a JAX tree path entry."""
    for attr in ("key", "idx", "name"):
        if hasattr(entry, attr):
            return getattr(entry, attr)
    return str(entry)


def _normalize_path(path: Any) -> tuple[Any, ...]:
    """Convert a JAX path object into a comparable tuple form."""
    return tuple(_path_entry_value(entry) for entry in path)


def no_ecc_leaf() -> jax.Array:
    """Sentinel leaf used when a parameter has no ECC buffer."""
    return jnp.zeros((), dtype=_NO_STATE_DTYPE)


def no_state_leaf() -> jax.Array:
    """Sentinel leaf used when an optimizer state slot is absent."""
    return jnp.zeros((), dtype=_NO_STATE_DTYPE)


def _has_state_leaf(x: Any) -> bool:
    """Return whether a serialized state leaf represents real state."""
    if _is_quantized_leaf(x):
        return True
    array = jnp.asarray(x)
    return not (array.shape == () and array.dtype == _NO_STATE_DTYPE)


def _is_sequence_of_selectors(value: Any) -> bool:
    """Return whether a param-group selector is an iterable of selectors."""
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes, tuple))


def _selector_matches(path: tuple[Any, ...], leaf: Any, selector: Any) -> bool:
    """Return whether one selector matches a parameter path and leaf."""
    if callable(selector):
        return bool(selector(path, leaf))
    if isinstance(selector, tuple):
        return path == selector
    if isinstance(selector, str):
        path_segments = [str(entry) for entry in path]
        if "/" in selector:
            return "/".join(path_segments) == selector
        return selector in path_segments
    return False


def _path_matches(path: tuple[Any, ...], leaf: Any, params_spec: Any) -> bool:
    """Return whether a param-group spec matches a parameter leaf."""
    if _is_sequence_of_selectors(params_spec):
        return any(_selector_matches(path, leaf, selector) for selector in params_spec)
    return _selector_matches(path, leaf, params_spec)


def _group_config_for_path(
    path: tuple[Any, ...],
    leaf: Any,
    defaults: dict[str, Any],
    param_groups: list[ParamGroupSpec] | None,
) -> dict[str, Any]:
    """Resolve the effective optimizer config for one parameter leaf."""
    config = dict(defaults)
    if param_groups is None:
        return config
    for group in param_groups:
        if "params" not in group:
            raise ValueError("Each param group must define a 'params' matcher")
        if _path_matches(path, leaf, group["params"]):
            config.update({k: v for k, v in group.items() if k != "params"})
            return config
    return config


def _validate_master_weight_bits(master_weight_bits: int | None) -> None:
    """Validate the supported master weight bit-widths."""
    if master_weight_bits not in (None, 24, 32):
        raise ValueError(
            f"master_weight_bits must be one of (None, 24, 32), got {master_weight_bits}"
        )


def _validate_fused_group_size(group_size: int, *, quantize: bool, fused: bool) -> None:
    """Validate group sizes required by the fused quantized kernels."""
    if fused and quantize and BLOCK_SIZE % group_size != 0:
        raise ValueError(
            f"group_size must divide BLOCK_SIZE ({BLOCK_SIZE}) when fused=True "
            f"and quantize=True, got group_size={group_size}"
        )


def _use_ecc_leaf(param: jax.Array, master_weight_bits: int | None) -> bool:
    """Return whether a parameter leaf should carry an ECC buffer."""
    if master_weight_bits is None:
        return False
    param_dtype = jnp.asarray(param).dtype
    if param_dtype not in _LOW_PRECISION_PARAM_DTYPES:
        return False
    return master_weight_bits > jnp.dtype(param_dtype).itemsize * 8


def _validate_fused_param_dtypes(
    params: Any,
    defaults: dict[str, Any],
    param_groups: list[ParamGroupSpec] | None,
    name: str,
) -> None:
    """Validate that all fused parameter leaves use supported dtypes."""
    paths, leaves, _ = _tree_leaves_with_paths(params)
    supported = ", ".join(str(dtype) for dtype in _FUSED_PARAM_DTYPES)
    for path, leaf in zip(paths, leaves, strict=True):
        config = _group_config_for_path(path, leaf, defaults=defaults, param_groups=param_groups)
        if not config["fused"]:
            continue
        param_dtype = jnp.asarray(leaf).dtype
        if param_dtype not in _FUSED_PARAM_DTYPES:
            path_str = "/".join(str(entry) for entry in path) or "<root>"
            raise ValueError(
                f"{name} fused=True only supports parameter dtypes ({supported}), "
                f"got {param_dtype} for parameter '{path_str}'"
            )


def _validate_meaningful_master_weight_bits(
    params: Any,
    defaults: dict[str, Any],
    param_groups: list[ParamGroupSpec] | None,
    name: str,
) -> None:
    """Reject ECC settings that never apply to any matched leaf."""
    paths, leaves, _ = _tree_leaves_with_paths(params)
    if not leaves:
        return

    defaults_master_weight_bits = defaults["master_weight_bits"]
    defaults_matched = False
    defaults_meaningful = False

    group_infos = []
    for idx, group in enumerate(param_groups or []):
        if "params" not in group:
            raise ValueError("Each param group must define a 'params' matcher")
        group_infos.append(
            {
                "index": idx,
                "group": group,
                "matched": False,
                "meaningful": False,
            }
        )

    for path, leaf in zip(paths, leaves, strict=True):
        matched_group_info = None
        for group_info in group_infos:
            group = group_info["group"]
            if _path_matches(path, leaf, group["params"]):
                matched_group_info = group_info
                break

        if matched_group_info is not None and "master_weight_bits" in matched_group_info["group"]:
            matched_group_info["matched"] = True
            master_weight_bits = matched_group_info["group"]["master_weight_bits"]
            if master_weight_bits is not None and _use_ecc_leaf(leaf, master_weight_bits):
                matched_group_info["meaningful"] = True
            continue

        defaults_matched = True
        if defaults_master_weight_bits is not None and _use_ecc_leaf(leaf, defaults_master_weight_bits):
            defaults_meaningful = True

    if defaults_matched and defaults_master_weight_bits is not None and not defaults_meaningful:
        raise ValueError(
            f"{name} master_weight_bits={defaults_master_weight_bits} has no effect because "
            "all matched parameters are already fp32 or otherwise ineligible for ECC"
        )

    for group_info in group_infos:
        master_weight_bits = group_info["group"].get("master_weight_bits")
        if group_info["matched"] and master_weight_bits is not None and not group_info["meaningful"]:
            raise ValueError(
                f"{name} param_groups[{group_info['index']}]['master_weight_bits']="
                f"{master_weight_bits} has no effect because all matched parameters are "
                "already fp32 or otherwise ineligible for ECC"
            )


def _tree_leaves_with_paths(tree: Any) -> tuple[list[tuple[Any, ...]], list[Any], Any]:
    """Flatten a pytree while keeping normalized paths for each leaf."""
    path_leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    paths = [_normalize_path(path) for path, _ in path_leaves]
    leaves = [leaf for _, leaf in path_leaves]
    return paths, leaves, treedef


def _resolve_learning_rate(
    learning_rate: ScalarOrSchedule,
    count: jax.Array,
) -> jax.Array:
    """Materialize a scalar learning rate from a value or schedule."""
    if callable(learning_rate):
        return jnp.asarray(learning_rate(count), dtype=jnp.float32)
    return jnp.asarray(learning_rate, dtype=jnp.float32)


def _init_leaf_state(
    param: jax.Array,
    group_size: int,
    quantize: bool,
    master_weight_bits: int | None,
) -> tuple[Any, Any, Any]:
    """Initialize Adam-style state for one parameter leaf."""
    zeros = jnp.zeros_like(param, dtype=jnp.float32)
    if quantize:
        mu = quantize_momentum(zeros, group_size)
        nu = quantize_variance(zeros, group_size)
    else:
        mu = zeros
        nu = zeros
    if _use_ecc_leaf(param, master_weight_bits):
        return mu, nu, jnp.zeros_like(
            param,
            dtype=jnp.int8 if master_weight_bits == 24 else jnp.int16,
        )
    return mu, nu, no_ecc_leaf()


def _materialize_momentum(mu: Any, group_size: int) -> jax.Array:
    """Convert stored momentum state into fp32 form."""
    if isinstance(mu, QuantizedArray):
        return dequantize_momentum(mu, group_size)
    return jnp.asarray(mu, dtype=jnp.float32)


def _materialize_variance(nu: Any, group_size: int) -> jax.Array:
    """Convert stored variance state into fp32 form."""
    if isinstance(nu, QuantizedArray):
        return dequantize_variance(nu, group_size)
    return jnp.asarray(nu, dtype=jnp.float32)


def _store_momentum(mu_f32: jax.Array, quantize: bool, group_size: int) -> Any:
    """Store fp32 momentum in quantized or full form."""
    if quantize:
        return quantize_momentum(mu_f32, group_size)
    return mu_f32


def _store_variance(nu_f32: jax.Array, quantize: bool, group_size: int) -> Any:
    """Store fp32 variance in quantized or full form."""
    if quantize:
        return quantize_variance(nu_f32, group_size)
    return nu_f32


def _init_momentum_leaf_state(
    param: jax.Array,
    group_size: int,
    quantize: bool,
    master_weight_bits: int | None,
    use_momentum: bool,
) -> tuple[Any, Any]:
    """Initialize momentum-only state for one parameter leaf."""
    if use_momentum:
        zeros = jnp.zeros_like(param, dtype=jnp.float32)
        if quantize:
            mu = quantize_momentum(zeros, group_size)
        else:
            mu = zeros
    else:
        mu = no_state_leaf()
    if _use_ecc_leaf(param, master_weight_bits):
        return mu, jnp.zeros_like(
            param,
            dtype=jnp.int8 if master_weight_bits == 24 else jnp.int16,
        )
    return mu, no_ecc_leaf()


def _materialize_optional_momentum(mu: Any, group_size: int) -> jax.Array | None:
    """Return fp32 momentum when present, otherwise `None`."""
    if _is_quantized_leaf(mu):
        return _materialize_momentum(mu, group_size)
    if not _has_state_leaf(mu):
        return None
    return _materialize_momentum(mu, group_size)


def _store_optional_momentum(mu_f32: jax.Array | None, quantize: bool, group_size: int) -> Any:
    """Store optional momentum state, preserving the no-state sentinel."""
    if mu_f32 is None:
        return no_state_leaf()
    return _store_momentum(mu_f32, quantize, group_size)


def _serialize_state_leaf(leaf: Any) -> dict[str, Any]:
    """Serialize one optimizer-state leaf into a checkpointable dict."""
    if isinstance(leaf, QuantizedArray):
        return {
            "kind": "quantized",
            "values": jnp.asarray(leaf.values),
            "scales": jnp.asarray(leaf.scales),
        }
    return {"kind": "array", "value": jnp.asarray(leaf)}


def _deserialize_state_leaf(leaf: dict[str, Any]) -> Any:
    """Deserialize one optimizer-state leaf from a state dict entry."""
    kind = leaf["kind"]
    if kind == "quantized":
        return QuantizedArray(
            values=jnp.asarray(leaf["values"]),
            scales=jnp.asarray(leaf["scales"]),
        )
    if kind == "array":
        return jnp.asarray(leaf["value"])
    raise ValueError(f"Unsupported serialized state leaf kind: {kind}")


def _tree_state_dict(tree: Any) -> dict[str, Any]:
    """Serialize an optimizer-state pytree, preserving quantized leaves."""
    leaves, treedef = jax.tree_util.tree_flatten(tree, is_leaf=_is_quantized_leaf)
    return {
        "treedef": treedef,
        "leaves": [_serialize_state_leaf(leaf) for leaf in leaves],
    }


def _tree_from_state_dict(state_dict: dict[str, Any]) -> Any:
    """Restore an optimizer-state pytree from serialized leaves."""
    leaves = [_deserialize_state_leaf(leaf) for leaf in state_dict["leaves"]]
    return state_dict["treedef"].unflatten(leaves)


def _low_precision_dtype_constants(param_dtype: jnp.dtype) -> tuple[int, int]:
    """Return dtype constants used by low-precision reconstruction code."""
    param_dtype = jnp.dtype(param_dtype)
    if param_dtype not in _DTYPE_CONSTANTS:
        raise ValueError(f"Unsupported fused parameter dtype: {param_dtype}")
    constants = _DTYPE_CONSTANTS[param_dtype]
    return constants["mantissa_bits"], constants["min_normal_exponent"]


def _fused_ecc_constants(
    param_dtype: jnp.dtype,
    ecc_dtype: jnp.dtype,
    use_ecc: bool,
) -> tuple[float, int, int]:
    """Return ECC scaling and dtype constants for fused kernels."""
    if not use_ecc:
        return 1.0, 0, 0
    ecc_max = ECC_MAX_INT8 if jnp.dtype(ecc_dtype) == jnp.dtype(jnp.int8) else ECC_MAX_INT16
    mantissa_bits, min_normal_exponent = _low_precision_dtype_constants(param_dtype)
    return ecc_max, mantissa_bits, min_normal_exponent


def make_leaf_layout(param: jax.Array, group_size: int) -> LeafKernelLayout:
    """Compute the flattened layout metadata for one parameter leaf."""
    size = param.size
    num_groups = (size + group_size - 1) // group_size
    return LeafKernelLayout(
        shape=param.shape,
        size=size,
        num_groups=num_groups,
        group_size=group_size,
    )


def pack_leaf_state_quantized(
    grad: jax.Array,
    param: jax.Array,
    ecc: jax.Array,
    mu: QuantizedArray,
    nu: QuantizedArray,
    group_size: int,
    ecc_dtype: jnp.dtype,
    param_dtype: jnp.dtype,
) -> tuple[LeafKernelLayout, LeafKernelQuantizedState]:
    """Pack one quantized leaf's state for a fused kernel call."""
    layout = make_leaf_layout(param, group_size)
    ecc_array = jnp.zeros_like(param, dtype=ecc_dtype)
    if jnp.asarray(ecc).shape != ():
        ecc_array = jnp.asarray(ecc, dtype=ecc_dtype)
    return layout, LeafKernelQuantizedState(
        grad=jnp.ravel(jnp.asarray(grad, dtype=jnp.float32)),
        param=jnp.ravel(jnp.asarray(param, dtype=param_dtype)),
        ecc=jnp.ravel(ecc_array),
        mu_values=jnp.ravel(jnp.asarray(mu.values, dtype=jnp.int8)),
        mu_scales=jnp.asarray(mu.scales, dtype=jnp.float16).reshape((layout.num_groups,)),
        nu_values=jnp.ravel(jnp.asarray(nu.values, dtype=jnp.uint8)),
        nu_scales=jnp.asarray(nu.scales, dtype=jnp.float16).reshape((layout.num_groups,)),
    )


def pack_leaf_state_full(
    grad: jax.Array,
    param: jax.Array,
    ecc: jax.Array,
    mu: jax.Array,
    nu: jax.Array,
    ecc_dtype: jnp.dtype,
    param_dtype: jnp.dtype,
) -> tuple[LeafKernelLayout, LeafKernelFullState]:
    """Pack one full-precision leaf's state for a fused kernel call."""
    layout = make_leaf_layout(param, GROUP_SIZE)
    ecc_array = jnp.zeros_like(param, dtype=ecc_dtype)
    if jnp.asarray(ecc).shape != ():
        ecc_array = jnp.asarray(ecc, dtype=ecc_dtype)
    return layout, LeafKernelFullState(
        grad=jnp.ravel(jnp.asarray(grad, dtype=jnp.float32)),
        param=jnp.ravel(jnp.asarray(param, dtype=param_dtype)),
        ecc=jnp.ravel(ecc_array),
        mu=jnp.ravel(jnp.asarray(mu, dtype=jnp.float32)),
        nu=jnp.ravel(jnp.asarray(nu, dtype=jnp.float32)),
    )


def _round_away_from_zero(x: jax.Array) -> jax.Array:
    """Round values to the nearest integer away from zero."""
    return jnp.sign(x) * jnp.floor(jnp.abs(x) + 0.5)


def _log_half_ulp_low_precision(
    x_lp: jax.Array,
    mantissa_bits: int,
    min_normal_exponent: int,
) -> jax.Array:
    """Compute the base-2 log of each low-precision value's half-ULP."""
    exponent_bias = 1 - min_normal_exponent
    x_bits = jnp.abs(x_lp).view(jnp.uint16).astype(jnp.int32)
    exponent_bits = x_bits >> mantissa_bits
    unbiased_exponent = jnp.where(
        exponent_bits == 0,
        jnp.full_like(exponent_bits, min_normal_exponent),
        exponent_bits - exponent_bias,
    )
    return unbiased_exponent - mantissa_bits - 1


def _reconstruct_from_split(
    param_lp: jax.Array,
    ecc: jax.Array,
    ecc_max: float,
    mantissa_bits: int,
    min_normal_exponent: int,
) -> jax.Array:
    """Reconstruct fp32 values from low-precision params and ECC."""
    log_scale = _log_half_ulp_low_precision(
        param_lp,
        mantissa_bits=mantissa_bits,
        min_normal_exponent=min_normal_exponent,
    )
    correction = ecc.astype(jnp.float32) / ecc_max
    error = jnp.ldexp(correction, log_scale)
    return param_lp.astype(jnp.float32) + error


def _split_to_low_precision_ecc(
    param_f32: jax.Array,
    param_dtype: jnp.dtype,
    ecc_dtype: jnp.dtype,
    ecc_max: float,
    mantissa_bits: int,
    min_normal_exponent: int,
) -> tuple[jax.Array, jax.Array]:
    """Split fp32 values into low-precision params and normalized ECC."""
    param_lp = param_f32.astype(param_dtype)
    error = param_f32 - param_lp.astype(jnp.float32)
    log_scale = _log_half_ulp_low_precision(
        param_lp,
        mantissa_bits=mantissa_bits,
        min_normal_exponent=min_normal_exponent,
    )
    error_norm = jnp.ldexp(error, -log_scale)
    ecc = _round_away_from_zero(jnp.clip(error_norm, -1.0, 1.0) * ecc_max).astype(ecc_dtype)
    return param_lp, ecc
