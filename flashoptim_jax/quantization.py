"""Momentum and variance quantization helpers for optimizer state."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class QuantizedArray(NamedTuple):
    """Quantized values paired with per-group scales."""
    values: jax.Array
    scales: jax.Array


def _round_to_nearest(x: jax.Array) -> jax.Array:
    """Round positive values to the nearest integer."""
    return jnp.floor(x + 0.5)


def _pad_groups(x: jax.Array, group_size: int) -> tuple[jax.Array, int]:
    """Flatten and pad an array so it reshapes into fixed-size groups."""
    flat = jnp.ravel(jnp.asarray(x, dtype=jnp.float32))
    pad = (-flat.size) % group_size
    if pad:
        flat = jnp.pad(flat, (0, pad))
    return flat.reshape((-1, group_size)), pad


def _unpad_groups(x: jax.Array, pad: int, shape: tuple[int, ...]) -> jax.Array:
    """Remove trailing padding and restore the original array shape."""
    flat = x.reshape((-1,))
    if pad:
        flat = flat[:-pad]
    return flat.reshape(shape)


def quantize_momentum(x: jax.Array, group_size: int = 32) -> QuantizedArray:
    """Quantize a momentum tensor with per-group symmetric scaling."""
    groups, pad = _pad_groups(x, group_size)
    scales = jnp.max(jnp.abs(groups), axis=1, keepdims=True)
    safe_scales = jnp.where(scales > 0, scales, 1.0)
    normalized = groups / safe_scales
    transformed = 2.0 * normalized / (1.0 + jnp.abs(normalized))
    quantized = _round_to_nearest(transformed * 127.0)
    values = _unpad_groups(jnp.clip(quantized, -127.0, 127.0), pad, x.shape).astype(jnp.int8)
    return QuantizedArray(values=values, scales=scales.reshape((-1,)).astype(jnp.float16))


def dequantize_momentum(x: QuantizedArray, group_size: int = 32) -> jax.Array:
    """Reconstruct an approximate momentum tensor from quantized storage."""
    groups, pad = _pad_groups(x.values, group_size)
    transformed = groups.astype(jnp.float32) / 127.0
    normalized = transformed / (2.0 - jnp.abs(transformed))
    scales = x.scales.astype(jnp.float32).reshape((-1, 1))
    dequantized = normalized * scales
    return _unpad_groups(dequantized, pad, x.values.shape)


def quantize_variance(x: jax.Array, group_size: int = 32) -> QuantizedArray:
    """Quantize a variance tensor after taking its square root."""
    x = jnp.sqrt(jnp.asarray(x, dtype=jnp.float32))
    groups, pad = _pad_groups(x, group_size)
    scales = jnp.max(groups, axis=1, keepdims=True)
    safe_scales = jnp.where(scales > 0, scales, 1.0)
    transformed = groups / safe_scales
    quantized = _round_to_nearest(transformed * 255.0)
    values = _unpad_groups(jnp.clip(quantized, 0.0, 255.0), pad, x.shape).astype(jnp.uint8)
    return QuantizedArray(values=values, scales=scales.reshape((-1,)).astype(jnp.float16))


def dequantize_variance(x: QuantizedArray, group_size: int = 32) -> jax.Array:
    """Reconstruct an approximate variance tensor from quantized storage."""
    groups, pad = _pad_groups(x.values, group_size)
    transformed = groups.astype(jnp.float32) / 255.0
    scales = x.scales.astype(jnp.float32).reshape((-1, 1))
    dequantized = transformed * scales
    return _unpad_groups(dequantized * dequantized, pad, x.values.shape)
