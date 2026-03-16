import numpy as np
import jax.numpy as jnp

from flashoptim_jax.quantization import (
    _round_to_nearest,
    dequantize_momentum,
    dequantize_variance,
    quantize_momentum,
    quantize_variance,
)


def test_momentum_quantization_roundtrip():
    x = jnp.linspace(-3.0, 3.0, 65, dtype=jnp.float32)

    q = quantize_momentum(x)
    x_recon = dequantize_momentum(q)

    mae = np.abs(np.asarray(x_recon) - np.asarray(x)).mean()
    cos = np.dot(np.asarray(x), np.asarray(x_recon)) / (
        np.linalg.norm(np.asarray(x)) * np.linalg.norm(np.asarray(x_recon))
    )

    assert q.values.dtype == jnp.int8
    assert q.scales.dtype == jnp.float16
    assert mae < 0.05
    assert cos > 0.999


def test_variance_quantization_tail_scale_matches_tail_absmax():
    x = jnp.linspace(0.0, 10.0, 65, dtype=jnp.float32)

    q = quantize_variance(x)
    x_recon = dequantize_variance(q)
    tail = np.sqrt(np.asarray(x)[64:])

    assert q.values.dtype == jnp.uint8
    assert np.isclose(np.asarray(q.scales[-1], dtype=np.float32), tail.max(), rtol=1e-3)
    assert np.abs(np.asarray(x_recon) - np.asarray(x)).mean() < 0.1


def test_round_to_nearest_matches_torch_tie_behavior():
    rounded = _round_to_nearest(jnp.array([-63.5, -62.5, 62.5, 63.5], dtype=jnp.float32))
    np.testing.assert_array_equal(np.asarray(rounded), np.array([-63.0, -62.0, 63.0, 64.0]))
