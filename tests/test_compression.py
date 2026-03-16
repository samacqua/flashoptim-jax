import numpy as np
import jax.numpy as jnp

from flashoptim_jax.compression import reconstruct_leaf, reconstruct_weights, split_leaf


def test_split_reconstruct_improves_on_plain_bf16():
    x = jnp.array(
        [-3.25, -1.125, -0.2, 0.0, 0.3, 1.75, 7.125, 1000.125],
        dtype=jnp.float32,
    )

    x_lp, ecc = split_leaf(x)
    x_recon = reconstruct_leaf(x_lp, ecc)

    plain_error = np.abs(np.asarray(x) - np.asarray(x_lp, dtype=np.float32))
    ecc_error = np.abs(np.asarray(x) - np.asarray(x_recon))

    assert x_lp.dtype == jnp.bfloat16
    assert ecc.dtype == jnp.int8
    assert np.all(ecc_error <= plain_error + 1e-12)
    assert np.any(ecc_error < plain_error)


def test_split_reconstruct_preserves_exact_bf16_values():
    x = jnp.array([0.0, 1.0, -2.0, 3.5], dtype=jnp.bfloat16).astype(jnp.float32)

    x_lp, ecc = split_leaf(x)
    x_recon = reconstruct_leaf(x_lp, ecc)

    np.testing.assert_array_equal(np.asarray(ecc), np.zeros_like(np.asarray(ecc)))
    np.testing.assert_array_equal(np.asarray(x_recon), np.asarray(x))


def test_split_leaf_supports_fp16_with_int16_ecc():
    x = jnp.array([-3.25, -1.125, 0.3, 7.125], dtype=jnp.float32)

    x_lp, ecc = split_leaf(x, narrow_dtype=jnp.float16, master_weight_bits=32)
    x_recon = reconstruct_leaf(x_lp, ecc)

    assert x_lp.dtype == jnp.float16
    assert ecc.dtype == jnp.int16
    np.testing.assert_allclose(np.asarray(x_recon), np.asarray(x), atol=1e-4, rtol=1e-4)


def test_reconstruct_weights_skips_non_ecc_leaves():
    params = {
        "w": jnp.array([1.0, -2.0], dtype=jnp.bfloat16),
        "bn": jnp.array([0.25, -0.5], dtype=jnp.float32),
    }
    ecc = {
        "w": jnp.array([0, 0], dtype=jnp.int8),
        "bn": jnp.zeros((), dtype=jnp.int32),
    }

    recon = reconstruct_weights(params, ecc)

    assert recon["w"].dtype == jnp.float32
    assert recon["bn"].dtype == jnp.float32
    np.testing.assert_array_equal(np.asarray(recon["bn"]), np.asarray(params["bn"]))
