import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from flashoptim_jax import flash_sgd
from flashoptim_jax.compression import reconstruct_leaf, reconstruct_weights
from flashoptim_jax.quantization import QuantizedArray


def _tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    return np.sqrt(sum(float(jnp.vdot(x, x)) for x in leaves))


def _run_flash_steps(tx, params, grads, steps):
    state = tx.init(params)
    for _ in range(steps):
        params, state = tx.step(params, state, grads)
    return params, state


def _run_flash_donating_steps(tx, params, grads, steps):
    state = tx.init(params)
    step_fn = jax.jit(tx.step, donate_argnums=(0, 1))
    for _ in range(steps):
        params, state = step_fn(params, state, grads)
    return params, state


def test_flash_sgd_tracks_optax_sgd_over_multiple_steps():
    params_f32 = {
        "w": jnp.array([[0.5, -1.0], [1.5, -0.25]], dtype=jnp.float32),
        "b": jnp.array([0.1, -0.2], dtype=jnp.float32),
    }
    params_flash = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params_f32)
    grads = {
        "w": jnp.array([[0.3, -0.1], [0.2, -0.4]], dtype=jnp.float32),
        "b": jnp.array([0.05, -0.02], dtype=jnp.float32),
    }

    flash = flash_sgd(learning_rate=1e-2, momentum=0.9, fused=False)
    flash_state = flash.init(params_flash)

    reference = optax.sgd(learning_rate=1e-2, momentum=0.9)
    reference_state = reference.init(params_f32)

    for _ in range(10):
        params_flash, flash_state = flash.step(params_flash, flash_state, grads)

        reference_updates, reference_state = reference.update(
            grads, reference_state, params_f32,
        )
        params_f32 = optax.apply_updates(params_f32, reference_updates)

    flash_master = reconstruct_weights(params_flash, flash_state.ecc)
    diff = jax.tree_util.tree_map(lambda a, b: a - b, flash_master, params_f32)
    assert _tree_l2_norm(diff) < 0.05


def test_flash_sgd_returns_bf16_params():
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.bfloat16)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}
    tx = flash_sgd(learning_rate=1e-3, momentum=0.9, fused=False)
    state = tx.init(params)
    params, _ = tx.step(params, state, grads)
    assert params["x"].dtype == jnp.bfloat16


def test_flash_sgd_quantizes_momentum_for_bf16_leaves():
    params = {"x": jnp.linspace(-1.0, 1.0, 128, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.full((128,), 0.03, dtype=jnp.float32)}

    tx = flash_sgd(learning_rate=1e-2, momentum=0.9, fused=False)
    state = tx.init(params)
    _, state = tx.step(params, state, grads)
    assert isinstance(state.mu["x"], QuantizedArray)


def test_flash_sgd_supports_unquantized_states():
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.bfloat16)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}

    tx = flash_sgd(learning_rate=1e-3, momentum=0.9, quantize=False, fused=False)
    state = tx.init(params)
    params, state = tx.step(params, state, grads)
    assert state.mu["x"].dtype == jnp.float32
    assert state.ecc["x"].dtype == jnp.int8
    assert params["x"].dtype == jnp.bfloat16


def test_flash_sgd_no_momentum():
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.bfloat16)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}

    tx = flash_sgd(learning_rate=1e-3, momentum=0.0, fused=False)
    state = tx.init(params)
    params, state = tx.step(params, state, grads)
    assert params["x"].dtype == jnp.bfloat16


def test_fused_flash_sgd_matches_unfused():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, fused=False)
    fused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, fused=True)

    unfused_state = unfused_tx.init(params)
    fused_state = fused_tx.init(params)

    unfused_params, unfused_state = unfused_tx.step(params, unfused_state, grads)
    fused_params, fused_state = fused_tx.step(params, fused_state, grads)

    unfused_master = reconstruct_weights(unfused_params, unfused_state.ecc)
    fused_master = reconstruct_weights(fused_params, fused_state.ecc)

    np.testing.assert_allclose(
        np.asarray(fused_master["x"]),
        np.asarray(unfused_master["x"]),
        atol=5e-3,
        rtol=5e-3,
    )
    assert fused_params["x"].dtype == jnp.bfloat16


def test_fused_flash_sgd_donation_matches_unfused():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, fused=False)
    fused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, fused=True)

    unfused_params, unfused_state = _run_flash_steps(unfused_tx, params, grads, 5)
    fused_params, fused_state = _run_flash_donating_steps(fused_tx, params, grads, 5)

    unfused_master = reconstruct_weights(unfused_params, unfused_state.ecc)
    fused_master = reconstruct_weights(fused_params, fused_state.ecc)

    np.testing.assert_allclose(
        np.asarray(fused_master["x"]),
        np.asarray(unfused_master["x"]),
        atol=5e-3,
        rtol=5e-3,
    )


def test_fused_flash_sgd_matches_unfused_without_ecc():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, master_weight_bits=None, fused=False)
    fused_tx = flash_sgd(
        learning_rate=1e-3, momentum=0.9, master_weight_bits=None,
        fused=True,
    )

    unfused_state = unfused_tx.init(params)
    fused_state = fused_tx.init(params)

    unfused_params, unfused_state = unfused_tx.step(params, unfused_state, grads)
    fused_params, fused_state = fused_tx.step(params, fused_state, grads)

    np.testing.assert_allclose(
        np.asarray(fused_params["x"]),
        np.asarray(unfused_params["x"]),
        atol=5e-3,
        rtol=5e-3,
    )
    assert fused_state.ecc["x"].dtype == jnp.int32


def test_fused_flash_sgd_matches_unfused_with_unquantized_states():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, quantize=False, fused=False)
    fused_tx = flash_sgd(
        learning_rate=1e-3, momentum=0.9, quantize=False,
        fused=True,
    )

    unfused_state = unfused_tx.init(params)
    fused_state = fused_tx.init(params)

    unfused_params, unfused_state = unfused_tx.step(params, unfused_state, grads)
    fused_params, fused_state = fused_tx.step(params, fused_state, grads)

    unfused_master = reconstruct_weights(unfused_params, unfused_state.ecc)
    fused_master = reconstruct_weights(fused_params, fused_state.ecc)

    np.testing.assert_allclose(
        np.asarray(fused_master["x"]),
        np.asarray(unfused_master["x"]),
        atol=5e-3,
        rtol=5e-3,
    )
    assert fused_state.mu["x"].dtype == jnp.float32


def test_fused_flash_sgd_matches_unfused_with_32bit_master_weights():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, master_weight_bits=32, fused=False)
    fused_tx = flash_sgd(
        learning_rate=1e-3, momentum=0.9, master_weight_bits=32,
        fused=True,
    )

    unfused_state = unfused_tx.init(params)
    fused_state = fused_tx.init(params)

    unfused_params, unfused_state = unfused_tx.step(params, unfused_state, grads)
    fused_params, fused_state = fused_tx.step(params, fused_state, grads)

    unfused_master = reconstruct_weights(unfused_params, unfused_state.ecc)
    fused_master = reconstruct_weights(fused_params, fused_state.ecc)

    np.testing.assert_allclose(
        np.asarray(fused_master["x"]),
        np.asarray(unfused_master["x"]),
        atol=5e-3,
        rtol=5e-3,
    )
    assert fused_state.ecc["x"].dtype == jnp.int16


def test_fused_flash_sgd_supports_fp16_params():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.float16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, master_weight_bits=32, fused=False)
    fused_tx = flash_sgd(
        learning_rate=1e-3, momentum=0.9, master_weight_bits=32,
        fused=True,
    )

    unfused_state = unfused_tx.init(params)
    fused_state = fused_tx.init(params)

    unfused_params, unfused_state = unfused_tx.step(params, unfused_state, grads)
    fused_params, fused_state = fused_tx.step(params, fused_state, grads)

    unfused_master = reconstruct_weights(unfused_params, unfused_state.ecc)
    fused_master = reconstruct_weights(fused_params, fused_state.ecc)

    np.testing.assert_allclose(
        np.asarray(fused_master["x"]),
        np.asarray(unfused_master["x"]),
        atol=5e-3,
        rtol=5e-3,
    )
    assert fused_params["x"].dtype == jnp.float16


def test_fused_flash_sgd_supports_fp32_params():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, master_weight_bits=None, fused=False)
    fused_tx = flash_sgd(
        learning_rate=1e-3,
        momentum=0.9,
        master_weight_bits=None,
        fused=True,

    )

    unfused_state = unfused_tx.init(params)
    fused_state = fused_tx.init(params)

    unfused_params, unfused_state = unfused_tx.step(params, unfused_state, grads)
    fused_params, fused_state = fused_tx.step(params, fused_state, grads)

    np.testing.assert_allclose(
        np.asarray(fused_params["x"]),
        np.asarray(unfused_params["x"]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert fused_params["x"].dtype == jnp.float32
    assert isinstance(fused_state.mu["x"], QuantizedArray)
    assert fused_state.ecc["x"].dtype == jnp.int32


def test_flash_sgd_rejects_unsupported_fused_param_dtype():
    params = {"x": jnp.array([1, -2, 3], dtype=jnp.int32)}
    tx = flash_sgd(learning_rate=1e-3, momentum=0.9, fused=True)
    with pytest.raises(ValueError, match="fused=True only supports parameter dtypes"):
        tx.init(params)


def test_fused_flash_sgd_on_mixed_leaf_sizes():
    params_f32 = {
        "big": jnp.linspace(-1.0, 1.0, 65 * 65, dtype=jnp.float32).reshape(65, 65),
        "small": jnp.array([0.5, -0.75, 0.25], dtype=jnp.float32),
    }
    grads = {
        "big": jnp.full((65, 65), 0.03, dtype=jnp.float32),
        "small": jnp.array([0.1, -0.05, 0.02], dtype=jnp.float32),
    }
    params_unfused = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params_f32)
    params_fused = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params_f32)

    unfused_tx = flash_sgd(learning_rate=1e-2, momentum=0.9, fused=False)
    fused_tx = flash_sgd(
        learning_rate=1e-2, momentum=0.9,
        fused=True,
    )

    params_unfused, state_unfused = _run_flash_steps(unfused_tx, params_unfused, grads, 5)
    params_fused, state_fused = _run_flash_steps(fused_tx, params_fused, grads, 5)

    master_unfused = reconstruct_weights(params_unfused, state_unfused.ecc)
    master_fused = reconstruct_weights(params_fused, state_fused.ecc)

    fused_vs_unfused = jax.tree_util.tree_map(lambda a, b: a - b, master_fused, master_unfused)
    assert _tree_l2_norm(fused_vs_unfused) < 0.5


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="compiled fused path requires GPU")
def test_compiled_fused_flash_sgd_matches_unfused():
    params = {"x": jnp.linspace(-2.0, 2.0, 4096, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 4096, dtype=jnp.float32)}

    unfused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, fused=False)
    fused_tx = flash_sgd(
        learning_rate=1e-3, momentum=0.9,
        fused=True,
    )

    unfused_state = unfused_tx.init(params)
    fused_state = fused_tx.init(params)
    fused_step_fn = jax.jit(lambda p, s, g: fused_tx.step(p, s, g))

    unfused_params, unfused_state = unfused_tx.step(params, unfused_state, grads)
    fused_params, fused_state = fused_step_fn(params, fused_state, grads)
    fused_params = jax.block_until_ready(fused_params)
    fused_state = jax.block_until_ready(fused_state)

    unfused_master = reconstruct_weights(unfused_params, unfused_state.ecc)
    fused_master = reconstruct_weights(fused_params, fused_state.ecc)

    np.testing.assert_allclose(
        np.asarray(fused_master["x"]),
        np.asarray(unfused_master["x"]),
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="compiled fused path requires GPU")
def test_compiled_flash_sgd_donation_matches_unfused():
    params = {"x": jnp.linspace(-2.0, 2.0, 4096, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 4096, dtype=jnp.float32)}

    unfused_tx = flash_sgd(learning_rate=1e-3, momentum=0.9, fused=False)
    fused_tx = flash_sgd(
        learning_rate=1e-3, momentum=0.9,
        fused=True,
    )

    unfused_state = unfused_tx.init(params)
    fused_state = fused_tx.init(params)
    fused_step = jax.jit(fused_tx.step, donate_argnums=(0, 1))

    unfused_params, unfused_state = unfused_tx.step(params, unfused_state, grads)

    fused_params, fused_state = fused_step(params, fused_state, grads)
    fused_params = jax.block_until_ready(fused_params)
    fused_state = jax.block_until_ready(fused_state)

    unfused_master = reconstruct_weights(unfused_params, unfused_state.ecc)
    fused_master = reconstruct_weights(fused_params, fused_state.ecc)

    np.testing.assert_allclose(
        np.asarray(fused_master["x"]),
        np.asarray(unfused_master["x"]),
        atol=5e-3,
        rtol=5e-3,
    )
