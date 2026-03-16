import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from flashoptim_jax import flash_adamw
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


def test_flash_adamw_tracks_optax_adamw_over_multiple_steps():
    params_f32 = {
        "w": jnp.array([[0.5, -1.0], [1.5, -0.25]], dtype=jnp.float32),
        "b": jnp.array([0.1, -0.2], dtype=jnp.float32),
    }
    params_flash = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params_f32)
    grads = {
        "w": jnp.array([[0.3, -0.1], [0.2, -0.4]], dtype=jnp.float32),
        "b": jnp.array([0.05, -0.02], dtype=jnp.float32),
    }

    flash = flash_adamw(learning_rate=1e-2, weight_decay=1e-2)
    flash_state = flash.init(params_flash)

    reference = optax.adamw(learning_rate=1e-2, weight_decay=1e-2)
    reference_state = reference.init(params_f32)

    for _ in range(10):
        params_flash, flash_state = flash.step(params_flash, flash_state, grads)

        reference_updates, reference_state = reference.update(
            grads,
            reference_state,
            params_f32,
        )
        params_f32 = optax.apply_updates(params_f32, reference_updates)

    flash_master = reconstruct_weights(params_flash, flash_state.ecc)
    diff = jax.tree_util.tree_map(lambda a, b: a - b, flash_master, params_f32)

    assert _tree_l2_norm(diff) < 0.05


def test_flash_adamw_returns_bf16_params():
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.bfloat16)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}
    tx = flash_adamw(learning_rate=1e-3)
    state = tx.init(params)

    params, _ = tx.step(params, state, grads)

    assert params["x"].dtype == jnp.bfloat16


def test_flash_adamw_quantizes_states_for_non_bf16_leaves():
    params_ref = {
        "w": jnp.linspace(-1.0, 1.0, 128, dtype=jnp.float32).reshape(32, 4),
        "bn": jnp.array([1.0, 0.5, -0.25, 0.75], dtype=jnp.float32),
    }
    params_flash = {
        "w": params_ref["w"].astype(jnp.bfloat16),
        "bn": params_ref["bn"],
    }
    grads = {
        "w": jnp.full((32, 4), 0.03, dtype=jnp.float32),
        "bn": jnp.array([0.01, -0.02, 0.03, -0.04], dtype=jnp.float32),
    }

    flash_tx = flash_adamw(learning_rate=1e-2, weight_decay=1e-2, fused=False)
    reference_tx = optax.adamw(learning_rate=1e-2, weight_decay=1e-2)

    flash_state = flash_tx.init(params_flash)
    reference_state = reference_tx.init(params_ref)

    assert flash_state.ecc["bn"].dtype == jnp.int32
    assert flash_state.ecc["bn"].shape == ()
    assert isinstance(flash_state.mu["bn"], QuantizedArray)
    assert isinstance(flash_state.nu["bn"], QuantizedArray)

    for _ in range(10):
        params_flash, flash_state = flash_tx.step(params_flash, flash_state, grads)

        reference_updates, reference_state = reference_tx.update(
            grads,
            reference_state,
            params_ref,
        )
        params_ref = optax.apply_updates(params_ref, reference_updates)

    flash_master = {
        "w": reconstruct_leaf(params_flash["w"], flash_state.ecc["w"]),
        "bn": params_flash["bn"],
    }
    diff = jax.tree_util.tree_map(lambda a, b: a - b, flash_master, params_ref)

    np.testing.assert_allclose(
        np.asarray(params_flash["bn"]),
        np.asarray(params_ref["bn"]),
        atol=3e-4,
        rtol=1e-3,
    )
    assert _tree_l2_norm(diff) < 0.05


def test_flash_adamw_supports_unquantized_states():
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.bfloat16)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}

    tx = flash_adamw(learning_rate=1e-3, quantize=False, fused=False)
    state = tx.init(params)
    params, state = tx.step(params, state, grads)

    assert state.mu["x"].dtype == jnp.float32
    assert state.nu["x"].dtype == jnp.float32
    assert state.ecc["x"].dtype == jnp.int8
    assert params["x"].dtype == jnp.bfloat16


def test_flash_adamw_supports_32bit_master_weights():
    params_f32 = {"x": jnp.linspace(-2.0, 2.0, 64, dtype=jnp.float32)}
    params_flash = {"x": params_f32["x"].astype(jnp.float16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 64, dtype=jnp.float32)}

    flash_tx = flash_adamw(
        learning_rate=1e-3,
        master_weight_bits=32,
        fused=False,
    )
    reference_tx = optax.adamw(learning_rate=1e-3)

    flash_state = flash_tx.init(params_flash)
    reference_state = reference_tx.init(params_f32)

    assert flash_state.ecc["x"].dtype == jnp.int16

    for _ in range(5):
        params_flash, flash_state = flash_tx.step(params_flash, flash_state, grads)

        reference_updates, reference_state = reference_tx.update(
            grads,
            reference_state,
            params_f32,
        )
        params_f32 = optax.apply_updates(params_f32, reference_updates)

    flash_master = reconstruct_leaf(params_flash["x"], flash_state.ecc["x"])
    np.testing.assert_allclose(
        np.asarray(flash_master),
        np.asarray(params_f32["x"]),
        atol=2e-2,
        rtol=2e-2,
    )


def test_flash_adamw_master_weight_bits_none_disables_ecc():
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.bfloat16)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}

    tx = flash_adamw(learning_rate=1e-3, master_weight_bits=None, fused=False)
    state = tx.init(params)
    _, state = tx.step(params, state, grads)

    assert state.ecc["x"].dtype == jnp.int32


def test_flash_adamw_schedule_uses_zero_based_counts():
    calls = []

    def schedule(count):
        calls.append(int(np.asarray(count)))
        return jnp.asarray(1e-3, dtype=jnp.float32)

    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.bfloat16)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}
    tx = flash_adamw(learning_rate=schedule, fused=False)
    state = tx.init(params)

    params, state = tx.step(params, state, grads)
    params, state = tx.step(params, state, grads)

    assert calls == [0, 1]
    assert int(np.asarray(state.count)) == 2


def test_flash_adamw_rejects_invalid_fused_group_size():
    with pytest.raises(ValueError, match="group_size must divide BLOCK_SIZE"):
        flash_adamw(
            learning_rate=1e-3,
            group_size=40,
            fused=True,
        )


def test_flash_adamw_rejects_unsupported_fused_param_dtype():
    params = {"x": jnp.array([1, -2, 3], dtype=jnp.int32)}
    tx = flash_adamw(learning_rate=1e-3, fused=True)
    with pytest.raises(ValueError, match="fused=True only supports parameter dtypes"):
        tx.init(params)


def test_fused_flash_adamw_tracks_unfused_and_optax_on_mixed_leaf_sizes():
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

    unfused_tx = flash_adamw(learning_rate=1e-2, weight_decay=1e-2, fused=False)
    fused_tx = flash_adamw(
        learning_rate=1e-2,
        weight_decay=1e-2,
        fused=True,
    )
    reference_tx = optax.adamw(learning_rate=1e-2, weight_decay=1e-2)

    params_unfused, state_unfused = _run_flash_steps(unfused_tx, params_unfused, grads, 5)
    params_fused, state_fused = _run_flash_steps(fused_tx, params_fused, grads, 5)

    reference_state = reference_tx.init(params_f32)
    for _ in range(5):
        reference_updates, reference_state = reference_tx.update(
            grads,
            reference_state,
            params_f32,
        )
        params_f32 = optax.apply_updates(params_f32, reference_updates)

    master_unfused = reconstruct_weights(params_unfused, state_unfused.ecc)
    master_fused = reconstruct_weights(params_fused, state_fused.ecc)

    fused_vs_unfused = jax.tree_util.tree_map(lambda a, b: a - b, master_fused, master_unfused)
    fused_vs_reference = jax.tree_util.tree_map(lambda a, b: a - b, master_fused, params_f32)

    assert _tree_l2_norm(fused_vs_unfused) < 0.5
    assert _tree_l2_norm(fused_vs_reference) < 0.5


def test_fused_flash_adamw_matches_unfused_on_tail_group_leaf():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_adamw(learning_rate=1e-3, fused=False)
    fused_tx = flash_adamw(learning_rate=1e-3, fused=True)

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


def test_fused_flash_adamw_matches_unfused_without_ecc():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_adamw(learning_rate=1e-3, master_weight_bits=None, fused=False)
    fused_tx = flash_adamw(
        learning_rate=1e-3,
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
        atol=5e-3,
        rtol=5e-3,
    )
    assert fused_state.ecc["x"].dtype == jnp.int32


def test_fused_flash_adamw_matches_unfused_with_unquantized_states():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_adamw(learning_rate=1e-3, quantize=False, fused=False)
    fused_tx = flash_adamw(
        learning_rate=1e-3,
        quantize=False,
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
    assert fused_state.nu["x"].dtype == jnp.float32


def test_fused_flash_adamw_matches_unfused_with_32bit_master_weights():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_adamw(learning_rate=1e-3, master_weight_bits=32, fused=False)
    fused_tx = flash_adamw(
        learning_rate=1e-3,
        master_weight_bits=32,
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


def test_fused_flash_adamw_matches_unfused_with_32bit_master_and_full_states():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_adamw(
        learning_rate=1e-3,
        master_weight_bits=32,
        quantize=False,
        fused=False,
    )
    fused_tx = flash_adamw(
        learning_rate=1e-3,
        master_weight_bits=32,
        quantize=False,
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
    assert fused_state.mu["x"].dtype == jnp.float32
    assert fused_state.nu["x"].dtype == jnp.float32


def test_fused_flash_adamw_supports_fp16_params():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.float16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_adamw(learning_rate=1e-3, master_weight_bits=32, fused=False)
    fused_tx = flash_adamw(
        learning_rate=1e-3,
        master_weight_bits=32,
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


def test_fused_flash_adamw_supports_fp32_params():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}

    unfused_tx = flash_adamw(learning_rate=1e-3, master_weight_bits=None, fused=False)
    fused_tx = flash_adamw(
        learning_rate=1e-3,
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


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="compiled fused path requires GPU")
def test_compiled_fused_flash_adamw_matches_unfused():
    params = {"x": jnp.linspace(-2.0, 2.0, 4096, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 4096, dtype=jnp.float32)}

    unfused_tx = flash_adamw(learning_rate=1e-3, fused=False)
    fused_tx = flash_adamw(
        learning_rate=1e-3,
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
