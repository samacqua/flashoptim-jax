import jax
import jax.numpy as jnp
import numpy as np
import optax

from flashoptim_jax import (
    QuantizedArray,
    flash_adamw,
    flash_adamw_state_dict,
    flash_lion,
    flash_lion_state_dict,
    flash_sgd,
    flash_sgd_state_dict,
    load_flash_adamw_state_dict,
    load_flash_lion_state_dict,
    load_flash_sgd_state_dict,
)
from flashoptim_jax.compression import reconstruct_weights, set_fp32_params


def _assert_optimizer_trees_equal(left, right):
    left_leaves, left_treedef = jax.tree_util.tree_flatten(
        left,
        is_leaf=lambda x: isinstance(x, QuantizedArray),
    )
    right_leaves, right_treedef = jax.tree_util.tree_flatten(
        right,
        is_leaf=lambda x: isinstance(x, QuantizedArray),
    )
    assert left_treedef == right_treedef
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if isinstance(left_leaf, QuantizedArray):
            assert isinstance(right_leaf, QuantizedArray)
            np.testing.assert_array_equal(
                np.asarray(left_leaf.values),
                np.asarray(right_leaf.values),
            )
            np.testing.assert_array_equal(
                np.asarray(left_leaf.scales),
                np.asarray(right_leaf.scales),
            )
        else:
            np.testing.assert_array_equal(np.asarray(left_leaf), np.asarray(right_leaf))


def test_flash_adamw_state_dict_roundtrip():
    params = {
        "w": jnp.linspace(-2.0, 2.0, 64, dtype=jnp.float32).astype(jnp.bfloat16),
        "b": jnp.array([0.5, -0.75], dtype=jnp.float32),
    }
    grads = {
        "w": jnp.linspace(-0.3, 0.4, 64, dtype=jnp.float32),
        "b": jnp.array([0.1, -0.2], dtype=jnp.float32),
    }
    tx = flash_adamw(learning_rate=1e-3, fused=False)
    state = tx.init(params)
    _, state = tx.step(params, state, grads)

    restored = load_flash_adamw_state_dict(flash_adamw_state_dict(state))

    assert int(np.asarray(restored.count)) == int(np.asarray(state.count))
    _assert_optimizer_trees_equal(restored.mu, state.mu)
    _assert_optimizer_trees_equal(restored.nu, state.nu)
    _assert_optimizer_trees_equal(restored.ecc, state.ecc)


def test_flash_sgd_state_dict_roundtrip():
    params = {
        "w": jnp.linspace(-2.0, 2.0, 64, dtype=jnp.float32).astype(jnp.bfloat16),
        "b": jnp.array([0.5, -0.75], dtype=jnp.float32),
    }
    grads = {
        "w": jnp.linspace(-0.3, 0.4, 64, dtype=jnp.float32),
        "b": jnp.array([0.1, -0.2], dtype=jnp.float32),
    }
    tx = flash_sgd(learning_rate=1e-3, momentum=0.9, fused=False)
    state = tx.init(params)
    _, state = tx.step(params, state, grads)

    restored = load_flash_sgd_state_dict(flash_sgd_state_dict(state))

    assert int(np.asarray(restored.count)) == int(np.asarray(state.count))
    _assert_optimizer_trees_equal(restored.mu, state.mu)
    _assert_optimizer_trees_equal(restored.ecc, state.ecc)


def test_flash_lion_state_dict_roundtrip():
    params = {
        "w": jnp.linspace(-2.0, 2.0, 64, dtype=jnp.float32).astype(jnp.bfloat16),
        "b": jnp.array([0.5, -0.75], dtype=jnp.float32),
    }
    grads = {
        "w": jnp.linspace(-0.3, 0.4, 64, dtype=jnp.float32),
        "b": jnp.array([0.1, -0.2], dtype=jnp.float32),
    }
    tx = flash_lion(learning_rate=1e-4, fused=False)
    state = tx.init(params)
    _, state = tx.step(params, state, grads)

    restored = load_flash_lion_state_dict(flash_lion_state_dict(state))

    assert int(np.asarray(restored.count)) == int(np.asarray(state.count))
    _assert_optimizer_trees_equal(restored.mu, state.mu)
    _assert_optimizer_trees_equal(restored.ecc, state.ecc)


def test_flash_adamw_fp32_param_roundtrip():
    params = {
        "w": jnp.linspace(-2.0, 2.0, 64, dtype=jnp.float32).astype(jnp.bfloat16),
        "norm": jnp.array([1.0, 0.5, -0.25], dtype=jnp.float32),
    }
    grads = {
        "w": jnp.linspace(-0.3, 0.4, 64, dtype=jnp.float32),
        "norm": jnp.array([0.1, -0.2, 0.3], dtype=jnp.float32),
    }
    tx = flash_adamw(learning_rate=1e-3, fused=False)
    state = tx.init(params)
    params, state = tx.step(params, state, grads)

    fp32_params = reconstruct_weights(params, state.ecc)
    restored_params, restored_ecc = set_fp32_params(
        fp32_params,
        param_template=params,
        master_weight_bits=24,
    )
    reconstructed = reconstruct_weights(restored_params, restored_ecc)

    np.testing.assert_allclose(
        np.asarray(reconstructed["w"]),
        np.asarray(fp32_params["w"]),
        atol=1e-4,
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(reconstructed["norm"]),
        np.asarray(fp32_params["norm"]),
        atol=1e-7,
        rtol=1e-7,
    )
    assert restored_params["w"].dtype == params["w"].dtype
    assert restored_params["norm"].dtype == params["norm"].dtype
