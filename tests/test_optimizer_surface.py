import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from flashoptim_jax import flash_adam, flash_adamw, flash_lion, flash_sgd, flash_sgdw
from flashoptim_jax.compression import reconstruct_weights


def _adam_step(param, grad, mu, nu, *, lr, b1, b2, eps, weight_decay, decoupled, step):
    if not decoupled and weight_decay:
        grad = grad + weight_decay * param
    mu = b1 * mu + (1.0 - b1) * grad
    nu = b2 * nu + (1.0 - b2) * (grad * grad)
    if decoupled and weight_decay:
        param = param * (1.0 - lr * weight_decay)
    bias_correction1 = 1.0 - b1**step
    bias_correction2 = 1.0 - b2**step
    param = param - (lr / bias_correction1) * mu / (jnp.sqrt(nu) / jnp.sqrt(bias_correction2) + eps)
    return param, mu, nu


def _lion_step(param, grad, mu, *, lr, b1, b2, weight_decay):
    if weight_decay:
        param = param * (1.0 - lr * weight_decay)
    update = jnp.sign(b1 * mu + (1.0 - b1) * grad)
    param = param - lr * update
    mu = b2 * mu + (1.0 - b2) * grad
    return param, mu


def _sgd_step(param, grad, mu, *, lr, momentum, dampening, weight_decay, nesterov, decoupled):
    if not decoupled and weight_decay:
        grad = grad + weight_decay * param
    if momentum > 0.0:
        mu = grad if mu is None else momentum * mu + (1.0 - dampening) * grad
        direction = grad + momentum * mu if nesterov else mu
    else:
        direction = grad
    if decoupled and weight_decay:
        param = param * (1.0 - lr * weight_decay)
    param = param - lr * direction
    return param, mu


_OPTIMIZER_FACTORIES = (
    (flash_adam, {}),
    (flash_adamw, {}),
    (flash_lion, {}),
    (flash_sgd, {"momentum": 0.9}),
    (flash_sgdw, {"momentum": 0.9}),
)


def test_flash_adam_matches_reference():
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}
    tx = flash_adam(
        learning_rate=1e-3,
        weight_decay=1e-2,
        quantize=False,
        master_weight_bits=None,
        fused=False,
    )
    state = tx.init(params)
    actual, _ = tx.step(params, state, grads)

    expected, _, _ = _adam_step(
        params["x"],
        grads["x"],
        jnp.zeros_like(params["x"]),
        jnp.zeros_like(params["x"]),
        lr=jnp.asarray(1e-3, dtype=jnp.float32),
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=1e-2,
        decoupled=False,
        step=1,
    )
    np.testing.assert_allclose(np.asarray(actual["x"]), np.asarray(expected), atol=1e-7, rtol=1e-7)


def test_flash_lion_matches_reference():
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}
    tx = flash_lion(
        learning_rate=1e-4,
        weight_decay=1e-2,
        quantize=False,
        master_weight_bits=None,
    )
    state = tx.init(params)
    actual, _ = tx.step(params, state, grads)

    expected, _ = _lion_step(
        params["x"],
        grads["x"],
        jnp.zeros_like(params["x"]),
        lr=jnp.asarray(1e-4, dtype=jnp.float32),
        b1=0.9,
        b2=0.99,
        weight_decay=1e-2,
    )
    np.testing.assert_allclose(np.asarray(actual["x"]), np.asarray(expected), atol=1e-7, rtol=1e-7)


def test_flash_sgd_matches_reference_with_nesterov():
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}
    tx = flash_sgd(
        learning_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-2,
        nesterov=True,
        quantize=False,
        master_weight_bits=None,
    )
    state = tx.init(params)
    actual, _ = tx.step(params, state, grads)

    expected, _ = _sgd_step(
        params["x"],
        grads["x"],
        None,
        lr=jnp.asarray(1e-3, dtype=jnp.float32),
        momentum=0.9,
        dampening=0.0,
        weight_decay=1e-2,
        nesterov=True,
        decoupled=False,
    )
    np.testing.assert_allclose(np.asarray(actual["x"]), np.asarray(expected), atol=1e-7, rtol=1e-7)


def test_flash_sgdw_matches_reference():
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)}
    grads = {"x": jnp.array([0.2, -0.1, 0.05], dtype=jnp.float32)}
    tx = flash_sgdw(
        learning_rate=1e-3,
        momentum=0.9,
        weight_decay=1e-2,
        quantize=False,
        master_weight_bits=None,
    )
    state = tx.init(params)
    actual, _ = tx.step(params, state, grads)

    expected, _ = _sgd_step(
        params["x"],
        grads["x"],
        None,
        lr=jnp.asarray(1e-3, dtype=jnp.float32),
        momentum=0.9,
        dampening=0.0,
        weight_decay=1e-2,
        nesterov=False,
        decoupled=True,
    )
    np.testing.assert_allclose(np.asarray(actual["x"]), np.asarray(expected), atol=1e-7, rtol=1e-7)


def test_flash_adam_fused_matches_unfused():
    params = {"x": jnp.linspace(-2.0, 2.0, 65, dtype=jnp.float32).astype(jnp.bfloat16)}
    grads = {"x": jnp.linspace(-0.3, 0.4, 65, dtype=jnp.float32)}
    unfused_tx = flash_adam(learning_rate=1e-3, fused=False)
    fused_tx = flash_adam(learning_rate=1e-3, fused=True)

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


@pytest.mark.parametrize(
    ("factory", "extra_kwargs"),
    _OPTIMIZER_FACTORIES,
    ids=["adam", "adamw", "lion", "sgd", "sgdw"],
)
def test_flash_optimizers_reject_invalid_fused_group_size(factory, extra_kwargs):
    with pytest.raises(ValueError, match="group_size must divide BLOCK_SIZE"):
        factory(
            learning_rate=1e-3,
            quantize=True,
            master_weight_bits=None,
            group_size=40,
            fused=True,
            **extra_kwargs,
        )


@pytest.mark.parametrize(
    ("factory", "extra_kwargs"),
    _OPTIMIZER_FACTORIES,
    ids=["adam", "adamw", "lion", "sgd", "sgdw"],
)
def test_flash_optimizers_reject_meaningless_master_weight_bits(factory, extra_kwargs):
    params = {"x": jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)}
    tx = factory(
        learning_rate=1e-3,
        master_weight_bits=32,
        fused=False,
        **extra_kwargs,
    )
    with pytest.raises(ValueError, match="has no effect"):
        tx.init(params)


def test_flash_adamw_rejects_meaningless_param_group_master_weight_bits():
    params = {
        "trunk": {"w": jnp.array([1.0, -2.0], dtype=jnp.bfloat16)},
        "head": {"w": jnp.array([0.5, -0.75], dtype=jnp.float32)},
    }
    tx = flash_adamw(
        learning_rate=1e-3,
        master_weight_bits=24,
        fused=False,
        param_groups=[
            {
                "params": ["head"],
                "master_weight_bits": 32,
            }
        ],
    )
    with pytest.raises(ValueError, match=r"param_groups\[0\]\['master_weight_bits'\]=32 has no effect"):
        tx.init(params)


def test_flash_adamw_param_groups_override_hparams():
    params = {
        "trunk": {"w": jnp.array([1.0, -2.0], dtype=jnp.float32)},
        "head": {"w": jnp.array([0.5, -0.75], dtype=jnp.float32)},
    }
    grads = {
        "trunk": {"w": jnp.array([0.2, -0.1], dtype=jnp.float32)},
        "head": {"w": jnp.array([0.4, -0.3], dtype=jnp.float32)},
    }
    tx = flash_adamw(
        learning_rate=1e-3,
        weight_decay=1e-2,
        quantize=False,
        master_weight_bits=None,
        fused=False,
        param_groups=[
            {
                "params": ["head"],
                "learning_rate": 5e-3,
                "weight_decay": 0.0,
            }
        ],
    )
    state = tx.init(params)
    actual, _ = tx.step(params, state, grads)

    trunk_expected, _, _ = _adam_step(
        params["trunk"]["w"],
        grads["trunk"]["w"],
        jnp.zeros_like(params["trunk"]["w"]),
        jnp.zeros_like(params["trunk"]["w"]),
        lr=jnp.asarray(1e-3, dtype=jnp.float32),
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=1e-2,
        decoupled=True,
        step=1,
    )
    head_expected, _, _ = _adam_step(
        params["head"]["w"],
        grads["head"]["w"],
        jnp.zeros_like(params["head"]["w"]),
        jnp.zeros_like(params["head"]["w"]),
        lr=jnp.asarray(5e-3, dtype=jnp.float32),
        b1=0.9,
        b2=0.999,
        eps=1e-8,
        weight_decay=0.0,
        decoupled=True,
        step=1,
    )

    np.testing.assert_allclose(np.asarray(actual["trunk"]["w"]), np.asarray(trunk_expected), atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(np.asarray(actual["head"]["w"]), np.asarray(head_expected), atol=1e-7, rtol=1e-7)
