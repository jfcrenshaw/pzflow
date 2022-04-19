import jax.numpy as jnp
import pytest
from jax import jit, random

from pzflow.bijectors import *

x = jnp.array(
    [
        [0.2, 0.1, -0.3, 0.5, 0.1, -0.4, -0.3],
        [0.6, 0.5, 0.2, 0.2, -0.4, -0.1, 0.7],
        [0.9, 0.2, -0.3, 0.3, 0.4, -0.4, -0.1],
    ]
)


@pytest.mark.parametrize(
    "bijector,args,conditions",
    [
        (ColorTransform, (3, [1, 3, 5]), jnp.zeros((3, 1))),
        (Reverse, (), jnp.zeros((3, 1))),
        (Roll, (2,), jnp.zeros((3, 1))),
        (Scale, (2.0,), jnp.zeros((3, 1))),
        (Shuffle, (), jnp.zeros((3, 1))),
        (InvSoftplus, (0,), jnp.zeros((3, 1))),
        (InvSoftplus, ([1, 3], [2.0, 12.0]), jnp.zeros((3, 1))),
        (
            StandardScaler,
            (jnp.linspace(-1, 1, 7), jnp.linspace(1, 8, 7)),
            jnp.zeros((3, 1)),
        ),
        (Chain, (Reverse(), Scale(1 / 6), Roll(-1)), jnp.zeros((3, 1))),
        (NeuralSplineCoupling, (), jnp.zeros((3, 1))),
        (
            NeuralSplineCoupling,
            (16, 3, 2, 128, 3),
            jnp.arange(9).reshape(3, 3),
        ),
        (RollingSplineCoupling, (2,), jnp.zeros((3, 1))),
        (
            RollingSplineCoupling,
            (2, 1, 16, 3, 2, 128, None, 0, True),
            jnp.zeros((3, 1)),
        ),
        (ShiftBounds, (-0.5, 0.9, 5), jnp.zeros((3, 1))),
        (
            ShiftBounds,
            (-1 * jnp.ones(7), 1.1 * jnp.ones(7), 3),
            jnp.zeros((3, 1)),
        ),
    ],
)
class TestBijectors:
    def test_returns_correct_shape(self, bijector, args, conditions):
        init_fun, bijector_info = bijector(*args)
        params, forward_fun, inverse_fun = init_fun(
            random.PRNGKey(0), x.shape[-1]
        )

        fwd_outputs, fwd_log_det = forward_fun(
            params, x, conditions=conditions
        )
        assert fwd_outputs.shape == x.shape
        assert fwd_log_det.shape == x.shape[:1]

        inv_outputs, inv_log_det = inverse_fun(
            params, x, conditions=conditions
        )
        assert inv_outputs.shape == x.shape
        assert inv_log_det.shape == x.shape[:1]

    def test_is_bijective(self, bijector, args, conditions):
        init_fun, bijector_info = bijector(*args)
        params, forward_fun, inverse_fun = init_fun(
            random.PRNGKey(0), x.shape[-1]
        )

        fwd_outputs, fwd_log_det = forward_fun(
            params, x, conditions=conditions
        )
        inv_outputs, inv_log_det = inverse_fun(
            params, fwd_outputs, conditions=conditions
        )

        print(inv_outputs)
        assert jnp.allclose(inv_outputs, x, atol=1e-6)
        assert jnp.allclose(fwd_log_det, -inv_log_det, atol=1e-6)

    def test_is_jittable(self, bijector, args, conditions):
        init_fun, bijector_info = bijector(*args)
        params, forward_fun, inverse_fun = init_fun(
            random.PRNGKey(0), x.shape[-1]
        )

        fwd_outputs_1, fwd_log_det_1 = forward_fun(
            params, x, conditions=conditions
        )
        forward_fun = jit(forward_fun)
        fwd_outputs_2, fwd_log_det_2 = forward_fun(
            params, x, conditions=conditions
        )

        inv_outputs_1, inv_log_det_1 = inverse_fun(
            params, x, conditions=conditions
        )
        inverse_fun = jit(inverse_fun)
        inv_outputs_2, inv_log_det_2 = inverse_fun(
            params, x, conditions=conditions
        )


@pytest.mark.parametrize(
    "bijector,args",
    [
        (ColorTransform, (0, [1, 2, 3, 4])),
        (ColorTransform, (1.3, [1, 2, 3, 4])),
        (ColorTransform, (1, [2, 3, 4])),
        (Roll, (2.4,)),
        (Scale, (2,)),
        (Scale, (jnp.arange(7),)),
        (InvSoftplus, ([0, 1, 2], [1.0, 2.0])),
        (
            RollingSplineCoupling,
            (2, 1, 16, 3, 2, 128, None, 0, "fake"),
        ),
        (ShiftBounds, (4, 2, 1)),
        (ShiftBounds, (jnp.array([0, 1]), 2, 1)),
    ],
)
def test_bad_inputs(bijector, args):
    with pytest.raises(ValueError):
        bijector(*args)


def test_uniform_dequantizer_returns_correct_shape():
    init_fun, bijector_info = UniformDequantizer([1, 3, 4])
    params, forward_fun, inverse_fun = init_fun(random.PRNGKey(0), x.shape[-1])

    conditions = jnp.zeros((3, 1))
    fwd_outputs, fwd_log_det = forward_fun(params, x, conditions=conditions)
    assert fwd_outputs.shape == x.shape
    assert fwd_log_det.shape == x.shape[:1]

    inv_outputs, inv_log_det = inverse_fun(params, x, conditions=conditions)
    assert inv_outputs.shape == x.shape
    assert inv_log_det.shape == x.shape[:1]
