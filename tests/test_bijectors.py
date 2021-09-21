import pytest
from pzflow.bijectors import *
import jax.numpy as np
from jax import random, jit


x = np.array(
    [
        [0.2, 0.1, -0.3, 0.5, 0.1, -0.4, -0.3],
        [0.6, 0.5, 0.2, 0.2, -0.4, -0.1, 0.7],
        [0.9, 0.2, -0.3, 0.3, 0.4, -0.4, -0.1],
    ]
)


@pytest.mark.parametrize(
    "bijector,args,conditions",
    [
        (ColorTransform, (3, [1, 3, 5]), np.zeros((3, 1))),
        (Reverse, (), np.zeros((3, 1))),
        (Roll, (2,), np.zeros((3, 1))),
        (Scale, (2.0,), np.zeros((3, 1))),
        (Shuffle, (), np.zeros((3, 1))),
        (InvSoftplus, (0,), np.zeros((3, 1))),
        (InvSoftplus, ([1, 3], [2.0, 12.0]), np.zeros((3, 1))),
        (
            StandardScaler,
            (np.linspace(-1, 1, 7), np.linspace(1, 8, 7)),
            np.zeros((3, 1)),
        ),
        (Chain, (Reverse(), Scale(1 / 6), Roll(-1)), np.zeros((3, 1))),
        (NeuralSplineCoupling, (), np.zeros((3, 1))),
        (NeuralSplineCoupling, (16, 3, 2, 128, 3), np.arange(9).reshape(3, 3)),
        (RollingSplineCoupling, (2,), np.zeros((3, 1))),
        (RollingSplineCoupling, (2, 1, 16, 3, 2, 128, None, 0, True), np.zeros((3, 1))),
        (ShiftBounds, (-0.5, 0.9, 5), np.zeros((3, 1))),
        (ShiftBounds, (-1 * np.ones(7), 1.1 * np.ones(7), 3), np.zeros((3, 1))),
    ],
)
class TestBijectors:
    def test_returns_correct_shape(self, bijector, args, conditions):
        init_fun, bijector_info = bijector(*args)
        params, forward_fun, inverse_fun = init_fun(random.PRNGKey(0), x.shape[-1])

        fwd_outputs, fwd_log_det = forward_fun(params, x, conditions=conditions)
        assert fwd_outputs.shape == x.shape
        assert fwd_log_det.shape == x.shape[:1]

        inv_outputs, inv_log_det = inverse_fun(params, x, conditions=conditions)
        assert inv_outputs.shape == x.shape
        assert inv_log_det.shape == x.shape[:1]

    def test_is_bijective(self, bijector, args, conditions):
        init_fun, bijector_info = bijector(*args)
        params, forward_fun, inverse_fun = init_fun(random.PRNGKey(0), x.shape[-1])

        fwd_outputs, fwd_log_det = forward_fun(params, x, conditions=conditions)
        inv_outputs, inv_log_det = inverse_fun(
            params, fwd_outputs, conditions=conditions
        )

        print(inv_outputs)
        assert np.allclose(inv_outputs, x, atol=1e-6)
        assert np.allclose(fwd_log_det, -inv_log_det, atol=1e-6)

    def test_is_jittable(self, bijector, args, conditions):
        init_fun, bijector_info = bijector(*args)
        params, forward_fun, inverse_fun = init_fun(random.PRNGKey(0), x.shape[-1])

        fwd_outputs_1, fwd_log_det_1 = forward_fun(params, x, conditions=conditions)
        forward_fun = jit(forward_fun)
        fwd_outputs_2, fwd_log_det_2 = forward_fun(params, x, conditions=conditions)

        inv_outputs_1, inv_log_det_1 = inverse_fun(params, x, conditions=conditions)
        inverse_fun = jit(inverse_fun)
        inv_outputs_2, inv_log_det_2 = inverse_fun(params, x, conditions=conditions)


@pytest.mark.parametrize(
    "bijector,args",
    [
        (ColorTransform, (0, [1, 2, 3, 4])),
        (ColorTransform, (1.3, [1, 2, 3, 4])),
        (ColorTransform, (1, [2, 3, 4])),
        (Roll, (2.4,)),
        (Scale, (2,)),
        (Scale, (np.arange(7),)),
        (InvSoftplus, ([0, 1, 2], [1.0, 2.0])),
        (
            RollingSplineCoupling,
            (2, 1, 16, 3, 2, 128, None, 0, "fake"),
        ),
        (ShiftBounds, (4, 2, 1)),
        (ShiftBounds, (np.array([0, 1]), 2, 1)),
    ],
)
def test_bad_inputs(bijector, args):
    with pytest.raises(ValueError):
        bijector(*args)


@pytest.mark.parametrize("column_idx", [(None), ([1, 3, 5])])
def test_uniform_dequantizer_returns_correct_shape(column_idx):
    init_fun, bijector_info = UniformDequantizer(column_idx)
    params, forward_fun, inverse_fun = init_fun(random.PRNGKey(0), x.shape[-1])

    conditions = np.zeros((3, 1))
    fwd_outputs, fwd_log_det = forward_fun(params, x, conditions=conditions)
    assert fwd_outputs.shape == x.shape
    assert fwd_log_det.shape == x.shape[:1]

    inv_outputs, inv_log_det = inverse_fun(params, x, conditions=conditions)
    assert inv_outputs.shape == x.shape
    assert inv_log_det.shape == x.shape[:1]