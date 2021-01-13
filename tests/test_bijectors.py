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
    "bijector,args",
    [
        (ColorTransform, (3, 20, 5)),
        (Reverse, ()),
        (Roll, (2,)),
        (Scale, (2,)),
        (Shuffle, ()),
        (Chain, (Reverse(), Scale(1 / 6), Roll(-1))),
        (NeuralSplineCoupling, ()),
        (RollingSplineCoupling, (2,)),
    ],
)
class TestBijectors:
    def test_returns_correct_shape(self, bijector, args):
        init_fun = bijector(*args)
        params, forward_fun, inverse_fun = init_fun(random.PRNGKey(0), x.shape[-1])

        fwd_outputs, fwd_log_det = forward_fun(params, x)
        assert fwd_outputs.shape == x.shape
        assert fwd_log_det.shape == x.shape[:1]

        inv_outputs, inv_log_det = inverse_fun(params, x)
        assert inv_outputs.shape == x.shape
        assert inv_log_det.shape == x.shape[:1]

    def test_is_bijective(self, bijector, args):
        init_fun = bijector(*args)
        params, forward_fun, inverse_fun = init_fun(random.PRNGKey(0), x.shape[-1])

        fwd_outputs, fwd_log_det = forward_fun(params, x)
        inv_outputs, inv_log_det = inverse_fun(params, fwd_outputs)

        assert np.allclose(inv_outputs, x, atol=1e-6)
        assert np.allclose(fwd_log_det, -inv_log_det, atol=1e-6)

    def test_is_jittable(self, bijector, args):
        init_fun = bijector(*args)
        params, forward_fun, inverse_fun = init_fun(random.PRNGKey(0), x.shape[-1])

        fwd_outputs_1, fwd_log_det_1 = forward_fun(params, x)
        forward_fun = jit(forward_fun)
        fwd_outputs_2, fwd_log_det_2 = forward_fun(params, x)

        inv_outputs_1, inv_log_det_1 = inverse_fun(params, x)
        inverse_fun = jit(inverse_fun)
        inv_outputs_2, inv_log_det_2 = inverse_fun(params, x)