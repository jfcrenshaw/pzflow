import pytest
from pzflow.bijectors import *
import jax.numpy as np
from jax import random

x = np.array(
    [
        [0.2, 24, 23, 27, 26, 24, 24],
        [1.4, 26, 26, 17, 14, 36, 23],
        [3.4, -1, 12, 74, 44, -6, 97],
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

        assert np.allclose(inv_outputs, x)
        assert np.allclose(fwd_log_det, -inv_log_det)