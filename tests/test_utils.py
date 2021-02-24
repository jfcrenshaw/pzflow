import jax.numpy as np
from jax import random, ops
from pzflow.bijectors import *
from pzflow.utils import *
import pytest


def test_build_bijector_from_info():

    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    init_fun, info1 = Chain(
        Reverse(),
        Chain(ColorTransform(1, [1, 2, 3]), Roll(-1)),
        InvSoftplus(0, 1),
        Scale(-0.5),
        Chain(Roll(), Scale(-4.0)),
    )

    params, forward_fun, inverse_fun = init_fun(random.PRNGKey(0), 4)
    xfwd1, log_det1 = forward_fun(params, x)

    init_fun, info2 = build_bijector_from_info(info1)
    assert info1 == info2

    params, forward_fun, inverse_fun = init_fun(random.PRNGKey(0), 4)
    xfwd2, log_det2 = forward_fun(params, x)
    assert np.allclose(xfwd1, xfwd2)
    assert np.allclose(log_det1, log_det2)

    invx, inv_log_det = inverse_fun(params, xfwd2)
    assert np.allclose(x, invx)
    assert np.allclose(log_det2, -inv_log_det)


def test_sub_diag_indices_correct():
    x = np.array([[[0, 0], [0, 0]], [[1, 1], [1, 1]], [[2, 2], [2, 2]]])
    y = np.array([[[1, 0], [0, 1]], [[2, 1], [1, 2]], [[3, 2], [2, 3]]])
    idx = sub_diag_indices(x)
    x = ops.index_update(x, idx, x[idx] + 1)

    assert np.allclose(x, y)


@pytest.mark.parametrize(
    "x",
    [np.ones(2), np.ones((2, 2)), np.ones((2, 2, 2, 2))],
)
def test_sub_diag_indices_bad_input(x):
    with pytest.raises(ValueError):
        idx = sub_diag_indices(x)