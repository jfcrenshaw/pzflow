from pzflow.bijectors import Reverse, Scale
import jax.numpy as np
from jax import random


def test_Reverse():
    rng = random.PRNGKey(0)
    x = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    init_fun = Reverse()
    params, forward_fun, inverse_fun = init_fun(rng, x.shape[-1])
    outputs, log_det = forward_fun(params, x)

    assert np.allclose(outputs, x[:, ::-1])
    assert np.allclose(log_det, np.zeros(x.shape[0]))

    outputs, log_det = inverse_fun(params, outputs)
    assert np.allclose(outputs, x)
    assert np.allclose(log_det, np.zeros(x.shape[0]))


def test_Scale():
    rng = random.PRNGKey(0)
    x = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])

    init_fun = Scale(scale=2)
    params, forward_fun, inverse_fun = init_fun(rng, x.shape[-1])
    outputs, fwd_log_det = forward_fun(params, x)

    assert np.allclose(outputs, 2 * x)
    assert np.allclose(fwd_log_det, np.log(2 ** x.shape[-1]) * np.ones(x.shape[0]))

    outputs, inv_log_det = inverse_fun(params, outputs)
    assert np.allclose(outputs, x)
    assert np.allclose(inv_log_det, -np.log(2 ** x.shape[-1]) * np.ones(x.shape[0]))

    assert np.allclose(fwd_log_det + inv_log_det, np.zeros(fwd_log_det.shape))
