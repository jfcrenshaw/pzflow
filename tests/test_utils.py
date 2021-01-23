import jax.numpy as np
from jax import random
from pzflow.bijectors import *
from pzflow.utils import Normal, build_bijector_from_info


def test_returns_correct_shapes():

    input_dim = 2
    normal = Normal(input_dim)

    nsamples = 8
    samples = normal.sample(nsamples)
    assert samples.shape == (nsamples, input_dim)

    log_prob = normal.log_prob(samples)
    assert log_prob.shape == (nsamples,)


def test_control_sample_randomness():

    input_dim = 2
    normal = Normal(input_dim)

    nsamples = 8
    s1 = normal.sample(nsamples)
    s2 = normal.sample(nsamples)
    assert ~np.all(np.isclose(s1, s2))

    s1 = normal.sample(nsamples, seed=0)
    s2 = normal.sample(nsamples, seed=0)
    assert np.allclose(s1, s2)


def test_build_bijector_from_info():

    x = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    init_fun, info1 = Chain(
        Reverse(),
        Chain(ColorTransform(0, 1, 1), Roll(-1)),
        Scale(-0.5),
        Softplus(0, 1),
        Chain(Roll(), Scale(-4)),
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
