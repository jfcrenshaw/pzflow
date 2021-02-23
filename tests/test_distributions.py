import pytest
import jax.numpy as np
from pzflow.distributions import *


@pytest.mark.parametrize("distribution", [Normal, Tdist])
def test_returns_correct_shapes(distribution):

    input_dim = 2
    dist = distribution(input_dim)

    nsamples = 8
    samples = dist.sample(4, nsamples)
    assert samples.shape == (nsamples, input_dim)

    log_prob = dist.log_prob(4, samples)
    assert log_prob.shape == (nsamples,)


@pytest.mark.parametrize("distribution", [Normal, Tdist])
def test_control_sample_randomness(distribution):

    input_dim = 2
    dist = distribution(input_dim)

    nsamples = 8
    s1 = dist.sample(4, nsamples)
    s2 = dist.sample(4, nsamples)
    assert ~np.all(np.isclose(s1, s2))

    s1 = dist.sample(4, nsamples, seed=0)
    s2 = dist.sample(4, nsamples, seed=0)
    assert np.allclose(s1, s2)


def test_normal_cov():

    dist = Normal(2)

    nsamples = 2
    samples = dist.sample((), nsamples)

    cov = np.array([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])
    log_prob = dist.log_prob(4, samples, cov=cov)
    assert log_prob.shape == (nsamples,)