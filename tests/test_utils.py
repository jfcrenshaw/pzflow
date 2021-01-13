import jax.numpy as np
from pzflow.utils import Normal


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