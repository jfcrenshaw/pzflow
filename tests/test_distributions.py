import pytest
import jax.numpy as np
from pzflow.distributions import *


@pytest.mark.parametrize(
    "distribution,inputs,params",
    [
        (Normal, (2,), ()),
        (Tdist, (2,), np.log(30.0)),
        (Uniform, ((0, 1), (0, 1)), ()),
        (Joint, (Normal(1), Uniform((0, 1))), ((), ())),
        (Joint, (Normal(1), Tdist(1)), ((), np.log(30.0))),
        (Joint, (Joint(Normal(1), Uniform((0, 1))).info[1]), ((), ())),
    ],
)
class TestDistributions:
    def test_returns_correct_shapes(self, distribution, inputs, params):

        dist = distribution(*inputs)

        nsamples = 8
        samples = dist.sample(params, nsamples)
        assert samples.shape == (nsamples, 2)

        log_prob = dist.log_prob(params, samples)
        assert log_prob.shape == (nsamples,)

    def test_control_sample_randomness(self, distribution, inputs, params):

        dist = distribution(*inputs)

        nsamples = 8
        s1 = dist.sample(params, nsamples)
        s2 = dist.sample(params, nsamples)
        assert ~np.all(np.isclose(s1, s2))

        s1 = dist.sample(params, nsamples, seed=0)
        s2 = dist.sample(params, nsamples, seed=0)
        assert np.allclose(s1, s2)


def test_normal_cov():

    dist = Normal(2)

    nsamples = 2
    samples = dist.sample((), nsamples)

    cov = np.array([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])
    log_prob = dist.log_prob(4, samples, cov=cov)
    assert log_prob.shape == (nsamples,)


@pytest.mark.parametrize(
    "inputs",
    [
        ((-1, 1, 2),),
        ((2, 1),),
    ],
)
def test_uniform_bad_inputs(inputs):
    with pytest.raises(ValueError):
        Uniform(*inputs)
