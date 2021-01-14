import pytest
import jax.numpy as np
from pzflow.utils import Normal, LSSTErrorModel
import pandas as pd


def test_Normal_returns_correct_shapes():

    input_dim = 2
    normal = Normal(input_dim)

    nsamples = 8
    samples = normal.sample(nsamples)
    assert samples.shape == (nsamples, input_dim)

    log_prob = normal.log_prob(samples)
    assert log_prob.shape == (nsamples,)


def test_Normal_control_sample_randomness():

    input_dim = 2
    normal = Normal(input_dim)

    nsamples = 8
    s1 = normal.sample(nsamples)
    s2 = normal.sample(nsamples)
    assert ~np.all(np.isclose(s1, s2))

    s1 = normal.sample(nsamples, seed=0)
    s2 = normal.sample(nsamples, seed=0)
    assert np.allclose(s1, s2)


@pytest.mark.parametrize(
    "limiting_mags,err_params",
    [
        ("fake", None),
        (None, "fake"),
    ],
)
def test_LSSTErrorModel_inputs_not_dictionary(limiting_mags, err_params):
    with pytest.raises(ValueError):
        err_model = LSSTErrorModel(limiting_mags=limiting_mags, err_params=err_params)


@pytest.mark.parametrize(
    "limiting_mags,err_params,undetected_flag",
    [
        ({"u": "fake", "g": 1}, {"u": 1, "g": 1}, 99),
        ({"u": 1, "g": 1}, {"u": "fake", "g": 1}, 99),
        ({"u": 1, "g": 1}, {"u": 1, "g": 1}, "fake"),
        ({"u": 1, "g": 1}, {"u": 1}, 99),
        ({"u": 1}, {"u": 1, "g": 1}, 99),
    ],
)
def test_LSSTErrorModel_bad_inputs(limiting_mags, err_params, undetected_flag):
    with pytest.raises(AssertionError):
        err_model = LSSTErrorModel(
            limiting_mags=limiting_mags,
            err_params=err_params,
            undetected_flag=undetected_flag,
        )


def test_LSSTErrorModel_returns_correct_shape():
    data = pd.DataFrame(
        [
            [0.1, 24, 25, 26, 26, 27, 28],
            [0.2, 25, 24, 24, 25, 24, 23],
            [1.2, 26, 26, 25, 25, 27, 28],
        ],
        columns=["redshift", "u", "g", "r", "i", "z", "y"],
    )

    err_data = LSSTErrorModel()(data)

    assert err_data.shape == (data.shape[0], 2 * data.shape[1] - 1)
