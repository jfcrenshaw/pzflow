import jax.numpy as np
import pandas as pd
from pzflow import examples
from pzflow import Flow


def test_get_twomoons_data():
    data = examples.get_twomoons_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (10000, 2)


def test_get_galaxy_data():
    data = examples.get_galaxy_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (100000, 7)


def test_get_city_data():
    data = examples.get_city_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (47966, 5)


def test_get_example_flow():
    flow = examples.get_example_flow()
    assert isinstance(flow, Flow)
    assert isinstance(flow.info, str)

    samples = flow.sample(2)
    flow.log_prob(samples)

    grid = np.arange(0, 2.5, 0.5)
    flow.posterior(samples, column="redshift", grid=grid)
