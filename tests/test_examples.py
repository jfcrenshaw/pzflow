"""Tests for pzflow.examples."""

import jax.numpy as jnp
import pandas as pd

from pzflow import Flow, examples


def test_get_twomoons_data() -> None:
    """get_twomoons_data returns a DataFrame of the expected shape."""
    data = examples.get_twomoons_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (100_000, 2)


def test_get_galaxy_data() -> None:
    """get_galaxy_data returns a DataFrame of the expected shape."""
    data = examples.get_galaxy_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (100_000, 7)


def test_get_city_data() -> None:
    """get_city_data returns a DataFrame of the expected shape."""
    data = examples.get_city_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (47_966, 5)


def test_get_checkerboard_data() -> None:
    """get_checkerboard_data returns a DataFrame of the expected shape."""
    data = examples.get_checkerboard_data()
    assert isinstance(data, pd.DataFrame)
    assert data.shape == (100_000, 2)


def test_get_example_flow() -> None:
    """get_example_flow returns a trained Flow with usable sample/log_prob/posterior."""
    flow = examples.get_example_flow()
    assert isinstance(flow, Flow)
    assert isinstance(flow.info, str)

    samples = flow.sample(2)
    flow.log_prob(samples)

    grid = jnp.arange(0, 2.5, 0.5)
    flow.posterior(samples, column="redshift", grid=grid)
