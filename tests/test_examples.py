import jax.numpy as np
from pzflow import examples
from pzflow import Flow


def test_load_two_moons_data():
    columns, data = examples.two_moons_data()
    assert columns == ("x", "y")
    assert isinstance(data, np.ndarray)
    assert data.shape == (10000, 2)


def test_load_galaxy_data():
    columns, data = examples.galaxy_data()
    assert all(isinstance(col, str) for col in columns)
    assert isinstance(data, np.ndarray)
    assert data.shape == (100000, len(columns))


def test_load_example_flow():
    flow = examples.example_flow()
    assert isinstance(flow, Flow)
    assert isinstance(flow.info, str)