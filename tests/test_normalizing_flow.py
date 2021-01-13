import pytest
import jax.numpy as np
from pzflow import Flow
from pzflow.bijectors import Reverse


@pytest.mark.parametrize(
    "input_dim,bijector,file",
    [
        (None, None, None),
        (2, None, "file"),
        (None, 2, "file"),
        (2, 2, "file"),
        (-1, None, None),
        (0, None, None),
        (1.1, None, None),
    ],
)
def test_bad_inputs(input_dim, bijector, file):
    with pytest.raises(ValueError):
        flow = Flow(input_dim, bijector, file)


def test_returns_correct_shape():
    flow = Flow(2, Reverse())

    x = np.array([[1, 2], [3, 4]])

    assert flow.forward(x).shape == x.shape
    assert flow.inverse(x).shape == x.shape
    assert flow.sample(2).shape == x.shape
    assert flow.log_prob(x).shape == (x.shape[0],)

    grid = np.arange(0, 2.1, 0.12)
    assert flow.posterior(x, grid=grid).shape == (x.shape[0], grid.size)
    assert flow.posterior(x[:, 1:], grid=grid).shape == (x.shape[0], grid.size)

    assert len(flow.train(x, epochs=11, verbose=True)) == 12


def test_flow_bijection():
    flow = Flow(2, Reverse(), info=["random", 42])

    x = np.array([[1, 2], [3, 4]])
    xrev = np.array([[2, 1], [4, 3]])

    assert np.allclose(flow.forward(x), xrev)
    assert np.allclose(flow.inverse(flow.forward(x)), x)
    assert flow.info == ["random", 42]


def test_load_flow(tmp_path):
    flow = Flow(2, Reverse(), info=["random", 42])
    file = tmp_path / "test-flow.dill"
    flow.save(str(file))

    file = tmp_path / "test-flow.dill"
    flow = Flow(file=str(file))

    x = np.array([[1, 2], [3, 4]])
    xrev = np.array([[2, 1], [4, 3]])

    assert np.allclose(flow.forward(x), xrev)
    assert np.allclose(flow.inverse(flow.forward(x)), x)
    assert flow.info == ["random", 42]


def test_control_sample_randomness():
    flow = Flow(2, Reverse())

    assert np.all(~np.isclose(flow.sample(2), flow.sample(2)))
    assert np.allclose(flow.sample(2, seed=0), flow.sample(2, seed=0))


@pytest.mark.parametrize(
    "inputs,column,mode",
    [
        (np.zeros((2, 2)), 0, "fake"),
        (np.zeros((2, 2)), 0, "insert"),
        (np.zeros((2, 1)), 0, "replace"),
        (np.zeros((2, 3)), 0, "auto"),
        (np.zeros((2, 2)), 0.0, "auto"),
    ],
)
def test_posterior_inputs(inputs, column, mode):
    flow = Flow(2, Reverse())
    with pytest.raises(ValueError):
        flow.posterior(inputs, column=column, mode=mode)