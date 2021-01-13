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

    zmin, zmax, dz = -1, 1, 0.1
    zs = np.arange(zmin, zmax + dz, dz)
    assert flow.pz_estimate(x, zmin, zmax, dz).shape == (x.shape[0], zs.size)

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
