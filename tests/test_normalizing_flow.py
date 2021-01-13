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


def test_simple_flow(tmp_path):
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
