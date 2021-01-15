import pytest
import jax.numpy as np
import pandas as pd
from pzflow import Flow
from pzflow.bijectors import Reverse


@pytest.mark.parametrize(
    "data_columns,bijector,info,file",
    [
        (None, None, None, None),
        (("x", "y"), None, None, None),
        (None, Reverse(), None, None),
        (("x", "y"), None, None, "file"),
        (None, Reverse(), None, "file"),
        (None, None, "fake", "file"),
    ],
)
def test_bad_inputs(data_columns, bijector, info, file):
    with pytest.raises(ValueError):
        Flow(data_columns, bijector, info, file)


def test_returns_correct_shape():
    columns = ("x", "y")
    flow = Flow(columns, Reverse())

    xarray = np.array([[1, 2], [3, 4]])
    x = pd.DataFrame(xarray, columns=columns)

    xfwd, xfwd_log_det = flow._forward(flow._params, xarray)
    assert xfwd.shape == x.shape
    assert xfwd_log_det.shape == (x.shape[0],)

    xinv, xinv_log_det = flow._inverse(flow._params, xarray)
    assert xinv.shape == x.shape
    assert xinv_log_det.shape == (x.shape[0],)

    nsamples = 4
    assert flow.sample(nsamples).shape == (nsamples, x.shape[1])
    assert flow.log_prob(x).shape == (x.shape[0],)

    grid = np.arange(0, 2.1, 0.12)
    pdfs = flow.posterior(x, column="x", grid=grid)
    assert pdfs.shape == (x.shape[0], grid.size)
    pdfs = flow.posterior(x.iloc[:, 1:], column="x", grid=grid)
    assert pdfs.shape == (x.shape[0], grid.size)

    assert len(flow.train(x, epochs=11, verbose=True)) == 12


def test_flow_bijection():
    columns = ("x", "y")
    flow = Flow(columns, Reverse())

    x = np.array([[1, 2], [3, 4]])
    xrev = np.array([[2, 1], [4, 3]])

    assert np.allclose(flow._forward(flow._params, x)[0], xrev)
    assert np.allclose(
        flow._inverse(flow._params, flow._forward(flow._params, x)[0])[0], x
    )
    assert np.allclose(
        flow._forward(flow._params, x)[1], flow._inverse(flow._params, x)[1]
    )


def test_load_flow(tmp_path):
    columns = ("x", "y")
    flow = Flow(columns, Reverse(), info=["random", 42])

    file = tmp_path / "test-flow"
    flow.save(str(file))

    file = tmp_path / "test-flow.dill"
    flow = Flow(file=str(file))

    x = np.array([[1, 2], [3, 4]])
    xrev = np.array([[2, 1], [4, 3]])

    assert np.allclose(flow._forward(flow._params, x)[0], xrev)
    assert np.allclose(
        flow._inverse(flow._params, flow._forward(flow._params, x)[0])[0], x
    )
    assert np.allclose(
        flow._forward(flow._params, x)[1], flow._inverse(flow._params, x)[1]
    )
    assert flow.info == ["random", 42]


def test_control_sample_randomness():
    columns = ("x", "y")
    flow = Flow(columns, Reverse())

    assert np.all(~np.isclose(flow.sample(2), flow.sample(2)))
    assert np.allclose(flow.sample(2, seed=0), flow.sample(2, seed=0))
