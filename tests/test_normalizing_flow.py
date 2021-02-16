import pytest
import jax.numpy as np
import pandas as pd
from pzflow import Flow
from pzflow.bijectors import Chain, Reverse, Scale


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
    columns = ("redshift", "y")
    flow = Flow(columns, Reverse())

    xarray = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=columns)

    x_with_errs = flow._array_with_errs(x)
    assert x_with_errs.shape == (3, 4)
    x_with_errs = flow._array_with_errs(x, skip="redshift")
    assert x_with_errs.shape == (3, 3)

    xfwd, xfwd_log_det = flow._forward(flow._params, xarray)
    assert xfwd.shape == x.shape
    assert xfwd_log_det.shape == (x.shape[0],)

    xinv, xinv_log_det = flow._inverse(flow._params, xarray)
    assert xinv.shape == x.shape
    assert xinv_log_det.shape == (x.shape[0],)

    Jinv = flow._Jinv(flow._params, xarray)
    assert Jinv.shape == (3, 2, 2)

    nsamples = 4
    assert flow.sample(nsamples).shape == (nsamples, x.shape[1])
    assert flow.log_prob(x).shape == (x.shape[0],)
    assert flow.log_prob(x, convolve_err=True).shape == (x.shape[0],)

    grid = np.arange(0, 2.1, 0.12)
    pdfs = flow.posterior(x, column="y", grid=grid)
    assert pdfs.shape == (x.shape[0], grid.size)
    pdfs = flow.posterior(x.iloc[:, 1:], column="redshift", grid=grid)
    assert pdfs.shape == (x.shape[0], grid.size)
    pdfs = flow.posterior(x.iloc[:, 1:], column="redshift", grid=grid, batch_size=2)
    assert pdfs.shape == (x.shape[0], grid.size)
    pdfs = flow.posterior(x, column="y", grid=grid, convolve_err=True)
    assert pdfs.shape == (x.shape[0], grid.size)

    assert len(flow.train(x, epochs=11, verbose=True)) == 12
    assert (
        len(flow.train(x, epochs=11, convolve_err=True, burn_in_epochs=4, verbose=True))
        == 17
    )


def test_columns_with_errs():
    columns = ("redshift", "y")
    flow = Flow(columns, Reverse())

    xarray = np.array([[1, 2, 0.4, 0.2], [3, 4, 0.1, 0.9]])
    x = pd.DataFrame(xarray, columns=("redshift", "y", "y_err", "redshift_err"))
    x_with_errs = flow._array_with_errs(x)
    assert np.allclose(x_with_errs, np.array([[1, 2, 0.2, 0.4], [3, 4, 0.9, 0.1]]))

    xarray = np.array([[1, 2, 0.4], [3, 4, 0.1]])
    x = pd.DataFrame(xarray, columns=("redshift", "y", "y_err"))
    x_with_errs = flow._array_with_errs(x)
    assert np.allclose(x_with_errs, np.array([[1, 2, 0, 0.4], [3, 4, 0, 0.1]]))

    xarray = np.array([[1, 2, 0.4, 0.2], [3, 4, 0.1, 0.9]])
    x = pd.DataFrame(xarray, columns=("redshift", "y", "y_err", "redshift_err"))
    x_with_errs = flow._array_with_errs(x, skip="redshift")
    assert np.allclose(x_with_errs, np.array([[2, 0, 0.4], [4, 0, 0.1]]))


def test_Jinv():
    columns = ("redshift", "y")
    flow = Flow(columns, Chain(Reverse(), Scale(2)))
    xarray = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    Jinv = flow._Jinv(flow._params, xarray)
    assert np.allclose(
        Jinv,
        np.array([[[0, 0.5], [0.5, 0]], [[0, 0.5], [0.5, 0]], [[0, 0.5], [0.5, 0]]]),
    )


def test_posterior_batch():
    columns = ("redshift", "y")
    flow = Flow(columns, Reverse())

    xarray = np.array([[1, 2], [3, 4], [5, 6]])
    x = pd.DataFrame(xarray, columns=columns)

    grid = np.arange(0, 2.1, 0.12)
    pdfs = flow.posterior(x.iloc[:, 1:], column="redshift", grid=grid)
    pdfs_batched = flow.posterior(
        x.iloc[:, 1:], column="redshift", grid=grid, batch_size=2
    )
    assert np.allclose(pdfs, pdfs_batched)


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

    file = tmp_path / "test-flow.pkl"
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


@pytest.mark.parametrize(
    "epochs,burn_in_epochs,loss_fn,",
    [
        (-1, 10, None),
        (2.4, 10, None),
        ("a", 10, None),
        (10, -1, None),
        (10, 1.4, None),
        (10, "a", None),
        (10, 10, lambda x: x ** 2),
    ],
)
def test_train_bad_inputs(epochs, burn_in_epochs, loss_fn):
    columns = ("redshift", "y")
    flow = Flow(columns, Reverse())

    xarray = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=columns)

    with pytest.raises(ValueError):
        flow.train(
            x,
            epochs=epochs,
            burn_in_epochs=burn_in_epochs,
            loss_fn=lambda x: x ** 2,
            convolve_err=True,
        )
