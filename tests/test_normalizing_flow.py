import pytest
import jax.numpy as np
from jax import random, vmap
import pandas as pd
from pzflow import Flow
from pzflow.bijectors import Chain, Reverse, Scale
from pzflow.distributions import *


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
        Flow(data_columns, bijector=bijector, info=info, file=file)


@pytest.mark.parametrize(
    "flow",
    [
        Flow(("redshift", "y"), Reverse(), latent=Normal(2)),
        Flow(("redshift", "y"), Reverse(), latent=Tdist(2)),
        Flow(("redshift", "y"), Reverse(), latent=Uniform((-3, 3), (-3, 3))),
    ],
)
def test_returns_correct_shape(flow):
    xarray = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=("redshift", "y"))

    conditions = flow._get_conditions(x, xarray.shape[0])

    xfwd, xfwd_log_det = flow._forward(flow._params, xarray, conditions=conditions)
    assert xfwd.shape == x.shape
    assert xfwd_log_det.shape == (x.shape[0],)

    xinv, xinv_log_det = flow._inverse(flow._params, xarray, conditions=conditions)
    assert xinv.shape == x.shape
    assert xinv_log_det.shape == (x.shape[0],)

    nsamples = 4
    assert flow.sample(nsamples).shape == (nsamples, x.shape[1])
    assert flow.log_prob(x).shape == (x.shape[0],)

    grid = np.arange(0, 2.1, 0.12)
    pdfs = flow.posterior(x, column="y", grid=grid)
    assert pdfs.shape == (x.shape[0], grid.size)
    pdfs = flow.posterior(x.iloc[:, 1:], column="redshift", grid=grid)
    assert pdfs.shape == (x.shape[0], grid.size)
    pdfs = flow.posterior(x.iloc[:, 1:], column="redshift", grid=grid, batch_size=2)
    assert pdfs.shape == (x.shape[0], grid.size)

    assert len(flow.train(x, epochs=11, verbose=True)) == 12
    assert len(flow.train(x, epochs=11, verbose=True, sample_errs=True)) == 12


def test_error_convolution():

    flow = Flow(("redshift", "y"), Reverse(), latent=Normal(2))

    xarray = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=("redshift", "y"))

    xarray_with_err = np.array(
        [[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.2, 0.3], [5.0, 6.0, 0.1, 0.2]]
    )
    x_with_err = pd.DataFrame(
        xarray_with_err, columns=("redshift", "y", "redshift_err", "y_err")
    )

    assert flow.log_prob(x, nsamples=10).shape == (x.shape[0],)
    assert np.allclose(
        flow.log_prob(x, nsamples=10, seed=0),
        flow.log_prob(x),
    )
    assert ~np.allclose(
        flow.log_prob(x_with_err, nsamples=10, seed=0),
        flow.log_prob(x_with_err),
    )
    assert np.allclose(
        flow.log_prob(x_with_err, nsamples=10, seed=0),
        flow.log_prob(x_with_err, nsamples=10, seed=0),
    )
    assert ~np.allclose(
        flow.log_prob(x_with_err, nsamples=10, seed=0),
        flow.log_prob(x_with_err, nsamples=10, seed=1),
    )
    assert ~np.allclose(
        flow.log_prob(x_with_err, nsamples=10),
        flow.log_prob(x_with_err, nsamples=10),
    )

    grid = np.arange(0, 2.1, 0.12)
    pdfs = flow.posterior(x, column="y", grid=grid, nsamples=10)
    assert pdfs.shape == (x.shape[0], grid.size)
    assert np.allclose(
        flow.posterior(x, column="y", grid=grid, nsamples=10, seed=0),
        flow.posterior(x, column="y", grid=grid),
    )
    assert np.allclose(
        flow.posterior(x_with_err, column="y", grid=grid, nsamples=10, seed=0),
        flow.posterior(x_with_err, column="y", grid=grid, nsamples=10, seed=0),
    )

    # now I will compare values against manual calculations
    rng = random.PRNGKey(0)
    xsample = random.multivariate_normal(
        rng,
        xarray_with_err[:, :2],
        vmap(np.diag)(xarray_with_err[:, 2:] ** 2),
        shape=(10, 3),
    ).reshape(-1, 2, order="F")
    xsample = pd.DataFrame(xsample, columns=("redshift", "y"))
    # check log_prob
    manual_conv = np.log(np.exp(flow.log_prob(xsample).reshape(3, -1)).mean(axis=1))
    auto_conv = flow.log_prob(x_with_err, nsamples=10, seed=0)
    assert np.allclose(auto_conv, manual_conv)

    # I didn't write a test checking posterior values. Should probably
    # do that at some point...


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
    "epochs,loss_fn,",
    [
        (-1, None),
        (2.4, None),
        ("a", None),
    ],
)
def test_train_bad_inputs(epochs, loss_fn):
    columns = ("redshift", "y")
    flow = Flow(columns, Reverse())

    xarray = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=columns)

    with pytest.raises(ValueError):
        flow.train(
            x,
            epochs=epochs,
            loss_fn=loss_fn,
        )


def test_conditional_sample():

    flow = Flow(("x", "y"), Reverse(), conditional_columns=("a", "b"))
    x = np.arange(12).reshape(-1, 4)
    x = pd.DataFrame(x, columns=("x", "y", "a", "b"))

    conditions = flow._get_conditions(x, x.shape[0])
    assert conditions.shape == (x.shape[0], 2)

    with pytest.raises(ValueError):
        flow.sample(4)

    samples = flow.sample(4, conditions=x)
    assert samples.shape == (4 * x.shape[0], 4)

    samples = flow.sample(4, conditions=x, save_conditions=False)
    assert samples.shape == (4 * x.shape[0], 2)


def test_train_no_errs_same():
    columns = ("redshift", "y")
    flow = Flow(columns, Reverse())

    xarray = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=columns)

    losses1 = flow.train(x, sample_errs=True)
    losses2 = flow.train(x, sample_errs=False)
    assert np.allclose(losses1, losses2)