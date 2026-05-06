"""Tests for pzflow.Flow."""

import pickle
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from jax import random

from pzflow import Flow
from pzflow.bijectors import Reverse, RollingSplineCoupling
from pzflow.distributions import (
    CentBeta,
    Normal,
    Tdist,
    Uniform,
)
from pzflow.examples import get_twomoons_data


@pytest.mark.parametrize(
    "data_columns,latent",
    [
        (None, None),
        (None, Normal(1)),
        (("x",), Normal(2)),
    ],
)
def test_bad_inputs(data_columns: Any, latent: Any) -> None:
    """Invalid constructor arguments raise ValueError."""
    with pytest.raises(ValueError):
        Flow(data_columns, latent=latent)


@pytest.mark.parametrize(
    "flow",
    [
        Flow(("redshift", "y"), Reverse(), latent=Normal(2)),
        Flow(("redshift", "y"), Reverse(), latent=Tdist(2)),
        Flow(("redshift", "y"), Reverse(), latent=Uniform(2, 10)),
        Flow(("redshift", "y"), Reverse(), latent=CentBeta(2, 10)),
    ],
)
def test_returns_correct_shape(flow: Any) -> None:
    """Forward, inverse, sample, log_prob, and posterior all return expected shapes."""
    xarray = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=("redshift", "y"))

    conditions = flow._get_conditions(x)

    xfwd, xfwd_log_det = flow._forward(
        flow._params, xarray, conditions=conditions
    )
    assert xfwd.shape == x.shape
    assert xfwd_log_det.shape == (x.shape[0],)

    xinv, xinv_log_det = flow._inverse(
        flow._params, xarray, conditions=conditions
    )
    assert xinv.shape == x.shape
    assert xinv_log_det.shape == (x.shape[0],)

    nsamples = 4
    assert flow.sample(nsamples).shape == (nsamples, x.shape[1])
    assert flow.log_prob(x).shape == (x.shape[0],)

    grid = jnp.arange(0, 2.1, 0.12)
    pdfs = flow.posterior(x, column="y", grid=grid)
    assert pdfs.shape == (x.shape[0], grid.size)
    pdfs = flow.posterior(x.iloc[:, 1:], column="redshift", grid=grid)
    assert pdfs.shape == (x.shape[0], grid.size)
    pdfs = flow.posterior(
        x.iloc[:, 1:], column="redshift", grid=grid, batch_size=2
    )
    assert pdfs.shape == (x.shape[0], grid.size)

    assert len(flow.train(x, epochs=11, verbose=True)) == 12
    assert (
        len(flow.train(x, epochs=11, verbose=True, convolve_errs=True)) == 12
    )


@pytest.mark.parametrize(
    "flag",
    [
        99,
        np.nan,
    ],
)
def test_posterior_with_marginalization(flag: Any) -> None:
    """Posterior with marg_rules returns the correct shape for flagged columns."""
    flow = Flow(("a", "b", "c", "d"), Reverse())

    x = pd.DataFrame(
        np.arange(16).reshape(-1, 4), columns=("a", "b", "c", "d")
    )
    grid = np.arange(0, 2.1, 0.12)

    marg_rules = {
        "flag": flag,
        "b": lambda row: np.linspace(0, 1, 2),
        "c": lambda row: np.linspace(1, 2, 3),
    }

    x["b"] = flag * np.ones(x.shape[0])
    pdfs = flow.posterior(x, column="a", grid=grid, marg_rules=marg_rules)
    assert pdfs.shape == (x.shape[0], grid.size)

    x["c"] = flag * np.ones(x.shape[0])
    pdfs = flow.posterior(x, column="a", grid=grid, marg_rules=marg_rules)
    assert pdfs.shape == (x.shape[0], grid.size)


@pytest.mark.parametrize(
    "flow,x,x_with_err",
    [
        (
            Flow(
                ("redshift", "y"), RollingSplineCoupling(2), latent=Normal(2)
            ),
            pd.DataFrame(
                np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                columns=("redshift", "y"),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [1.0, 2.0, 0.1, 0.2],
                        [3.0, 4.0, 0.2, 0.3],
                        [5.0, 6.0, 0.1, 0.2],
                    ]
                ),
                columns=("redshift", "y", "redshift_err", "y_err"),
            ),
        ),
        (
            Flow(
                ("redshift", "y"),
                RollingSplineCoupling(2, n_conditions=2),
                latent=Normal(2),
                conditional_columns=("a", "b"),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [1.0, 2.0, 10, 20],
                        [3.0, 4.0, 30, 40],
                        [5.0, 6.0, 50, 60],
                    ]
                ),
                columns=("redshift", "y", "a", "b"),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [1.0, 2.0, 10, 20, 0.1, 0.2, 1, 2],
                        [3.0, 4.0, 30, 40, 0.2, 0.3, 3, 4],
                        [5.0, 6.0, 50, 60, 0.1, 0.2, 5, 6],
                    ]
                ),
                columns=(
                    "redshift",
                    "y",
                    "a",
                    "b",
                    "redshift_err",
                    "y_err",
                    "a_err",
                    "b_err",
                ),
            ),
        ),
        (
            Flow(
                ("redshift", "y"),
                RollingSplineCoupling(2, n_conditions=1),
                latent=Normal(2),
                conditional_columns=("a",),
            ),
            pd.DataFrame(
                np.array([[1.0, 2.0, 10], [3.0, 4.0, 30], [5.0, 6.0, 50]]),
                columns=("redshift", "y", "a"),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [1.0, 2.0, 10, 0.1, 0.2, 1],
                        [3.0, 4.0, 30, 0.2, 0.3, 3],
                        [5.0, 6.0, 50, 0.1, 0.2, 5],
                    ]
                ),
                columns=(
                    "redshift",
                    "y",
                    "a",
                    "redshift_err",
                    "y_err",
                    "a_err",
                ),
            ),
        ),
        (
            Flow(
                ("y",),
                RollingSplineCoupling(1, n_conditions=2),
                latent=Normal(1),
                conditional_columns=("a", "b"),
            ),
            pd.DataFrame(
                np.array([[1.0, 10, 20], [3.0, 30, 40], [5.0, 50, 60]]),
                columns=("y", "a", "b"),
            ),
            pd.DataFrame(
                np.array(
                    [
                        [1.0, 10, 20, 0.1, 1, 2],
                        [3.0, 30, 40, 0.2, 3, 4],
                        [5.0, 50, 60, 0.1, 5, 6],
                    ]
                ),
                columns=(
                    "y",
                    "a",
                    "b",
                    "y_err",
                    "a_err",
                    "b_err",
                ),
            ),
        ),
    ],
)
def test_error_convolution(flow: Any, x: Any, x_with_err: Any) -> None:
    """Error convolution returns correct shapes and is reproducible with a fixed seed."""
    assert flow.log_prob(x, err_samples=10).shape == (x.shape[0],)
    assert jnp.allclose(
        flow.log_prob(x, err_samples=10, seed=0),
        flow.log_prob(x),
    )
    assert ~jnp.allclose(
        flow.log_prob(x_with_err, err_samples=10, seed=0),
        flow.log_prob(x_with_err),
    )
    assert jnp.allclose(
        flow.log_prob(x_with_err, err_samples=10, seed=0),
        flow.log_prob(x_with_err, err_samples=10, seed=0),
    )
    assert ~jnp.allclose(
        flow.log_prob(x_with_err, err_samples=10, seed=0),
        flow.log_prob(x_with_err, err_samples=10, seed=1),
    )
    assert ~jnp.allclose(
        flow.log_prob(x_with_err, err_samples=10),
        flow.log_prob(x_with_err, err_samples=10),
    )

    grid = jnp.arange(0, 2.1, 0.12)
    pdfs = flow.posterior(x, column="y", grid=grid, err_samples=10)
    assert pdfs.shape == (x.shape[0], grid.size)
    assert jnp.allclose(
        flow.posterior(x, column="y", grid=grid, err_samples=10, seed=0),
        flow.posterior(x, column="y", grid=grid),
        rtol=1e-4,
    )
    assert jnp.allclose(
        flow.posterior(
            x_with_err, column="y", grid=grid, err_samples=10, seed=0
        ),
        flow.posterior(
            x_with_err, column="y", grid=grid, err_samples=10, seed=0
        ),
    )


def test_posterior_batch() -> None:
    """Batched posterior computation matches unbatched results."""
    columns = ("redshift", "y")
    flow = Flow(columns, Reverse())

    xarray = np.array([[1, 2], [3, 4], [5, 6]])
    x = pd.DataFrame(xarray, columns=columns)

    grid = jnp.arange(0, 2.1, 0.12)
    pdfs = flow.posterior(x.iloc[:, 1:], column="redshift", grid=grid)
    pdfs_batched = flow.posterior(
        x.iloc[:, 1:], column="redshift", grid=grid, batch_size=2
    )
    assert jnp.allclose(pdfs, pdfs_batched)


def test_flow_bijection() -> None:
    """Forward composed with inverse recovers the input."""
    columns = ("x", "y")
    flow = Flow(columns, Reverse())

    x = jnp.array([[1, 2], [3, 4]])
    xrev = jnp.array([[2, 1], [4, 3]])

    assert jnp.allclose(flow._forward(flow._params, x)[0], xrev)
    assert jnp.allclose(
        flow._inverse(flow._params, flow._forward(flow._params, x)[0])[0], x
    )
    assert jnp.allclose(
        flow._forward(flow._params, x)[1], flow._inverse(flow._params, x)[1]
    )


def test_load_flow(tmp_path: Any) -> None:
    """Flow can be saved and loaded via file, and wrong class raises TypeError."""
    columns = ("x", "y")
    flow = Flow(columns, Reverse(), info=["random", 42])

    file = tmp_path / "test-flow.pzflow.pkl"
    flow.save(str(file))

    flow = Flow.from_file(str(file))

    x = jnp.array([[1, 2], [3, 4]])
    xrev = jnp.array([[2, 1], [4, 3]])

    assert jnp.allclose(flow._forward(flow._params, x)[0], xrev)
    assert jnp.allclose(
        flow._inverse(flow._params, flow._forward(flow._params, x)[0])[0], x
    )
    assert jnp.allclose(
        flow._forward(flow._params, x)[1], flow._inverse(flow._params, x)[1]
    )
    assert flow.info == ["random", 42]

    with open(str(file), "rb") as handle:
        save_dict = pickle.load(handle).__getstate__()
    save_dict["class"] = "FlowEnsemble"
    with open(str(file), "wb") as handle:
        pickle.dump(save_dict, handle)
    with pytest.raises(TypeError):
        Flow.from_file(str(file))


def test_pickle_flow(tmp_path: Any) -> None:
    """Flow can be pickled and unpickled with identical bijection behaviour."""
    columns = ("x", "y")
    flow = Flow(columns, Reverse(), info=["random", 42])

    file = tmp_path / "test-flow.pzflow.pkl"
    with open(str(file), "wb") as f:
        pickle.dump(flow, f)

    with open(str(file), "rb") as f:
        flow = pickle.load(f)

    x = jnp.array([[1, 2], [3, 4]])
    xrev = jnp.array([[2, 1], [4, 3]])

    assert jnp.allclose(flow._forward(flow._params, x)[0], xrev)
    assert jnp.allclose(
        flow._inverse(flow._params, flow._forward(flow._params, x)[0])[0], x
    )
    assert jnp.allclose(
        flow._forward(flow._params, x)[1], flow._inverse(flow._params, x)[1]
    )
    assert flow.info == ["random", 42]


def test_control_sample_randomness() -> None:
    """Samples differ between calls but are reproducible with a fixed seed."""
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
def test_train_bad_inputs(epochs: Any, loss_fn: Any) -> None:
    """Invalid epoch values raise ValueError during training."""
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


def test_conditional_sample() -> None:
    """Conditional sampling returns the correct number of samples and columns."""
    flow = Flow(("x", "y"), Reverse(), conditional_columns=("a", "b"))
    x = np.arange(12).reshape(-1, 4)
    x = pd.DataFrame(x, columns=("x", "y", "a", "b"))

    conditions = flow._get_conditions(x)
    assert conditions.shape == (x.shape[0], 2)

    with pytest.raises(ValueError):
        flow.sample(4)

    samples = flow.sample(4, conditions=x)
    assert samples.shape == (4 * x.shape[0], 4)

    samples = flow.sample(4, conditions=x, save_conditions=False)
    assert samples.shape == (4 * x.shape[0], 2)


def test_train_no_errs_same() -> None:
    """Training with and without error convolution produces identical losses when no errors present."""
    columns = ("redshift", "y")
    flow = Flow(columns, Reverse())

    xarray = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=columns)

    losses1 = flow.train(x, convolve_errs=True)
    losses2 = flow.train(x, convolve_errs=False)
    assert jnp.allclose(jnp.array(losses1), jnp.array(losses2))


def test_get_err_samples() -> None:
    """_get_err_samples draws correctly-shaped samples and respects the skip parameter."""
    rng = random.PRNGKey(0)

    # check Gaussian data samples
    columns = ("x", "y")
    flow = Flow(columns, Reverse())
    xarray = np.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.3, 0.4]])
    x = pd.DataFrame(xarray, columns=("x", "y", "x_err", "y_err"))
    samples = flow._get_err_samples(rng, x, 10)
    assert samples.shape == (20, 2)

    # test skip
    xarray = np.array([[1.0, 2.0, 0, 0]])
    x = pd.DataFrame(xarray, columns=("x", "y", "x_err", "y_err"))
    samples = flow._get_err_samples(rng, x, 10, skip="y")
    assert jnp.allclose(samples, jnp.ones((10, 1)))
    samples = flow._get_err_samples(rng, x, 10, skip="x")
    assert jnp.allclose(samples, 2 * jnp.ones((10, 1)))

    # check Gaussian conditional samples
    flow = Flow(("x"), Reverse(), conditional_columns=("y"))
    samples = flow._get_err_samples(rng, x, 10, kind="conditions")
    assert jnp.allclose(samples, 2 * jnp.ones((10, 1)))

    # check incorrect kind value
    with pytest.raises(ValueError):
        flow._get_err_samples(rng, x, 10, kind="wrong")

    # check constant shift data samples
    columns = ("x", "y")
    shift_err_model = lambda key, X, Xerr, nsamples: jnp.repeat(
        X + Xerr, nsamples, axis=0
    ).reshape(X.shape[0], nsamples, X.shape[1])
    flow = Flow(columns, Reverse(), data_error_model=shift_err_model)
    xarray = np.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.3, 0.4]])
    x = pd.DataFrame(xarray, columns=("x", "y", "x_err", "y_err"))
    samples = flow._get_err_samples(rng, x, 10)
    assert samples.shape == (20, 2)
    assert jnp.allclose(
        samples,
        shift_err_model(None, xarray[:, :2], xarray[:, 2:], 10).reshape(20, 2),
    )

    # check constant shift conditional samples
    flow = Flow(
        ("x"),
        Reverse(),
        conditional_columns=("y"),
        condition_error_model=shift_err_model,
    )
    samples = flow._get_err_samples(rng, x, 10, kind="conditions")
    assert jnp.allclose(
        samples, jnp.repeat(jnp.array([[2.2], [4.4]]), 10, axis=0)
    )


def test_train_w_conditions() -> None:
    """Training a conditional flow updates condition_means and condition_stds correctly."""
    xarray = np.array(
        [[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.3, 0.4], [5.0, 6.0, 0.5, 0.6]]
    )
    x = pd.DataFrame(xarray, columns=("redshift", "y", "a", "b"))

    flow = Flow(
        ("redshift", "y"),
        Reverse(),
        latent=Normal(2),
        conditional_columns=("a", "b"),
    )
    assert len(flow.train(x, epochs=11)) == 12

    assert jnp.allclose(flow._condition_means, xarray[:, 2:].mean(axis=0))
    assert jnp.allclose(flow._condition_stds, xarray[:, 2:].std(axis=0))


def test_patience() -> None:
    """Early stopping halts training after patience epochs without improvement."""
    columns = ("redshift", "y")
    flow = Flow(columns, Reverse())

    xarray = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=columns)

    losses = flow.train(x, patience=2)
    assert len(losses) == 4


def test_latent_with_wrong_dimension() -> None:
    """Mismatched latent dimension raises ValueError at construction time."""
    cols = ["x", "y"]
    latent = Uniform(3)

    with pytest.raises(ValueError):
        Flow(data_columns=cols, latent=latent, bijector=Reverse())


def test_bijector_not_set() -> None:
    """Calling sample or posterior before setting a bijector raises ValueError."""
    flow = Flow(["x", "y"])

    with pytest.raises(ValueError):
        flow.sample(1)

    with pytest.raises(ValueError):
        x = np.linspace(0, 1, 12)
        df = pd.DataFrame(x.reshape(-1, 2), columns=("x", "y"))
        flow.posterior(df, column="x", grid=x)


def test_default_bijector() -> None:
    """Training without a pre-set bijector uses the default and produces finite losses."""
    flow = Flow(["x", "y"])

    losses = flow.train(get_twomoons_data())
    assert all(~np.isnan(losses))


def test_validation_train() -> None:
    """Training with a validation set returns both train and val loss lists."""
    data = get_twomoons_data()[:10]
    train_set = data[:8]
    val_set = data[8:]

    flow = Flow(train_set.columns, Reverse())
    losses = flow.train(
        train_set,
        val_set,
        verbose=True,
        epochs=3,
        best_params=False,
    )
    assert len(losses[0]) == 4
    assert len(losses[1]) == 4


def test_nan_train_stop() -> None:
    """Training on all-NaN data stops after the first epoch."""
    data = jnp.nan * jnp.ones((4, 2))
    data = pd.DataFrame(data, columns=["x", "y"])

    flow = Flow(data.columns, Reverse())
    losses = flow.train(data)
    assert len(losses) == 2


def test_train_weights() -> None:
    """Training with sample weights produces correct-length loss lists."""
    data = get_twomoons_data()[:10]
    train_set = data[:8]
    val_set = data[8:]
    train_weight = np.linspace(1, 2, len(train_set))
    val_weight = np.linspace(1, 2, len(val_set))

    flow = Flow(train_set.columns, Reverse())
    losses = flow.train(
        train_set,
        val_set,
        verbose=True,
        epochs=3,
        best_params=False,
        train_weight=train_weight,
        val_weight=val_weight,
    )
    assert len(losses[0]) == 4
    assert len(losses[1]) == 4


def test_no_initial_loss() -> None:
    """Training with initial_loss=False omits the initial loss from the returned list."""
    data = get_twomoons_data()[:10]
    train_set = data[:8]
    val_set = data[8:]

    flow = Flow(train_set.columns, Reverse())
    losses = flow.train(
        train_set,
        val_set,
        verbose=True,
        epochs=3,
        best_params=False,
        initial_loss=False,
    )
    assert len(losses[0]) == 3
    assert len(losses[1]) == 3
