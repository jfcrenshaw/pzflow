import dill as pickle
import jax.numpy as np
import numpy as onp
import pandas as pd
import pytest
from jax import random
from pzflow import Flow
from pzflow.bijectors import Reverse, RollingSplineCoupling
from pzflow.distributions import *


@pytest.mark.parametrize(
    "data_columns,bijector,info,file,_dictionary",
    [
        (None, None, None, None, None),
        (("x", "y"), None, None, None, None),
        (None, Reverse(), None, None, None),
        (("x", "y"), None, None, "file", None),
        (None, Reverse(), None, "file", None),
        (None, None, "fake", "file", None),
        (("x", "y"), Reverse(), None, None, "dict"),
        (None, None, None, "file", "dict"),
    ],
)
def test_bad_inputs(data_columns, bijector, info, file, _dictionary):
    with pytest.raises(ValueError):
        Flow(
            data_columns,
            bijector=bijector,
            info=info,
            file=file,
            _dictionary=_dictionary,
        )


@pytest.mark.parametrize(
    "flow",
    [
        Flow(("redshift", "y"), Reverse(), latent=Normal(2)),
        Flow(("redshift", "y"), Reverse(), latent=Tdist(2)),
        Flow(("redshift", "y"), Reverse(), latent=Uniform((-3, 3), (-3, 3))),
        Flow(("redshift", "y"), Reverse(), latent=CentBeta(2)),
    ],
)
def test_returns_correct_shape(flow):
    xarray = onp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=("redshift", "y"))

    conditions = flow._get_conditions(x)

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
    assert len(flow.train(x, epochs=11, verbose=True, convolve_errs=True)) == 12


@pytest.mark.parametrize(
    "flag",
    [
        99,
        onp.nan,
    ],
)
def test_posterior_with_marginalization(flag):

    flow = Flow(("a", "b", "c", "d"), Reverse())

    # test posteriors with marginalization
    x = pd.DataFrame(onp.arange(16).reshape(-1, 4), columns=("a", "b", "c", "d"))
    grid = onp.arange(0, 2.1, 0.12)

    marg_rules = {
        "flag": flag,
        "b": lambda row: onp.linspace(0, 1, 2),
        "c": lambda row: onp.linspace(1, 2, 3),
    }

    x["b"] = flag * onp.ones(x.shape[0])
    pdfs = flow.posterior(x, column="a", grid=grid, marg_rules=marg_rules)
    assert pdfs.shape == (x.shape[0], grid.size)

    x["c"] = flag * onp.ones(x.shape[0])
    pdfs = flow.posterior(x, column="a", grid=grid, marg_rules=marg_rules)
    assert pdfs.shape == (x.shape[0], grid.size)


@pytest.mark.parametrize(
    "flow,x,x_with_err",
    [
        (
            Flow(("redshift", "y"), RollingSplineCoupling(2), latent=Normal(2)),
            pd.DataFrame(
                onp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
                columns=("redshift", "y"),
            ),
            pd.DataFrame(
                onp.array(
                    [[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.2, 0.3], [5.0, 6.0, 0.1, 0.2]]
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
                onp.array([[1.0, 2.0, 10, 20], [3.0, 4.0, 30, 40], [5.0, 6.0, 50, 60]]),
                columns=("redshift", "y", "a", "b"),
            ),
            pd.DataFrame(
                onp.array(
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
                onp.array([[1.0, 2.0, 10], [3.0, 4.0, 30], [5.0, 6.0, 50]]),
                columns=("redshift", "y", "a"),
            ),
            pd.DataFrame(
                onp.array(
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
                onp.array([[1.0, 10, 20], [3.0, 30, 40], [5.0, 50, 60]]),
                columns=("y", "a", "b"),
            ),
            pd.DataFrame(
                onp.array(
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
def test_error_convolution(flow, x, x_with_err):

    assert flow.log_prob(x, err_samples=10).shape == (x.shape[0],)
    assert np.allclose(
        flow.log_prob(x, err_samples=10, seed=0),
        flow.log_prob(x),
    )
    assert ~np.allclose(
        flow.log_prob(x_with_err, err_samples=10, seed=0),
        flow.log_prob(x_with_err),
    )
    assert np.allclose(
        flow.log_prob(x_with_err, err_samples=10, seed=0),
        flow.log_prob(x_with_err, err_samples=10, seed=0),
    )
    assert ~np.allclose(
        flow.log_prob(x_with_err, err_samples=10, seed=0),
        flow.log_prob(x_with_err, err_samples=10, seed=1),
    )
    assert ~np.allclose(
        flow.log_prob(x_with_err, err_samples=10),
        flow.log_prob(x_with_err, err_samples=10),
    )

    grid = np.arange(0, 2.1, 0.12)
    pdfs = flow.posterior(x, column="y", grid=grid, err_samples=10)
    assert pdfs.shape == (x.shape[0], grid.size)
    assert np.allclose(
        flow.posterior(x, column="y", grid=grid, err_samples=10, seed=0),
        flow.posterior(x, column="y", grid=grid),
        rtol=1e-4,
    )
    assert np.allclose(
        flow.posterior(x_with_err, column="y", grid=grid, err_samples=10, seed=0),
        flow.posterior(x_with_err, column="y", grid=grid, err_samples=10, seed=0),
    )


def test_posterior_batch():
    columns = ("redshift", "y")
    flow = Flow(columns, Reverse())

    xarray = onp.array([[1, 2], [3, 4], [5, 6]])
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

    file = tmp_path / "test-flow.pzflow.pkl"
    flow.save(str(file))

    file = tmp_path / "test-flow.pzflow.pkl"
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

    with open(str(file), "rb") as handle:
        save_dict = pickle.load(handle)
    save_dict["class"] = "FlowEnsemble"
    with open(str(file), "wb") as handle:
        pickle.dump(save_dict, handle, recurse=True)
    with pytest.raises(TypeError):
        Flow(file=str(file))


def test_control_sample_randomness():
    columns = ("x", "y")
    flow = Flow(columns, Reverse())

    assert onp.all(~onp.isclose(flow.sample(2), flow.sample(2)))
    assert onp.allclose(flow.sample(2, seed=0), flow.sample(2, seed=0))


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

    xarray = onp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=columns)

    with pytest.raises(ValueError):
        flow.train(
            x,
            epochs=epochs,
            loss_fn=loss_fn,
        )


def test_conditional_sample():

    flow = Flow(("x", "y"), Reverse(), conditional_columns=("a", "b"))
    x = onp.arange(12).reshape(-1, 4)
    x = pd.DataFrame(x, columns=("x", "y", "a", "b"))

    conditions = flow._get_conditions(x)
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

    xarray = onp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    x = pd.DataFrame(xarray, columns=columns)

    losses1 = flow.train(x, convolve_errs=True)
    losses2 = flow.train(x, convolve_errs=False)
    assert np.allclose(np.array(losses1), np.array(losses2))


def test_get_err_samples():

    rng = random.PRNGKey(0)

    # check Gaussian data samples
    columns = ("x", "y")
    flow = Flow(columns, Reverse())
    xarray = onp.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.3, 0.4]])
    x = pd.DataFrame(xarray, columns=("x", "y", "x_err", "y_err"))
    samples = flow._get_err_samples(rng, x, 10)
    assert samples.shape == (20, 2)

    # test skip
    xarray = onp.array([[1.0, 2.0, 0, 0]])
    x = pd.DataFrame(xarray, columns=("x", "y", "x_err", "y_err"))
    samples = flow._get_err_samples(rng, x, 10, skip="y")
    assert np.allclose(samples, np.ones((10, 1)))
    samples = flow._get_err_samples(rng, x, 10, skip="x")
    assert np.allclose(samples, 2 * np.ones((10, 1)))

    # check Gaussian conditional samples
    flow = Flow(("x"), Reverse(), conditional_columns=("y"))
    samples = flow._get_err_samples(rng, x, 10, type="conditions")
    assert np.allclose(samples, 2 * np.ones((10, 1)))

    # check incorrect type
    with pytest.raises(ValueError):
        flow._get_err_samples(rng, x, 10, type="wrong")

    # check constant shift data samples
    columns = ("x", "y")
    shift_err_model = lambda key, X, Xerr, nsamples: np.repeat(
        X + Xerr, nsamples, axis=0
    ).reshape(X.shape[0], nsamples, X.shape[1])
    flow = Flow(columns, Reverse(), data_error_model=shift_err_model)
    xarray = onp.array([[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.3, 0.4]])
    x = pd.DataFrame(xarray, columns=("x", "y", "x_err", "y_err"))
    samples = flow._get_err_samples(rng, x, 10)
    assert samples.shape == (20, 2)
    assert np.allclose(
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
    samples = flow._get_err_samples(rng, x, 10, type="conditions")
    assert np.allclose(samples, np.repeat(np.array([[2.2], [4.4]]), 10, axis=0))


def test_train_w_conditions():

    xarray = onp.array(
        [[1.0, 2.0, 0.1, 0.2], [3.0, 4.0, 0.3, 0.4], [5.0, 6.0, 0.5, 0.6]]
    )
    x = pd.DataFrame(xarray, columns=("redshift", "y", "a", "b"))

    flow = Flow(
        ("redshift", "y"), Reverse(), latent=Normal(2), conditional_columns=("a", "b")
    )
    assert len(flow.train(x, epochs=11)) == 12

    print("------->>>>>")
    print(flow._condition_stds, "\n\n")
    print(xarray[:, 2:].std(axis=0))
    assert np.allclose(flow._condition_means, xarray[:, 2:].mean(axis=0))
    assert np.allclose(flow._condition_stds, xarray[:, 2:].std(axis=0))
