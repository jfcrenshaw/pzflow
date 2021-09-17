import dill as pickle
import jax.numpy as np
import pandas as pd
import pytest
from jax import random
from pzflow import Flow, FlowEnsemble
from pzflow.bijectors import Reverse, RollingSplineCoupling

flowEns = FlowEnsemble(("x", "y"), RollingSplineCoupling(nlayers=2), N=2)
flow0 = Flow(("x", "y"), RollingSplineCoupling(nlayers=2), seed=0)
flow1 = Flow(("x", "y"), RollingSplineCoupling(nlayers=2), seed=1)

xarray = np.arange(6).reshape(3, 2) / 10
x = pd.DataFrame(xarray, columns=("x", "y"))


def test_log_prob():

    lpEns = flowEns.log_prob(x, returnEnsemble=True)
    assert lpEns.shape == (3, 2)

    lp0 = flow0.log_prob(x)
    lp1 = flow1.log_prob(x)
    assert np.allclose(lpEns[:, 0], lp0)
    assert np.allclose(lpEns[:, 1], lp1)

    lpEnsMean = flowEns.log_prob(x)
    assert lpEnsMean.shape == lp0.shape

    manualMean = np.log(np.mean(np.array([np.exp(lp0), np.exp(lp1)]), axis=0))
    assert np.allclose(lpEnsMean, manualMean)


def test_posterior():

    grid = np.linspace(-1, 1, 5)

    pEns = flowEns.posterior(x, "x", grid, returnEnsemble=True)
    assert pEns.shape == (3, 2, grid.size)

    p0 = flow0.posterior(x, "x", grid)
    p1 = flow1.posterior(x, "x", grid)
    assert np.allclose(pEns[:, 0, :], p0)
    assert np.allclose(pEns[:, 1, :], p1)

    pEnsMean = flowEns.posterior(x, "x", grid)
    assert pEnsMean.shape == p0.shape

    p0 = flow0.posterior(x, "x", grid, normalize=False)
    p1 = flow1.posterior(x, "x", grid, normalize=False)
    manualMean = (p0 + p1) / 2
    manualMean = manualMean / np.trapz(y=manualMean, x=grid).reshape(-1, 1)
    assert np.allclose(pEnsMean, manualMean)


def test_sample():

    # first test everything with returnEnsemble=False
    sEns = flowEns.sample(10, seed=0).values
    assert sEns.shape == (10, 2)

    s0 = flow0.sample(5, seed=0)
    s1 = flow1.sample(5, seed=0)
    sManual = np.vstack([s0.values, s1.values])
    assert np.allclose(sEns[sEns[:, 0].argsort()], sManual[sManual[:, 0].argsort()])

    # now test everything with returnEnsemble=True
    sEns = flowEns.sample(10, seed=0, returnEnsemble=True).values
    assert sEns.shape == (20, 2)

    s0 = flow0.sample(10, seed=0)
    s1 = flow1.sample(10, seed=0)
    sManual = np.vstack([s0.values, s1.values])
    assert np.allclose(sEns, sManual)


def test_conditional_sample():

    cEns = FlowEnsemble(
        ("x", "y"),
        RollingSplineCoupling(nlayers=2, n_conditions=2),
        conditional_columns=("a", "b"),
        N=2,
    )

    # test with nsamples = 1, fewer samples than flows
    conditions = pd.DataFrame(np.arange(2).reshape(-1, 2), columns=("a", "b"))
    samples = cEns.sample(nsamples=1, conditions=conditions, save_conditions=False)
    assert samples.shape == (1, 2)

    # test with nsamples = 1, more samples than flows
    conditions = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=("a", "b"))
    samples = cEns.sample(nsamples=1, conditions=conditions, save_conditions=False)
    assert samples.shape == (5, 2)

    # test with nsamples = 2, more samples than flows
    conditions = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=("a", "b"))
    samples = cEns.sample(nsamples=2, conditions=conditions, save_conditions=False)
    assert samples.shape == (10, 2)

    # test with returnEnsemble=True
    conditions = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=("a", "b"))
    samples = cEns.sample(
        nsamples=1, conditions=conditions, save_conditions=False, returnEnsemble=True
    )
    assert samples.shape == (10, 2)


def test_train():

    data = random.normal(random.PRNGKey(0), shape=(100, 2))
    data = pd.DataFrame(data, columns=("x", "y"))

    loss_dict = flowEns.train(data, epochs=4, batch_size=50, verbose=True)
    losses0 = flow0.train(data, epochs=4, batch_size=50)
    losses1 = flow1.train(data, epochs=4, batch_size=50)
    assert np.allclose(loss_dict["Flow 0"], losses0)
    assert np.allclose(loss_dict["Flow 1"], losses1)


def test_load_ensemble(tmp_path):

    flowEns = FlowEnsemble(("x", "y"), RollingSplineCoupling(nlayers=2), N=2)

    preSave = flowEns.sample(10, seed=0)

    file = tmp_path / "test-ensemble.pzflow.pkl"
    flowEns.save(str(file))

    file = tmp_path / "test-ensemble.pzflow.pkl"
    flowEns = FlowEnsemble(file=str(file))

    postSave = flowEns.sample(10, seed=0)

    assert np.allclose(preSave.values, postSave.values)

    with open(str(file), "rb") as handle:
        save_dict = pickle.load(handle)
    save_dict["class"] = "Flow"
    with open(str(file), "wb") as handle:
        pickle.dump(save_dict, handle, recurse=True)
    with pytest.raises(TypeError):
        FlowEnsemble(file=str(file))


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
        FlowEnsemble(data_columns, bijector=bijector, info=info, file=file)
