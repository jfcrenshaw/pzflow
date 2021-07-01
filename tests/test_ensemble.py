from pzflow import Flow, FlowEnsemble
from pzflow.bijectors import NeuralSplineCoupling
import jax.numpy as np
import pandas as pd

flowEns = FlowEnsemble(("x", "y"), NeuralSplineCoupling(), N=2)
flow0 = Flow(("x", "y"), NeuralSplineCoupling(), seed=0)
flow1 = Flow(("x", "y"), NeuralSplineCoupling(), seed=1)

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
