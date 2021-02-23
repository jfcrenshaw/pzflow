import numpy as onp

import jax.numpy as np
from jax import random
from jax.scipy.special import gammaln

from pzflow.bijectors import Pytree


def mahalanobis_and_logdet(x, mean, cov):
    """Calculate mahalanobis distance and log_det of cov.

    Uses scipy method, explained here:
    http://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/
    """
    vals, vecs = np.linalg.eigh(cov)
    U = vecs * np.sqrt(1 / vals[..., None])
    dev = x - mean
    maha = np.square(np.dot(dev, U)).reshape(x.shape[0], -1).sum(axis=-1)
    log_det = np.log(vals).sum(axis=-1)
    return maha, log_det


class Normal:
    """A multivariate Gaussian distribution with mean zero and unit variance."""

    def __init__(self, input_dim: int):
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the distribution.
        """
        self.input_dim = input_dim
        self.type = "Normal"
        self._params = ()

    def log_prob(
        self, params: Pytree, inputs: np.ndarray, cov: np.ndarray = None
    ) -> np.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        params : a Jax pytree
            Empty pytree -- this distribution doesn't have learnable parameters.
            This parameter is present to ensure a consistent interface.
        inputs : np.ndarray
            Input data for which log probability density is calculated.
        cov : np.ndarray, default=None
            Covariance matrix for the log probability calculation.

        Returns
        -------
        np.ndarray
            Device array of shape (inputs.shape[0],).

        Notes
        -----
        jax.scipy.stats.multivariate_normal.log_pdf doesn't seem to work with
        different covariances for each input. To get around this, I implemented
        the method from the original scipy code.
        See:
        https://github.com/scipy/scipy/blob/v1.6.0/scipy/stats/_multivariate.py
        or
        http://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/
        """
        mean = np.zeros(self.input_dim)
        if cov is None:
            cov = np.identity(self.input_dim)
        maha, log_det = mahalanobis_and_logdet(inputs, mean, cov)

        log_prob = -0.5 * (inputs.shape[-1] * np.log(2 * np.pi) + log_det + maha)
        return log_prob

    def sample(self, params: Pytree, nsamples: int, seed: int = None) -> np.ndarray:
        """Returns samples from the distribution.

        Parameters
        ----------
        params : a Jax pytree
            Empty pytree -- this distribution doesn't have learnable parameters.
            This parameter is present to ensure a consistent interface.
        nsamples : int
            The number of samples to be returned.
        seed : int, optional
            Sets the random seed for the samples.

        Returns
        -------
        np.ndarray
            Device array of shape (nsamples, self.input_dim).
        """
        seed = onp.random.randint(1e18) if seed is None else seed
        return random.multivariate_normal(
            key=random.PRNGKey(seed),
            mean=np.zeros(self.input_dim),
            cov=np.identity(self.input_dim),
            shape=(nsamples,),
        )


class Tdist:
    """A multivariate T distribution with mean zero and unit scale matrix."""

    def __init__(self, input_dim: int):
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the distribution.
        """
        self.input_dim = input_dim
        self.type = "Tdist"
        self._params = np.log(30.0)

    def log_prob(self, params: Pytree, inputs: np.ndarray) -> np.ndarray:
        """Calculates log probability density of inputs.

        Uses method explained here:
        http://gregorygundersen.com/blog/2020/01/20/multivariate-t/

        Parameters
        ----------
        params : float
            The degrees of freedom (nu) of the t-distribution.
        inputs : np.ndarray
            Input data for which log probability density is calculated.

        Returns
        -------
        np.ndarray
            Device array of shape (inputs.shape[0],).
        """
        mean = np.zeros(self.input_dim)
        cov = np.identity(self.input_dim)
        nu = np.exp(params)
        maha, log_det = mahalanobis_and_logdet(inputs, mean, cov)
        t = 0.5 * (nu + self.input_dim)
        A = gammaln(t)
        B = gammaln(0.5 * nu)
        C = self.input_dim / 2.0 * np.log(nu * np.pi)
        D = 0.5 * log_det
        E = -t * np.log(1 + (1.0 / nu) * maha)

        return A - B - C - D + E

    def sample(self, params: Pytree, nsamples: int, seed: int = None) -> np.ndarray:
        """Returns samples from the distribution.

        Parameters
        ----------
        params : a Jax pytree
            Empty pytree -- this distribution doesn't have learnable parameters.
            This parameter is present to ensure a consistent interface.
        nsamples : int
            The number of samples to be returned.
        seed : int, optional
            Sets the random seed for the samples.

        Returns
        -------
        np.ndarray
            Device array of shape (nsamples, self.input_dim).
        """
        mean = np.zeros(self.input_dim)
        nu = np.exp(params)
        seed = onp.random.randint(1e18) if seed is None else seed
        rng = onp.random.default_rng(seed)
        x = np.array(rng.chisquare(nu, nsamples) / nu)
        z = random.multivariate_normal(
            key=random.PRNGKey(seed),
            mean=np.zeros(self.input_dim),
            cov=np.identity(self.input_dim),
            shape=(nsamples,),
        )
        samples = mean + z / np.sqrt(x)[:, None]
        return samples