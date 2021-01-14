import jax.numpy as np
import numpy as onp
from jax import random
from jax.scipy.stats import multivariate_normal


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

    def log_prob(self, inputs: np.ndarray) -> np.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Input data for which log probability density is calculated.

        Returns
        -------
        np.ndarray
            Device array of shape (inputs.shape[0],).
        """
        return multivariate_normal.logpdf(
            x=inputs,
            mean=np.zeros(self.input_dim),
            cov=np.identity(self.input_dim),
        )

    def sample(self, nsamples: int, seed: int = None) -> np.ndarray:
        """Returns samples from the distribution.

        Parameters
        ----------
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
