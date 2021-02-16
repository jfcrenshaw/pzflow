from typing import Callable, Tuple

import jax.numpy as np
import numpy as onp
from jax import random
from jax.experimental.stax import Dense, Relu, serial, LeakyRelu

from pzflow import bijectors


def build_bijector_from_info(info):
    """Build a Bijector from a Bijector_Info object"""

    # recurse through chains
    if info[0] == "Chain":
        return bijectors.Chain(*(build_bijector_from_info(i) for i in info[1]))
    # build individual bijector from name and parameters
    else:
        return getattr(bijectors, info[0])(*info[1])


def DenseReluNetwork(
    out_dim: int, hidden_layers: int, hidden_dim: int
) -> Tuple[Callable, Callable]:
    """Create a dense neural network with Relu after hidden layers.

    Parameters
    ----------
    out_dim : int
        The output dimension.
    hidden_layers : int
        The number of hidden layers
    hidden_dim : int
        The dimension of the hidden layers

    Returns
    -------
    init_fun : function
        The function that initializes the network. Note that this is the
        init_function defined in the Jax stax module, which is different
        from the functions of my InitFunction class.
    forward_fun : function
        The function that passes the inputs through the neural network.
    """
    init_fun, forward_fun = serial(
        *(Dense(hidden_dim), LeakyRelu) * hidden_layers,
        Dense(out_dim),
    )
    return init_fun, forward_fun


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

    def log_prob(self, inputs: np.ndarray, cov: np.ndarray = None) -> np.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
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
        the method from the original scipy code. See:
        https://github.com/scipy/scipy/blob/v1.6.0/scipy/stats/_multivariate.py
        """
        if cov is not None:
            s, u = np.linalg.eigh(cov)
            U = u * np.sqrt(1 / s[..., None])
            log_det = np.log(s).sum(axis=-1)
            xT_sigInv_x = np.sum((U @ inputs[..., None]) ** 2, axis=(1, 2))
        else:
            log_det = 0
            xT_sigInv_x = (inputs ** 2).sum(axis=-1)

        log_prob = (
            -1 / 2 * (inputs.shape[-1] * np.log(2 * np.pi) + log_det + xT_sigInv_x)
        )
        return log_prob

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


def sub_diag_indices(inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return indices for diagonal of 2D blocks in 3D array"""
    if inputs.ndim != 3:
        raise ValueError("Input must be a 3D array.")
    nblocks = inputs.shape[0]
    ndiag = min(inputs.shape[1], inputs.shape[2])
    idx = (
        np.repeat(np.arange(nblocks), ndiag),
        np.tile(np.arange(ndiag), nblocks),
        np.tile(np.arange(ndiag), nblocks),
    )
    return idx