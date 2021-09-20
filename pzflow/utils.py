from typing import Callable, Tuple

import jax.numpy as np
from jax import random
from jax.experimental.stax import Dense, LeakyRelu, serial

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


def gaussian_error_model(
    key, X: np.ndarray, Xerr: np.ndarray, nsamples: int
) -> np.ndarray:
    """
    Default Gaussian error model were X are the means and Xerr are the stds.
    """

    eps = random.normal(key, shape=(X.shape[0], nsamples, X.shape[1]))

    return X[:, None, :] + eps * Xerr[:, None, :]


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
