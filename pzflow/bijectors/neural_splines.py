import jax.numpy as np
from jax import random, ops
from jax.experimental.stax import serial, Dense, Relu
from jax.nn import softmax, softplus
from typing import Callable


def _FCNN(out_dim: int, hidden_dim: int):
    return serial(
        Dense(hidden_dim),
        Relu,
        Dense(hidden_dim),
        Relu,
        Dense(out_dim),
    )


def _RationalQuadraticSpline(
    inputs,
    bin_widths,
    knot_heights,
    knot_derivatives,
    tail_bound: float = 3,
    inverse: bool = False,
):
    return inputs, np.zeros(inputs.shape[0])


def NeuralSplineCoupling(K: int = 5, B: float = 3, hidden_dim: int = 8) -> Callable:
    def init_fun(rng, input_dim, **kwargs):

        upper_dim = input_dim // 2  # variables that determine NN params
        lower_dim = input_dim - upper_dim  # variables transformed by the NN

        # create the neural network that will take in the upper dimensions and
        # will return the spline parameters to transform the lower dimensions
        network_init_fun, network_apply_fun = _FCNN((3 * K - 1) * lower_dim, hidden_dim)
        _, network_params = network_init_fun(rng, (upper_dim,))

        def forward_fun(params, inputs):
            upper, lower = inputs[:, :upper_dim], inputs[:, upper_dim:]
            # widths, heights, derivatives = function(upper variables)
            outputs = network_apply_fun(params, upper)
            outputs = np.reshape(outputs, [-1, lower_dim, 3 * K - 1])
            W, H, D = np.split(outputs, [K, 2 * K], axis=2)
            W = 2 * B * softmax(W)
            H = 2 * B * softmax(H)
            D = softplus(D)
            # transform the lower variables with the Rational Quadratic Spline
            lower, log_det = _RationalQuadraticSpline(lower, W, H, D, B, inverse=False)
            outputs = np.concatenate([upper, lower], axis=1)
            return outputs, log_det

        def inverse_fun(params, inputs):
            upper, lower = inputs[:, :upper_dim], inputs[:, upper_dim:]
            # widths, heights, derivatives = function(upper variables)
            outputs = network_apply_fun(params, upper)
            outputs = np.reshape(outputs, [-1, lower_dim, 3 * K - 1])
            W, H, D = np.split(outputs, [K, 2 * K], axis=2)
            W = 2 * B * softmax(W)
            H = 2 * B * softmax(H)
            D = softplus(D)
            # transform the lower variables with the Rational Quadratic Spline
            lower, log_det = _RationalQuadraticSpline(lower, W, H, D, B, inverse=True)
            outputs = np.concatenate([upper, lower], axis=1)
            return outputs, log_det

        return network_params, forward_fun, inverse_fun

    return init_fun
