from textwrap import dedent
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
    W,
    H,
    D,
    B: float = 3,
    inverse: bool = False,
):
    # knot x-positions
    xk = np.pad(
        -B + np.cumsum(W, axis=-1),
        [(0, 0)] * (len(W.shape) - 1) + [(1, 0)],
        mode="constant",
        constant_values=-B,
    )
    # knot y-positions
    yk = np.pad(
        -B + np.cumsum(H, axis=-1),
        [(0, 0)] * (len(H.shape) - 1) + [(1, 0)],
        mode="constant",
        constant_values=-B,
    )
    # knot derivatives
    dk = np.pad(
        D,
        [(0, 0)] * (len(D.shape) - 1) + [(1, 1)],
        mode="constant",
        constant_values=1,
    )
    # knot slopes
    sk = H / W

    # any out-of-bounds inputs will have identity applied
    # for now we will replace these inputs with dummy inputs
    # so that the spline doesn't cause any problems.
    # at the end, we will replace them with their original values
    out_of_bounds = (inputs <= -B) | (inputs >= B)
    masked_inputs = np.where(out_of_bounds, -B, inputs)

    # find bin for each input
    if inverse:
        idx = np.sum(yk <= masked_inputs[..., None], axis=-1)[..., None] - 1
    else:
        idx = np.sum(xk <= masked_inputs[..., None], axis=-1)[..., None] - 1

    # get kx, ky, kyp1, kd, kdp1, kw, ks for the bin corresponding to each input
    input_xk = np.take_along_axis(xk, idx, -1)[..., 0]
    input_yk = np.take_along_axis(yk, idx, -1)[..., 0]
    input_dk = np.take_along_axis(dk, idx, -1)[..., 0]
    input_dkp1 = np.take_along_axis(dk, idx + 1, -1)[..., 0]
    input_wk = np.take_along_axis(W, idx, -1)[..., 0]
    input_hk = np.take_along_axis(H, idx, -1)[..., 0]
    input_sk = np.take_along_axis(sk, idx, -1)[..., 0]

    if inverse:
        # quadratic formula coefficients
        a = (input_hk) * (input_sk - input_dk) + (masked_inputs - input_yk) * (
            input_dkp1 + input_dk - 2 * input_sk
        )
        b = (input_hk) * input_dk + (masked_inputs - input_yk) * (
            input_dkp1 + input_dk - 2 * input_sk
        )
        c = -input_sk * (masked_inputs - input_yk)

        relx = 2 * c / (-b - np.sqrt(b ** 2 - 4 * a * c))
        outputs = np.where(out_of_bounds, inputs, relx * input_wk + input_xk)

        # calculate the log determinant
        dnum = (
            input_dkp1 * relx ** 2
            + 2 * input_sk * relx * (1 - relx)
            + input_dk * (1 - relx) ** 2
        )
        dden = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        log_det = 2 * np.log(input_sk) + np.log(dnum) - 2 * np.log(dden)
        log_det = log_det.sum(axis=1)

        return outputs, -log_det

    else:
        # calculate spline
        relx = (masked_inputs - input_xk) / input_wk
        num = input_hk * (input_sk * relx ** 2 + input_dk * relx * (1 - relx))
        den = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        spline_val = input_yk + num / den
        # replace out-of-bounds values
        outputs = np.where(out_of_bounds, inputs, spline_val)

        # calculate the log determinant
        dnum = (
            input_dkp1 * relx ** 2
            + 2 * input_sk * relx * (1 - relx)
            + input_dk * (1 - relx) ** 2
        )
        dden = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        log_det = 2 * np.log(input_sk) + np.log(dnum) - 2 * np.log(dden)
        log_det = log_det.sum(axis=1)

        return outputs, log_det


def NeuralSplineCoupling(K: int = 8, B: float = 3, hidden_dim: int = 8) -> Callable:
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
            outputs = np.hstack((upper, lower))
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
            outputs = np.hstack((upper, lower))
            return outputs, log_det

        return network_params, forward_fun, inverse_fun

    return init_fun
