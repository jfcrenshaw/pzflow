from typing import Tuple

import jax.numpy as np
from jax.nn import softmax, softplus
from pzflow.bijectors.bijectors import (
    Bijector,
    Bijector_Info,
    Chain,
    ForwardFunction,
    InitFunction,
    InverseFunction,
    Roll,
)
from pzflow.utils import DenseReluNetwork


def _RationalQuadraticSpline(
    inputs: np.ndarray,
    W: np.ndarray,
    H: np.ndarray,
    D: np.ndarray,
    B: float,
    inverse: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rational quadratic spline to inputs and return outputs with log_det.

    Applies the piecewise rational quadratic spline developed in [1].

    Parameters
    ----------
    inputs : np.ndarray
        The inputs to be transformed.
    W : np.ndarray
        The widths of the spline bins.
    H : np.ndarray
        The heights of the spline bins.
    D : np.ndarray
        The derivatives of the inner spline knots.
    B : float
        Range of the splines.
        Outside of (-B,B), the transformation is just the identity.
    inverse : bool, default=False
        If True, perform the inverse transformation.
        Otherwise perform the forward transformation.

    Returns
    -------
    outputs : np.ndarray
        The result of applying the splines to the inputs.
    log_det : np.ndarray
        The log determinant of the Jacobian at the inputs.

    References
    ----------
    [1] Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios.
        Neural Spline Flows. arXiv:1906.04032, 2019.
        https://arxiv.org/abs/1906.04032
    """
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
        # [1] Appendix A.3
        # quadratic formula coefficients
        a = (input_hk) * (input_sk - input_dk) + (masked_inputs - input_yk) * (
            input_dkp1 + input_dk - 2 * input_sk
        )
        b = (input_hk) * input_dk - (masked_inputs - input_yk) * (
            input_dkp1 + input_dk - 2 * input_sk
        )
        c = -input_sk * (masked_inputs - input_yk)

        relx = 2 * c / (-b - np.sqrt(b ** 2 - 4 * a * c))
        outputs = np.where(out_of_bounds, inputs, relx * input_wk + input_xk)

        # [1] Appendix A.2
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
        # [1] Appendix A.1
        # calculate spline
        relx = (masked_inputs - input_xk) / input_wk
        num = input_hk * (input_sk * relx ** 2 + input_dk * relx * (1 - relx))
        den = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        spline_val = input_yk + num / den
        # replace out-of-bounds values
        outputs = np.where(out_of_bounds, inputs, spline_val)

        # [1] Appendix A.2
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


@Bijector
def NeuralSplineCoupling(
    K: int = 16, B: float = 3, hidden_layers: int = 2, hidden_dim: int = 128
) -> Tuple[InitFunction, Bijector_Info]:
    """A coupling layer bijection with rational quadratic splines.

    This Bijector is a Coupling Layer [1,2], and as such only transforms
    the second half of input dimensions. In order to transform all of
    dimensions, you need multiple Couplings interspersed with Bijectors
    that change the order of inputs dimensions, e.g., Reverse, Shuffle,
    Roll, etc.

    NeuralSplineCoupling uses piecewise rational quadratic splines,
    as developed in [3].

    Parameters
    ----------
    K : int, default=16
        Number of bins in the spline (the number of knots is K+1).
    B : float, default=3
        Range of the splines.
        Outside of (-B,B), the transformation is just the identity.
    hidden_layers : int, default=2
        The number of hidden layers in the neural network used to calculate
        the positions and derivatives of the spline knots.
    hidden_dim : int, default=128
        The width of the hidden layers in the neural network used to
        calculate the positions and derivatives of the spline knots.

    Returns
    -------
    InitFunction
        The InitFunction of the NeuralSplineCoupling Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.

    References
    ----------
    [1] Laurent Dinh, David Krueger, Yoshua Bengio. NICE: Non-linear
        Independent Components Estimation. arXiv: 1605.08803, 2015.
        http://arxiv.org/abs/1605.08803
    [2] Laurent Dinh, Jascha Sohl-Dickstein, Samy Bengio.
        Density Estimation Using Real NVP. arXiv: 1605.08803, 2017.
        http://arxiv.org/abs/1605.08803
    [3] Conor Durkan, Artur Bekasov, Iain Murray, George Papamakarios.
        Neural Spline Flows. arXiv:1906.04032, 2019.
        https://arxiv.org/abs/1906.04032
    """

    bijector_info = ("NeuralSplineCoupling", (K, B, hidden_layers, hidden_dim))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):

        upper_dim = input_dim // 2  # variables that determine NN params
        lower_dim = input_dim - upper_dim  # variables transformed by the NN

        # create the neural network that will take in the upper dimensions and
        # will return the spline parameters to transform the lower dimensions
        network_init_fun, network_apply_fun = DenseReluNetwork(
            (3 * K - 1) * lower_dim, hidden_layers, hidden_dim
        )
        _, network_params = network_init_fun(rng, (upper_dim,))

        # calculate spline parameters as a function of the upper variables
        def spline_params(params, upper):
            outputs = network_apply_fun(params, upper)
            outputs = np.reshape(outputs, [-1, lower_dim, 3 * K - 1])
            W, H, D = np.split(outputs, [K, 2 * K], axis=2)
            W = 2 * B * softmax(W)
            H = 2 * B * softmax(H)
            D = softplus(D)
            return W, H, D

        @ForwardFunction
        def forward_fun(params, inputs):
            # lower dimensions are transformed as function of upper dimensions
            upper, lower = inputs[:, :upper_dim], inputs[:, upper_dim:]
            # widths, heights, derivatives = function(upper dimensions)
            W, H, D = spline_params(params, upper)
            # transform the lower dimensions with the Rational Quadratic Spline
            lower, log_det = _RationalQuadraticSpline(lower, W, H, D, B, inverse=False)
            outputs = np.hstack((upper, lower))
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs):
            # lower dimensions are transformed as function of upper dimensions
            upper, lower = inputs[:, :upper_dim], inputs[:, upper_dim:]
            # widths, heights, derivatives = function(upper dimensions)
            W, H, D = spline_params(params, upper)
            # transform the lower dimensions with the Rational Quadratic Spline
            lower, log_det = _RationalQuadraticSpline(lower, W, H, D, B, inverse=True)
            outputs = np.hstack((upper, lower))
            return outputs, log_det

        return network_params, forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def RollingSplineCoupling(
    nlayers: int,
    K: int = 16,
    B: float = 3,
    hidden_layers: int = 2,
    hidden_dim: int = 128,
) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that alternates NeuralSplineCouplings and Roll bijections.

    Parameters
    ----------
    nlayers : int
        The number of (NeuralSplineCoupling(), Roll()) couplets in the chain.
    K : int, default=16
        Number of bins in the RollingSplineCoupling.
    B : float, default=3
        Range of the splines in the RollingSplineCoupling.
    hidden_layers : int, default=2
        The number of hidden layers in the neural network used to calculate
        the bins and derivatives in the RollingSplineCoupling.
    hidden_dim : int, default=128
        The width of the hidden layers in the neural network used to
        calculate the bins and derivatives in the RollingSplineCoupling.

    Returns
    -------
    InitFunction
        The InitFunction of the RollingSplineCoupling Bijector.
    Bijector_Info
        Nested tuple of the Bijector name and input parameters. This allows
        it to be recreated later.

    """
    return Chain(
        *(NeuralSplineCoupling(K, B, hidden_layers, hidden_dim), Roll()) * nlayers
    )
