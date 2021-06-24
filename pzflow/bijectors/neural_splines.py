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
    periodic: bool = False,
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
    periodic : bool, default=False
        Whether to make this a periodic, Circular Spline [2].

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
    [2] Rezende, Danilo Jimenez et al.
        Normalizing Flows on Tori and Spheres. arxiv:2002.02428, 2020
        http://arxiv.org/abs/2002.02428
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
    if periodic:
        dk = np.pad(D, [(0, 0)] * (len(D.shape) - 1) + [(1, 0)], mode="wrap")
    else:
        dk = np.pad(
            D,
            [(0, 0)] * (len(D.shape) - 1) + [(1, 1)],
            mode="constant",
            constant_values=1,
        )
    # knot slopes
    sk = H / W

    # if not periodic, out-of-bounds inputs will have identity applied
    # if periodic, we map the input into the appropriate region inside
    # the period. For now, we will pretend all inputs are periodic.
    # This makes sure that out-of-bounds inputs don't cause problems
    # with the spline, but for the non-periodic case, we will replace
    # these with their original values at the end
    out_of_bounds = (inputs <= -B) | (inputs >= B)
    masked_inputs = np.where(out_of_bounds, np.abs(inputs) - B, inputs)

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
        outputs = relx * input_wk + input_xk
        # if not periodic, replace out-of-bounds values with original values
        if not periodic:
            outputs = np.where(out_of_bounds, inputs, outputs)

        # [1] Appendix A.2
        # calculate the log determinant
        dnum = (
            input_dkp1 * relx ** 2
            + 2 * input_sk * relx * (1 - relx)
            + input_dk * (1 - relx) ** 2
        )
        dden = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        log_det = 2 * np.log(input_sk) + np.log(dnum) - 2 * np.log(dden)
        # if not periodic, replace log_det for out-of-bounds values = 0
        if not periodic:
            log_det = np.where(out_of_bounds, 0, log_det)
        log_det = log_det.sum(axis=1)

        return outputs, -log_det

    else:
        # [1] Appendix A.1
        # calculate spline
        relx = (masked_inputs - input_xk) / input_wk
        num = input_hk * (input_sk * relx ** 2 + input_dk * relx * (1 - relx))
        den = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        outputs = input_yk + num / den
        # if not periodic, replace out-of-bounds values with original values
        if not periodic:
            outputs = np.where(out_of_bounds, inputs, outputs)

        # [1] Appendix A.2
        # calculate the log determinant
        dnum = (
            input_dkp1 * relx ** 2
            + 2 * input_sk * relx * (1 - relx)
            + input_dk * (1 - relx) ** 2
        )
        dden = input_sk + (input_dkp1 + input_dk - 2 * input_sk) * relx * (1 - relx)
        log_det = 2 * np.log(input_sk) + np.log(dnum) - 2 * np.log(dden)
        # if not periodic, replace log_det for out-of-bounds values = 0
        if not periodic:
            log_det = np.where(out_of_bounds, 0, log_det)
        log_det = log_det.sum(axis=1)

        return outputs, log_det


@Bijector
def NeuralSplineCoupling(
    K: int = 16,
    B: float = 5,
    hidden_layers: int = 2,
    hidden_dim: int = 128,
    transformed_dim: int = None,
    n_conditions: int = 0,
    periodic: bool = False,
) -> Tuple[InitFunction, Bijector_Info]:
    """A coupling layer bijection with rational quadratic splines.

    This Bijector is a Coupling Layer [1,2], and as such only transforms
    the second half of input dimensions (or the last N dimensions, where
    N = transformed_dim). In order to transform all of the dimensions,
    you need multiple Couplings interspersed with Bijectors that change
    the order of inputs dimensions, e.g., Reverse, Shuffle, Roll, etc.

    NeuralSplineCoupling uses piecewise rational quadratic splines,
    as developed in [3].

    If periodic=True, then this is a Circular Spline as described in [4].

    Parameters
    ----------
    K : int, default=16
        Number of bins in the spline (the number of knots is K+1).
    B : float, default=5
        Range of the splines.
        If periodic=False, outside of (-B,B), the transformation is just
        the identity. If periodic=True, the input is mapped into the
        appropriate location in the range (-B,B).
    hidden_layers : int, default=2
        The number of hidden layers in the neural network used to calculate
        the positions and derivatives of the spline knots.
    hidden_dim : int, default=128
        The width of the hidden layers in the neural network used to
        calculate the positions and derivatives of the spline knots.
    transformed_dim : int, optional
        The number of dimensions transformed by the splines.
        Default is ceiling(input_dim /2).
    n_conditions : int, default=0
        The number of variables to condition the bijection on.
    periodic : bool, default=False
        Whether to make this a periodic, Circular Spline [4].

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
    [4] Rezende, Danilo Jimenez et al.
        Normalizing Flows on Tori and Spheres. arxiv:2002.02428, 2020
        http://arxiv.org/abs/2002.02428
    """

    if not isinstance(periodic, bool):
        raise ValueError("`periodic` must be True or False.")

    bijector_info = (
        "NeuralSplineCoupling",
        (K, B, hidden_layers, hidden_dim, transformed_dim, n_conditions, periodic),
    )

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):

        if transformed_dim is None:
            upper_dim = input_dim // 2  # variables that determine NN params
            lower_dim = input_dim - upper_dim  # variables transformed by the NN
        else:
            upper_dim = input_dim - transformed_dim
            lower_dim = transformed_dim

        # create the neural network that will take in the upper dimensions and
        # will return the spline parameters to transform the lower dimensions
        network_init_fun, network_apply_fun = DenseReluNetwork(
            (3 * K - 1 + int(periodic)) * lower_dim, hidden_layers, hidden_dim
        )
        _, network_params = network_init_fun(rng, (upper_dim + n_conditions,))

        # calculate spline parameters as a function of the upper variables
        def spline_params(params, upper, conditions):
            key = np.hstack((upper, conditions))[:, : upper_dim + n_conditions]
            outputs = network_apply_fun(params, key)
            outputs = np.reshape(outputs, [-1, lower_dim, 3 * K - 1 + int(periodic)])
            W, H, D = np.split(outputs, [K, 2 * K], axis=2)
            W = 2 * B * softmax(W)
            H = 2 * B * softmax(H)
            D = softplus(D)
            return W, H, D

        @ForwardFunction
        def forward_fun(params, inputs, conditions, **kwargs):
            # lower dimensions are transformed as function of upper dimensions
            upper, lower = inputs[:, :upper_dim], inputs[:, upper_dim:]
            # widths, heights, derivatives = function(upper dimensions)
            W, H, D = spline_params(params, upper, conditions)
            # transform the lower dimensions with the Rational Quadratic Spline
            lower, log_det = _RationalQuadraticSpline(
                lower, W, H, D, B, periodic, inverse=False
            )
            outputs = np.hstack((upper, lower))
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, conditions, **kwargs):
            # lower dimensions are transformed as function of upper dimensions
            upper, lower = inputs[:, :upper_dim], inputs[:, upper_dim:]
            # widths, heights, derivatives = function(upper dimensions)
            W, H, D = spline_params(params, upper, conditions)
            # transform the lower dimensions with the Rational Quadratic Spline
            lower, log_det = _RationalQuadraticSpline(
                lower, W, H, D, B, periodic, inverse=True
            )
            outputs = np.hstack((upper, lower))
            return outputs, log_det

        return network_params, forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def RollingSplineCoupling(
    nlayers: int,
    shift: int = 1,
    K: int = 16,
    B: float = 5,
    hidden_layers: int = 2,
    hidden_dim: int = 128,
    transformed_dim: int = None,
    n_conditions: int = 0,
    periodic: bool = False,
) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that alternates NeuralSplineCouplings and Roll bijections.

    Parameters
    ----------
    nlayers : int
        The number of (NeuralSplineCoupling(), Roll()) couplets in the chain.
    shift : int
        How far the inputs are shifted on each Roll().
    K : int, default=16
        Number of bins in the RollingSplineCoupling.
    B : float, default=5
        Range of the splines in the RollingSplineCoupling.
        If periodic=False, outside of (-B,B), the transformation is just
        the identity. If periodic=True, the input is mapped into the
        appropriate location in the range (-B,B).
    hidden_layers : int, default=2
        The number of hidden layers in the neural network used to calculate
        the bins and derivatives in the RollingSplineCoupling.
    hidden_dim : int, default=128
        The width of the hidden layers in the neural network used to
        calculate the bins and derivatives in the RollingSplineCoupling.
    transformed_dim : int, optional
        The number of dimensions transformed by the splines.
        Default is ceiling(input_dim /2).
    n_conditions : int, default=0
        The number of variables to condition the bijection on.
    periodic : bool, default=False
        Whether to make this a periodic, Circular Spline

    Returns
    -------
    InitFunction
        The InitFunction of the RollingSplineCoupling Bijector.
    Bijector_Info
        Nested tuple of the Bijector name and input parameters. This allows
        it to be recreated later.

    """
    return Chain(
        *(
            NeuralSplineCoupling(
                K=K,
                B=B,
                hidden_layers=hidden_layers,
                hidden_dim=hidden_dim,
                transformed_dim=transformed_dim,
                n_conditions=n_conditions,
                periodic=periodic,
            ),
            Roll(shift),
        )
        * nlayers
    )
