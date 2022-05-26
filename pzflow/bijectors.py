"""Define the bijectors used in the normalizing flows."""
from functools import update_wrapper
from typing import Callable, Sequence, Tuple, Union

import jax.numpy as jnp
from jax import random
from jax.nn import softmax, softplus

from pzflow.utils import DenseReluNetwork, RationalQuadraticSpline

# define a type alias for Jax Pytrees
Pytree = Union[tuple, list]
Bijector_Info = Tuple[str, tuple]


class ForwardFunction:
    """Return the output and log_det of the forward bijection on the inputs.

    ForwardFunction of a Bijector, originally returned by the
    InitFunction of the Bijector.

    Parameters
    ----------
    params : a Jax pytree
        A pytree of bijector parameters.
        This usually looks like a nested tuple or list of parameters.
    inputs : jnp.ndarray
        The data to be transformed by the bijection.

    Returns
    -------
    outputs : jnp.ndarray
        Result of the forward bijection applied to the inputs.
    log_det : jnp.ndarray
        The log determinant of the Jacobian evaluated at the inputs.
    """

    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(
        self, params: Pytree, inputs: jnp.ndarray, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._func(params, inputs, **kwargs)


class InverseFunction:
    """Return the output and log_det of the inverse bijection on the inputs.

    InverseFunction of a Bijector, originally returned by the
    InitFunction of the Bijector.

    Parameters
    ----------
    params : a Jax pytree
        A pytree of bijector parameters.
        This usually looks like a nested tuple or list of parameters.
    inputs : jnp.ndarray
        The data to be transformed by the bijection.

    Returns
    -------
    outputs : jnp.ndarray
        Result of the inverse bijection applied to the inputs.
    log_det : jnp.ndarray
        The log determinant of the Jacobian evaluated at the inputs.
    """

    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(
        self, params: Pytree, inputs: jnp.ndarray, **kwargs
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self._func(params, inputs, **kwargs)


class InitFunction:
    """Initialize the corresponding Bijector.

    InitFunction returned by the initialization of a Bijector.

    Parameters
    ----------
    rng : jnp.ndarray
        A Random Number Key from jax.random.PRNGKey.
    input_dim : int
        The input dimension of the bijection.

    Returns
    -------
    params : a Jax pytree
        A pytree of bijector parameters.
        This usually looks like a nested tuple or list of parameters.
    forward_fun : ForwardFunction
        The forward function of the Bijector.
    inverse_fun : InverseFunction
        The inverse function of the Bijector.
    """

    def __init__(self, func: Callable) -> None:
        self._func = func

    def __call__(
        self, rng: jnp.ndarray, input_dim: int, **kwargs
    ) -> Tuple[Pytree, ForwardFunction, InverseFunction]:
        return self._func(rng, input_dim, **kwargs)


class Bijector:
    """Wrapper class for bijector functions"""

    def __init__(self, func: Callable) -> None:
        self._func = func
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> Tuple[InitFunction, Bijector_Info]:
        return self._func(*args, **kwargs)


@Bijector
def Chain(
    *inputs: Sequence[Tuple[InitFunction, Bijector_Info]]
) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that chains multiple InitFunctions into a single InitFunction.

    Parameters
    ----------
    inputs : (Bijector1(), Bijector2(), ...)
        A container of Bijector calls to be chained together.

    Returns
    -------
    InitFunction
        The InitFunction of the total chained Bijector.
    Bijector_Info
        Tuple('Chain', Tuple(Bijector_Info for each bijection in the chain))
        This allows the chain to be recreated later.
    """

    init_funs = tuple(i[0] for i in inputs)
    bijector_info = ("Chain", tuple(i[1] for i in inputs))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):

        all_params, forward_funs, inverse_funs = [], [], []
        for init_f in init_funs:
            rng, layer_rng = random.split(rng)
            param, forward_f, inverse_f = init_f(layer_rng, input_dim)

            all_params.append(param)
            forward_funs.append(forward_f)
            inverse_funs.append(inverse_f)

        def bijector_chain(params, bijectors, inputs, **kwargs):
            log_dets = jnp.zeros(inputs.shape[0])
            for bijector, param in zip(bijectors, params):
                inputs, log_det = bijector(param, inputs, **kwargs)
                log_dets += log_det
            return inputs, log_dets

        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            return bijector_chain(params, forward_funs, inputs, **kwargs)

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            return bijector_chain(
                params[::-1], inverse_funs[::-1], inputs, **kwargs
            )

        return all_params, forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def ColorTransform(
    ref_idx: int, mag_idx: int
) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that calculates photometric colors from magnitudes.

    Using ColorTransform restricts and impacts the order of columns in the
    corresponding normalizing flow. See the notes below for an example.

    Parameters
    ----------
    ref_idx : int
        The index corresponding to the column of the reference band, which
        serves as a proxy for overall luminosity.
    mag_idx : arraylike of int
        The indices of the magnitude columns from which colors will be calculated.

    Returns
    -------
    InitFunction
        The InitFunction of the ColorTransform Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.

    Notes
    -----
    ColorTransform requires careful management of column order in the bijector.
    This is best explained with an example:

    Assume we have data
    [redshift, u, g, ellipticity, r, i, z, y, mass]
    Then
    ColorTransform(ref_idx=4, mag_idx=[1, 2, 4, 5, 6, 7])
    will output
    [redshift, ellipticity, mass, r, u-g, g-r, r-i, i-z, z-y]

    Notice how the non-magnitude columns are aggregated at the front of the
    array, maintaining their relative order from the original array.
    These values are then followed by the reference magnitude, and the new colors.

    Also notice that the magnitudes indices in mag_idx are assumed to be
    adjacent colors. E.g. mag_idx=[1, 2, 5, 4, 6, 7] would have produced
    the colors [u-g, g-i, i-r, r-z, z-y]. You can chain multiple ColorTransforms
    back-to-back to create colors in a non-adjacent manner.
    """

    # validate parameters
    if ref_idx <= 0:
        raise ValueError("ref_idx must be a positive integer.")
    if not isinstance(ref_idx, int):
        raise ValueError("ref_idx must be an integer.")
    if ref_idx not in mag_idx:
        raise ValueError("ref_idx must be in mag_idx.")

    bijector_info = ("ColorTransform", (ref_idx, mag_idx))

    # convert mag_idx to an array
    mag_idx = jnp.array(mag_idx)

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):

        # array of all the indices
        all_idx = jnp.arange(input_dim)
        # indices for columns to stick at the front
        front_idx = jnp.setdiff1d(all_idx, mag_idx)
        # the index corresponding to the first magnitude
        mag0_idx = len(front_idx)

        # the new column order
        new_idx = jnp.concatenate((front_idx, mag_idx))
        # the new column for the reference magnitude
        new_ref = jnp.where(new_idx == ref_idx)[0][0]

        # define a convenience function for the forward_fun below
        # if the first magnitude is the reference mag, do nothing
        if ref_idx == mag_idx[0]:

            def mag0(outputs):
                return outputs

        # if the first magnitude is not the reference mag,
        # then we need to calculate the first magnitude (mag[0])
        else:

            def mag0(outputs):
                return outputs.at[:, mag0_idx].set(
                    outputs[:, mag0_idx] + outputs[:, new_ref],
                    indices_are_sorted=True,
                    unique_indices=True,
                )

        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            # re-order columns and calculate colors
            outputs = jnp.hstack(
                (
                    inputs[:, front_idx],  # other values
                    inputs[:, ref_idx, None],  # ref mag
                    -jnp.diff(inputs[:, mag_idx]),  # colors
                )
            )
            # determinant of Jacobian is zero
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            # convert all colors to be in terms of the first magnitude, mag[0]
            outputs = jnp.hstack(
                (
                    inputs[:, 0:mag0_idx],  # other values unchanged
                    inputs[:, mag0_idx, None],  # reference mag unchanged
                    jnp.cumsum(
                        inputs[:, mag0_idx + 1 :], axis=-1
                    ),  # all colors mag[i-1] - mag[i] --> mag[0] - mag[i]
                )
            )
            # calculate mag[0]
            outputs = mag0(outputs)
            # mag[i] = mag[0] - (mag[0] - mag[i])
            outputs = outputs.at[:, mag0_idx + 1 :].set(
                outputs[:, mag0_idx, None] - outputs[:, mag0_idx + 1 :],
                indices_are_sorted=True,
                unique_indices=True,
            )
            # return to original ordering
            outputs = outputs[:, jnp.argsort(new_idx)]
            # determinant of Jacobian is zero
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def InvSoftplus(
    column_idx: int, sharpness: float = 1
) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that applies inverse softplus to the specified column(s).

    Applying the inverse softplus ensures that samples from that column will
    always be non-negative. This is because samples are the output of the
    inverse bijection -- so samples will have a softplus applied to them.

    Parameters
    ----------
    column_idx : int
        An index or iterable of indices corresponding to the column(s)
        you wish to be transformed.
    sharpness : float; default=1
        The sharpness(es) of the softplus transformation. If more than one
        is provided, the list of sharpnesses must be of the same length as
        column_idx.

    Returns
    -------
    InitFunction
        The InitFunction of the Softplus Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.
    """

    idx = jnp.atleast_1d(column_idx)
    k = jnp.atleast_1d(sharpness)
    if len(idx) != len(k) and len(k) != 1:
        raise ValueError(
            "Please provide either a single sharpness or one for each column index."
        )

    bijector_info = ("InvSoftplus", (column_idx, sharpness))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = inputs.at[:, idx].set(
                jnp.log(-1 + jnp.exp(k * inputs[:, idx])) / k,
            )
            log_det = jnp.log(1 + jnp.exp(-k * outputs[:, idx])).sum(axis=1)
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs.at[:, idx].set(
                jnp.log(1 + jnp.exp(k * inputs[:, idx])) / k,
            )
            log_det = -jnp.log(1 + jnp.exp(-k * inputs[:, idx])).sum(axis=1)
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


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
    K : int; default=16
        Number of bins in the spline (the number of knots is K+1).
    B : float; default=5
        Range of the splines.
        If periodic=False, outside of (-B,B), the transformation is just
        the identity. If periodic=True, the input is mapped into the
        appropriate location in the range (-B,B).
    hidden_layers : int; default=2
        The number of hidden layers in the neural network used to calculate
        the positions and derivatives of the spline knots.
    hidden_dim : int; default=128
        The width of the hidden layers in the neural network used to
        calculate the positions and derivatives of the spline knots.
    transformed_dim : int; optional
        The number of dimensions transformed by the splines.
        Default is ceiling(input_dim /2).
    n_conditions : int; default=0
        The number of variables to condition the bijection on.
    periodic : bool; default=False
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
        (
            K,
            B,
            hidden_layers,
            hidden_dim,
            transformed_dim,
            n_conditions,
            periodic,
        ),
    )

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):

        if transformed_dim is None:
            upper_dim = input_dim // 2  # variables that determine NN params
            lower_dim = (
                input_dim - upper_dim
            )  # variables transformed by the NN
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
            key = jnp.hstack((upper, conditions))[
                :, : upper_dim + n_conditions
            ]
            outputs = network_apply_fun(params, key)
            outputs = jnp.reshape(
                outputs, [-1, lower_dim, 3 * K - 1 + int(periodic)]
            )
            W, H, D = jnp.split(outputs, [K, 2 * K], axis=2)
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
            lower, log_det = RationalQuadraticSpline(
                lower, W, H, D, B, periodic, inverse=False
            )
            outputs = jnp.hstack((upper, lower))
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, conditions, **kwargs):
            # lower dimensions are transformed as function of upper dimensions
            upper, lower = inputs[:, :upper_dim], inputs[:, upper_dim:]
            # widths, heights, derivatives = function(upper dimensions)
            W, H, D = spline_params(params, upper, conditions)
            # transform the lower dimensions with the Rational Quadratic Spline
            lower, log_det = RationalQuadraticSpline(
                lower, W, H, D, B, periodic, inverse=True
            )
            outputs = jnp.hstack((upper, lower))
            return outputs, log_det

        return network_params, forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def Reverse() -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that reverses the order of inputs.

    Returns
    -------
    InitFunction
        The InitFunction of the the Reverse Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.
    """

    bijector_info = ("Reverse", ())

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = inputs[:, ::-1]
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs[:, ::-1]
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def Roll(shift: int = 1) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that rolls inputs along their last column using jnp.roll.

    Parameters
    ----------
    shift : int; default=1
        The number of places to roll.

    Returns
    -------
    InitFunction
        The InitFunction of the the Roll Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.
    """

    if not isinstance(shift, int):
        raise ValueError("shift must be an integer.")

    bijector_info = ("Roll", (shift,))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = jnp.roll(inputs, shift=shift, axis=-1)
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = jnp.roll(inputs, shift=-shift, axis=-1)
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

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
    K : int; default=16
        Number of bins in the RollingSplineCoupling.
    B : float; default=5
        Range of the splines in the RollingSplineCoupling.
        If periodic=False, outside of (-B,B), the transformation is just
        the identity. If periodic=True, the input is mapped into the
        appropriate location in the range (-B,B).
    hidden_layers : int; default=2
        The number of hidden layers in the neural network used to calculate
        the bins and derivatives in the RollingSplineCoupling.
    hidden_dim : int; default=128
        The width of the hidden layers in the neural network used to
        calculate the bins and derivatives in the RollingSplineCoupling.
    transformed_dim : int; optional
        The number of dimensions transformed by the splines.
        Default is ceiling(input_dim /2).
    n_conditions : int; default=0
        The number of variables to condition the bijection on.
    periodic : bool; default=False
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


@Bijector
def Scale(scale: float) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that multiplies inputs by a scalar.

    Parameters
    ----------
    scale : float
        Factor by which to scale inputs.

    Returns
    -------
    InitFunction
        The InitFunction of the the Scale Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.
    """

    if isinstance(scale, jnp.ndarray):
        if scale.dtype != jnp.float32:
            raise ValueError("scale must be a float or array of floats.")
    elif not isinstance(scale, float):
        raise ValueError("scale must be a float or array of floats.")

    bijector_info = ("Scale", (scale,))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = scale * inputs
            log_det = jnp.log(scale ** inputs.shape[-1]) * jnp.ones(
                inputs.shape[0]
            )
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = 1 / scale * inputs
            log_det = -jnp.log(scale ** inputs.shape[-1]) * jnp.ones(
                inputs.shape[0]
            )
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def ShiftBounds(
    min: float, max: float, B: float = 5
) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector shifts the bounds of inputs so the lie in the range (-B, B).

    Parameters
    ----------
    min : float
        The minimum of the input range.
    min : float
        The maximum of the input range.
    B : float; default=5
        The extent of the output bounds, which will be (-B, B).

    Returns
    -------
    InitFunction
        The InitFunction of the ShiftBounds Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.
    """

    min = jnp.atleast_1d(min)
    max = jnp.atleast_1d(max)
    if len(min) != len(max):
        raise ValueError(
            "Lengths of min and max do not match. "
            + "Please provide either a single min and max, "
            + "or a min and max for each dimension."
        )
    if (min > max).any():
        raise ValueError("All mins must be less than maxes.")

    bijector_info = ("ShiftBounds", (min, max, B))

    mean = (max + min) / 2
    half_range = (max - min) / 2

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = B * (inputs - mean) / half_range
            log_det = jnp.log(jnp.prod(B / half_range)) * jnp.ones(
                inputs.shape[0]
            )
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs * half_range / B + mean
            log_det = jnp.log(jnp.prod(half_range / B)) * jnp.ones(
                inputs.shape[0]
            )
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def Shuffle() -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that randomly permutes inputs.

    Returns
    -------
    InitFunction
        The InitFunction of the Shuffle Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.
    """

    bijector_info = ("Shuffle", ())

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):

        perm = random.permutation(rng, jnp.arange(input_dim))
        inv_perm = jnp.argsort(perm)

        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = inputs[:, perm]
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs[:, inv_perm]
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def StandardScaler(
    means: jnp.array, stds: jnp.array
) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that applies standard scaling to each input.

    Each input dimension i has an associated mean u_i and standard dev s_i.
    Each input is rescaled as (input[i] - u_i)/s_i, so that each input dimension
    has mean zero and unit variance.

    Parameters
    ----------
    means : jnp.ndarray
        The mean of each column.
    stds : jnp.ndarray
        The standard deviation of each column.

    Returns
    -------
    InitFunction
        The InitFunction of the StandardScaler Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.
    """

    bijector_info = ("StandardScaler", (means, stds))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = (inputs - means) / stds
            log_det = jnp.log(1 / jnp.prod(stds)) * jnp.ones(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs * stds + means
            log_det = jnp.log(jnp.prod(stds)) * jnp.ones(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def UniformDequantizer(column_idx: int) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that dequantizes discrete variables with uniform noise.

    Dequantizers are necessary for modeling discrete values with a flow.
    Note that this isn't technically a bijector.

    Parameters
    ----------
    column_idx : int
        An index or iterable of indices corresponding to the column(s) with
        discrete values.

    Returns
    -------
    InitFunction
        The InitFunction of the UniformDequantizer Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.
    """

    bijector_info = ("UniformDequantizer", (column_idx,))
    column_idx = jnp.array(column_idx)

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            u = random.uniform(
                random.PRNGKey(0), shape=inputs[:, column_idx].shape
            )
            outputs = inputs.astype(float)
            outputs.at[:, column_idx].set(outputs[:, column_idx] + u)
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs.at[:, column_idx].set(
                jnp.floor(inputs[:, column_idx])
            )
            log_det = jnp.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info
