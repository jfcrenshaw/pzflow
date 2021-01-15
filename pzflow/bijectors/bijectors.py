from functools import update_wrapper
from typing import Callable, Sequence, Tuple, Union

import jax.numpy as np
from jax import ops, random

# define a type alias for Jax Pytrees
Pytree = Union[tuple, list]


class ForwardFunction:
    """Return the output and log_det of the forward bijection on the inputs.

    ForwardFunction of a Bijector, originally returned by the
    InitFunction of the Bijector.

    Parameters
    ----------
    params : a Jax pytree
        A pytree of bijector parameters.
        This usually looks like a nested tuple or list of parameters.
    inputs : np.ndarray
        The data to be transformed by the bijection.

    Returns
    -------
    outputs : np.ndarray
        Result of the forward bijection applied to the inputs.
    log_det : np.ndarray
        The log determinant of the Jacobian evaluated at the inputs.
    """

    def __init__(self, func: Callable):
        self._func = func

    def __call__(
        self, params: Pytree, inputs: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
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
    inputs : np.ndarray
        The data to be transformed by the bijection.

    Returns
    -------
    outputs : np.ndarray
        Result of the inverse bijection applied to the inputs.
    log_det : np.ndarray
        The log determinant of the Jacobian evaluated at the inputs.
    """

    def __init__(self, func: Callable):
        self._func = func

    def __call__(
        self, params: Pytree, inputs: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._func(params, inputs, **kwargs)


class InitFunction:
    """Initialize the corresponding Bijector.

    InitFunction returned by the initialization of a Bijector.

    Parameters
    ----------
    rng : np.ndarray
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

    def __init__(self, func: Callable):
        self._func = func

    def __call__(
        self, rng: np.ndarray, input_dim: int, **kwargs
    ) -> Tuple[Pytree, ForwardFunction, InverseFunction]:
        return self._func(rng, input_dim, **kwargs)


class Bijector:
    """Wrapper class for bijector functions"""

    def __init__(self, func: Callable):
        self._func = func
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> InitFunction:
        return self._func(*args, **kwargs)


@Bijector
def Chain(*init_funs: Sequence[InitFunction]) -> InitFunction:
    """Bijector that chains multiple InitFunctions into a single InitFunction.

    Parameters
    ----------
    init_funs : Sequence[InitFunction]
        A contained of Bijector InitFunctions to be chained together.

    Returns
    -------
    InitFunction
        The InitFunction of the total chained Bijector.
    """

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):

        all_params, forward_funs, inverse_funs = [], [], []
        for init_f in init_funs:
            rng, layer_rng = random.split(rng)
            param, forward_f, inverse_f = init_f(layer_rng, input_dim)

            all_params.append(param)
            forward_funs.append(forward_f)
            inverse_funs.append(inverse_f)

        def bijector_chain(params, bijectors, inputs):
            log_dets = np.zeros(inputs.shape[0])
            for bijector, param in zip(bijectors, params):
                inputs, log_det = bijector(param, inputs, **kwargs)
                log_dets += log_det
            return inputs, log_dets

        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            return bijector_chain(params, forward_funs, inputs)

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            return bijector_chain(params[::-1], inverse_funs[::-1], inputs)

        return all_params, forward_fun, inverse_fun

    return init_fun


@Bijector
def ColorTransform(
    ref_idx: int, ref_mean: float, ref_stdd: float, z_sharp: float = 10
) -> InitFunction:
    """Bijector that converts colors to magnitudes and constrains redshift positive.

    Using ColorTransform restricts the order of columns in the corresponding
    normalizing flow. Redshift must be the first column, and the following
    columns must be adjacent color magnitudes,
    e.g.: redshift, R, u-g, g-r, r-i --> redshift, u, g, r, i

    Parameters
    ----------
    ref_idx : int
        The index corresponding to the column of the reference band.
    ref_mean : float
        The mean magnitude of the reference band.
    ref_std : float
        The standard deviation of the reference band.
    z_sharp : float, default=10
        The sharpness of the softplus applied to the redshift column.

    Returns
    -------
    InitFunction
        The InitFunction of the the ColorTransform Bijector.

    Notes
    -----
    This Bijector takes a redshift parameter, a normalized reference magnitude,
    and a series of galaxy colors, and converts them to redshift and galaxy
    magnitudes. Here is an example of the bijection:

    redshift_param, R, u-g, g-r, r-i, i-z, z-y --> redshift, u, g, r, i, z, y

    where
    redshift = softplus(redshift_param)
    r = R * ref_std + ref_mean

    This transformation is useful at the very end of your Bijector Chain,
    as redshifts correlate with galaxy colors more directly than galaxy
    magnitudes. In addition, the softplus applied to the redshift parameter
    ensures that the sampled redshifts are always positive.

    In the example above, the r band was used as the reference magnitude to
    serve as a proxy for overall galaxy luminosity. In this example, this
    would be achieved by setting ref_idx=3 as that is the index of the column
    corresponding to the r band in my data. You also need to provide the mean
    and standard deviation of the r band as ref_mean and ref_std, respectively.
    You can use another band by changing these values appropriately. E.g., in
    the example above, you can set ref_idx=4 for the i band.

    Note that using ColorTransform restricts the order of columns in the
    corresponding normalizing flow. Redshift must be the first column, and
    the following columns must be adjacent color magnitudes,
    e.g.: redshift, R, u-g, g-r, r-i --> redshift, u, g, r, i
    """

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            # calculate reference magnitude,
            # and convert all colors to be in terms of the first magnitude, mag[0]
            outputs = np.hstack(
                (
                    np.log(1 + np.exp(z_sharp * inputs[:, 0, None]))
                    / z_sharp,  # softplus to force redshift positive
                    inputs[:, 1, None] * ref_stdd + ref_mean,  # reference mag
                    np.cumsum(inputs[:, 2:], axis=-1),  # all colors --> mag[0] - mag[i]
                )
            )
            # calculate mag[0]
            outputs = ops.index_update(
                outputs, ops.index[:, 1], outputs[:, 1] + outputs[:, ref_idx]
            )
            # mag[i] = mag[0] - (mag[0] - mag[i])
            outputs = ops.index_update(
                outputs, ops.index[:, 2:], outputs[:, 1, None] - outputs[:, 2:]
            )
            log_det = np.log(ref_stdd * (1 - np.exp(-z_sharp * outputs[:, 0])))
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = np.hstack(
                (
                    np.log(-1 + np.exp(z_sharp * inputs[:, 0, None]))
                    / z_sharp,  # inverse of softplus
                    (inputs[:, ref_idx, None] - ref_mean) / ref_stdd,  # ref mag
                    -np.diff(inputs[:, 1:]),  # colors
                )
            )
            log_det = -np.log(ref_stdd * (1 - np.exp(-z_sharp * inputs[:, 0])))
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun


@Bijector
def Reverse() -> InitFunction:
    """Bijector that reverses the order of inputs.

    Returns
    -------
    InitFunction
        The InitFunction of the the Reverse Bijector.
    """

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            return inputs[:, ::-1], np.zeros(inputs.shape[0])

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, ::-1], np.zeros(inputs.shape[0])

        return (), forward_fun, inverse_fun

    return init_fun


@Bijector
def Roll(shift: int = 1) -> InitFunction:
    """Bijector that rolls inputs along their last column using np.roll.

    Parameters
    ----------
    shift : int, default=1
        The number of places to roll.

    Returns
    -------
    InitFunction
        The InitFunction of the the Roll Bijector.
    """

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            return np.roll(inputs, shift=shift, axis=-1), np.zeros(inputs.shape[0])

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            return np.roll(inputs, shift=-shift, axis=-1), np.zeros(inputs.shape[0])

        return (), forward_fun, inverse_fun

    return init_fun


@Bijector
def Scale(scale: float) -> InitFunction:
    """Bijector that multiplies inputs by a scalar.

    Parameters
    ----------
    scale : float
        Factor by which to scale inputs.

    Returns
    -------
    InitFunction
        The InitFunction of the the Scale Bijector.
    """

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = scale * inputs
            log_det = np.log(scale ** inputs.shape[-1]) * np.ones(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = 1 / scale * inputs
            log_det = -np.log(scale ** inputs.shape[-1]) * np.ones(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun


@Bijector
def Shuffle() -> InitFunction:
    """Bijector that randomly permutes inputs.

    Returns
    -------
    InitFunction
        The InitFunction of the Shuffle Bijector.
    """

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):

        perm = random.permutation(rng, np.arange(input_dim))
        inv_perm = np.argsort(perm)

        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            return inputs[:, perm], np.zeros(inputs.shape[0])

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, inv_perm], np.zeros(inputs.shape[0])

        return (), forward_fun, inverse_fun

    return init_fun
