from functools import update_wrapper
from typing import Callable, Sequence, Tuple, Union

import jax.numpy as np
from jax import ops, random

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

    return init_fun, bijector_info


@Bijector
def ColorTransform(
    ref_idx: int, ref_mean: float, ref_std: float
) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that converts photometric colors to magnitudes.

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

    Returns
    -------
    InitFunction
        The InitFunction of the the ColorTransform Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.

    Notes
    -----
    This Bijector takes redshift, a normalized reference magnitude, and a
    series of galaxy colors, and converts them to redshift and galaxy
    magnitudes. Here is an example of the bijection:

    redshift, R, u-g, g-r, r-i, i-z, z-y --> redshift, u, g, r, i, z, y

    where r = R * ref_std + ref_mean

    This transformation is useful at the end of your Bijector Chain, as
    redshifts correlate with galaxy colors more directly than galaxy
    magnitudes.

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

    bijector_info = ("ColorTransform", (ref_idx, ref_mean, ref_std))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            # calculate reference magnitude,
            # and convert all colors to be in terms of the first magnitude, mag[0]
            outputs = np.hstack(
                (
                    inputs[:, 0, None],  # redshift unchanged
                    inputs[:, 1, None] * ref_std + ref_mean,  # reference mag
                    np.cumsum(inputs[:, 2:], axis=-1),  # all colors --> mag[0] - mag[i]
                )
            )
            # calculate mag[0]
            outputs = ops.index_update(
                outputs,
                ops.index[:, 1],
                outputs[:, 1] + outputs[:, ref_idx],
                indices_are_sorted=True,
                unique_indices=True,
            )
            # mag[i] = mag[0] - (mag[0] - mag[i])
            outputs = ops.index_update(
                outputs,
                ops.index[:, 2:],
                outputs[:, 1, None] - outputs[:, 2:],
                indices_are_sorted=True,
                unique_indices=True,
            )
            log_det = np.log(ref_std) * np.ones(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = np.hstack(
                (
                    inputs[:, 0, None],  # redshift
                    (inputs[:, ref_idx, None] - ref_mean) / ref_std,  # ref mag
                    -np.diff(inputs[:, 1:]),  # colors
                )
            )
            log_det = -np.log(ref_std) * np.ones(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

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
            log_det = np.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs[:, ::-1]
            log_det = np.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def Roll(shift: int = 1) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that rolls inputs along their last column using np.roll.

    Parameters
    ----------
    shift : int, default=1
        The number of places to roll.

    Returns
    -------
    InitFunction
        The InitFunction of the the Roll Bijector.
    Bijector_Info
        Tuple of the Bijector name and the input parameters.
        This allows it to be recreated later.
    """

    bijector_info = ("Roll", (shift,))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = np.roll(inputs, shift=shift, axis=-1)
            log_det = np.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = np.roll(inputs, shift=-shift, axis=-1)
            log_det = np.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


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

    bijector_info = ("Scale", (scale,))

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

        perm = random.permutation(rng, np.arange(input_dim))
        inv_perm = np.argsort(perm)

        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = inputs[:, perm]
            log_det = np.zeros(inputs.shape[0])
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = inputs[:, inv_perm]
            log_det = np.zeros(inputs.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info


@Bijector
def Softplus(
    column_idx: int, sharpness: float = 1
) -> Tuple[InitFunction, Bijector_Info]:
    """Bijector that applies a softplus function to the specified column(s).

    Applying the softplus ensures that samples from that column will always
    be non-negative.

    Parameters
    ----------
    column_idx : int
        An index or iterable of indices corresponding to the column(s)
        you wish to be transformed.
    sharpness : float, default=1
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

    idx = np.atleast_1d(column_idx)
    k = np.atleast_1d(sharpness)
    if len(idx) != len(k) and len(k) != 1:
        raise ValueError(
            "Please provide either a single sharpness or one for each column index."
        )

    bijector_info = ("Softplus", (column_idx, sharpness))

    @InitFunction
    def init_fun(rng, input_dim, **kwargs):
        @ForwardFunction
        def forward_fun(params, inputs, **kwargs):
            outputs = ops.index_update(
                inputs,
                ops.index[:, idx],
                np.log(1 + np.exp(k * inputs[:, idx])) / k,
            )
            log_det = -np.log(1 + np.exp(-k * inputs[ops.index[:, idx]])).sum(axis=1)
            return outputs, log_det

        @InverseFunction
        def inverse_fun(params, inputs, **kwargs):
            outputs = ops.index_update(
                inputs,
                ops.index[:, idx],
                np.log(-1 + np.exp(k * inputs[:, idx])) / k,
            )
            log_det = np.log(1 + np.exp(-k * outputs[ops.index[:, idx]])).sum(axis=1)
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun, bijector_info