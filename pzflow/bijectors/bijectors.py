from functools import update_wrapper
from typing import Callable, Sequence, Tuple, Union

import jax.numpy as np
from jax import ops, random


Pytree = Union[tuple, list]


class ForwardFunction:
    def __init__(self, func: Callable):
        self._func = func

    def __call__(
        self, params: Pytree, inputs: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._func(params, inputs, **kwargs)


class InverseFunction:
    def __init__(self, func: Callable):
        self._func = func

    def __call__(
        self, params: Pytree, inputs: np.ndarray, **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._func(params, inputs, **kwargs)


class InitFunction:
    def __init__(self, func: Callable):
        self._func = func

    def __call__(
        self, rng: np.ndarray, input_dim: int, **kwargs
    ) -> Tuple[Pytree, ForwardFunction, InverseFunction]:
        return self._func(rng, input_dim, **kwargs)


class Bijector:
    def __init__(self, func: Callable):
        self._func = func
        update_wrapper(self, func)

    def __call__(self, *args, **kwargs) -> InitFunction:
        return self._func(*args, **kwargs)


@Bijector
def Chain(*init_funs: Sequence[InitFunction]) -> InitFunction:
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
