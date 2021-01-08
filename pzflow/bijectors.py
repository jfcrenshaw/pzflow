import jax.numpy as np
from jax import random, ops


def Scale(scale):
    def init_fun(rng, input_dim, **kwargs):
        def forward_fun(params, inputs, **kwargs):
            return scale * inputs, np.log(np.abs(scale)) * np.ones(inputs.shape[0])

        def inverse_fun(params, inputs, **kwargs):
            return inputs / scale, np.log(np.abs(1 / scale)) * np.ones(inputs.shape[0])

        return (), forward_fun, inverse_fun

    return init_fun


def Reverse():
    def init_fun(rng, input_dim, **kwargs):
        def forward_fun(params, inputs, **kwargs):
            return inputs[:, ::-1], np.zeros(inputs.shape[0])

        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, ::-1], np.zeros(inputs.shape[0])

        return (), forward_fun, inverse_fun

    return init_fun


def Shuffle():
    def init_fun(rng, input_dim, **kwargs):

        perm = random.permutation(rng, np.arange(input_dim))
        inv_perm = np.argsort(perm)

        def forward_fun(params, inputs, **kwargs):
            return inputs[:, perm], np.zeros(inputs.shape[0])

        def inverse_fun(params, inputs, **kwargs):
            return inputs[:, inv_perm], np.zeros(inputs.shape[0])

        return (), forward_fun, inverse_fun

    return init_fun


def Roll(shift=1):
    def init_fun(rng, input_dim, **kwargs):
        def forward_fun(params, inputs, **kwargs):
            return np.roll(inputs, shift=shift, axis=-1), np.zeros(inputs.shape[0])

        def inverse_fun(params, inputs, **kwargs):
            return np.roll(inputs, shift=-shift, axis=-1), np.zeros(inputs.shape[0])

        return (), forward_fun, inverse_fun

    return init_fun


def ColorTransform(ref_idx, ref_mean, ref_stdd):
    def init_fun(rng, input_dim, **kwargs):
        def forward_fun(params, inputs, **kwargs):
            # calculate reference magnitude,
            # and convert all colors to be in terms of the first magnitude, mag[0]
            outputs = np.hstack(
                (
                    inputs[:, 0, None],  # redshift unchanged
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

            log_det = 1 / np.log(1 / ref_stdd) * np.ones(data.shape[0])

            return outputs, log_det

        def inverse_fun(params, inputs, **kwargs):
            outputs = np.hstack(
                (
                    inputs[:, 0, None],  # redshift
                    (inputs[:, ref_idx, None] - ref_mean) / ref_stdd,  # ref mag
                    -np.diff(inputs[:, 1:]),  # colors
                )
            )
            log_det = np.log(1 / ref_stdd) * np.ones(data.shape[0])
            return outputs, log_det

        return (), forward_fun, inverse_fun

    return init_fun


def Chain(*init_funs):
    def init_fun(rng, input_dim, **kwargs):

        all_params, forward_funs, inverse_funs = [], [], []
        for init_f in init_funs:
            rng, layer_rng = random.split(rng)
            param, forward_fun, inverse_fun = init_f(layer_rng, input_dim)

            all_params.append(param)
            forward_funs.append(forward_fun)
            inverse_funs.append(inverse_fun)

        def bijector_chain(params, bijectors, inputs):
            log_dets = np.zeros(inputs.shape[0])
            for bijector, param in zip(bijectors, params):
                inputs, log_det = bijector(param, inputs, **kwargs)
                log_dets += log_det
            return inputs, log_det

        def forward_fun(params, inputs, **kwargs):
            return bijector_chain(params, forward_funs, inputs)

        def inverse_fun(params, inputs, **kwargs):
            return bijector_chain(params[::-1], inverse_funs[::-1], inputs)

        return all_params, forward_fun, inverse_fun

    return init_fun