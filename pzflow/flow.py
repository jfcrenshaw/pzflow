import jax.numpy as np
from jax import random, grad, jit
from jax.experimental import optimizers
from jax.scipy.stats import multivariate_normal
import numpy as onp
import itertools
from pzflow.bijectors import DefaultBijector
from typing import Callable


class _Normal:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def log_prob(self, inputs: np.ndarray):
        return multivariate_normal.logpdf(
            x=inputs,
            mean=np.zeros(self.input_dim),
            cov=np.identity(self.input_dim),
        )

    def sample(self, nsamples: int, seed: int = None):
        seed = onp.random.randint(1e18) if seed is None else seed
        return random.multivariate_normal(
            key=random.PRNGKey(seed),
            mean=np.zeros(self.input_dim),
            cov=np.identity(self.input_dim),
            shape=(nsamples,),
        )


class Flow:
    def __init__(self, input_dim: int, bijector: Callable = None, file: str = None):

        # add some checks here to make sure that input_dim or file is supplied

        bijector = DefaultBijector(input_dim) if bijector is None else bijector
        self.input_dim = input_dim

        self.prior = _Normal(input_dim)

        params, forward_fun, inverse_fun = bijector(random.PRNGKey(0), input_dim)
        self.params = params
        self._forward = forward_fun
        self._inverse = inverse_fun

    def forward(self, inputs):
        return self._forward(self.params, inputs)[0]

    def inverse(self, inputs):
        return self._inverse(self.params, inputs)[0]

    def sample(self, nsamples: int = 1, seed: int = None):
        u = self.prior.sample(nsamples, seed)
        x = self.forward(u)
        return x

    def log_prob(self, inputs: np.ndarray):
        u, log_det = self._inverse(self.params, inputs)
        log_prob = self.prior.log_prob(u)
        return log_prob + log_det

    def pz_estimate(self, inputs, zmin=0, zmax=2, dz=0.02):
        zs = np.arange(zmin, zmax + dz, dz)
        X = np.stack(
            (
                np.tile(zs, inputs.shape[0]),
                *[np.repeat(inputs[:, i], len(zs)) for i in range(1, inputs.shape[1])],
            ),
            axis=-1,
        )
        log_prob = self.log_prob(X).reshape((inputs.shape[0], len(zs)))
        pdfs = np.exp(log_prob)
        pdfs = pdfs / (pdfs * dz).sum(axis=1).reshape(-1, 1)
        return pdfs

    def train(
        self,
        inputs,
        epochs: int = 200,
        batch_size: int = 512,
        step_size: float = 1e-3,
        seed: int = 0,
        verbose: bool = False,
    ):

        opt_init, opt_update, get_params = optimizers.adam(step_size=step_size)
        opt_state = opt_init(self.params)

        @jit
        def loss(params, inputs):
            u, log_det = self._inverse(params, inputs)
            log_prob = self.prior.log_prob(u)
            return -np.mean(log_prob + log_det)

        @jit
        def step(i, opt_state, inputs):
            params = get_params(opt_state)
            gradients = grad(loss)(params, inputs)
            return opt_update(i, gradients, opt_state)

        losses = []
        itercount = itertools.count()
        rng = random.PRNGKey(seed)
        for epoch in range(epochs):
            permute_rng, rng = random.split(rng)
            X = random.permutation(permute_rng, inputs)
            for batch_idx in range(0, len(X), batch_size):
                opt_state = step(
                    next(itercount), opt_state, X[batch_idx : batch_idx + batch_size]
                )

            params = get_params(opt_state)
            losses.append(loss(params, inputs))

            if verbose and epoch % int(0.05 * epochs) == 0:
                print(f"{losses[-1]:.4f}")

        self.params = get_params(opt_state)
        return losses

    def save(self):
        # this will save the flow to a file that can be loaded during
        # flow construction
        # make sure that only a file or input dim/bijector are supplied
        pass