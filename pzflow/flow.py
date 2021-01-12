import itertools
from typing import Callable

import dill
import jax.numpy as np
from jax import grad, jit, random
from jax.experimental.optimizers import Optimizer, adam

from pzflow.bijectors import RollingSplineCoupling
from pzflow.utils import Normal


class Flow:
    def __init__(
        self, input_dim: int = None, bijector: Callable = None, file: str = None
    ):

        if input_dim is None and file is None:
            raise ValueError("User must provide either input_dim or file")

        if file is not None and any((input_dim != None, bijector != None)):
            raise ValueError(
                "If file is provided, please do not provide input_dim or bijector"
            )

        if file is not None:
            with open(file, "rb") as handle:
                save_dict = dill.load(handle)
            self.input_dim = save_dict["input_dim"]
            self._bijector = save_dict["bijector"]
            self.params = save_dict["params"]
            _, forward_fun, inverse_fun = self._bijector(
                random.PRNGKey(0), self.input_dim
            )
        elif isinstance(input_dim, int) and input_dim > 0:
            self.input_dim = input_dim
            self._bijector = (
                RollingSplineCoupling(self.input_dim) if bijector is None else bijector
            )
            self.params, forward_fun, inverse_fun = self._bijector(
                random.PRNGKey(0), input_dim
            )
        else:
            raise ValueError("input_dim must be a positive integer")

        self._forward = forward_fun
        self._inverse = inverse_fun

        self.prior = Normal(self.input_dim)

    def save(self, file: str):
        save_dict = {
            "input_dim": self.input_dim,
            "bijector": self._bijector,
            "params": self.params,
        }
        with open(file, "wb") as handle:
            dill.dump(save_dict, handle, recurse=True)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return self._forward(self.params, inputs)[0]

    def inverse(self, inputs: np.ndarray) -> np.ndarray:
        return self._inverse(self.params, inputs)[0]

    def sample(self, nsamples: int = 1, seed: int = None) -> np.ndarray:
        u = self.prior.sample(nsamples, seed)
        x = self.forward(u)
        return x

    def log_prob(self, inputs: np.ndarray) -> np.ndarray:
        u, log_det = self._inverse(self.params, inputs)
        log_prob = self.prior.log_prob(u)
        return log_prob + log_det

    def pz_estimate(self, inputs: np.ndarray, zmin=0, zmax=2, dz=0.02) -> np.ndarray:
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
        inputs: np.ndarray,
        epochs: int = 200,
        batch_size: int = 512,
        optimizer: Optimizer = adam(step_size=1e-3),
        seed: int = 0,
        verbose: bool = False,
    ) -> list:

        opt_init, opt_update, get_params = optimizer
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
