import itertools
from typing import Callable, Any, Sequence

import dill
import jax.numpy as np
from jax import grad, jit, random, ops
from jax.experimental.optimizers import Optimizer, adam
import pandas as pd

from pzflow.bijectors import RollingSplineCoupling
from pzflow.utils import Normal


class Flow:
    def __init__(
        self,
        data_columns: Sequence = None,
        bijector: Callable = None,
        file: str = None,
        info: Any = None,
    ):

        if data_columns is None and file is None:
            raise ValueError("User must provide either data_columns or file")

        if file is not None and any((data_columns != None, bijector != None)):
            raise ValueError(
                "If file is provided, please do not provide data_columns or bijector"
            )

        if file is not None:
            with open(file, "rb") as handle:
                save_dict = dill.load(handle)
            self.data_columns = save_dict["data_columns"]
            self._input_dim = len(self.data_columns)
            self.info = save_dict["info"]
            self._bijector = save_dict["bijector"]
            self._params = save_dict["params"]
            _, self._forward, self._inverse = self._bijector(
                random.PRNGKey(0), self._input_dim
            )
        else:
            self.data_columns = tuple(data_columns)
            self._input_dim = len(self.data_columns)
            self.info = info
            self._bijector = (
                RollingSplineCoupling(self._input_dim) if bijector is None else bijector
            )
            self._params, self._forward, self._inverse = self._bijector(
                random.PRNGKey(0), self._input_dim
            )

        self.prior = Normal(self._input_dim)

    def save(self, file: str):
        save_dict = {
            # "input_dim": self.input_dim,
            "data_columns": self.data_columns,
            "info": self.info,
            "bijector": self._bijector,
            "params": self._params,
        }
        with open(file, "wb") as handle:
            dill.dump(save_dict, handle, recurse=True)

    def sample(self, nsamples: int = 1, seed: int = None) -> np.ndarray:
        u = self.prior.sample(nsamples, seed)
        x = self._forward(self._params, u)[0]
        x = pd.DataFrame(x, columns=self.data_columns)
        return x

    def log_prob(self, inputs: pd.DataFrame) -> np.ndarray:
        columns = list(self.data_columns)
        inputs = np.array(inputs[columns].values)
        u, log_det = self._inverse(self._params, inputs)
        log_prob = self.prior.log_prob(u)
        return np.nan_to_num(log_prob + log_det, nan=np.NINF)

    def posterior(
        self,
        inputs: pd.DataFrame,
        column: str = "redshift",
        grid: np.ndarray = np.arange(0, 2.02, 0.02),
    ) -> np.ndarray:

        columns = list(self.data_columns)
        idx = columns.index(column)
        columns.remove(column)

        inputs = np.array(inputs[columns].values)

        nrows = inputs.shape[0]

        inputs = np.hstack(
            (
                np.repeat(inputs[:, :idx], len(grid), axis=0),
                np.tile(grid, nrows)[:, None],
                np.repeat(inputs[:, idx:], len(grid), axis=0),
            )
        )

        u, log_det = self._inverse(self._params, inputs)
        log_prob = self.prior.log_prob(u)
        log_prob = np.nan_to_num(log_prob + log_det, nan=np.NINF)
        log_prob = log_prob.reshape((nrows, len(grid)))

        pdfs = np.exp(log_prob)
        pdfs = pdfs / np.trapz(y=pdfs, x=grid).reshape(-1, 1)

        return np.nan_to_num(pdfs, nan=0.0)

    def train(
        self,
        inputs: pd.DataFrame,
        epochs: int = 200,
        batch_size: int = 512,
        optimizer: Optimizer = adam(step_size=1e-3),
        seed: int = 0,
        verbose: bool = False,
    ) -> list:

        columns = list(self.data_columns)
        inputs = np.array(inputs[columns].values)

        opt_init, opt_update, get_params = optimizer
        opt_state = opt_init(self._params)

        @jit
        def loss(params, x):
            u, log_det = self._inverse(params, x)
            log_prob = self.prior.log_prob(u)
            return -np.mean(log_prob + log_det)

        @jit
        def step(i, opt_state, x):
            params = get_params(opt_state)
            gradients = grad(loss)(params, x)
            return opt_update(i, gradients, opt_state)

        losses = [loss(self._params, inputs)]
        if verbose:
            print(f"{losses[-1]:.4f}")

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

            if verbose and epoch % max(int(0.05 * epochs), 1) == 0:
                print(f"{losses[-1]:.4f}")

        self._params = get_params(opt_state)
        return losses
