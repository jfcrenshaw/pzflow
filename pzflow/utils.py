import jax.numpy as np
import numpy as onp
from jax import random
from jax.scipy.stats import multivariate_normal
from numbers import Number
import pandas as pd


class Normal:
    def __init__(self, input_dim: int):
        self.input_dim = input_dim

    def log_prob(self, inputs: np.ndarray) -> np.ndarray:
        return multivariate_normal.logpdf(
            x=inputs,
            mean=np.zeros(self.input_dim),
            cov=np.identity(self.input_dim),
        )

    def sample(self, nsamples: int, seed: int = None) -> np.ndarray:
        seed = onp.random.randint(1e18) if seed is None else seed
        return random.multivariate_normal(
            key=random.PRNGKey(seed),
            mean=np.zeros(self.input_dim),
            cov=np.identity(self.input_dim),
            shape=(nsamples,),
        )


class LSSTErrorModel:
    def __init__(
        self,
        limiting_mags: dict = None,
        err_params: dict = None,
        undetected_flag: int = 99,
    ):

        if isinstance(limiting_mags, dict):
            self.limiting_mags = limiting_mags
        elif limiting_mags is not None:
            raise ValueError("limiting_mags must be a dictionary")
        else:
            # defaults are 10 year 5-sigma point source depth
            # from https://www.lsst.org/scientists/keynumbers
            self.limiting_mags = {
                "u": 26.1,
                "g": 27.4,
                "r": 27.5,
                "i": 26.8,
                "z": 26.1,
                "y": 27.9,
            }

        if isinstance(err_params, dict):
            self.err_params = err_params
        elif err_params is not None:
            raise ValueError("err_params must be a dictionary")
        else:
            # defaults are gamma values in Table 2
            # from https://arxiv.org/pdf/0805.2366.pdf
            self.err_params = {
                "u": 0.038,
                "g": 0.039,
                "r": 0.039,
                "i": 0.039,
                "z": 0.039,
                "y": 0.039,
            }

        self.undetected_flag = undetected_flag

        # check that the keys match
        err_str = "limiting_mags and err_params have different keys"
        assert self.limiting_mags.keys() == self.err_params.keys(), err_str

        # check that all values are numbers
        all_numbers = all(
            isinstance(val, Number) for val in self.limiting_mags.values()
        )
        err_str = "All limiting magnitudes must be numbers"
        assert all_numbers, err_str
        all_numbers = all(isinstance(val, Number) for val in self.err_params.values())
        err_str = "All error parameters must be numbers"
        assert all_numbers, err_str
        all_numbers = isinstance(self.undetected_flag, Number)
        err_str = (
            "The undetected flag for mags beyond the 5-sigma limit must be a number"
        )
        assert all_numbers, err_str

    def __call__(self, data: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        # Gaussian errors using Equation 5
        # from https://arxiv.org/pdf/0805.2366.pdf
        # then flag all magnitudes beyond 5-sig limit

        data = data.copy()

        rng = onp.random.default_rng(seed)

        for band in self.limiting_mags.keys():

            # calculate err with Eq 5
            m5 = self.limiting_mags[band]
            gamma = self.err_params[band]
            x = 10 ** (0.4 * (data[band] - m5))
            err = onp.sqrt((0.04 - gamma) * x + gamma * x ** 2)

            # Add errs to galaxies within limiting mag
            data[f"{band}_err"] = err
            rand_err = rng.normal(0, err)
            rand_err[data[band] > m5] = 0
            data[band] += rand_err

            # flag mags beyond limiting mag
            data.loc[
                data.eval(f"{band} > {m5}"), (band, f"{band}_err")
            ] = self.undetected_flag

        return data
