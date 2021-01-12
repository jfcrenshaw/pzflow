import jax.numpy as np
import numpy as onp
from jax import random
from jax.scipy.stats import multivariate_normal


class Normal:
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
