"""Define the latent distributions used in the normalizing flows."""
import sys
from abc import ABC, abstractmethod
from typing import Union

import jax.numpy as jnp
import numpy as np
from jax import random
from jax.scipy.special import gammaln
from jax.scipy.stats import beta, multivariate_normal

from pzflow.bijectors import Pytree

epsilon = sys.float_info.epsilon


class LatentDist(ABC):
    """Base class for latent distributions."""

    info = ("LatentDist", ())

    @abstractmethod
    def log_prob(self, params: Pytree, inputs: jnp.ndarray) -> jnp.ndarray:
        """Calculate log-probability of the inputs."""

    @abstractmethod
    def sample(
        self, params: Pytree, nsamples: int, seed: int = None
    ) -> jnp.ndarray:
        """Sample from the distribution."""


def _mahalanobis_and_logdet(x: jnp.array, cov: jnp.array) -> tuple:
    # Calculate mahalanobis distance and log_det of cov.
    # Uses scipy method, explained here:
    # http://gregorygundersen.com/blog/2019/10/30/scipy-multivariate/

    vals, vecs = jnp.linalg.eigh(cov)
    U = vecs * jnp.sqrt(1 / vals[..., None])
    maha = jnp.square(U @ x[..., None]).reshape(x.shape[0], -1).sum(axis=1)
    log_det = jnp.log(vals).sum(axis=-1)
    return maha, log_det


class CentBeta(LatentDist):
    """A centered Beta distribution.

    This distribution is just a regular Beta distribution, scaled and shifted
    to have support on the domain [-B, B] in each dimension.

    Alpha and beta parameters for each dimension are learned during training.
    """

    def __init__(self, input_dim: int, B: float = 5) -> None:
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the distribution.
        B : float; default=5
            The distribution has support (-B, B) along each dimension.
        """
        self.input_dim = input_dim
        self.B = B

        # save dist info
        self._params = tuple([(0.0, 0.0) for i in range(input_dim)])
        self.info = ("CentBeta", (input_dim, B))

    def log_prob(self, params: Pytree, inputs: jnp.ndarray) -> jnp.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        params : a Jax pytree
            Tuple of ((a1, b1), (a2, b2), ...) where aN,bN are log(alpha),log(beta)
            for the Nth dimension.
        inputs : jnp.ndarray
            Input data for which log probability density is calculated.

        Returns
        -------
        jnp.ndarray
            Device array of shape (inputs.shape[0],).
        """
        log_prob = jnp.hstack(
            [
                beta.logpdf(
                    inputs[:, i],
                    a=jnp.exp(params[i][0]),
                    b=jnp.exp(params[i][1]),
                    loc=-self.B,
                    scale=2 * self.B,
                ).reshape(-1, 1)
                for i in range(self.input_dim)
            ]
        ).sum(axis=1)

        return log_prob

    def sample(
        self, params: Pytree, nsamples: int, seed: int = None
    ) -> jnp.ndarray:
        """Returns samples from the distribution.

        Parameters
        ----------
        params : a Jax pytree
            Tuple of ((a1, b1), (a2, b2), ...) where aN,bN are log(alpha),log(beta)
            for the Nth dimension.
        nsamples : int
            The number of samples to be returned.
        seed : int; optional
            Sets the random seed for the samples.

        Returns
        -------
        jnp.ndarray
            Device array of shape (nsamples, self.input_dim).
        """
        seed = np.random.randint(1e18) if seed is None else seed
        seeds = random.split(random.PRNGKey(seed), self.input_dim)
        samples = jnp.hstack(
            [
                random.beta(
                    seeds[i],
                    jnp.exp(params[i][0]),
                    jnp.exp(params[i][1]),
                    shape=(nsamples, 1),
                )
                for i in range(self.input_dim)
            ]
        )
        return 2 * self.B * (samples - 0.5)

class CentBeta13(LatentDist):
    """A centered Beta distribution with alpha, beta = 13.

    This distribution is just a regular Beta distribution, scaled and shifted
    to have support on the domain [-B, B] in each dimension.

    Alpha, beta = 13 means that the distribution looks like a Gaussian
    distribution, but with hard cutoffs at +/- B.
    """

    def __init__(self, input_dim: int, B: float = 5) -> None:
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the distribution.
        B : float; default=5
            The distribution has support (-B, B) along each dimension.
        """
        self.input_dim = input_dim
        self.B = B

        # save dist info
        self._params = tuple([(0.0, 0.0) for i in range(input_dim)])
        self.info = ("CentBeta22", (input_dim, B))
        self.a = 13
        self.b = 13

    def log_prob(self, params: Pytree, inputs: jnp.ndarray) -> jnp.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        params : a Jax pytree
            Empty pytree -- this distribution doesn't have learnable parameters.
            This parameter is present to ensure a consistent interface.
        inputs : jnp.ndarray
            Input data for which log probability density is calculated.

        Returns
        -------
        jnp.ndarray
            Device array of shape (inputs.shape[0],).
        """
        log_prob = jnp.hstack(
            [
                beta.logpdf(
                    inputs[:, i],
                    a=self.a,
                    b=self.b,
                    loc=-self.B,
                    scale=2 * self.B,
                ).reshape(-1, 1)
                for i in range(self.input_dim)
            ]
        ).sum(axis=1)

        return log_prob

    def sample(
        self, params: Pytree, nsamples: int, seed: int = None
    ) -> jnp.ndarray:
        """Returns samples from the distribution.

        Parameters
        ----------
        params : a Jax pytree
            Empty pytree -- this distribution doesn't have learnable parameters.
            This parameter is present to ensure a consistent interface.
        nsamples : int
            The number of samples to be returned.
        seed : int; optional
            Sets the random seed for the samples.

        Returns
        -------
        jnp.ndarray
            Device array of shape (nsamples, self.input_dim).
        """
        seed = np.random.randint(1e18) if seed is None else seed
        seeds = random.split(random.PRNGKey(seed), self.input_dim)
        samples = jnp.hstack(
            [
                random.beta(
                    seeds[i],
                    self.a,
                    self.b,
                    shape=(nsamples, 1),
                )
                for i in range(self.input_dim)
            ]
        )
        return 2 * self.B * (samples - 0.5)



class Normal(LatentDist):
    """A multivariate Gaussian distribution with mean zero and unit variance.

    Note this distribution has infinite support, so it is not recommended that
    you use it with the spline coupling layers, which have compact support.
    If you do use the two together, you should set the support of the spline
    layers (using the spline parameter B) to be large enough that you rarely
    draw Gaussian samples outside the support of the splines.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the distribution.
        """
        self.input_dim = input_dim

        # save dist info
        self._params = ()
        self.info = ("Normal", (input_dim,))

    def log_prob(self, params: Pytree, inputs: jnp.ndarray) -> jnp.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        params : a Jax pytree
            Empty pytree -- this distribution doesn't have learnable parameters.
            This parameter is present to ensure a consistent interface.
        inputs : jnp.ndarray
            Input data for which log probability density is calculated.

        Returns
        -------
        jnp.ndarray
            Device array of shape (inputs.shape[0],).
        """
        return multivariate_normal.logpdf(
            inputs,
            mean=jnp.zeros(self.input_dim),
            cov=jnp.identity(self.input_dim),
        )

    def sample(
        self, params: Pytree, nsamples: int, seed: int = None
    ) -> jnp.ndarray:
        """Returns samples from the distribution.

        Parameters
        ----------
        params : a Jax pytree
            Empty pytree -- this distribution doesn't have learnable parameters.
            This parameter is present to ensure a consistent interface.
        nsamples : int
            The number of samples to be returned.
        seed : int; optional
            Sets the random seed for the samples.

        Returns
        -------
        jnp.ndarray
            Device array of shape (nsamples, self.input_dim).
        """
        seed = np.random.randint(1e18) if seed is None else seed
        return random.multivariate_normal(
            key=random.PRNGKey(seed),
            mean=jnp.zeros(self.input_dim),
            cov=jnp.identity(self.input_dim),
            shape=(nsamples,),
        )


class Tdist(LatentDist):
    """A multivariate T distribution with mean zero and unit scale matrix.

    The number of degrees of freedom (i.e. the weight of the tails) is learned
    during training.

    Note this distribution has infinite support and potentially large tails,
    so it is not recommended to use this distribution with the spline coupling
    layers, which have compact support.
    """

    def __init__(self, input_dim: int) -> None:
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the distribution.
        """
        self.input_dim = input_dim

        # save dist info
        self._params = jnp.log(30.0)
        self.info = ("Tdist", (input_dim,))

    def log_prob(self, params: Pytree, inputs: jnp.ndarray) -> jnp.ndarray:
        """Calculates log probability density of inputs.

        Uses method explained here:
        http://gregorygundersen.com/blog/2020/01/20/multivariate-t/

        Parameters
        ----------
        params : float
            The degrees of freedom (nu) of the t-distribution.
        inputs : jnp.ndarray
            Input data for which log probability density is calculated.

        Returns
        -------
        jnp.ndarray
            Device array of shape (inputs.shape[0],).
        """
        cov = jnp.identity(self.input_dim)
        nu = jnp.exp(params)
        maha, log_det = _mahalanobis_and_logdet(inputs, cov)
        t = 0.5 * (nu + self.input_dim)
        A = gammaln(t)
        B = gammaln(0.5 * nu)
        C = self.input_dim / 2.0 * jnp.log(nu * jnp.pi)
        D = 0.5 * log_det
        E = -t * jnp.log(1 + (1.0 / nu) * maha)

        return A - B - C - D + E

    def sample(
        self, params: Pytree, nsamples: int, seed: int = None
    ) -> jnp.ndarray:
        """Returns samples from the distribution.

        Parameters
        ----------
        params : float
            The degrees of freedom (nu) of the t-distribution.
        nsamples : int
            The number of samples to be returned.
        seed : int; optional
            Sets the random seed for the samples.

        Returns
        -------
        jnp.ndarray
            Device array of shape (nsamples, self.input_dim).
        """
        mean = jnp.zeros(self.input_dim)
        nu = jnp.exp(params)

        seed = np.random.randint(1e18) if seed is None else seed
        rng = np.random.default_rng(int(seed))
        x = jnp.array(rng.chisquare(nu, nsamples) / nu)
        z = random.multivariate_normal(
            key=random.PRNGKey(seed),
            mean=jnp.zeros(self.input_dim),
            cov=jnp.identity(self.input_dim),
            shape=(nsamples,),
        )
        samples = mean + z / jnp.sqrt(x)[:, None]
        return samples


class Uniform(LatentDist):
    """A multivariate uniform distribution with support [-B, B]."""

    def __init__(self, input_dim: int, B: float = 5) -> None:
        """
        Parameters
        ----------
        input_dim : int
            The dimension of the distribution.
        B : float; default=5
            The distribution has support (-B, B) along each dimension.
        """
        self.input_dim = input_dim
        self.B = B

        # save dist info
        self._params = ()
        self.info = ("Uniform", (input_dim, B))

    def log_prob(self, params: Pytree, inputs: jnp.ndarray) -> jnp.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        params : Jax Pytree
            Empty pytree -- this distribution doesn't have learnable parameters.
            This parameter is present to ensure a consistent interface.
        inputs : jnp.ndarray
            Input data for which log probability density is calculated.

        Returns
        -------
        jnp.ndarray
            Device array of shape (inputs.shape[0],).
        """

        # which inputs are inside the support of the distribution
        mask = (
            ((inputs >= -self.B) & (inputs <= self.B))
            .astype(float)
            .prod(axis=-1)
        )

        # calculate log_prob
        prob = mask / (2 * self.B) ** self.input_dim
        prob = jnp.where(prob == 0, epsilon, prob)
        log_prob = jnp.log(prob)

        return log_prob

    def sample(
        self, params: Pytree, nsamples: int, seed: int = None
    ) -> jnp.ndarray:
        """Returns samples from the distribution.

        Parameters
        ----------
        params : a Jax pytree
            Empty pytree -- this distribution doesn't have learnable parameters.
            This parameter is present to ensure a consistent interface.
        nsamples : int
            The number of samples to be returned.
        seed : int; optional
            Sets the random seed for the samples.

        Returns
        -------
        jnp.ndarray
            Device array of shape (nsamples, self.input_dim).
        """
        seed = np.random.randint(1e18) if seed is None else seed
        samples = random.uniform(
            random.PRNGKey(seed),
            shape=(nsamples, self.input_dim),
            minval=-self.B,
            maxval=self.B,
        )
        return jnp.array(samples)


class Joint(LatentDist):
    """A joint distribution built from other distributions.

    Note that each of the other distributions already have support for
    multiple dimensions. This is only useful if you want to combine
    different distributions for different dimensions, e.g. if your first
    dimension has a Uniform latent space and the second dimension has a
    CentBeta latent space.
    """

    def __init__(self, *inputs: Union[LatentDist, tuple]) -> None:
        """
        Parameters
        ----------
        inputs: LatentDist or tuple
            The latent distributions to join together.
        """

        # if Joint info provided, use that for setup
        if inputs[0] == "Joint info":
            self.dists = [globals()[dist[0]](*dist[1]) for dist in inputs[1]]
        # otherwise, assume it's a list of distributions
        else:
            self.dists = inputs

        # save info
        self._params = [dist._params for dist in self.dists]
        self.input_dim = sum([dist.input_dim for dist in self.dists])
        self.info = (
            "Joint",
            ("Joint info", [dist.info for dist in self.dists]),
        )

        # save the indices at which inputs will be split for log_prob
        # they must be concretely saved ahead-of-time so that jax trace
        # works properly when jitting
        self._splits = jnp.cumsum(
            jnp.array([dist.input_dim for dist in self.dists])
        )[:-1]

    def log_prob(self, params: Pytree, inputs: jnp.ndarray) -> jnp.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        params : Jax Pytree
            Parameters for the distributions.
        inputs : jnp.ndarray
            Input data for which log probability density is calculated.

        Returns
        -------
        jnp.ndarray
            Device array of shape (inputs.shape[0],).
        """

        # split inputs for corresponding distribution
        inputs = jnp.split(inputs, self._splits, axis=1)

        # calculate log_prob with respect to each sub-distribution,
        # then sum all the log_probs for each input
        log_prob = jnp.hstack(
            [
                self.dists[i].log_prob(params[i], inputs[i]).reshape(-1, 1)
                for i in range(len(self.dists))
            ]
        ).sum(axis=1)

        return log_prob

    def sample(
        self, params: Pytree, nsamples: int, seed: int = None
    ) -> jnp.ndarray:
        """Returns samples from the distribution.

        Parameters
        ----------
        params : a Jax pytree
            Parameters for the distributions.
        nsamples : int
            The number of samples to be returned.
        seed : int; optional
            Sets the random seed for the samples.

        Returns
        -------
        jnp.ndarray
            Device array of shape (nsamples, self.input_dim).
        """

        seed = np.random.randint(1e18) if seed is None else seed
        seeds = random.randint(
            random.PRNGKey(seed), (len(self.dists),), 0, int(1e9)
        )
        samples = jnp.hstack(
            [
                self.dists[i]
                .sample(params[i], nsamples, seeds[i])
                .reshape(nsamples, -1)
                for i in range(len(self.dists))
            ]
        )

        return samples
