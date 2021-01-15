import itertools
from typing import Any, Sequence

import dill
import jax.numpy as np
import pandas as pd
from jax import grad, jit, random
from jax.experimental.optimizers import Optimizer, adam

from pzflow.bijectors import InitFunction, RollingSplineCoupling
from pzflow.utils import Normal


class Flow:
    """A normalizing flow that models galaxy magnitudes and redshifts.

    Attributes
    ----------
    data_columns : tuple
        List of DataFrame columns that the flow expects/produces.
    info : Any
        Object containing any kind of info included with the flow.
        Often describes the data the flow is trained on.
    prior : pzflow.utils.Normal
        The Gaussian distribution the flow samples from before the bijection.
        Has it's own sample and log_prob methods.
    """

    def __init__(
        self,
        data_columns: Sequence[str] = None,
        bijector: InitFunction = None,
        info: Any = None,
        file: str = None,
    ):
        """
        Note that while all of the init parameters are technically optional,
        you must provide either data_columns and bijector OR file.
        In addition, if a file is provided, all other parameters must be None.

        Parameters
        ----------
        data_columns : Sequence[str], optional
            Tuple, list, or other container of column names.
            These are the columns the flow expects/produces in DataFrames.
        bijector : InitFunction, optional
            A bijector InitFunction that initializes the bijector that
            maps the Gaussian prior onto the data distribution.
            Can be the output of any Bijector, e.g. Reverse(), Chain(...), etc.
        info : Any, optional
            An object to attach to the info attribute
        file : str, optional
            Path to file from which to load a pretrained flow.
            If a file is provided, all other parameters must be None.
        """

        # validate parameters
        if data_columns is None and bijector is None and file is None:
            raise ValueError("You must provide data_columns and bijector OR file.")
        elif data_columns is not None and bijector is None:
            raise ValueError("Please also provide a bijector.")
        elif data_columns is None and bijector is not None:
            raise ValueError("Please also provide data_columns.")
        elif file is not None and any(
            (data_columns != None, bijector != None, info != None)
        ):
            raise ValueError(
                "If providing a file, please do not provide any other parameters."
            )

        # if file is provided, load everything from the file
        if file is not None:
            with open(file, "rb") as handle:
                save_dict = dill.load(handle)
            # load params from the file
            self.data_columns = save_dict["data_columns"]
            self._input_dim = len(self.data_columns)
            self.info = save_dict["info"]
            self._bijector = save_dict["bijector"]
            self._params = save_dict["params"]
            # initialize the forward and inverse functions
            _, self._forward, self._inverse = self._bijector(
                random.PRNGKey(0), self._input_dim
            )
        # if no file is provided, use provided parameters
        else:
            self.data_columns = tuple(data_columns)
            self._input_dim = len(self.data_columns)
            self.info = info
            self._bijector = bijector
            # initialize the bijector with random params
            self._params, self._forward, self._inverse = self._bijector(
                random.PRNGKey(0), self._input_dim
            )

        # use a standard Gaussian as the prior distribution
        self.prior = Normal(self._input_dim)

    def log_prob(self, inputs: pd.DataFrame) -> np.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        inputs : pd.DataFrame
            Input data for which log probability density is calculated.
            Every column in self.data_columns must be present.
            If other columns are present, they are ignored.

        Returns
        -------
        np.ndarray
            Device array of shape (inputs.shape[0],).
        """
        # convert data to an array with columns ordered
        columns = list(self.data_columns)
        inputs = np.array(inputs[columns].values)
        # calculate log_prob
        u, log_det = self._inverse(self._params, inputs)
        log_prob = self.prior.log_prob(u) + log_det
        # set NaN's to negative infinity (i.e. zero probability)
        log_prob = np.nan_to_num(log_prob, nan=np.NINF)
        return log_prob

    def posterior(
        self,
        inputs: pd.DataFrame,
        column: str,
        grid: np.ndarray,
    ) -> np.ndarray:
        """Calculates posterior distributions for the provided column.

        Calculates the conditional posterior distribution, assuming the
        data values in the other columns of the DataFrame.

        Parameters
        ----------
        inputs : pd.DataFrame
            Data on which the posterior distributions are conditioned.
            Must have columns matching self.data_columns, *except*
            for the column specified for the posterior (see below).
        column : str
            Name of the column for which the posterior distribution
            is calculated. Must be one of the columns in self.data_columns.
            However, whether or not this column is one of the columns in
            `inputs` is irrelevant.
        grid : np.ndarray
            Grid on which to calculate the posterior.

        Returns
        -------
        np.ndarray
            Device array of shape (inputs.shape[0], grid.size).
        """

        # get the index of the provided column, and remove it from the list
        columns = list(self.data_columns)
        idx = columns.index(column)
        columns.remove(column)

        # convert data (sans the provided column)
        # to an array with columns ordered
        inputs = np.array(inputs[columns].values)

        nrows = inputs.shape[0]

        # make a new copy of each row for each value of the column
        # for which we are calculating the posterior
        inputs = np.hstack(
            (
                np.repeat(inputs[:, :idx], len(grid), axis=0),
                np.tile(grid, nrows)[:, None],
                np.repeat(inputs[:, idx:], len(grid), axis=0),
            )
        )

        # calculate probability densities
        u, log_det = self._inverse(self._params, inputs)
        log_prob = self.prior.log_prob(u)
        log_prob = np.nan_to_num(log_prob + log_det, nan=np.NINF)
        pdfs = np.exp(log_prob)

        # reshape so that each row is a posterior
        pdfs = pdfs.reshape((nrows, len(grid)))
        # normalize so they integrate to one
        pdfs = pdfs / np.trapz(y=pdfs, x=grid).reshape(-1, 1)
        # set NaN's equal to zero probability
        pdfs = np.nan_to_num(pdfs, nan=0.0)
        return pdfs

    def sample(self, nsamples: int = 1, seed: int = None) -> pd.DataFrame:
        """Returns samples from the normalizing flow.

        Parameters
        ----------
        nsamples : int, default=1
            The number of samples to be returned.
        seed : int, optional
            Sets the random seed for the samples.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame with columns flow.data_columns and
            number of rows equal to nsamples.
        """
        u = self.prior.sample(nsamples, seed)
        x = self._forward(self._params, u)[0]
        x = pd.DataFrame(x, columns=self.data_columns)
        return x

    def save(self, file: str):
        """Saves the flow to a file.

        Pickles the flow and saves it to a file that can be passed as
        the `file` argument during flow instantiation.

        Parameters
        ----------
        file : str
            Path to where the flow will be saved.
            Extension `.dill` will be appended if not already present.
        """
        save_dict = {
            "data_columns": self.data_columns,
            "info": self.info,
            "bijector": self._bijector,
            "params": self._params,
        }
        if not file.endswith(".dill"):
            file += ".dill"
        with open(file, "wb") as handle:
            dill.dump(save_dict, handle, recurse=True)

    def train(
        self,
        inputs: pd.DataFrame,
        epochs: int = 25,
        batch_size: int = 1024,
        optimizer: Optimizer = None,
        seed: int = 0,
        verbose: bool = False,
    ) -> list:
        """Trains the normalizing flow on the provided inputs.

        Calculates the conditional posterior distribution, assuming the
        data values in the other columns of the DataFrame.

        Parameters
        ----------
        inputs : pd.DataFrame
            Data on which to train the normalizing flow.
            Must have columns matching self.data_columns.
        epochs : int, default=25
            Number of epochs to train.
        batch_size : int, default=1024
            Batch size for training.
        optimizer: jax Optimizer, default=adam(step_size=1e-3)
            An optimizer from jax.experimental.optimizers.
        seed : int, default=0
            A random seed to control the batching.
        verbose : bool, default=False
            If true, print the training loss every 5% of epochs.

        Returns
        -------
        list
            List of training losses from every epoch.
        """

        # convert data to an array with columns ordered
        columns = list(self.data_columns)
        inputs = np.array(inputs[columns].values)

        # initialize the optimizer
        optimizer = adam(step_size=1e-3) if optimizer is None else optimizer
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

        # save the initial loss
        losses = [loss(self._params, inputs)]
        if verbose:
            print(f"{losses[-1]:.4f}")

        # loop through training
        itercount = itertools.count()
        rng = random.PRNGKey(seed)
        for epoch in range(epochs):
            # new permutation of batches
            permute_rng, rng = random.split(rng)
            X = random.permutation(permute_rng, inputs)
            # loop through batches and step optimizer
            for batch_idx in range(0, len(X), batch_size):
                opt_state = step(
                    next(itercount), opt_state, X[batch_idx : batch_idx + batch_size]
                )

            # save end-of-epoch training loss
            params = get_params(opt_state)
            losses.append(loss(params, inputs))

            if verbose and epoch % max(int(0.05 * epochs), 1) == 0:
                print(f"{losses[-1]:.4f}")

        # update the flow parameters with the final training state
        self._params = get_params(opt_state)
        return losses
