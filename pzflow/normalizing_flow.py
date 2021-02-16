import itertools
import pickle
from typing import Any, Callable, Sequence, Tuple

import jax.numpy as np
import pandas as pd
from jax import grad, jit, ops, random, jacfwd, vmap
from jax.experimental.optimizers import Optimizer, adam

from pzflow.bijectors import Bijector_Info, InitFunction, Pytree
from pzflow.utils import build_bijector_from_info, Normal, sub_diag_indices


class Flow:
    """A normalizing flow that models tabular data.

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
        bijector: Tuple[InitFunction, Bijector_Info] = None,
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
        bijector : Bijector Call, optional
            A Bijector call that consists of the bijector InitFunction that
            initializes the bijector and the tuple of Bijector Info.
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
                save_dict = pickle.load(handle)
            # load params from the file
            self.data_columns = save_dict["data_columns"]
            self._input_dim = len(self.data_columns)
            self.info = save_dict["info"]
            self._bijector_info = save_dict["bijector_info"]
            self._params = save_dict["params"]

            init_fun, _ = build_bijector_from_info(self._bijector_info)
            _, self._forward, self._inverse = init_fun(
                random.PRNGKey(0), self._input_dim
            )

        # if no file is provided, use provided parameters
        else:
            self.data_columns = tuple(data_columns)
            self._input_dim = len(self.data_columns)
            self.info = info
            init_fun, self._bijector_info = bijector
            # initialize the bijector with random params
            self._params, self._forward, self._inverse = init_fun(
                random.PRNGKey(0), self._input_dim
            )

        # use a standard Gaussian as the prior distribution
        self.prior = Normal(self._input_dim)

    def _array_with_errs(self, inputs: pd.DataFrame, skip: str = None):
        """Convert pandas DataFrame to Jax array with columns for errors.

        Skip can be one of the columns in self.data_columns. If provided,
        that data column isn't returned (but its zero-error column is).
        This is a useful utility for the posterior method.
        """
        X = inputs.copy()
        for col in self.data_columns:
            # if errors are provided for the column, fill in zeros
            if f"{col}_err" not in inputs.columns or col == skip:
                X[f"{col}_err"] = np.zeros(X.shape[0])
        # get list of columns in correct order
        cols_with_errs = list(self.data_columns)
        if skip is not None:
            cols_with_errs.remove(skip)
        cols_with_errs += [col + "_err" for col in self.data_columns]
        # convert to jax array
        X = np.array(X[cols_with_errs].values)
        return X

    def _Jinv(self, params: Pytree, inputs: np.ndarray) -> np.ndarray:
        """Calculates the Jacobian of the inverse bijection"""

        # calculates jacobian for a single input
        # first we define a lambda that calculates the inverse
        # (but drops the log_det). then we take the Jacobian of that
        # evaluated at the vector y. the [None, :] and .squeeze() are
        # just making sure the inputs and outputs are of the correct shape
        def J(y):
            return jacfwd(lambda x: self._inverse(params, x)[0])(y[None, :]).squeeze()

        # now we can vectorize with Jax and apply to whole set of inputs at once
        return vmap(J)(inputs)

    def _log_prob(self, params: Pytree, inputs: np.ndarray) -> np.ndarray:
        """Log prob for arrays."""
        # calculate log_prob
        u, log_det = self._inverse(params, inputs)
        log_prob = self.prior.log_prob(u) + log_det
        # set NaN's to negative infinity (i.e. zero probability)
        log_prob = np.nan_to_num(log_prob, nan=np.NINF)
        return log_prob

    def _log_prob_convolved(self, params: Pytree, inputs: np.ndarray) -> np.ndarray:
        """Log prob for arrays, with error convolution"""

        # separate data from data errs
        ncols = len(self.data_columns)
        X, Xerr = inputs[:, :ncols], inputs[:, ncols:]

        # inverse and log determinant
        u, log_det = self._inverse(params, X)

        # Jacobian of inverse bijection
        Jinv = self._Jinv(params, X)
        # calculate modified covariances
        sig_u = Jinv @ (Xerr[..., None] * Jinv.transpose((0, 2, 1)))
        # add identity matrix to each covariance matrix
        idx = sub_diag_indices(sig_u)
        sig = ops.index_update(sig_u, idx, sig_u[idx] + 1)

        # calculate log_prob w.r.t the prior, with the new covariances
        log_prob = self.prior.log_prob(u, sig) + log_det
        # set NaN's to negative infinity (i.e. zero probability)
        log_prob = np.nan_to_num(log_prob, nan=np.NINF)
        return log_prob

    def log_prob(self, inputs: pd.DataFrame, convolve_err=False) -> np.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        inputs : pd.DataFrame
            Input data for which log probability density is calculated.
            Every column in self.data_columns must be present.
            If other columns are present, they are ignored.
        convolve_err : boolean, default=False
            Whether to analytically convolve Gaussian errors.
            Looks for in `inputs` for columns with names ending in `_err`.
            I.e., the error for column `u` needs to be in the column `u_err`.
            Zero error assumed for any missing error columns.

        Returns
        -------
        np.ndarray
            Device array of shape (inputs.shape[0],).
        """
        if not convolve_err:
            # convert data to an array with columns ordered
            columns = list(self.data_columns)
            X = np.array(inputs[columns].values)
            return self._log_prob(self._params, X)
        else:
            # convert data to an array with columns ordered
            X = self._array_with_errs(inputs)
            return self._log_prob_convolved(self._params, X)

    def posterior(
        self,
        inputs: pd.DataFrame,
        column: str,
        grid: np.ndarray,
        convolve_err: bool = False,
        batch_size: int = None,
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
        convolve_err : boolean, default=False
            Whether to analytically convolve Gaussian errors in the posterior.
            Looks for in `inputs` for columns with names ending in `_err`.
            I.e., the error for column `u` needs to be in the column `u_err`.
            Zero error assumed for any missing error columns.
        batch_size : int, default=None
            Size of batches in which to calculate posteriors. If None, all
            posteriors are calculated simultaneously. Simultaneous calculation
            is faster, but memory intensive for large data sets.

        Returns
        -------
        np.ndarray
            Device array of shape (inputs.shape[0], grid.size).
        """

        # get the index of the provided column, and remove it from the list
        columns = list(self.data_columns)
        idx = columns.index(column)
        columns.remove(column)

        nrows = inputs.shape[0]
        batch_size = nrows if batch_size is None else batch_size

        # convert data (sans the provided column) to array with columns ordered
        # and alias the required log_prob function
        if convolve_err:
            X = self._array_with_errs(inputs, skip=column)
            log_prob_fun = self._log_prob_convolved
        else:
            X = np.array(inputs[columns].values)
            log_prob_fun = self._log_prob

        pdfs = np.zeros((nrows, len(grid)))

        for batch_idx in range(0, nrows, batch_size):

            batch = X[batch_idx : batch_idx + batch_size]

            # make a new copy of each row for each value of the column
            # for which we are calculating the posterior
            batch = np.hstack(
                (
                    np.repeat(
                        batch[:, :idx],
                        len(grid),
                        axis=0,
                    ),
                    np.tile(grid, len(batch))[:, None],
                    np.repeat(
                        batch[:, idx:],
                        len(grid),
                        axis=0,
                    ),
                )
            )

            # calculate probability densities

            log_prob = log_prob_fun(self._params, batch).reshape((-1, len(grid)))
            pdfs = ops.index_update(
                pdfs,
                ops.index[batch_idx : batch_idx + batch_size, :],
                np.exp(log_prob),
                indices_are_sorted=True,
                unique_indices=True,
            )

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

        WARNING: Currently, this method only works for bijectors that are
        implemented in the `bijectors` module. If you want to save a flow
        with a custom bijector, you either need to add the it to that
        module, or handle the saving and loading on your end.

        Parameters
        ----------
        file : str
            Path to where the flow will be saved.
            Extension `.pkl` will be appended if not already present.
        """
        save_dict = {
            "data_columns": self.data_columns,
            "info": self.info,
            "bijector_info": self._bijector_info,
            "params": self._params,
        }
        if not file.endswith(".pkl"):
            file += ".pkl"
        with open(file, "wb") as handle:
            pickle.dump(save_dict, handle)

    def _train(
        self,
        inputs: np.ndarray,
        epochs: int,
        batch_size: int,
        optimizer: Optimizer,
        loss_fn: Callable,
        rng: np.ndarray,
        verbose: bool,
    ) -> list:
        """Private training loop that is called by the public train method."""

        # initialize the optimizer
        optimizer = adam(step_size=1e-3) if optimizer is None else optimizer
        opt_init, opt_update, get_params = optimizer
        opt_state = opt_init(self._params)

        # define the training step function
        @jit
        def step(i, opt_state, x):
            params = get_params(opt_state)
            gradients = grad(loss_fn)(params, x)
            return opt_update(i, gradients, opt_state)

        # save the initial loss
        losses = [loss_fn(self._params, inputs)]
        if verbose:
            print(f"{losses[-1]:.4f}")

        # loop through training
        itercount = itertools.count()
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
            losses.append(loss_fn(params, inputs))

            if verbose and epoch % max(int(0.05 * epochs), 1) == 0:
                print(f"{losses[-1]:.4f}")

        # update the flow parameters with the final training state
        self._params = get_params(opt_state)
        return losses

    def train(
        self,
        inputs: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 1024,
        optimizer: Optimizer = None,
        loss_fn: Callable = None,
        convolve_err: bool = False,
        burn_in_epochs: int = 0,
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
        epochs : int, default=50
            Number of epochs to train.
        batch_size : int, default=1024
            Batch size for training.
        optimizer : jax Optimizer, default=adam(step_size=1e-3)
            An optimizer from jax.experimental.optimizers.
        loss_fn : Callable, optional
            A function to calculate the loss: loss = loss_fn(params, x).
            If not provided, will be -mean(log_prob).
        convolve_err : boolean, default=False
            Whether to convolve Gaussian errors in the loss function.
            Looks for in `inputs` for columns with names ending in `_err`.
            I.e., the error for column `u` needs to be in the column `u_err`.
            Zero error assumed for any missing error columns.
            Note this only works with the default loss function and a
            Gaussian base distribution.
            WARNING this is still experimental -- training is unstable and
            often results in NaN's.

        burn_in_epochs : int, default=0
            The number of epochs to train without error convolution,
            before beginning to train with error convolution.
            E.g., if epochs=50 and convolve_burn_in=20, then the flow
            is trained for 20 epochs without error convolution, followed
            by 50 epochs with error convolution.
        seed : int, default=0
            A random seed to control the batching.
        verbose : bool, default=False
            If true, print the training loss every 5% of epochs.

        Returns
        -------
        list
            List of training losses from every epoch.
        """

        if convolve_err:
            print("WARNING: error convolution in training is still experimental.")

        # validate epochs
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")
        if not isinstance(burn_in_epochs, int) or burn_in_epochs < 0:
            raise ValueError("burn_in_epochs must be a non-negative integer.")

        # get random seeds for training loops
        rng = random.PRNGKey(seed)
        burn_in_rng, main_train_rng = random.split(rng)

        # if we are convolving errors, we will first perform burn-in
        # then will set up for the real training loop
        burn_in_losses = []
        if convolve_err:
            # make sure we are using the default loss function
            if loss_fn is not None:
                raise ValueError(
                    "Error convolution is currently only implemented for the default loss function."
                )
            # perform burn-in
            if burn_in_epochs > 0:
                if verbose:
                    print(f"Burning-in {burn_in_epochs} epochs \nLoss:")
                # use the loss function without error convolution
                @jit
                def loss_fn(params, x):
                    return -np.mean(self._log_prob(params, x))

                # convert data to an array with required columns
                columns = list(self.data_columns)
                X = np.array(inputs[columns].values)

                # run the training
                burn_in_losses = self._train(
                    X,
                    burn_in_epochs,
                    batch_size,
                    optimizer,
                    loss_fn,
                    burn_in_rng,
                    verbose,
                )
                if verbose:
                    print("Burn-in complete \n")

            # AFTER BURN-IN
            # switch to the convolved loss function
            @jit
            def loss_fn(params, x):
                return -np.mean(self._log_prob_convolved(params, x))

            # and get a data array with error columns
            X = self._array_with_errs(inputs)

        # if not performing error convolution,
        # simply get ready for the real training loop
        else:
            # if no loss_fn is provided, use the default loss function
            if loss_fn is None:

                @jit
                def loss_fn(params, x):
                    return -np.mean(self._log_prob(params, x))

            # convert data to an array with required columns
            columns = list(self.data_columns)
            X = np.array(inputs[columns].values)

        if verbose:
            print(f"Training {epochs} epochs \nLoss:")

        # normal training run
        main_train_losses = self._train(
            X,
            epochs,
            batch_size,
            optimizer,
            loss_fn,
            main_train_rng,
            verbose,
        )

        # combine the burn-in and main training losses
        losses = burn_in_losses + main_train_losses

        return losses
