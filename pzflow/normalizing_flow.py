import itertools
import pickle
from typing import Any, Callable, Sequence, Tuple

import jax.numpy as np
import pandas as pd
from jax import grad, jacfwd, jit, ops, random, vmap
from jax.experimental.optimizers import Optimizer, adam

from pzflow import distributions
from pzflow.bijectors import Bijector_Info, InitFunction, Pytree
from pzflow.utils import build_bijector_from_info, sub_diag_indices


class Flow:
    """A normalizing flow that models tabular data.

    Attributes
    ----------
    data_columns : tuple
        List of DataFrame columns that the flow expects/produces.
    conditional_columns : tuple
        List of DataFrame columns on which the flow is conditioned.
    info : Any
        Object containing any kind of info included with the flow.
        Often describes the data the flow is trained on.
    latent
        The latent distribution of the normalizing flow.
        Has it's own sample and log_prob methods.
    """

    def __init__(
        self,
        data_columns: Sequence[str] = None,
        bijector: Tuple[InitFunction, Bijector_Info] = None,
        conditional_columns: Sequence[str] = None,
        latent: str = None,
        info: Any = None,
        file: str = None,
    ):
        """Instantiate a normalizing flow.

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
        conditional_columns : Sequence[str], optional
            Names of columns on which to condition the normalizing flow.
        latent : str, optional
            The latent distribution for the normalizing flow. Possible values
            are `Normal` or `Tdist`. Default is `Normal`.
        info : Any, optional
            An object to attach to the info attribute.
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
            (
                data_columns != None,
                bijector != None,
                conditional_columns != None,
                latent != None,
                info != None,
            )
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
            self.conditional_columns = save_dict["conditional_columns"]
            self._input_dim = len(self.data_columns)
            self.info = save_dict["info"]
            self._bijector_info = save_dict["bijector_info"]
            self._params = save_dict["params"]

            # set up the latent distribution
            self.latent = getattr(distributions, save_dict["latent"])(self._input_dim)

            # set up the bijector
            init_fun, _ = build_bijector_from_info(self._bijector_info)
            _, self._forward, self._inverse = init_fun(
                random.PRNGKey(0), self._input_dim
            )

        # if no file is provided, use provided parameters
        else:
            self.data_columns = tuple(data_columns)
            self._input_dim = len(self.data_columns)
            self.info = info

            if conditional_columns is None:
                self.conditional_columns = None
            else:
                self.conditional_columns = tuple(conditional_columns)

            # set up the latent distribution
            latent = "Normal" if latent is None else latent
            self.latent = getattr(distributions, latent)(self._input_dim)

            # set up the bijector with random params
            init_fun, self._bijector_info = bijector
            bijector_params, self._forward, self._inverse = init_fun(
                random.PRNGKey(0), self._input_dim
            )
            self._params = (self.latent._params, bijector_params)

    def _array_with_errs(self, inputs: pd.DataFrame, skip: str = None) -> np.ndarray:
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

    def _get_conditions(
        self, inputs: pd.DataFrame = None, nrows: int = None
    ) -> np.ndarray:
        """Return an array of the bijector conditions."""

        # if this isn't a conditional flow, just return empty conditions
        if self.conditional_columns is None:
            conditions = np.zeros((nrows, 1))
            return conditions
        # if this a conditional flow, return an array of the conditions
        else:
            columns = list(self.conditional_columns)
            conditions = np.array(inputs[columns].values)
            return conditions

    def _jacobian(
        self, params: Pytree, inputs: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        """Calculates the Jacobian of the forward bijection"""

        # calculates jacobian for a single input
        # first we define a lambda that calculates the forward
        # (but drops the log_det). then we take the Jacobian of that
        # evaluated at the vector y. the [None, :] and .squeeze() are
        # just making sure the inputs and outputs are of the correct shape
        def J(y, c):
            return jacfwd(lambda x: self._inverse(params, x, conditions=c[None, :])[0])(
                y[None, :]
            ).squeeze()

        # now we can vectorize with Jax and apply to whole set of inputs at once
        return vmap(J)(inputs, conditions)

    def _log_prob(
        self, params: Pytree, inputs: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        """Log prob for arrays."""
        # calculate log_prob
        u, log_det = self._forward(params[1], inputs, conditions=conditions)
        log_prob = self.latent.log_prob(params[0], u) + log_det
        # set NaN's to negative infinity (i.e. zero probability)
        log_prob = np.nan_to_num(log_prob, nan=np.NINF)
        return log_prob

    def _log_prob_convolved(
        self, params: Pytree, inputs: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        """Log prob for arrays, with error convolution"""

        # separate data from data errs
        ncols = len(self.data_columns)
        X, Xerr = inputs[:, :ncols], inputs[:, ncols:]

        # forward and log determinant
        u, log_det = self._forward(params[1], X, conditions=conditions)

        # Jacobian of inverse bijection
        J = self._jacobian(params[1], X, conditions)
        # calculate modified covariances
        sig_u = J @ (Xerr[..., None] * J.transpose((0, 2, 1)))
        # add identity matrix to each covariance matrix
        idx = sub_diag_indices(sig_u)
        sig = ops.index_update(sig_u, idx, sig_u[idx] + 1)

        # calculate log_prob w.r.t the latent distribution, with the new covariances
        log_prob = self.latent.log_prob(params[0], u, sig) + log_det
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
            If self.conditional_columns is not None, those must be present
            as well. If other columns are present, they are ignored.
        convolve_err : boolean, default=False
            Whether to analytically convolve Gaussian errors.
            Looks for in `inputs` for columns with names ending in `_err`.
            I.e., the error for column `u` needs to be in the column `u_err`.
            Zero error assumed for any missing error columns.
            WARNING: This is still experimental.

        Returns
        -------
        np.ndarray
            Device array of shape (inputs.shape[0],).
        """
        if convolve_err and not isinstance(self.latent, distributions.Normal):
            raise ValueError(
                "Currently can only convolve error when using a Normal latent distribution."
            )
        if convolve_err:
            print("WARNING: Error convolution is still experimental.")

        if not convolve_err:
            # convert data to an array with columns ordered
            columns = list(self.data_columns)
            X = np.array(inputs[columns].values)
            # get conditions
            conditions = self._get_conditions(inputs, len(inputs))
            # calculate log_prob
            return self._log_prob(self._params, X, conditions)
        else:
            # convert data to an array with columns ordered
            X = self._array_with_errs(inputs)
            # get conditions
            conditions = self._get_conditions(inputs, len(inputs))
            # calculate log_prob
            return self._log_prob_convolved(self._params, X, conditions)

    def posterior(
        self,
        inputs: pd.DataFrame,
        column: str,
        grid: np.ndarray,
        normalize: bool = True,
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
        normalize : boolean, default=True
            Whether to normalize the posterior so that it integrates to 1.
        convolve_err : boolean, default=False
            Whether to analytically convolve Gaussian errors in the posterior.
            Looks for in `inputs` for columns with names ending in `_err`.
            I.e., the error for column `u` needs to be in the column `u_err`.
            Zero error assumed for any missing error columns.
            WARNING: This is still experimental.
        batch_size : int, default=None
            Size of batches in which to calculate posteriors. If None, all
            posteriors are calculated simultaneously. Simultaneous calculation
            is faster, but memory intensive for large data sets.

        Returns
        -------
        np.ndarray
            Device array of shape (inputs.shape[0], grid.size).
        """
        if convolve_err and not isinstance(self.latent, distributions.Normal):
            raise ValueError(
                "Currently can only convolve error when using a Normal latent distribution."
            )
        if convolve_err:
            print("WARNING: Error convolution is still experimental.")

        # get the index of the provided column, and remove it from the list
        columns = list(self.data_columns)
        idx = columns.index(column)
        columns.remove(column)

        nrows = inputs.shape[0]
        batch_size = nrows if batch_size is None else batch_size

        # 1. convert data (sans the provided column) to array with columns ordered
        # 2. if this is a conditional flow, get array of conditions
        # 3. alias the required log_prob function
        if convolve_err:
            X = self._array_with_errs(inputs, skip=column)
            conditions = self._get_conditions(inputs, len(inputs))
            log_prob_fun = self._log_prob_convolved
        else:
            X = np.array(inputs[columns].values)
            conditions = self._get_conditions(inputs, len(inputs))
            log_prob_fun = self._log_prob

        # empty array to hold pdfs
        pdfs = np.zeros((nrows, len(grid)))

        # loop through batches
        for batch_idx in range(0, nrows, batch_size):

            # get the data batch
            # and, if this is a conditional flow, the correpsonding conditions
            batch = X[batch_idx : batch_idx + batch_size]
            batch_conditions = conditions[batch_idx : batch_idx + batch_size]

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

            # make similar copies of the conditions
            batch_conditions = np.repeat(batch_conditions, len(grid), axis=0)

            # calculate probability densities
            log_prob = log_prob_fun(self._params, batch, batch_conditions).reshape(
                (-1, len(grid))
            )
            pdfs = ops.index_update(
                pdfs,
                ops.index[batch_idx : batch_idx + batch_size, :],
                np.exp(log_prob),
                indices_are_sorted=True,
                unique_indices=True,
            )

        # reshape so that each row is a posterior
        pdfs = pdfs.reshape((nrows, len(grid)))
        if normalize:
            # normalize so they integrate to one
            pdfs = pdfs / np.trapz(y=pdfs, x=grid).reshape(-1, 1)
        # set NaN's equal to zero probability
        pdfs = np.nan_to_num(pdfs, nan=0.0)
        return pdfs

    def sample(
        self,
        nsamples: int = 1,
        conditions: pd.DataFrame = None,
        save_conditions: bool = True,
        seed: int = None,
    ) -> pd.DataFrame:
        """Returns samples from the normalizing flow.

        Parameters
        ----------
        nsamples : int, default=1
            The number of samples to be returned.
        conditions : pd.DataFrame, optional
            If this is a conditional flow, you must pass conditions for
            each sample. nsamples will be drawn for each row in conditions.
        save_conditions : bool, default=True
            If true, conditions will be saved in the DataFrame of samples
            that is returned.
        seed : int, optional
            Sets the random seed for the samples.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of samples.
        """
        if self.conditional_columns is not None and conditions is None:
            raise ValueError(
                f"Must provide the following conditions\n{self.conditional_columns}"
            )

        # if this is a conditional flow, get the conditions
        conditions = self._get_conditions(conditions, nsamples)
        if self.conditional_columns is not None:
            # repeat each condition nsamples-times
            conditions = np.repeat(conditions, nsamples, axis=0)
            nsamples = conditions.shape[0]

        # draw from latent distribution
        u = self.latent.sample(self._params[0], nsamples, seed)
        # take the inverse back to the data distribution
        x = self._inverse(self._params[1], u, conditions=conditions)[0]

        # if not conditional, or save_conditions is False, this is all we need
        if self.conditional_columns is None or save_conditions is False:
            x = pd.DataFrame(x, columns=self.data_columns)
        # but if conditional and save_conditions is True,
        # save conditions with samples
        else:
            x = pd.DataFrame(
                np.hstack((x, conditions)),
                columns=self.data_columns + self.conditional_columns,
            )

        # return the samples!
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
            "conditional_columns": self.conditional_columns,
            "info": self.info,
            "bijector_info": self._bijector_info,
            "latent": self.latent.type,
            "params": self._params,
        }
        if not file.endswith(".pkl"):
            file += ".pkl"
        with open(file, "wb") as handle:
            pickle.dump(save_dict, handle)

    def _train(
        self,
        inputs: np.ndarray,
        conditions: np.ndarray,
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
        def step(i, opt_state, x, c):
            params = get_params(opt_state)
            gradients = grad(loss_fn)(params, x, c)
            return opt_update(i, gradients, opt_state)

        # save the initial loss
        losses = [loss_fn(self._params, inputs, conditions)]
        if verbose:
            print(f"{losses[-1]:.4f}")

        # loop through training
        itercount = itertools.count()
        for epoch in range(epochs):
            # new permutation of batches
            permute_rng, rng = random.split(rng)
            idx = random.permutation(permute_rng, inputs.shape[0])
            X = inputs[idx]
            C = conditions[idx]
            # loop through batches and step optimizer
            for batch_idx in range(0, len(X), batch_size):
                opt_state = step(
                    next(itercount),
                    opt_state,
                    X[batch_idx : batch_idx + batch_size],
                    C[batch_idx : batch_idx + batch_size],
                )

            # save end-of-epoch training loss
            params = get_params(opt_state)
            losses.append(loss_fn(params, inputs, conditions))

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
            Gaussian latent distribution.
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

        if convolve_err and not isinstance(self.latent, distributions.Normal):
            raise ValueError(
                "Currently can only convolve error when using a Normal latent distribution."
            )

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
                def loss_fn(params, x, c):
                    return -np.mean(self._log_prob(params, x, c))

                # convert data to an array with required columns
                columns = list(self.data_columns)
                X = np.array(inputs[columns].values)
                conditions = self._get_conditions(inputs, inputs.shape[0])

                # run the training
                burn_in_losses = self._train(
                    X,
                    conditions,
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
            def loss_fn(params, x, c):
                return -np.mean(self._log_prob_convolved(params, x, c))

            # and get a data array with error columns
            X = self._array_with_errs(inputs)
            conditions = self._get_conditions(inputs, inputs.shape[0])

        # if not performing error convolution,
        # simply get ready for the real training loop
        else:
            # if no loss_fn is provided, use the default loss function
            if loss_fn is None:

                @jit
                def loss_fn(params, x, c):
                    return -np.mean(self._log_prob(params, x, c))

            # convert data to an array with required columns
            columns = list(self.data_columns)
            X = np.array(inputs[columns].values)
            conditions = self._get_conditions(inputs, inputs.shape[0])

        if verbose:
            print(f"Training {epochs} epochs \nLoss:")

        # normal training run
        main_train_losses = self._train(
            X,
            conditions,
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
