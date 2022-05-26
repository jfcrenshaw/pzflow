"""Define the Flow object that defines the normalizing flow."""
from typing import Any, Callable, Sequence, Tuple

import dill as pickle
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from jax import grad, jit, random
from tqdm import tqdm

from pzflow import distributions
from pzflow.bijectors import (
    Bijector_Info,
    Chain,
    InitFunction,
    Pytree,
    RollingSplineCoupling,
    ShiftBounds,
)
from pzflow.utils import build_bijector_from_info, gaussian_error_model


class Flow:
    """A normalizing flow that models tabular data.

    Attributes
    ----------
    data_columns : tuple
        List of DataFrame columns that the flow expects/produces.
    conditional_columns : tuple
        List of DataFrame columns on which the flow is conditioned.
    latent : distributions.LatentDist
        The latent distribution of the normalizing flow.
        Has it's own sample and log_prob methods.
    data_error_model : Callable
        The error model for the data variables. See the docstring of
        __init__ for more details.
    condition_error_model : Callable
        The error model for the conditional variables. See the docstring
        of __init__ for more details.
    info : Any
        Object containing any kind of info included with the flow.
        Often describes the data the flow is trained on.
    """

    def __init__(
        self,
        data_columns: Sequence[str] = None,
        bijector: Tuple[InitFunction, Bijector_Info] = None,
        latent: distributions.LatentDist = None,
        conditional_columns: Sequence[str] = None,
        data_error_model: Callable = None,
        condition_error_model: Callable = None,
        autoscale_conditions: bool = True,
        seed: int = 0,
        info: Any = None,
        file: str = None,
        _dictionary: dict = None,
    ) -> None:
        """Instantiate a normalizing flow.

        Note that while all of the init parameters are technically optional,
        you must provide either data_columns OR file.
        In addition, if a file is provided, all other parameters must be None.

        Parameters
        ----------
        data_columns : Sequence[str]; optional
            Tuple, list, or other container of column names.
            These are the columns the flow expects/produces in DataFrames.
        bijector : Bijector Call; optional
            A Bijector call that consists of the bijector InitFunction that
            initializes the bijector and the tuple of Bijector Info.
            Can be the output of any Bijector, e.g. Reverse(), Chain(...), etc.
            If not provided, the bijector can be set later using
            flow.set_bijector, or by calling flow.train, in which case the
            default bijector will be used. The default bijector is
            ShiftBounds -> RollingSplineCoupling, where the range of shift
            bounds is learned from the training data, and the dimensions of
            RollingSplineCoupling is inferred. The default bijector assumes
            that the latent has support [-5, 5] for every dimension.
        latent : distributions.LatentDist; optional
            The latent distribution for the normalizing flow. Can be any of
            the distributions from pzflow.distributions. If not provided,
            a uniform distribution is used with input_dim = len(data_columns),
            and B=5.
        conditional_columns : Sequence[str]; optional
            Names of columns on which to condition the normalizing flow.
        data_error_model : Callable; optional
            A callable that defines the error model for data variables.
            data_error_model must take key, X, Xerr, nsamples as arguments:
                - key is a jax rng key, e.g. jax.random.PRNGKey(0)
                - X is 2D array of data variables, where the order of variables
                    matches the order of the columns in data_columns
                - Xerr is the corresponding 2D array of errors
                - nsamples is number of samples to draw from error distribution
            data_error_model must return an array of samples with the shape
            (X.shape[0], nsamples, X.shape[1]).
            If data_error_model is not provided, Gaussian error model assumed.
        condition_error_model : Callable; optional
            A callable that defines the error model for conditional variables.
            condition_error_model must take key, X, Xerr, nsamples, where:
                - key is a jax rng key, e.g. jax.random.PRNGKey(0)
                - X is 2D array of conditional variables, where the order of
                    variables matches order of columns in conditional_columns
                - Xerr is the corresponding 2D array of errors
                - nsamples is number of samples to draw from error distribution
            condition_error_model must return array of samples with shape
            (X.shape[0], nsamples, X.shape[1]).
            If condition_error_model is not provided, Gaussian error model
            assumed.
        autoscale_conditions : bool; default=True
            Sets whether or not conditions are automatically standard scaled
            when passed to a conditional flow. I recommend you leave as True.
        seed : int; default=0
            The random seed for initial parameters
        info : Any; optional
            An object to attach to the info attribute.
        file : str; optional
            Path to file from which to load a pretrained flow.
            If a file is provided, all other parameters must be None.
        """

        # validate parameters
        if data_columns is None and file is None and _dictionary is None:
            raise ValueError("You must provide data_columns OR file.")
        if any(
            (
                data_columns is not None,
                bijector is not None,
                conditional_columns is not None,
                latent is not None,
                data_error_model is not None,
                condition_error_model is not None,
                info is not None,
            )
        ):
            if file is not None:
                raise ValueError(
                    "If providing a file, please do not provide any other parameters."
                )
            if _dictionary is not None:
                raise ValueError(
                    "If providing a dictionary, please do not provide any other parameters."
                )
        if file is not None and _dictionary is not None:
            raise ValueError("Only provide file or _dictionary, not both.")

        # if file or dictionary is provided, load everything from it
        if file is not None or _dictionary is not None:

            save_dict = self._save_dict()
            if file is not None:
                with open(file, "rb") as handle:
                    save_dict.update(pickle.load(handle))
            else:
                save_dict.update(_dictionary)

            if save_dict["class"] != self.__class__.__name__:
                raise TypeError(
                    f"This save file isn't a {self.__class__.__name__}. "
                    f"It is a {save_dict['class']}"
                )

            # load columns and dimensions
            self.data_columns = save_dict["data_columns"]
            self.conditional_columns = save_dict["conditional_columns"]
            self._input_dim = len(self.data_columns)
            self.info = save_dict["info"]

            # load the latent distribution
            self._latent_info = save_dict["latent_info"]
            self.latent = getattr(distributions, self._latent_info[0])(
                *self._latent_info[1]
            )

            # load the error models
            self.data_error_model = save_dict["data_error_model"]
            self.condition_error_model = save_dict["condition_error_model"]

            # load the bijector
            self._bijector_info = save_dict["bijector_info"]
            if self._bijector_info is not None:
                init_fun, _ = build_bijector_from_info(self._bijector_info)
                _, self._forward, self._inverse = init_fun(
                    random.PRNGKey(0), self._input_dim
                )
            self._params = save_dict["params"]

            # load the conditional means and stds
            self._condition_means = save_dict["condition_means"]
            self._condition_stds = save_dict["condition_stds"]

            # set whether or not to automatically standard scale any
            # conditions passed to the normalizing flow
            self._autoscale_conditions = save_dict["autoscale_conditions"]

        # if no file is provided, use provided parameters
        else:
            self.data_columns = tuple(data_columns)
            self._input_dim = len(self.data_columns)
            self.info = info

            if conditional_columns is None:
                self.conditional_columns = None
                self._condition_means = None
                self._condition_stds = None
            else:
                self.conditional_columns = tuple(conditional_columns)
                self._condition_means = jnp.zeros(
                    len(self.conditional_columns)
                )
                self._condition_stds = jnp.ones(len(self.conditional_columns))

            # set whether or not to automatically standard scale any
            # conditions passed to the normalizing flow
            self._autoscale_conditions = autoscale_conditions

            # set up the latent distribution
            if latent is None:
                self.latent = distributions.Uniform(self._input_dim, 5)
            else:
                self.latent = latent
            self._latent_info = self.latent.info

            # make sure the latent distribution and data_columns have the
            # same number of dimensions
            if self.latent.input_dim != len(data_columns):
                raise ValueError(
                    f"The latent distribution has {self.latent.input_dim} "
                    f"dimensions, but data_columns has {len(data_columns)} "
                    "dimensions. They must match!"
                )

            # set up the error models
            if data_error_model is None:
                self.data_error_model = gaussian_error_model
            else:
                self.data_error_model = data_error_model
            if condition_error_model is None:
                self.condition_error_model = gaussian_error_model
            else:
                self.condition_error_model = condition_error_model

            # set up the bijector
            if bijector is not None:
                self.set_bijector(bijector, seed=seed)
            # if no bijector was provided, set bijector_info to None
            else:
                self._bijector_info = None

    def _check_bijector(self) -> None:
        if self._bijector_info is None:
            raise ValueError(
                "The bijector has not been set up yet! "
                "You can do this by calling "
                "flow.set_bijector(bijector, params), "
                "or by calling train, in which case the default "
                "bijector will be used."
            )

    def set_bijector(
        self,
        bijector: Tuple[InitFunction, Bijector_Info],
        params: Pytree = None,
        seed: int = 0,
    ) -> None:
        """Set the bijector.

        Parameters
        ----------
        bijector : Bijector Call
            A Bijector call that consists of the bijector InitFunction that
            initializes the bijector and the tuple of Bijector Info.
            Can be the output of any Bijector, e.g. Reverse(), Chain(...), etc.
        params : Pytree; optional
            A Pytree of bijector parameters. If not provided, the bijector
            will be initialized with random parameters.
        seed: int; default=0
            A random seed for initializing the bijector with random parameters.
        """

        # set up the bijector
        init_fun, self._bijector_info = bijector
        bijector_params, self._forward, self._inverse = init_fun(
            random.PRNGKey(seed), self._input_dim
        )

        # check if params were passed
        bijector_params = params if params is not None else bijector_params

        # save the bijector params along with the latent params
        self._params = (self.latent._params, bijector_params)

    def _set_default_bijector(self, inputs: pd.DataFrame, seed: int = 0) -> None:
        # Set the default bijector
        # which is ShiftBounds -> RollingSplineCoupling

        # get the min/max for each data column
        data = inputs[list(self.data_columns)].to_numpy()
        mins = data.min(axis=0)
        maxs = data.max(axis=0)

        # determine how many conditional columns we have
        n_conditions = (
            0
            if self.conditional_columns is None
            else len(self.conditional_columns)
        )

        self.set_bijector(
            Chain(
                ShiftBounds(mins, maxs, 4.),
                RollingSplineCoupling(
                    len(self.data_columns), n_conditions=n_conditions
                ),
            ),
            seed=seed,
        )

    def _get_conditions(self, inputs: pd.DataFrame) -> jnp.ndarray:
        # Return an array of the bijector conditions.

        # if this isn't a conditional flow, just return empty conditions
        if self.conditional_columns is None:
            conditions = jnp.zeros((inputs.shape[0], 1))
        # if this a conditional flow, return an array of the conditions
        else:
            columns = list(self.conditional_columns)
            conditions = jnp.array(inputs[columns].to_numpy())
            conditions = (
                conditions - self._condition_means
            ) / self._condition_stds
        return conditions

    def _get_err_samples(
        self,
        key,
        inputs: pd.DataFrame,
        err_samples: int,
        type: str = "data",
        skip: str = None,
    ) -> jnp.ndarray:
        # Draw error samples for each row of inputs.

        X = inputs.copy()

        # get list of columns
        if type == "data":
            columns = list(self.data_columns)
            error_model = self.data_error_model
        elif type == "conditions":
            if self.conditional_columns is None:
                return jnp.zeros((err_samples * X.shape[0], 1))
            else:
                columns = list(self.conditional_columns)
                error_model = self.condition_error_model
        else:
            raise ValueError("type must be `data` or `conditions`.")

        # make sure all relevant variables have error columns
        for col in columns:
            # if errors not provided for the column, fill in zeros
            if f"{col}_err" not in inputs.columns and col != skip:
                X[f"{col}_err"] = jnp.zeros(X.shape[0])
            # if we are skipping this column, fill in nan's
            elif col == skip:
                X[col] = jnp.nan * jnp.zeros(X.shape[0])
                X[f"{col}_err"] = jnp.nan * jnp.zeros(X.shape[0])

        # pull out relevant columns
        err_columns = [col + "_err" for col in columns]
        X, Xerr = jnp.array(X[columns].to_numpy()), jnp.array(
            X[err_columns].to_numpy()
        )

        # generate samples
        Xsamples = error_model(key, X, Xerr, err_samples)
        Xsamples = Xsamples.reshape(X.shape[0] * err_samples, X.shape[1])

        # delete the column corresponding to skip
        if skip is not None:
            idx = columns.index(skip)
            Xsamples = jnp.delete(Xsamples, idx, axis=1)

        # if these are samples of conditions, standard scale them!
        if type == "conditions":
            Xsamples = (
                Xsamples - self._condition_means
            ) / self._condition_stds

        return Xsamples

    def _log_prob(
        self, params: Pytree, inputs: jnp.ndarray, conditions: jnp.ndarray
    ) -> jnp.ndarray:
        # Log prob for arrays.

        # calculate log_prob
        u, log_det = self._forward(params[1], inputs, conditions=conditions)
        log_prob = self.latent.log_prob(params[0], u) + log_det
        # set NaN's to negative infinity (i.e. zero probability)
        log_prob = jnp.nan_to_num(log_prob, nan=jnp.NINF)
        return log_prob

    def log_prob(
        self, inputs: pd.DataFrame, err_samples: int = None, seed: int = None
    ) -> jnp.ndarray:
        """Calculates log probability density of inputs.

        Parameters
        ----------
        inputs : pd.DataFrame
            Input data for which log probability density is calculated.
            Every column in self.data_columns must be present.
            If self.conditional_columns is not None, those must be present
            as well. If other columns are present, they are ignored.
        err_samples : int; default=None
            Number of samples from the error distribution to average over for
            the log_prob calculation. If provided, Gaussian errors are assumed,
            and method will look for error columns in `inputs`. Error columns
            must end in `_err`. E.g. the error column for the variable `u` must
            be `u_err`. Zero error assumed for any missing error columns.
        seed : int; default=None
            Random seed for drawing the samples with Gaussian errors.

        Returns
        -------
        jnp.ndarray
            Device array of shape (inputs.shape[0],).
        """

        # check that the bijector exists
        self._check_bijector()

        if err_samples is None:
            # convert data to an array with columns ordered
            columns = list(self.data_columns)
            X = jnp.array(inputs[columns].to_numpy())
            # get conditions
            conditions = self._get_conditions(inputs)
            # calculate log_prob
            return self._log_prob(self._params, X, conditions)

        else:
            # validate nsamples
            assert isinstance(
                err_samples, int
            ), "err_samples must be a positive integer."
            assert err_samples > 0, "err_samples must be a positive integer."
            # get Gaussian samples
            seed = np.random.randint(1e18) if seed is None else seed
            key = random.PRNGKey(seed)
            X = self._get_err_samples(key, inputs, err_samples, type="data")
            C = self._get_err_samples(
                key, inputs, err_samples, type="conditions"
            )
            # calculate log_probs
            log_probs = self._log_prob(self._params, X, C)
            probs = jnp.exp(log_probs.reshape(-1, err_samples))
            return jnp.log(probs.mean(axis=1))

    def posterior(
        self,
        inputs: pd.DataFrame,
        column: str,
        grid: jnp.ndarray,
        marg_rules: dict = None,
        normalize: bool = True,
        err_samples: int = None,
        seed: int = None,
        batch_size: int = None,
        nan_to_zero: bool = True,
    ) -> jnp.ndarray:
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
        grid : jnp.ndarray
            Grid on which to calculate the posterior.
        marg_rules : dict; optional
            Dictionary with rules for marginalizing over missing variables.
            The dictionary must contain the key "flag", which gives the flag
            that indicates a missing value. E.g. if missing values are given
            the value 99, the dictionary should contain {"flag": 99}.
            The dictionary must also contain {"name": callable} for any
            variables that will need to be marginalized over, where name is
            the name of the variable, and callable is a callable that takes
            the row of variables nad returns a grid over which to marginalize
            the variable. E.g. {"y": lambda row: jnp.linspace(0, row["x"], 10)}.
            Note: the callable for a given name must *always* return an array
            of the same length, regardless of the input row.
        err_samples : int; default=None
            Number of samples from the error distribution to average over for
            the posterior calculation. If provided, Gaussian errors are assumed,
            and method will look for error columns in `inputs`. Error columns
            must end in `_err`. E.g. the error column for the variable `u` must
            be `u_err`. Zero error assumed for any missing error columns.
        seed : int; default=None
            Random seed for drawing the samples with Gaussian errors.
        batch_size : int; default=None
            Size of batches in which to calculate posteriors. If None, all
            posteriors are calculated simultaneously. Simultaneous calculation
            is faster, but memory intensive for large data sets.
        normalize : boolean; default=True
            Whether to normalize the posterior so that it integrates to 1.
        nan_to_zero : bool; default=True
            Whether to convert NaN's to zero probability in the final pdfs.

        Returns
        -------
        jnp.ndarray
            Device array of shape (inputs.shape[0], grid.size).
        """

        # check that the bijector exists
        self._check_bijector()

        # get the index of the provided column, and remove it from the list
        columns = list(self.data_columns)
        idx = columns.index(column)
        columns.remove(column)

        nrows = inputs.shape[0]
        batch_size = nrows if batch_size is None else batch_size

        # make sure indices run 0 -> nrows
        inputs = inputs.reset_index(drop=True)

        if err_samples is not None:
            # validate nsamples
            assert isinstance(
                err_samples, int
            ), "err_samples must be a positive integer."
            assert err_samples > 0, "err_samples must be a positive integer."
            # set the seed
            seed = np.random.randint(1e18) if seed is None else seed
            key = random.PRNGKey(seed)

        # empty array to hold pdfs
        pdfs = jnp.zeros((nrows, len(grid)))

        # if marginalization rules were passed, we will loop over the rules
        # and repeatedly call this method
        if marg_rules is not None:

            # if the flag is NaN, we must use jnp.isnan to check for flags
            if np.isnan(marg_rules["flag"]):

                def check_flags(data):
                    return np.isnan(data)

            # else we use jnp.isclose to check for flags
            else:

                def check_flags(data):
                    return np.isclose(data, marg_rules["flag"])

            # first calculate pdfs for unflagged rows
            unflagged_idx = inputs[
                ~check_flags(inputs[columns]).any(axis=1)
            ].index.tolist()
            unflagged_pdfs = self.posterior(
                inputs=inputs.iloc[unflagged_idx],
                column=column,
                grid=grid,
                err_samples=err_samples,
                seed=seed,
                batch_size=batch_size,
                normalize=False,
                nan_to_zero=nan_to_zero,
            )

            # save these pdfs in the big array
            pdfs = pdfs.at[unflagged_idx, :].set(
                unflagged_pdfs,
                indices_are_sorted=True,
                unique_indices=True,
            )

            # we will keep track of all the rows we've already calculated
            # posteriors for
            already_done = unflagged_idx

            # now we will loop over the rules in marg_rules
            for name, rule in marg_rules.items():

                # ignore the flag, because that's not a column in the data
                if name == "flag":
                    continue

                # get the list of new rows for which we need to calculate posteriors
                flagged_idx = inputs[check_flags(inputs[name])].index.tolist()
                flagged_idx = list(set(flagged_idx).difference(already_done))

                # if flagged_idx is empty, move on!
                if len(flagged_idx) == 0:
                    continue

                # get the marginalization grid for each row
                marg_grids = (
                    inputs.iloc[flagged_idx]
                    .apply(rule, axis=1, result_type="expand")
                    .to_numpy()
                )

                # make a new data frame with the marginalization grids replacing
                # the values of the flag in the column
                marg_inputs = pd.DataFrame(
                    np.repeat(
                        inputs.iloc[flagged_idx].to_numpy(),
                        marg_grids.shape[1],
                        axis=0,
                    ),
                    columns=inputs.columns,
                )
                marg_inputs[name] = marg_grids.reshape(marg_inputs.shape[0], 1)

                # remove the error column if it's present
                marg_inputs.drop(
                    f"{name}_err", axis=1, inplace=True, errors="ignore"
                )

                # calculate posteriors for these
                marg_pdfs = self.posterior(
                    inputs=marg_inputs,
                    column=column,
                    grid=grid,
                    marg_rules=marg_rules,
                    err_samples=err_samples,
                    seed=seed,
                    batch_size=batch_size,
                    normalize=False,
                    nan_to_zero=nan_to_zero,
                )

                # sum over the marginalized dimension
                marg_pdfs = marg_pdfs.reshape(
                    len(flagged_idx), marg_grids.shape[1], grid.size
                )
                marg_pdfs = marg_pdfs.sum(axis=1)

                # save the new pdfs in the big array
                pdfs = pdfs.at[flagged_idx, :].set(
                    marg_pdfs,
                    indices_are_sorted=True,
                    unique_indices=True,
                )

                # add these flagged indices to the list of rows already done
                already_done += flagged_idx

        # now for the main posterior calculation loop
        else:

            # loop through batches
            for batch_idx in range(0, nrows, batch_size):

                # get the data batch
                # and, if this is a conditional flow, the correpsonding conditions
                batch = inputs.iloc[batch_idx : batch_idx + batch_size]

                # if not drawing samples, just grab batch and conditions
                if err_samples is None:
                    conditions = self._get_conditions(batch)
                    batch = jnp.array(batch[columns].to_numpy())
                # if only drawing condition samples...
                elif len(self.data_columns) == 1:
                    conditions = self._get_err_samples(
                        key, batch, err_samples, type="conditions"
                    )
                    batch = jnp.repeat(
                        batch[columns].to_numpy(), err_samples, axis=0
                    )
                # if drawing data and condition samples...
                else:
                    conditions = self._get_err_samples(
                        key, batch, err_samples, type="conditions"
                    )
                    batch = self._get_err_samples(
                        key, batch, err_samples, skip=column, type="data"
                    )

                # make a new copy of each row for each value of the column
                # for which we are calculating the posterior
                batch = jnp.hstack(
                    (
                        jnp.repeat(
                            batch[:, :idx],
                            len(grid),
                            axis=0,
                        ),
                        jnp.tile(grid, len(batch))[:, None],
                        jnp.repeat(
                            batch[:, idx:],
                            len(grid),
                            axis=0,
                        ),
                    )
                )

                # make similar copies of the conditions
                conditions = jnp.repeat(conditions, len(grid), axis=0)

                # calculate probability densities
                log_prob = self._log_prob(
                    self._params, batch, conditions
                ).reshape((-1, len(grid)))
                prob = jnp.exp(log_prob)
                # if we were Gaussian sampling, average over the samples
                if err_samples is not None:
                    prob = prob.reshape(-1, err_samples, len(grid))
                    prob = prob.mean(axis=1)
                # add the pdfs to the bigger list
                pdfs = pdfs.at[batch_idx : batch_idx + batch_size, :].set(
                    prob,
                    indices_are_sorted=True,
                    unique_indices=True,
                )

        if normalize:
            # normalize so they integrate to one
            pdfs = pdfs / jnp.trapz(y=pdfs, x=grid).reshape(-1, 1)
        if nan_to_zero:
            # set NaN's equal to zero probability
            pdfs = jnp.nan_to_num(pdfs, nan=0.0)
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
        nsamples : int; default=1
            The number of samples to be returned.
        conditions : pd.DataFrame; optional
            If this is a conditional flow, you must pass conditions for
            each sample. nsamples will be drawn for each row in conditions.
        save_conditions : bool; default=True
            If true, conditions will be saved in the DataFrame of samples
            that is returned.
        seed : int; optional
            Sets the random seed for the samples.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of samples.
        """

        # check that the bijector exists
        self._check_bijector()

        # validate nsamples
        assert isinstance(
            nsamples, int
        ), "nsamples must be a positive integer."
        assert nsamples > 0, "nsamples must be a positive integer."

        if self.conditional_columns is not None and conditions is None:
            raise ValueError(
                f"Must provide the following conditions\n{self.conditional_columns}"
            )

        # if this isn't a conditional flow, get empty conditions
        if self.conditional_columns is None:
            conditions = jnp.zeros((nsamples, 1))
        # otherwise get conditions and make `nsamples` copies of each
        else:
            conditions_idx = list(conditions.index)
            conditions = self._get_conditions(conditions)
            conditions_idx = np.repeat(conditions_idx, nsamples)
            conditions = jnp.repeat(conditions, nsamples, axis=0)

        # draw from latent distribution
        u = self.latent.sample(self._params[0], conditions.shape[0], seed)
        # take the inverse back to the data distribution
        x = self._inverse(self._params[1], u, conditions=conditions)[0]
        # if not conditional, this is all we need
        if self.conditional_columns is None:
            x = pd.DataFrame(np.array(x), columns=self.data_columns)
        # but if conditional
        else:
            if save_conditions:
                # unscale the conditions
                conditions = (
                    conditions * self._condition_stds + self._condition_means
                )
                x = pd.DataFrame(
                    np.array(jnp.hstack((x, conditions))),
                    columns=self.data_columns + self.conditional_columns,
                ).set_index(conditions_idx)
            else:
                # reindex according to the conditions
                x = pd.DataFrame(
                    np.array(x), columns=self.data_columns
                ).set_index(conditions_idx)

        # return the samples!
        return x

    def _save_dict(self) -> None:
        ### Returns the dictionary of all flow params to be saved.
        save_dict = {"class": self.__class__.__name__}
        keys = [
            "data_columns",
            "conditional_columns",
            "condition_means",
            "condition_stds",
            "data_error_model",
            "condition_error_model",
            "autoscale_conditions",
            "info",
            "latent_info",
            "bijector_info",
            "params",
        ]
        for key in keys:
            try:
                save_dict[key] = getattr(self, key)
            except AttributeError:
                try:
                    save_dict[key] = getattr(self, "_" + key)
                except AttributeError:
                    save_dict[key] = None

        return save_dict

    def save(self, file: str) -> None:
        """Saves the flow to a file.

        Pickles the flow and saves it to a file that can be passed as
        the `file` argument during flow instantiation.

        WARNING: Currently, this method only works for bijectors that are
        implemented in the `bijectors` module. If you want to save a flow
        with a custom bijector, you either need to add the bijector to that
        module, or handle the saving and loading on your end.

        Parameters
        ----------
        file : str
            Path to where the flow will be saved.
            Extension `.pkl` will be appended if not already present.
        """
        save_dict = self._save_dict()

        with open(file, "wb") as handle:
            pickle.dump(save_dict, handle, recurse=True)

    def train(
        self,
        inputs: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 1024,
        optimizer: Callable = None,
        loss_fn: Callable = None,
        convolve_errs: bool = False,
        patience: int = None,
        seed: int = 0,
        verbose: bool = False,
        progress_bar: bool = False,
    ) -> list:
        """Trains the normalizing flow on the provided inputs.

        Parameters
        ----------
        inputs : pd.DataFrame
            Data on which to train the normalizing flow.
            Must have columns matching `self.data_columns`.
        epochs : int; default=100
            Number of epochs to train.
        batch_size : int; default=1024
            Batch size for training.
        optimizer : optax optimizer
            An optimizer from Optax. default = optax.adam(learning_rate=1e-3)
            see https://optax.readthedocs.io/en/latest/index.html for more.
        loss_fn : Callable; optional
            A function to calculate the loss: `loss = loss_fn(params, x)`.
            If not provided, will be `-mean(log_prob)`.
        convolve_errs : bool; default=False
            Whether to draw new data from the error distributions during
            each epoch of training. Method will look for error columns in
            `inputs`. Error columns must end in `_err`. E.g. the error column
            for the variable `u` must be `u_err`. Zero error assumed for
            any missing error columns. The error distribution is set during
            flow instantiation.
        patience : int; optional
            Factor that controls early stopping. Training will stop if the
            loss doesn't decrease for this number of epochs.
        seed : int; default=0
            A random seed to control the batching and the (optional)
            error sampling and creating the default bijector (the latter
            only happens if you didn't set up the bijector during Flow
            instantiation).
        verbose : bool; default=False
            If true, print the training loss every 5% of epochs.
        progress_bar : bool; default=False
            If true, display a tqdm progress bar during training.

        Returns
        -------
        list
            List of training losses from every epoch.
        """

        # split the seed
        rng = np.random.default_rng(seed)
        batch_seed, bijector_seed = rng.integers(1e9, size=2)

        # if the bijector is None, set the default bijector
        if self._bijector_info is None:
            self._set_default_bijector(inputs, seed=bijector_seed)

        # validate epochs
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("epochs must be a positive integer.")

        # if no loss_fn is provided, use the default loss function
        if loss_fn is None:

            @jit
            def loss_fn(params, x, c):
                return -jnp.mean(self._log_prob(params, x, c))

        # initialize the optimizer
        optimizer = (
            optax.adam(learning_rate=1e-3) if optimizer is None else optimizer
        )
        opt_state = optimizer.init(self._params)

        # pull out the model parameters
        model_params = self._params

        # define the training step function
        @jit
        def step(params, opt_state, x, c):
            gradients = grad(loss_fn)(params, x, c)
            updates, opt_state = optimizer.update(gradients, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        # get list of data columns
        columns = list(self.data_columns)

        # if this is a conditional flow, and autoscale_conditions == True
        # save the means and stds of the conditional columns
        if self.conditional_columns is not None and self._autoscale_conditions:
            self._condition_means = jnp.array(
                inputs[list(self.conditional_columns)].to_numpy().mean(axis=0)
            )
            condition_stds = jnp.array(
                inputs[list(self.conditional_columns)].to_numpy().std(axis=0)
            )
            self._condition_stds = jnp.where(
                condition_stds != 0, condition_stds, 1
            )

        # define a function to return batches
        if convolve_errs:

            def get_batch(sample_key, x, type):
                return self._get_err_samples(sample_key, x, 1, type=type)

        else:

            def get_batch(sample_key, x, type):
                if type == "conditions":
                    return self._get_conditions(x)
                else:
                    return jnp.array(x[columns].to_numpy())

        # get random seed for training loop
        key = random.PRNGKey(batch_seed)

        if verbose:
            print(f"Training {epochs} epochs \nLoss:")

        # save the initial loss
        X = jnp.array(inputs[columns].to_numpy())
        C = self._get_conditions(inputs)
        losses = [loss_fn(model_params, X, C)]
        if verbose:
            print(f"(0) {losses[-1]:.4f}")

        # initialize variables for early stopping
        best_loss = jnp.inf
        early_stopping_counter = 0

        # loop through training
        loop = tqdm(range(epochs)) if progress_bar else range(epochs)
        for epoch in loop:
            # new permutation of batches
            permute_key, sample_key, key = random.split(key, num=3)
            idx = random.permutation(permute_key, inputs.shape[0])
            X = inputs.iloc[idx]

            # loop through batches and step optimizer
            for batch_idx in range(0, len(X), batch_size):

                # if sampling from the error distribution, this returns a
                # Gaussian sample of the batch. Else just returns batch as a
                # jax array
                batch = get_batch(
                    sample_key,
                    X.iloc[batch_idx : batch_idx + batch_size],
                    type="data",
                )
                batch_conditions = get_batch(
                    sample_key,
                    X.iloc[batch_idx : batch_idx + batch_size],
                    type="conditions",
                )

                model_params, opt_state = step(
                    model_params,
                    opt_state,
                    batch,
                    batch_conditions,
                )

            # save end-of-epoch training loss
            losses.append(
                loss_fn(
                    model_params,
                    jnp.array(X[columns].to_numpy()),
                    self._get_conditions(X),
                )
            )

            # if verbose, print current loss
            if verbose and (
                epoch % max(int(0.05 * epochs), 1) == 0
                or (epoch + 1) == epochs
            ):
                print(f"({epoch+1}) {losses[-1]:.4f}")

            # if patience provided, we need to check for early stopping
            if patience is not None:

                # if loss didn't improve, increase counter
                # and check early stopping criterion
                if losses[-1] >= best_loss or jnp.isclose(
                    losses[-1], best_loss
                ):
                    early_stopping_counter += 1

                    # check if the early stopping criterion is met
                    if early_stopping_counter >= patience:
                        print(
                            "Early stopping criterion is met.",
                            f"Training stopping after epoch {epoch}.",
                        )
                        break
                # if this is the best loss, reset the counter
                else:
                    best_loss = losses[-1]
                    early_stopping_counter = 0

        # update the flow parameters with the final training state
        self._params = model_params

        return losses
