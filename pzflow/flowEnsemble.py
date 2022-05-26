"""Define FlowEnsemble object that holds an ensemble of normalizing flows."""
from typing import Any, Callable, Sequence, Tuple

import dill as pickle
import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax import random

from pzflow import Flow, distributions
from pzflow.bijectors import Bijector_Info, InitFunction


class FlowEnsemble:
    """An ensemble of normalizing flows.

    Attributes
    ----------
    data_columns : tuple
        List of DataFrame columns that the flows expect/produce.
    conditional_columns : tuple
        List of DataFrame columns on which the flows are conditioned.
    latent: distributions.LatentDist
        The latent distribution of the normalizing flows.
        Has it's own sample and log_prob methods.
    data_error_model : Callable
        The error model for the data variables. See the docstring of
        __init__ for more details.
    condition_error_model : Callable
        The error model for the conditional variables. See the docstring
        of __init__ for more details.
    info : Any
        Object containing any kind of info included with the ensemble.
        Often Reverse the data the flows are trained on.
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
        N: int = 1,
        info: Any = None,
        file: str = None,
    ) -> None:
        """Instantiate an ensemble of normalizing flows.

        Note that while all of the init parameters are technically optional,
        you must provide either data_columns and bijector OR file.
        In addition, if a file is provided, all other parameters must be None.

        Parameters
        ----------
        data_columns : Sequence[str]; optional
            Tuple, list, or other container of column names.
            These are the columns the flows expect/produce in DataFrames.
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
            Names of columns on which to condition the normalizing flows.
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
        N : int; default=1
            The number of flows in the ensemble.
        info : Any; optional
            An object to attach to the info attribute.
        file : str; optional
            Path to file from which to load a pretrained flow ensemble.
            If a file is provided, all other parameters must be None.
        """

        # validate parameters
        if data_columns is None and file is None:
            raise ValueError("You must provide data_columns OR file.")
        if file is not None and any(
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
            raise ValueError(
                "If providing a file, please do not provide any other parameters."
            )

        # if file is provided, load everything from the file
        if file is not None:

            # load the file
            with open(file, "rb") as handle:
                save_dict = pickle.load(handle)

            # make sure the saved file is for this class
            c = save_dict.pop("class")
            if c != self.__class__.__name__:
                raise TypeError(
                    f"This save file isn't a {self.__class__.__name__}. It is a {c}."
                )

            # load the ensemble from the dictionary
            self._ensemble = {
                name: Flow(_dictionary=flow_dict)
                for name, flow_dict in save_dict["ensemble"].items()
            }
            # load the metadata
            self.data_columns = save_dict["data_columns"]
            self.conditional_columns = save_dict["conditional_columns"]
            self.data_error_model = save_dict["data_error_model"]
            self.condition_error_model = save_dict["condition_error_model"]
            self.info = save_dict["info"]

            self._latent_info = save_dict["latent_info"]
            self.latent = getattr(distributions, self._latent_info[0])(
                *self._latent_info[1]
            )

        # otherwise create a new ensemble from the provided parameters
        else:
            # save the ensemble of flows
            self._ensemble = {
                f"Flow {i}": Flow(
                    data_columns=data_columns,
                    bijector=bijector,
                    conditional_columns=conditional_columns,
                    latent=latent,
                    data_error_model=data_error_model,
                    condition_error_model=condition_error_model,
                    autoscale_conditions=autoscale_conditions,
                    seed=i,
                    info=f"Flow {i}",
                )
                for i in range(N)
            }
            # save the metadata
            self.data_columns = data_columns
            self.conditional_columns = conditional_columns
            self.latent = self._ensemble["Flow 0"].latent
            self.data_error_model = data_error_model
            self.condition_error_model = condition_error_model
            self.info = info

    def log_prob(
        self,
        inputs: pd.DataFrame,
        err_samples: int = None,
        seed: int = None,
        returnEnsemble: bool = False,
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
        returnEnsemble : bool; default=False
            If True, returns log_prob for each flow in the ensemble as an
            array of shape (inputs.shape[0], N flows in ensemble).
            If False, the prob is averaged over the flows in the ensemble,
            and the log of this average is returned as an array of shape
            (inputs.shape[0],)

        Returns
        -------
        jnp.ndarray
            For shape, see returnEnsemble description above.
        """

        # calculate log_prob for each flow in the ensemble
        ensemble = jnp.array(
            [
                flow.log_prob(inputs, err_samples, seed)
                for flow in self._ensemble.values()
            ]
        )

        # re-arrange so that (axis 0, axis 1) = (inputs, flows in ensemble)
        ensemble = jnp.rollaxis(ensemble, axis=1)

        if returnEnsemble:
            # return the ensemble of log_probs
            return ensemble
        else:
            # return mean over ensemble
            # note we return log(mean prob) instead of just mean log_prob
            return jnp.log(jnp.exp(ensemble).mean(axis=1))

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
        returnEnsemble: bool = False,
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
        normalize : boolean; default=True
            Whether to normalize the posterior so that it integrates to 1.
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
        returnEnsemble : bool; default=False
            If True, returns posterior for each flow in the ensemble as an
            array of shape (inputs.shape[0], N flows in ensemble, grid.size).
            If False, the posterior is averaged over the flows in the ensemble,
            and returned as an array of shape (inputs.shape[0], grid.size)
        nan_to_zero : bool; default=True
            Whether to convert NaN's to zero probability in the final pdfs.

        Returns
        -------
        jnp.ndarray
            For shape, see returnEnsemble description above.
        """

        # calculate posterior for each flow in the ensemble
        ensemble = jnp.array(
            [
                flow.posterior(
                    inputs=inputs,
                    column=column,
                    grid=grid,
                    marg_rules=marg_rules,
                    err_samples=err_samples,
                    seed=seed,
                    batch_size=batch_size,
                    normalize=False,
                    nan_to_zero=nan_to_zero,
                )
                for flow in self._ensemble.values()
            ]
        )

        # re-arrange so that (axis 0, axis 1) = (inputs, flows in ensemble)
        ensemble = jnp.rollaxis(ensemble, axis=1)

        if returnEnsemble:
            # return the ensemble of posteriors
            if normalize:
                ensemble = ensemble.reshape(-1, grid.size)
                ensemble = ensemble / jnp.trapz(y=ensemble, x=grid).reshape(
                    -1, 1
                )
                ensemble = ensemble.reshape(inputs.shape[0], -1, grid.size)
            return ensemble
        else:
            # return mean over ensemble
            pdfs = ensemble.mean(axis=1)
            if normalize:
                pdfs = pdfs / jnp.trapz(y=pdfs, x=grid).reshape(-1, 1)
            return pdfs

    def sample(
        self,
        nsamples: int = 1,
        conditions: pd.DataFrame = None,
        save_conditions: bool = True,
        seed: int = None,
        returnEnsemble: bool = False,
    ) -> pd.DataFrame:
        """Returns samples from the ensemble.

        Parameters
        ----------
        nsamples : int; default=1
            The number of samples to be returned, either overall or per flow
            in the ensemble (see returnEnsemble below).
        conditions : pd.DataFrame; optional
            If this is a conditional flow, you must pass conditions for
            each sample. nsamples will be drawn for each row in conditions.
        save_conditions : bool; default=True
            If true, conditions will be saved in the DataFrame of samples
            that is returned.
        seed : int; optional
            Sets the random seed for the samples.
        returnEnsemble : bool; default=False
            If True, nsamples is drawn from each flow in the ensemble.
            If False, nsamples are drawn uniformly from the flows in the ensemble.

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame of samples.
        """

        if returnEnsemble:
            # return nsamples for each flow in the ensemble
            return pd.concat(
                [
                    flow.sample(nsamples, conditions, save_conditions, seed)
                    for flow in self._ensemble.values()
                ],
                keys=self._ensemble.keys(),
            )
        else:
            # if this isn't a conditional flow, sampling is straightforward
            if conditions is None:
                # return nsamples drawn uniformly from the flows in the ensemble
                N = int(jnp.ceil(nsamples / len(self._ensemble)))
                samples = pd.concat(
                    [
                        flow.sample(N, conditions, save_conditions, seed)
                        for flow in self._ensemble.values()
                    ]
                )
                return samples.sample(nsamples, random_state=seed).reset_index(
                    drop=True
                )
            # if this is a conditional flow, it's a little more complicated...
            else:
                # if nsamples > 1, we duplicate the rows of the conditions
                if nsamples > 1:
                    conditions = pd.concat([conditions] * nsamples)

                # now the main sampling algorithm
                seed = np.random.randint(1e18) if seed is None else seed
                # if we are drawing more samples than the number of flows in
                # the ensemble, then we will shuffle the conditions and randomly
                # assign them to one of the constituent flows
                if conditions.shape[0] > len(self._ensemble):
                    # shuffle the conditions
                    conditions_shuffled = conditions.sample(
                        frac=1.0, random_state=int(seed / 1e9)
                    )
                    # split conditions into ~equal sized chunks
                    chunks = np.array_split(
                        conditions_shuffled, len(self._ensemble)
                    )
                    # shuffle the chunks
                    chunks = [
                        chunks[i]
                        for i in random.permutation(
                            random.PRNGKey(seed), jnp.arange(len(chunks))
                        )
                    ]
                    # sample from each flow, and return all the samples
                    return pd.concat(
                        [
                            flow.sample(
                                1, chunk, save_conditions, seed
                            ).set_index(chunk.index)
                            for flow, chunk in zip(
                                self._ensemble.values(), chunks
                            )
                        ]
                    ).sort_index()
                # however, if there are more flows in the ensemble than samples
                # being drawn, then we will randomly select flows for each condition
                else:
                    rng = np.random.default_rng(seed)
                    # randomly select a flow to sample from for each condition
                    flows = rng.choice(
                        list(self._ensemble.values()),
                        size=conditions.shape[0],
                        replace=True,
                    )
                    # sample from each flow and return all the samples together
                    seeds = rng.integers(1e18, size=conditions.shape[0])
                    return pd.concat(
                        [
                            flow.sample(
                                1,
                                conditions[i : i + 1],
                                save_conditions,
                                new_seed,
                            )
                            for i, (flow, new_seed) in enumerate(
                                zip(flows, seeds)
                            )
                        ],
                    ).set_index(conditions.index)

    def save(self, file: str) -> None:
        """Saves the ensemble to a file.

        Pickles the ensemble and saves it to a file that can be passed as
        the `file` argument during flow instantiation.

        WARNING: Currently, this method only works for bijectors that are
        implemented in the `bijectors` module. If you want to save a flow
        with a custom bijector, you either need to add the bijector to that
        module, or handle the saving and loading on your end.

        Parameters
        ----------
        file : str
            Path to where the ensemble will be saved.
            Extension `.pkl` will be appended if not already present.
        """
        save_dict = {
            "data_columns": self.data_columns,
            "conditional_columns": self.conditional_columns,
            "latent_info": self.latent.info,
            "data_error_model": self.data_error_model,
            "condition_error_model": self.condition_error_model,
            "info": self.info,
            "class": self.__class__.__name__,
            "ensemble": {
                name: flow._save_dict()
                for name, flow in self._ensemble.items()
            },
        }

        with open(file, "wb") as handle:
            pickle.dump(save_dict, handle, recurse=True)

    def train(
        self,
        inputs: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 1024,
        optimizer: Callable = None,
        loss_fn: Callable = None,
        convolve_errs: bool = False,
        patience: int = None,
        seed: int = 0,
        verbose: bool = False,
        progress_bar: bool = False,
    ) -> dict:
        """Trains the normalizing flows on the provided inputs.

        Parameters
        ----------
        inputs : pd.DataFrame
            Data on which to train the normalizing flows.
            Must have columns matching self.data_columns.
        epochs : int; default=50
            Number of epochs to train.
        batch_size : int; default=1024
            Batch size for training.
        optimizer : optax optimizer
            An optimizer from Optax. default = optax.adam(learning_rate=1e-3)
            see https://optax.readthedocs.io/en/latest/index.html for more.
        loss_fn : Callable; optional
            A function to calculate the loss: loss = loss_fn(params, x).
            If not provided, will be -mean(log_prob).
        convolve_errs : bool; default=False
            Whether to draw new data from the error distributions during
            each epoch of training. Method will look for error columns in
            `inputs`. Error columns must end in `_err`. E.g. the error column
            for the variable `u` must be `u_err`. Zero error assumed for
            any missing error columns. The error distribution is set during
            ensemble instantiation.
        patience : int; optional
            Factor that controls early stopping. Training will stop if the
            loss doesn't decrease for this number of epochs.
        seed : int; default=0
            A random seed to control the batching and the (optional)
            error sampling.
        verbose : bool; default=False
            If true, print the training loss every 5% of epochs.
        progress_bar : bool; default=False
            If true, display a tqdm progress bar during training.

        Returns
        -------
        dict
            Dictionary of training losses from every epoch for each flow
            in the ensemble.
        """

        # generate random seeds for each flow
        rng = np.random.default_rng(seed)
        seeds = rng.integers(1e9, size=len(self._ensemble))

        loss_dict = dict()

        for i, (name, flow) in enumerate(self._ensemble.items()):

            if verbose:
                print(name)

            loss_dict[name] = flow.train(
                inputs=inputs,
                epochs=epochs,
                batch_size=batch_size,
                optimizer=optimizer,
                loss_fn=loss_fn,
                convolve_errs=convolve_errs,
                patience=patience,
                seed=seeds[i],
                verbose=verbose,
                progress_bar=progress_bar,
            )

        return loss_dict
