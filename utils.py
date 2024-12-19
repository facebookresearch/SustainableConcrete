#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines concrete strength data loaders, search space constraints, and other utilties.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from botorch.models import ModelListGP
from botorch.optim.initializers import sample_q_batches_from_polytope
from botorch.utils.multi_objective import is_non_dominated

from gpytorch import settings
from gpytorch.constraints import Interval
from torch import Tensor

# linear constraint type (ind, coeffs, value)
T_CONSTRAINT = Tuple[Tensor, Tensor, float]

_TOTAL_BINDER_NAMES = ["Cement", "Fly Ash", "Slag"]
_PASTE_CONTENT_NAMES = _TOTAL_BINDER_NAMES + ["Water"]
_BINDER_PLUS_AGGREGATE = _TOTAL_BINDER_NAMES + ["Fine Aggregate"]
_TOTAL_MASS_NAMES = _PASTE_CONTENT_NAMES + [
    "HRWR",
    "Coarse Aggregate",
    "Fine Aggregate",
]
DEFAULT_USED_COLUMNS = [
    "Mix ID",
    "Name",
    "Description",
    "Cement",
    "Fly Ash",
    "Slag",
    "Water",
    "HRWR",
    "Fine Aggregate",
    "Curing Temp (°C)",  # adding this here because last dimension is assumed to be time
    "Time",
    "GWP",  # the last four are output dimensions
    "Strength (Mean)",
    "Strength (Std)",
    "# of measurements",
]
_VERBOSE = False


class SustainableConcreteDataset(object):
    def __init__(
        self,
        X: Tensor,
        Y: Tensor,
        Ystd: Tensor,
        X_columns: List[str],
        Y_columns: List[str],
        Ystd_columns: List[str],
        bounds: Optional[Tensor] = None,
        batch_name_to_indices: Optional[Dict[str, List[int]]] = None,
    ):
        """An object to store, process, and access a concrete strength dataset.

        Args:
            X: `n x d`-dim Tensor of inputs, including composition dimensions and a time
                as the last dimension time = `X[:, -1]`.
            Y: `n x 2`-dim Tensor of outputs, where `Y[i, 0]` corresponds to the
                global warming potential (GWP) and `Y[i, 1]` corresponds to the
                empirical mean strength value corresponding to `X[i, :]`.
            Ystd: `n x 2`-dim Tensor of empirical standard deviations of `Y`.
            X_columns: A list of column names of `X`.
            Y_columns: A list of column names of `Y`.
            Ystd_columns: A list of column names of `Ystd`.
            bounds: A `2 x d`-dim Tensor of lower and upper bounds on the inputs `X`.
            batch_name_to_indices: A dictionary mapping experiment batch names to the
                indices of the corresponding samples in `X` and `Y`.

        Raises:
            ValueError: If the last columne of `X` is not time.
        """
        if X_columns[-1] != "Time":
            raise ValueError(
                f"Last dimension of X assumed to be time, but is {X_columns[-1]}."
            )

        # making sure we are not overwriting these
        self._X_columns = X_columns
        self._Y_columns = Y_columns
        self._Ystd_columns = Ystd_columns
        self._X = X
        self._Y = Y
        self._Ystd = Ystd
        self.bounds = bounds
        self._batch_name_to_indices = batch_name_to_indices

    @property
    def X(self) -> Tensor:
        """The `n x d`-dim input data `X`, where
        1) `X[i, :-1]` are the composition values of the ith sample.
        2) `X[i, -1]` is the time value of the ith sample.
        """
        return self._X

    @property
    def Y(self) -> Tensor:
        """The `n x 2`-dim output data `Y`, where
        1) `X[i, 0]` is the measured strength value for the ith sample.
        2) `X[i, 1]` is the GWP value of the ith sample.
        """
        return self._Y

    @property
    def Ystd(self) -> Tensor:
        """Getter for the `n x 2`-dim empirical standard deviation of the outputs.
        1) `Ystd[i, 0]` is the empirical standard deviation of the GWP values of the
            ith sample, and
        2) `Ystd[i, 1]` is the empirical standard deviation strength values for the
            ith sample.
        """
        return self._Ystd

    @property
    def Yvar(self) -> Tensor:
        """Convenience method for the empirical variance of the observations. See
        the documentation of Ystd for details.
        """
        return self.Ystd.square()

    @property
    def strength_data(self) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Returns the data with which to fit a strength model.

        Returns:
            A 4-Tuple of Tensors containing 1) the inputs `X` (composition and time),
            2) observed strengths `Y`, 3) empirical strength variances `Yvar`, and
            4) the `2 x d`-dim bounds on the inputs `X`.
        """
        return self.X, self.Y[:, [1]], self.Yvar[:, [1]], self.bounds

    def strength_data_by_time(self, time: float) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns the strength data for a specific time.

        Returns:
            A 3-Tuple of Tensors containing 1) the inputs X (*without* time since it is
            fixed), 2) strengths Y that are observed at `time`, and 3) empirical
            variances Yvar of Y.
        """
        X, Y, Yvar, _ = self.strength_data
        row_ind = torch.where(X[:, -1] == time)[0]
        return X[row_ind], Y[row_ind], Yvar[row_ind]

    @property
    def gwp_data(self) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """Returns the data with which to fit a strength model.

        Returns:
            A 4-Tuple of Tensors containing 1) the `n_unique x (d - 1)` unique
            compositions X *without* time since GWP does not depend on `time`, 2) the
            corresponding `n_unique x 1`-dim GWP values Y, 3) the `n_unique x 1`-dim
            empirical strength variances Yvar, and the `2 x (d - 1)`-dim bounds on X.
        """
        # removes duplicates due to multiple measurements in time, which is irrelevant for gwp
        unique_indices = self.unique_composition_indices
        X = self.X[unique_indices, :-1]
        Y = self.Y[unique_indices, 0].unsqueeze(-1)
        Yvar = self.Yvar[unique_indices, 0].unsqueeze(-1)
        X_bounds = None
        if self.bounds is not None:
            X_bounds = self.bounds[:, :-1]  # without time dimension
            if (X.amin(dim=0) < X_bounds[0, :]).any() or (
                X.amax(dim=0) > X_bounds[1, :]
            ).any():
                # raise Exception(
                print(
                    "Bounds do not hold in training data: "
                    f"{X_bounds[0, :], X.amin(dim=0) = }"
                    f"{X_bounds[1, :], X.amax(dim=0) = }"
                )
        return X, Y, Yvar, X_bounds

    @property
    def unique_compositions(self) -> Tuple[Tensor, Tensor]:
        """Returns the unique compositions and their reverse index mapping.

        Returns:
            A 2-Tuple of Tensors containing 1) the unique `n_unique x (d - 1)`-dim
            compositions `C` (without time), and 2) the reverse index mapping `rev`
            such that `C[rev]` is the original `X`.
        """
        c = self.X[:, :-1]
        c_unique, rev = c.unique(dim=0, sorted=False, return_inverse=True)
        return c_unique, rev

    @property
    def unique_composition_indices(self) -> List[int]:
        """Returns the indices of of the first occurance of each unique composition
        in `X`.

        Returns:
            A List of integer indices indicating the first occurance of each unique
            composition.
        """
        c, rev = self.unique_compositions
        rev = [r.item() for r in rev]  # converting to a list of python ints
        # indices of first occurances of unique compositions
        unique_indices = [rev.index(i) for i in range(len(c))]
        # sorting in ascending order, to be identical to collection order
        unique_indices.sort()
        return unique_indices

    def subselect_batch_names(self, names: List[str]) -> SustainableConcreteDataset:
        """Creates a subset of this dataset by selecting only the specified batch names.

        Args:
            names: A list of strings specifying the names of the batches to select.

        Returns:
            A SustainableConcreteDataset containing the selected batches.
        """
        all_inds = []
        new_batch_name_to_indices = {}
        for name, inds in self._batch_name_to_indices.items():
            if name in names:
                len_all = len(all_inds)
                new_batch_inds = list(range(len_all, len_all + len(inds)))
                new_batch_name_to_indices[name] = new_batch_inds
                all_inds.append(inds)

        return SustainableConcreteDataset(
            X=self.X[all_inds],
            Y=self.Y[all_inds],
            Ystd=self.Ystd[all_inds],
            X_columns=self.X_columns,
            Y_columns=self.Y_columns,
            Ystd_columns=self.Ystd_columns,
            batch_name_to_indices=new_batch_name_to_indices,
        )

    @property
    def X_columns(self) -> List[str]:
        """The names of the columns of `X`."""
        return self._X_columns

    @property
    def Y_columns(self) -> List[str]:
        """The names of the columns of `Y`."""
        return self._Y_columns

    @property
    def Ystd_columns(self) -> List[str]:
        """The names of the columns in `Ystd`."""
        return self._Ystd_columns


def load_concrete_strength(
    data_path: str = "data/concrete_strength.csv",
    verbose: bool = _VERBOSE,
    batch_names: Optional[List[str]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    used_columns: Optional[List[str]] = None,
) -> SustainableConcreteDataset:
    """A function to load concrete strength data from a CSV file.

    The assumptions of this function are as follows:
        - The first three columns are reserved for identifiers (e.g. Mix ID, Name, Description).
        - The fourth column to the last five columns are assumed to be composition data.
        - Immediately following the composition data is the time column, i.e. 5th to last.
        - Following the time column are four columns characterizing the output, i.e.:
            - "GWP"
            - "Strength (Mean)"
            - "Strength (Std)"
            - "# of measurements"

    To summarize, the columns format should be:
        ["Mix ID", "Name", "Description"]
        + ["Composition 1", ..., "Composition n"]  (names can be arbitrary.)
        + ["Time"]
        + ["GWP", "Strength (Mean)", "Strength (Std)", "# of measurements"]

    Args:
        data_path: The path to the data to be loaded. Defaults to "data/concrete_strength.csv".
        verbose: Toggles verbose printing of the operations applied to the data.
        batch_names: A list of strings specifying the names of the experimental batches
            that are to be loaded. If None, then all available batches will be loaded.
        dtype: A torch.dtype object specifying the desired datatype of the Tensors.
        device: A torch.device object specifying the desired device of the Tensors.
        used_columns: A list of strings specifying the names of the columns to be used.
            This can be used to bring the data into the desired format, outlined above.

    Returns:
        A SustainableConcreteDataset containing the strength and GWP data.
    """
    # loading csv into dataframe
    df = pd.read_csv(data_path, delimiter=",")

    if used_columns is not None:
        df = df[used_columns]

    # dropping any mix id that is not in batch names
    if (
        batch_names is not None
    ):  # TODO: make this safe! "contains" only works if the batch names are unique strings, not numbers
        not_in_names = df["Mix ID"].astype(bool)  # creating True series
        for batch_name in batch_names:
            not_in_names = not_in_names & (~df["Mix ID"].str.contains(batch_name))
        df = df.drop(df[not_in_names].index)

    if verbose:
        print(f"The data has {len(df)} rows and {len(df.columns)} columns, which are:")
        for column in df.columns.to_list():
            print("\t-", column)
        print()

    # first, remove rows and columns with missing data
    data_index = 3
    data_columns = df.columns[data_index:]
    is_missing = torch.tensor(df[data_columns].to_numpy()).isnan()
    n_missing = is_missing.sum(dim=0)
    missing_col_ind = n_missing > 0
    if missing_col_ind.any():
        if verbose:
            print(f"There are {missing_col_ind.sum()} columns with missing entries:")
            for name, missing in zip(
                data_columns[missing_col_ind], n_missing[missing_col_ind]
            ):
                print("\t-", name, "has", missing.item(), "missing entries.")
            print("")
            print("Removing missing rows with missing entries from data.")
        missing_row_ind = [i for i in range(len(df)) if is_missing[i].any()]
        if verbose:
            print(f"\t-Rows indices to be removed: {missing_row_ind = }")
        df = df.drop(missing_row_ind)
        if verbose:
            print(
                "\t-Number of missing values after deletion (Should be zero): "
                f"{torch.tensor(df[data_columns].to_numpy()).isnan().sum()}"
            )
            print("")

    # get batch names
    name_index = 0  # assumes mix ids are the first column of the table
    name_column = df.columns[name_index]
    mix_names = df[name_column].to_list()
    # this removes everything from the last underscore of the name
    batch_names = [name[: name.rfind("_")] for name in mix_names]
    # find unique batch names
    batch_names = unique_elements(batch_names)
    # maps batch_name to the indices of the mixes associated with the batch
    batch_name_to_indices = {
        batch_name: [
            i
            for i, name in enumerate(mix_names)
            if name[: len(batch_name)] == batch_name
        ]
        for batch_name in batch_names
    }
    if verbose:
        print("Found the following batch names:")
        for batch_name in batch_names:
            print("\t-", batch_name)
        print()

    # separating columns as inputs, outputs, and output uncertainties
    X_columns = df.columns[3:-4].to_list()
    Y_columns = ["GWP", "Strength (Mean)"]
    Ystd_columns = ["Strength (Std)"]
    if verbose:
        print("Separating model inputs and outputs:")
        print(f"Input columns: ")
        for col in X_columns:
            print("\t-", col)
        print(f"Output (Mean) columns")
        for col in Y_columns:
            print("\t-", col)
        print(f"Output (Std) columns")
        for col in Ystd_columns:
            print("\t-", col)
        print()

    # casting dataframe to torch tensors
    tkwargs = {"dtype": dtype, "device": device}
    X = torch.tensor(df[X_columns].to_numpy(), **tkwargs)
    Y = torch.tensor(df[Y_columns].to_numpy(), **tkwargs)

    if verbose:
        print(f"Negating GWP to frame as joint maximization problem.")
        print()
    Y[:, 0] = -Y[:, 0]

    if verbose:
        print(
            "Adding and setting standard deviation of GWP to uniformly small value "
            "since our estimates are deterministic."
        )
        print()
    if len(Ystd_columns) == 1:
        Ystd = torch.cat(
            (  # to use FixedNoiseGP with noiseless observations
                torch.full_like(Y[:, 0], 1e-3).unsqueeze(-1),
                torch.tensor(df[Ystd_columns].to_numpy(), **tkwargs),
            ),
            dim=-1,
        )
    else:  # assumed to have GWP as first data source
        Ystd[:, 0] = 1e-3  # allowing us to use the same model type (FixedNoiseGP)

    # dividing empirical standard deviations of strength by the number of measurements.
    if "# of measurements" in df.columns:
        if verbose:
            print(
                "Computing strength standard error of by "
                "dividing standard deviation by sqrt(# of measurements)."
            )
            print()
        n_measurements = torch.tensor(df["# of measurements"].to_numpy(), **tkwargs)
        Ystd[:, 1] /= n_measurements.sqrt()

    return SustainableConcreteDataset(
        X=X,
        Y=Y,
        Ystd=Ystd,
        X_columns=X_columns,
        Y_columns=Y_columns,
        Ystd_columns=Ystd_columns,
        batch_name_to_indices=batch_name_to_indices,
    )


def get_mortar_bounds(X_columns: List[str], verbose: bool = _VERBOSE) -> Tensor:
    """Returns bounds of columns in X for mortar mixes.

    Args:
        X_columns: Names of the columns in the input dataset.
        verbose: Whether to print what the lower and upper bounds are set to.

    Tensor:
        A `2 x d`-dim Tensor of lower and upper mortar bounds for each column of X.
    """
    bounds_dict = {
        "Cement": (0, 950),  # in grams, as opposed to the original concrete bounds
        "Fly Ash": (0, 950),
        "Slag": (0, 950),
        "Fine Aggregate": (925, 1775),  # fixed based on binder + aggregate constraint
        "Curing Temp (°C)": (0, 40),
        "Time": (0, 28),  # up to 28 days
    }

    min_binder = 100.0
    max_binder = 950.0
    bounds_dict.update(
        {
            "Water": (0.35 * min_binder, 0.5 * max_binder),
            "HRWR": (
                0,
                0.1 * max_binder,
            ),  # we are not optimizing this, but need this to fit the model
        }
    )
    bounds = torch.tensor([bounds_dict[col] for col in X_columns]).T
    if verbose:
        print("The lower and upper bounds for the respective variables are set to:")
        for col, bound in zip(X_columns, bounds.T):
            print(f"\t- {col}: [{bound[0].item()}, {bound[1].item()}]")
        print()
    return bounds


def get_mortar_constraints(
    X_columns, min_wb: float = 0.35, verbose: bool = _VERBOSE
) -> Tuple[List, List]:
    """Returns the linear equality and inequality constraints for mortar mixes.

    Args:
        X_columns: Names of columns in the input dataset.
        min_wb: Minimum water-binder ratio. Defaults to 0.35.
        verbose: Whether to print details on the constraints.

    Returns:
        A 2-Tuple of equality and inequality constraints.
    """
    # inequality constraints
    equality_dict = {
        # "Total Binder": 500.0,
        # "Fine Aggregate": 1375.0,
        "Total Binder + Fine Aggregate": 1875.0,
    }
    inequality_dict = {
        "Total Binder": (100.0, 950.0),
        "Water": (min_wb, 0.5),  # NOTE: as a proportion of total binder
    }
    if verbose:
        print("Adding linear equality constraints:")
        for key in equality_dict:
            print("\t-", key, ":", equality_dict[key])
        print(
            "NOTE: the paste content constraint is proportional to the total mass, "
            "and the water and HRWR constraints are proportional to the total binder."
        )
        print()

    equality_constraints = [
        # get_sum_equality_constraint(
        #     X_columns=X_columns,
        #     subset_names=_TOTAL_BINDER_NAMES,
        #     value=equality_dict["Total Binder"],
        # ),
        get_sum_equality_constraint(
            X_columns=X_columns,
            subset_names=_BINDER_PLUS_AGGREGATE,
            value=equality_dict["Total Binder + Fine Aggregate"],
        )
    ]
    inequality_constraints = [
        *get_binder_constraints(X_columns, *inequality_dict["Total Binder"]),
        # as long as binder is constant, the water constraint is just a bound (earlier)
        *get_water_constraints(X_columns, *inequality_dict["Water"]),
    ]
    return equality_constraints, inequality_constraints


def get_bounds(X_columns, verbose: bool = _VERBOSE) -> Tensor:
    """Returns bounds of columns in X for concrete mixes."""
    bounds_dict = {
        # NOTE: the pure cement baseline is outside of these bounds (~752), as is Dec_2022_2 (~211)
        "Cement": (300, 700),
        "Fly Ash": (0, 350),
        "Slag": (0, 450),
        "Coarse Aggregate": (800, 1950),
        "Fine Aggregate": (600, 1700),
        "Time": (0, 28),  # up to 28 days
    }

    min_binder, max_binder = 0, 0
    for name in _TOTAL_BINDER_NAMES:
        min_binder += bounds_dict[name][0]
        max_binder += bounds_dict[name][1]

    bounds_dict.update(
        {
            "Water": (0.2 * min_binder, 0.5 * max_binder),
            "HRWR": (0, 0.1 * max_binder),  # linear constraint also applies, see below
        }
    )
    bounds = torch.tensor([bounds_dict[col] for col in X_columns]).T
    if verbose:
        print("The lower and upper bounds for the respective variables are set to:")
        for col, bound in zip(X_columns, bounds.T):
            print(f"\t- {col}: [{bound[0].item()}, {bound[1].item()}]")
        print()
    return bounds


def get_concrete_constraints(X_columns, verbose: bool = _VERBOSE) -> List[T_CONSTRAINT]:
    # inequality constraints for concrete (vs. mortar) mixtures
    inequality_dict = {
        "Total Binder": (510, 1000),
        "Total Mass": (3600, 4400),
        "Paste Content": (0.16, 0.35),  # as a proportion of total mass
        "Water": (0.2, 0.5),  # as a proportion of total binder
        "HRWR": (0, 0.1),  # as a proportion of total binder
    }
    if verbose:
        print("Adding linear constraints with lower and upper limits:")
        for key in inequality_dict:
            print("\t-", key, ":", inequality_dict[key])
        print(
            "NOTE: the paste content constraint is proportional to the total mass, "
            "and the water and HRWR constraints are proportional to the total binder."
        )
        print()

    constraints = [
        *get_mass_constraints(X_columns, *inequality_dict["Total Mass"]),
        *get_binder_constraints(X_columns, *inequality_dict["Total Binder"]),
        *get_paste_constraints(X_columns, *inequality_dict["Paste Content"]),
        *get_water_constraints(X_columns, *inequality_dict["Water"]),
        *get_hrwr_constraints(X_columns, *inequality_dict["HRWR"]),
    ]
    return constraints


def get_mass_constraints(
    X_columns: List[str], lower: float, upper: float
) -> List[T_CONSTRAINT]:
    return get_sum_constraints(
        X_columns=X_columns, subset_names=_TOTAL_MASS_NAMES, lower=lower, upper=upper
    )


def get_binder_constraints(
    X_columns: List[str], lower: float, upper: float
) -> List[T_CONSTRAINT]:
    return get_sum_constraints(
        X_columns=X_columns, subset_names=_TOTAL_BINDER_NAMES, lower=lower, upper=upper
    )


def get_paste_constraints(
    X_columns: List[str], lower: float, upper: float
) -> List[T_CONSTRAINT]:
    # Paste content = (Cement + Slag + Fly Ash + Water)
    # Constraint: lower < (Paste content) / (Total Mass) < upper
    # i.e. a proportional sum constraint
    return get_proportional_sum_constraints(
        X_columns=X_columns,
        numerator_names=_PASTE_CONTENT_NAMES,
        denominator_names=_TOTAL_MASS_NAMES,
        lower=lower,
        upper=upper,
    )


def get_water_constraints(
    X_columns: List[str], lower: float, upper: float
) -> List[T_CONSTRAINT]:
    # Constraint: lower < (Water) / (Total Binder) < upper
    # i.e. a proportional sum constraint
    return get_proportional_sum_constraints(
        X_columns=X_columns,
        numerator_names=["Water"],
        denominator_names=_TOTAL_BINDER_NAMES,
        lower=lower,
        upper=upper,
    )


def get_hrwr_constraints(
    X_columns: List[str], lower: float, upper: float
) -> List[T_CONSTRAINT]:
    # Constraint: lower < (HRWR) / (Total Binder) < upper
    # i.e. a proportional sum constraint
    return get_proportional_sum_constraints(
        X_columns=X_columns,
        numerator_names=["HRWR"],
        denominator_names=_TOTAL_BINDER_NAMES,
        lower=lower,
        upper=upper,
    )


def get_sum_constraints(
    X_columns: List[str], subset_names: List[str], lower: float, upper: float
) -> List[T_CONSTRAINT]:
    lower_constraint = get_sum_equality_constraint(X_columns, subset_names, value=lower)
    upper_constraint = get_sum_equality_constraint(X_columns, subset_names, value=upper)
    # rephrasing the upper as a lower bound
    upper_constraint = (upper_constraint[0], -upper_constraint[1], -upper_constraint[2])
    return [lower_constraint, upper_constraint]


def get_sum_equality_constraint(
    X_columns: List[str], subset_names: List[str], value: float
) -> T_CONSTRAINT:
    _, coeffs = get_subset_sum_tensors(X_columns=X_columns, subset_names=subset_names)
    # can throw out indices for which coeffs is zero if we don't recombine coefficients
    nz_ind = coeffs != 0
    ind, coeffs = torch.arange(len(coeffs))[nz_ind], coeffs[nz_ind]
    return (ind, coeffs, value)


def get_proportional_sum_constraints(
    X_columns: List[str],
    numerator_names: List[str],
    denominator_names: List[str],
    lower: float,
    upper: float,
) -> List[T_CONSTRAINT]:
    """Converts a constraint on a fraction of two subset sums into a linear form,
    i.e. if the constraint is of the form

        `lower < (sum of numerator_names) / (sum of denominator_names) < upper`,

    then `(numerator) < upper * (denominator)` and so
    `upper * (denominator) - (numerator) > 0`, and
    `(numerator) - lower * (denominator) > 0`.

    Args:
        X_columns: The column (variable) names of the inputs `X`.
        numerator_names: The subset of variable names whose sum to use as the numerator.
        denominator_names: The subset of variable names whose sum to use as the denominator.
        lower: The lower limit of the fractional constraint.
        upper: The upper limit of the fractional constraint.

    Returns:
        A list of tuples of the form `(indices, coefficients, constant)` that represents
        the porportional sum constraint in its linear representation.
    """
    _, num_coeffs = get_subset_sum_tensors(
        X_columns=X_columns, subset_names=numerator_names
    )
    _, den_coeffs = get_subset_sum_tensors(
        X_columns=X_columns, subset_names=denominator_names
    )

    # upper constraint
    upper_coeffs = upper * den_coeffs - num_coeffs
    upper_nz_ind = upper_coeffs != 0
    upper_ind = torch.arange(len(upper_coeffs))[upper_nz_ind]
    upper_coeffs = upper_coeffs[upper_nz_ind]

    # lower constraint
    lower_coeffs = num_coeffs - lower * den_coeffs
    lower_nz_ind = lower_coeffs != 0
    lower_ind = torch.arange(len(lower_coeffs))[lower_nz_ind]
    lower_coeffs = lower_coeffs[lower_nz_ind]

    return [(upper_ind, upper_coeffs, 0.0), (lower_ind, lower_coeffs, 0.0)]


def get_subset_sum_tensors(
    X_columns: List[str], subset_names: List[str]
) -> Tuple[Tensor, Tensor]:
    """Returns indices and coefficients such that `X[indices].dot(coeffs) == X[indices].sum()`,
    where indices are the indices of subset_names in X_columns.

    Args:
        X_columns: The column (variable) names.
        subset_names: The subset of variable names whose sum to compute.

    Returns:
        A Tuple of Tensors `indices` and `coeffs` with which to compute the subset sum.
    """
    indices = [X_columns.index(name) for name in subset_names]
    coeffs = torch.zeros(len(X_columns))
    coeffs[indices] = 1
    return indices, coeffs


def get_reference_point() -> Tensor:
    # gwp = -430.0  # based on existing minimum in the data (pure cement)
    gwp = -150.0  # chosen to hone in on the greener and strong region
    strength_day_1 = 1000
    # strength_day_7 = 3000
    strength_day_28 = 5000
    return torch.tensor([gwp, strength_day_1, strength_day_28], dtype=torch.double)


def get_day_zero_data(X: Tensor, bounds: Optional[Tensor], n: int = 128):
    """Computes a tensor of n sobol points that satisfy the bounds, appended with a
    zeros tensor. Useful to condition the strength GP to be zero at day zero.

    Args:
        X: The input tensor.
        bounds: The bounds of the input tensor. If None, will be inferred from X.

    Returns:
        A tensor of n sobol points that satisfy the bounds, appended with a zeros
        tensor, corresponding to the strength at day zero.
    """
    if bounds is None:
        bounds = torch.stack((X.amin(dim=0), X.amax(dim=0)))

    d = bounds.shape[-1]
    sobol_engine = torch.quasirandom.SobolEngine(dimension=(d - 1))  # excluding time
    X_0 = sobol_engine.draw(n)
    X_0 = torch.cat((X_0, torch.zeros(n, 1)), dim=-1)  # append time (zero)
    a, b = bounds[0], bounds[1]
    X_0 = (b - a) * X_0 + a  # scaling according to bounds
    Y_0 = torch.zeros(n, 1)  #  zero strength
    Yvar_0 = torch.full((n, 1), 1e-4)  #  with large certainty
    return X_0, Y_0, Yvar_0


# NOTE: moved over from saas implementation.
# TODO: move this to GPyTorch.
class LogTransformedInterval(Interval):
    def __init__(
        self,
        lower_bound: Tensor,
        upper_bound: Tensor,
        initial_value: Optional[Tensor] = None,
    ):
        """Modification of the GPyTorch interval class.

        The Interval class in GPyTorch will map the parameter to the range [0, 1] before
        applying the inverse transform. We don't want to do this when using log as an
        inverse transform. This class will skip this step and apply the log transform
        directly to the parameter values so we can optimize log(parameter) under the bound
        constraints log(lower) <= log(parameter) <= log(upper).
        """
        super().__init__(
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            transform=torch.exp,
            inv_transform=torch.log,
            initial_value=initial_value,
        )

        # Save the untransformed initial value
        self.register_buffer(
            "initial_value_untransformed",
            torch.tensor(initial_value).to(self.lower_bound)
            if initial_value is not None
            else None,
        )

        if settings.debug.on():
            max_bound = torch.max(self.upper_bound)
            min_bound = torch.min(self.lower_bound)
            if max_bound == math.inf or min_bound == -math.inf:
                raise RuntimeError(
                    "Cannot make an Interval directly with non-finite bounds. Use a "
                    "derived class like GreaterThan or LessThan instead."
                )

    def transform(self, tensor):
        if not self.enforced:
            return tensor

        transformed_tensor = self._transform(tensor)
        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        if not self.enforced:
            return transformed_tensor

        tensor = self._inv_transform(transformed_tensor)
        return tensor


def unique_elements(x: List) -> List:
    """Returns unique elements of x in the same order as their first
    occurrance in the input list.
    """
    return list(dict.fromkeys(x))


def predict_pareto(
    model_list: ModelListGP,
    pareto_dims: List[int],
    ref_point: Tensor,
    bounds: Tensor,
    equality_constraints,
    inequality_constraints,
    num_candidates: int = 4096,
) -> Tuple[Tensor, Tensor]:
    """Use the `model_list` to approximate the predictive Pareto frontier of the
    output dimensions specified by `pareto_dims`.

    Args:
        model_list: A ModelListGP, usually generated by `SustainableConcreteModel`'s
            `get_model_list`.
        pareto_dims: A list of integers specifying two output dimensions for which to
            approximate the predicted Pareto frontier.
        ref_point: The reference point for computing the Pareto frontier.
        bounds: The bounds of the input variables of the model. NOTE: These bounds do
            not have to be the same as those used to train the model. In fact, an
            interesting application of this function is to use different bounds to
            get quantitative results for "what-if" scenarios.
        equality_constraints: Equality constraints. Similar to the bounds, these can be
            different than those used to train the model to explore "what-if" scenarios.
        inequality_constraints: Inequality constraints. Similar to the bounds, these can
            be different than those used to train the model to explore "what-if" scenarios.
        num_candidates: The number of random inputs to generate in order to approximate
            the Pareto frontier. The higher the number of candidates, the more accurate.

    Returns:
        A 2-Tuple of Tensors containing the predicted Pareto-optimal outputs and their
        predictive uncertainties, i.e. predictive standard deviations.
    """
    X = sample_q_batches_from_polytope(
        n=num_candidates,
        q=1,
        bounds=bounds,
        n_burnin=10000,
        n_thinning=2,  # don't actually need to thin for this problem
        seed=1234,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints,
    )
    post = model_list.posterior(X)
    Y = post.mean
    Ystd = post.variance.sqrt()
    X = X.squeeze(-2)  # squeezing q
    Y = Y.squeeze(-2)  # squeezing q
    Ystd = Ystd.squeeze(-2)  # squeezing q

    # subselect dimensions with which to compute Pareto frontier
    Y = Y[..., pareto_dims]
    Ystd = Ystd[..., pareto_dims]
    ref_point = ref_point[pareto_dims]

    # compute pareto optimal points
    is_pareto = is_non_dominated(Y)
    X, Y, Ystd = X[is_pareto], Y[is_pareto], Ystd[is_pareto]

    # remove any points that do not satisfy the reference point
    better_than_ref = (Y > ref_point).all(dim=-1)
    X, Y, Ystd = X[better_than_ref], Y[better_than_ref], Ystd[better_than_ref]
    # sort by firt dimension to enable easier plotting
    indices = Y[..., 0].argsort()
    X, Y, Ystd = X[indices], Y[indices], Ystd[indices]
    return Y, Ystd
