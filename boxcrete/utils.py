#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Defines concrete strength data loaders, search space constraints, and other utilities.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch
from botorch.models import ModelList
from botorch.optim.initializers import sample_q_batches_from_polytope
from botorch.utils.multi_objective import is_non_dominated
from torch import Tensor

logger = logging.getLogger(__name__)

# Path to the repository root, resolved from the package location.
# This allows data loading to work regardless of the current working directory.
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_DIR, "data", "boxcrete_data.csv")

# linear constraint type (ind, coeffs, value)
T_CONSTRAINT = tuple[Tensor, Tensor, float]

_TOTAL_BINDER_NAMES = ["Cement (kg/m3)", "Fly Ash (kg/m3)", "Slag (kg/m3)"]
_PASTE_CONTENT_NAMES = _TOTAL_BINDER_NAMES + ["Water (kg/m3)"]
_MORTAR_BINDER_PLUS_AGGREGATE = _TOTAL_BINDER_NAMES + ["Fine Aggregate (kg/m3)"]
_TOTAL_MASS_NAMES = _PASTE_CONTENT_NAMES + [
    "HRWR (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Fine Aggregate (kg/m3)",
]  # MRWR excluded: negligible contribution to total mass
DEFAULT_X_COLUMNS = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "MRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Material Source",
    "Temp (C)",
    "Time",  # last dimension is assumed to be time
]
DEFAULT_Y_COLUMNS = ["GWP", "Strength (Mean)"]
DEFAULT_YSTD_COLUMNS = ["Strength (Std)"]

MORTAR_BOUNDS_DICT = {
    "Cement (kg/m3)": (0, 950),
    "Fly Ash (kg/m3)": (0, 950),
    "Slag (kg/m3)": (0, 950),
    "Fine Aggregate (kg/m3)": (925, 1775),
    "Temp (C)": (0, 40),
    "Time": (0, 28),
}

CONCRETE_BOUNDS_DICT = {
    "Cement (kg/m3)": (0, 1000),
    "Fly Ash (kg/m3)": (0, 600),
    "Slag (kg/m3)": (0, 1300),
    "Coarse Aggregates (kg/m3)": (0, 1600),
    "Fine Aggregate (kg/m3)": (400, 2600),
    "Material Source": (0, 1),
    "MRWR (kg/m3)": (
        0,
        1,
    ),  # effectively zero in training data; small range avoids NaN in normalization
    "Temp (C)": (0, 40),
    "Time": (0, 28),
}

DEFAULT_BOUNDS_DICT = CONCRETE_BOUNDS_DICT

MORTAR_CONSTRAINTS = dict(
    equality_sums=[(_MORTAR_BINDER_PLUS_AGGREGATE, 1875.0)],
    binder_bounds=(100.0, 950.0),
    mass_bounds=None,
    paste_bounds=None,
    water_binder_bounds=(0.35, 0.5),
)

CONCRETE_CONSTRAINTS = dict()


class SustainableConcreteDataset:
    """A container for concrete strength and GWP data with composition inputs.

    Stores input features (composition + time), outputs (GWP and strength), and
    their uncertainties. Provides convenience methods for splitting data by time
    and by unique compositions.
    """

    def __init__(
        self,
        X: Tensor,
        Y: Tensor,
        Ystd: Tensor,
        X_columns: list[str],
        Y_columns: list[str],
        Ystd_columns: list[str],
        bounds: Tensor | None = None,
        batch_name_to_indices: dict[str, list[int]] | None = None,
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
            ValueError: If the last column of `X` is not time.
        """
        if X_columns[-1].lower() != "time":
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
        1) `Y[i, 0]` is the GWP value of the ith sample.
        2) `Y[i, 1]` is the measured strength value for the ith sample.
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
    def strength_data(self) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Returns the data with which to fit a strength model.

        Returns:
            A 4-tuple of Tensors containing 1) the inputs `X` (composition and time),
            2) observed strengths `Y`, 3) empirical strength variances `Yvar`, and
            4) the `2 x d`-dim bounds on the inputs `X`.
        """
        return self.X, self.Y[:, [1]], self.Yvar[:, [1]], self.bounds

    def strength_data_by_time(self, time: float) -> tuple[Tensor, Tensor, Tensor]:
        """Returns the strength data for a specific time.

        Args:
            time: The curing time (in days) to filter by.

        Returns:
            A 3-tuple of Tensors containing 1) the inputs X (*without* time since it is
            fixed), 2) strengths Y that are observed at `time`, and 3) empirical
            variances Yvar of Y.
        """
        X, Y, Yvar, _ = self.strength_data
        row_ind = torch.where(X[:, -1] == time)[0]
        return X[row_ind], Y[row_ind], Yvar[row_ind]

    @property
    def gwp_data(self) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Returns the data with which to fit a GWP model.

        Returns:
            A 4-tuple of Tensors containing 1) the `n_unique x (d - 1)` unique
            compositions X *without* time since GWP does not depend on `time`, 2) the
            corresponding `n_unique x 1`-dim GWP values Y, 3) the `n_unique x 1`-dim
            GWP variances Yvar, and the `2 x (d - 1)`-dim bounds on X.
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
                logger.warning(  # pragma: no cover
                    "Bounds do not hold in training data: "
                    f"{X_bounds[0, :], X.amin(dim=0) = }"
                    f"{X_bounds[1, :], X.amax(dim=0) = }"
                )
        return X, Y, Yvar, X_bounds

    @property
    def unique_compositions(self) -> tuple[Tensor, Tensor]:
        """Returns the unique compositions and their reverse index mapping.

        Returns:
            A 2-tuple of Tensors containing 1) the unique `n_unique x (d - 1)`-dim
            compositions `C` (without time), and 2) the reverse index mapping `rev`
            such that `C[rev]` is the original `X`.
        """
        c = self.X[:, :-1]
        c_unique, rev = c.unique(dim=0, sorted=False, return_inverse=True)
        return c_unique, rev

    @property
    def unique_composition_indices(self) -> list[int]:
        """Returns the indices of the first occurrence of each unique composition
        in `X`.

        Returns:
            A list of integer indices indicating the first occurrence of each unique
            composition.
        """
        c, rev = self.unique_compositions
        rev = [r.item() for r in rev]  # converting to a list of python ints
        # indices of first occurrences of unique compositions
        unique_indices = [rev.index(i) for i in range(len(c))]
        # sorting in ascending order, to be identical to collection order
        unique_indices.sort()
        return unique_indices

    def subselect_batch_names(self, names: list[str]) -> SustainableConcreteDataset:
        """Creates a subset of this dataset by selecting only the specified batch names.

        Args:
            names: A list of strings specifying the names of the batches to select.

        Returns:
            A SustainableConcreteDataset containing the selected batches.
        """
        all_inds = []
        new_batch_name_to_indices = {}
        if self._batch_name_to_indices is None:
            raise ValueError("batch_name_to_indices is None.")

        for name, inds in self._batch_name_to_indices.items():
            if name in names:
                len_all = len(all_inds)
                new_batch_inds = list(range(len_all, len_all + len(inds)))
                new_batch_name_to_indices[name] = new_batch_inds
                all_inds.extend(inds)

        return SustainableConcreteDataset(
            X=self.X[all_inds],
            Y=self.Y[all_inds],
            Ystd=self.Ystd[all_inds],
            X_columns=self.X_columns,
            Y_columns=self.Y_columns,
            Ystd_columns=self.Ystd_columns,
            bounds=self.bounds,
            batch_name_to_indices=new_batch_name_to_indices,
        )

    @property
    def X_columns(self) -> list[str]:
        """The names of the columns of `X`."""
        return self._X_columns

    @property
    def Y_columns(self) -> list[str]:
        """The names of the columns of `Y`."""
        return self._Y_columns

    @property
    def Ystd_columns(self) -> list[str]:
        """The names of the columns in `Ystd`."""
        return self._Ystd_columns


def load_concrete_strength(
    data_path: str | pd.DataFrame = DATA_PATH,
    batch_names: list[str] | None = None,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    mix_name_column: str = "Mix Name",
    X_columns: list[str] = DEFAULT_X_COLUMNS,
    Y_columns: list[str] = DEFAULT_Y_COLUMNS,
    Ystd_columns: list[str] = DEFAULT_YSTD_COLUMNS,
    process_batch_names_from_mix_name: bool = False,
    bounds_dict: dict[str, tuple[float, float]] = DEFAULT_BOUNDS_DICT,
) -> SustainableConcreteDataset:
    """Loads concrete strength data from a CSV file or DataFrame.

    The function expects the following column structure:
        - An identifier column (e.g. Mix Name).
        - Composition columns corresponding to `X_columns`.
        - Output columns: "GWP", "Strength (Mean)", "Strength (Std)".
        - Optionally "# of measurements" for computing standard errors.

    Args:
        data_path: Path to a CSV file or a pandas DataFrame. Defaults to
            `DATA_PATH` (``data/boxcrete_data.csv``).
        batch_names: Optional list of batch name substrings. If provided, only
            rows whose `mix_name_column` value contains one of these strings
            are kept.
        dtype: Desired torch dtype for the output tensors.
        device: Desired torch device for the output tensors.
        mix_name_column: Name of the column containing the mix identifier.
        X_columns: Column names to use as model inputs.
        Y_columns: Column names to use as model outputs.
        Ystd_columns: Column names to use as output standard deviations.
        process_batch_names_from_mix_name: Whether to parse batch names from
            ``mix_name_column`` using the ``<batch>_<number>`` convention.
        bounds_dict: Mapping from column name to ``(lower, upper)`` bounds.

    Returns:
        A SustainableConcreteDataset containing the loaded data.
    """
    # loading csv into dataframe
    if isinstance(data_path, str):
        df = pd.read_csv(data_path, delimiter=",")
    else:
        df = data_path

    # dropping any mix id that is not in batch names
    if batch_names is not None:
        not_in_names = df[mix_name_column].astype(bool)  # creating True series
        for batch_name in batch_names:
            not_in_names = not_in_names & (
                ~df[mix_name_column].str.contains(batch_name)
            )
        df = df.drop(df[not_in_names].index)

    logger.info(
        f"The data has {len(df)} rows and {len(df.columns)} columns, which are:"
    )
    for column in df.columns.to_list():
        logger.info("  - %s", column)

    # first, remove rows and columns with missing data
    data_columns = X_columns + Y_columns + Ystd_columns
    data_columns = np.array(data_columns)
    is_missing = torch.tensor(df[data_columns].to_numpy()).isnan()
    n_missing = is_missing.sum(dim=0)
    missing_col_ind = n_missing > 0
    if missing_col_ind.any():
        logger.info(f"There are {missing_col_ind.sum()} columns with missing entries:")
        logger.info(f"{missing_col_ind=}")
        logger.info(f"{data_columns=}")
        logger.info(f"{n_missing=}")
        for name, missing in zip(
            data_columns[missing_col_ind], n_missing[missing_col_ind]
        ):
            logger.info("  - %s has %s missing entries.", name, missing.item())
        logger.info("Removing missing rows with missing entries from data.")
        missing_row_ind = [i for i in range(len(df)) if is_missing[i].any()]
        logger.info(f"  -Rows indices to be removed: {missing_row_ind = }")
        df = df.drop(missing_row_ind)
        logger.info(
            "  -Number of missing values after deletion (Should be zero): "
            f"{torch.tensor(df[data_columns].to_numpy()).isnan().sum()}"
        )

    # assumes mix ids are the first column of the table
    if process_batch_names_from_mix_name:
        # get batch names assuming old name formatting
        mix_names = df[mix_name_column].to_list()
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
    else:
        batch_names = None
        batch_name_to_indices = None

    if batch_names is None:
        logger.info("Found no batch names.")
    else:
        logger.info("Found the following batch names:")
        for batch_name in batch_names:
            logger.info("  - %s", batch_name)

    # separating columns as inputs, outputs, and output uncertainties
    logger.info("Separating model inputs and outputs:")
    logger.info("Input columns: ")
    for col in X_columns:
        logger.info("  - %s", col)
    logger.info("Output (Mean) columns")
    for col in Y_columns:
        logger.info("  - %s", col)
    logger.info("Output (Std) columns")
    for col in Ystd_columns:
        logger.info("  - %s", col)

    # casting dataframe to torch tensors
    tkwargs = {"dtype": dtype, "device": device}
    X = torch.tensor(df[X_columns].to_numpy(), **tkwargs)
    Y = torch.tensor(df[Y_columns].to_numpy(), **tkwargs)

    logger.info("Negating GWP to frame as joint maximization problem.")
    Y[:, 0] = -Y[:, 0]

    logger.info(
        "Adding and setting standard deviation of GWP to uniformly small value "
        "since our estimates are deterministic."
    )
    if len(Ystd_columns) == 1:
        Ystd = torch.cat(
            (  # to use FixedNoiseGP with noiseless observations
                torch.full_like(Y[:, 0], 1e-3).unsqueeze(-1),
                torch.tensor(df[Ystd_columns].to_numpy(), **tkwargs),
            ),
            dim=-1,
        )
    else:
        raise NotImplementedError(
            "Multiple Ystd columns not supported yet."
        )  # pragma: no cover

    # dividing empirical standard deviations of strength by the number of measurements.
    if "# of measurements" in df.columns:
        logger.info(
            "Computing strength standard error of by "
            "dividing standard deviation by sqrt(# of measurements)."
        )
        n_measurements = torch.tensor(df["# of measurements"].to_numpy(), **tkwargs)
        Ystd[:, 1] = Ystd[:, 1] / n_measurements.sqrt()

    bounds = get_bounds(X_columns=X_columns, bounds_dict=bounds_dict)
    return SustainableConcreteDataset(
        X=X,
        Y=Y,
        Ystd=Ystd,
        X_columns=X_columns,
        Y_columns=Y_columns,
        Ystd_columns=Ystd_columns,
        bounds=bounds,
        batch_name_to_indices=batch_name_to_indices,
    )


def get_bounds(
    X_columns: list[str],
    bounds_dict: dict[str, tuple[float, float]] = DEFAULT_BOUNDS_DICT,
) -> Tensor:
    """Returns a ``2 x d`` bounds tensor for the given columns.

    Columns present in ``bounds_dict`` get their bounds directly.  For
    ``"Water (kg/m3)"`` and ``"HRWR (kg/m3)"``, bounds are derived from
    the binder range: Water ∈ [0.2 × min_binder, 0.5 × max_binder] and
    HRWR ∈ [0, 0.1 × max_binder].

    Args:
        X_columns: Column names of the input features.
        bounds_dict: Mapping from column name to ``(lower, upper)`` bounds.
            Defaults to ``DEFAULT_BOUNDS_DICT`` (concrete bounds).

    Returns:
        A ``2 x d``-dim Tensor of lower and upper bounds for each column.
    """
    min_binder = 0.0
    max_binder = 0.0
    for name in _TOTAL_BINDER_NAMES:
        if name in bounds_dict:
            min_binder += bounds_dict[name][0]
            max_binder += bounds_dict[name][1]

    bounds_dict = dict(bounds_dict)  # copy to avoid mutating the original
    bounds_dict.setdefault("Water (kg/m3)", (0.2 * min_binder, 0.5 * max_binder))
    bounds_dict.setdefault("HRWR (kg/m3)", (0, 0.1 * max_binder))

    # Columns not in bounds_dict get (0, 0) bounds (e.g. Coarse Aggregates in
    # mortar mode, or MRWR when not relevant).
    bounds = torch.tensor([bounds_dict.get(col, (0, 0)) for col in X_columns]).T
    logger.info("The lower and upper bounds for the respective variables are set to:")
    for col, bound in zip(X_columns, bounds.T):
        logger.info(f"  - {col}: [{bound[0].item()}, {bound[1].item()}]")
    return bounds


def get_constraints(
    X_columns: list[str],
    equality_sums: list[tuple[list[str], float]] | None = None,
    binder_bounds: tuple[float, float] | None = (510, 1000),
    mass_bounds: tuple[float, float] | None = (3600, 4400),
    paste_bounds: tuple[float, float] | None = (0.16, 0.35),
    water_binder_bounds: tuple[float, float] = (0.2, 0.5),
    hrwr_binder_bounds: tuple[float, float] | None = (0.0, 0.1),
) -> tuple[list[T_CONSTRAINT], list[T_CONSTRAINT]]:
    """Returns equality and inequality constraints for concrete/mortar optimisation.

    This single function replaces the former ``get_concrete_constraints`` and
    ``get_mortar_constraints``.  Each constraint group can be disabled by passing
    ``None``.  Preset configurations are available as ``MORTAR_CONSTRAINTS`` and
    ``CONCRETE_CONSTRAINTS`` dictionaries that can be unpacked into this function.

    Example usage::

        # Concrete (all defaults)
        eq, ineq = get_constraints(X_columns)

        # Mortar (preset)
        eq, ineq = get_constraints(X_columns, **MORTAR_CONSTRAINTS)

    Args:
        X_columns: Column names of the input features.
        equality_sums: Optional list of ``(subset_names, value)`` pairs that
            create sum-equality constraints.  Each entry constrains the sum of
            the named columns to equal ``value``.
        binder_bounds: ``(lower, upper)`` on total binder, or ``None`` to skip.
        mass_bounds: ``(lower, upper)`` on total mass, or ``None`` to skip.
        paste_bounds: ``(lower, upper)`` on paste/mass ratio, or ``None`` to skip.
        water_binder_bounds: ``(lower, upper)`` on water/binder ratio.
        hrwr_binder_bounds: ``(lower, upper)`` on HRWR/binder ratio, or ``None``
            to skip.

    Returns:
        A tuple of ``(equality_constraints, inequality_constraints)``.
    """
    logger.info("Adding linear constraints with lower and upper limits:")
    logger.info("  - Total Binder: %s", binder_bounds)
    logger.info("  - Total Mass: %s", mass_bounds)
    logger.info("  - Paste Content: %s", paste_bounds)
    logger.info("  - Water/Binder: %s", water_binder_bounds)
    logger.info("  - HRWR/Binder: %s", hrwr_binder_bounds)
    logger.info(
        "NOTE: the paste content constraint is proportional to the total mass, "
        "and the water and HRWR constraints are proportional to the total binder."
    )

    equality_constraints: list[T_CONSTRAINT] = []
    if equality_sums is not None:
        for subset_names, value in equality_sums:
            equality_constraints.append(
                get_sum_equality_constraint(
                    X_columns=X_columns,
                    subset_names=subset_names,
                    value=value,
                )
            )

    inequality_constraints: list[T_CONSTRAINT] = []

    if mass_bounds is not None:
        inequality_constraints.extend(
            get_sum_constraints(
                X_columns=X_columns,
                subset_names=_TOTAL_MASS_NAMES,
                lower=mass_bounds[0],
                upper=mass_bounds[1],
            )
        )

    if binder_bounds is not None:
        inequality_constraints.extend(
            get_sum_constraints(
                X_columns=X_columns,
                subset_names=_TOTAL_BINDER_NAMES,
                lower=binder_bounds[0],
                upper=binder_bounds[1],
            )
        )

    if paste_bounds is not None:
        inequality_constraints.extend(
            get_proportional_sum_constraints(
                X_columns=X_columns,
                numerator_names=_PASTE_CONTENT_NAMES,
                denominator_names=_TOTAL_MASS_NAMES,
                lower=paste_bounds[0],
                upper=paste_bounds[1],
            )
        )

    inequality_constraints.extend(
        get_proportional_sum_constraints(
            X_columns=X_columns,
            numerator_names=["Water (kg/m3)"],
            denominator_names=_TOTAL_BINDER_NAMES,
            lower=water_binder_bounds[0],
            upper=water_binder_bounds[1],
        )
    )

    if hrwr_binder_bounds is not None:
        inequality_constraints.extend(
            get_proportional_sum_constraints(
                X_columns=X_columns,
                numerator_names=["HRWR (kg/m3)"],
                denominator_names=_TOTAL_BINDER_NAMES,
                lower=hrwr_binder_bounds[0],
                upper=hrwr_binder_bounds[1],
            )
        )

    return equality_constraints, inequality_constraints


def get_cement_replacement_constraints(
    X_columns: list[str],
    lower: float,
    upper: float,
    binder_names: list[str] = _TOTAL_BINDER_NAMES,
) -> list[T_CONSTRAINT]:
    """Constrains the supplementary cementitious material (SCM) replacement ratio.

    The constraint enforces ``lower ≤ SCM / binder ≤ upper``, where SCM is the
    sum of all binder components except cement.

    Args:
        X_columns: Column names of the input features.
        lower: Lower bound on the SCM replacement ratio.
        upper: Upper bound on the SCM replacement ratio.
        binder_names: Names of the binder columns.

    Returns:
        A list of inequality constraint tuples.
    """
    scm_names = list(set(binder_names) - {"Cement (kg/m3)"})
    return get_proportional_sum_constraints(
        X_columns=X_columns,
        numerator_names=scm_names,
        denominator_names=binder_names,
        lower=lower,
        upper=upper,
    )


def get_total_water_reducer_constraints(
    X_columns: list[str], lower: float, upper: float
) -> list[T_CONSTRAINT]:
    """Constrains the total water reducer (HRWR + optional MRWR) to binder ratio.

    If ``"MRWR (kg/m3)"`` is present in ``X_columns`` it is included in the
    numerator; otherwise only ``"HRWR (kg/m3)"`` is used.

    Args:
        X_columns: Column names of the input features.
        lower: Lower bound on the water-reducer / binder ratio.
        upper: Upper bound on the water-reducer / binder ratio.

    Returns:
        A list of inequality constraint tuples.
    """
    numerator_names = ["HRWR (kg/m3)"]
    if "MRWR (kg/m3)" in X_columns:
        numerator_names.append("MRWR (kg/m3)")
    return get_proportional_sum_constraints(
        X_columns=X_columns,
        numerator_names=numerator_names,
        denominator_names=_TOTAL_BINDER_NAMES,
        lower=lower,
        upper=upper,
    )


def get_aggregate_constraint(
    X_columns: list[str], lower: float, upper: float
) -> list[T_CONSTRAINT]:
    """Constrains the fine-to-coarse aggregate ratio.

    Enforces ``lower ≤ Fine Aggregate / Coarse Aggregates ≤ upper``.

    Args:
        X_columns: Column names of the input features.
        lower: Lower bound on the fine/coarse aggregate ratio.
        upper: Upper bound on the fine/coarse aggregate ratio.

    Returns:
        A list of inequality constraint tuples.
    """
    return get_proportional_sum_constraints(
        X_columns=X_columns,
        numerator_names=["Fine Aggregate (kg/m3)"],
        denominator_names=["Coarse Aggregates (kg/m3)"],
        lower=lower,
        upper=upper,
    )


def get_sum_constraints(
    X_columns: list[str], subset_names: list[str], lower: float, upper: float
) -> list[T_CONSTRAINT]:
    """Creates inequality constraints bounding the sum of a subset of columns.

    Enforces ``lower ≤ sum(subset) ≤ upper``.

    Args:
        X_columns: Column names of the input features.
        subset_names: Columns whose sum to constrain.
        lower: Lower bound on the sum.
        upper: Upper bound on the sum.

    Returns:
        A list of two inequality constraint tuples (lower and upper).
    """
    lower_constraint = get_sum_equality_constraint(X_columns, subset_names, value=lower)
    upper_constraint = get_sum_equality_constraint(X_columns, subset_names, value=upper)
    # rephrasing the upper as a lower bound
    upper_constraint = (upper_constraint[0], -upper_constraint[1], -upper_constraint[2])
    return [lower_constraint, upper_constraint]


def get_sum_equality_constraint(
    X_columns: list[str], subset_names: list[str], value: float
) -> T_CONSTRAINT:
    """Creates an equality constraint on the sum of a subset of columns.

    Enforces ``sum(subset) == value``.

    Args:
        X_columns: Column names of the input features.
        subset_names: Columns whose sum to constrain.
        value: The required sum value.

    Returns:
        A constraint tuple ``(indices, coefficients, value)``.
    """
    _, coeffs = get_subset_sum_tensors(X_columns=X_columns, subset_names=subset_names)
    # can throw out indices for which coeffs is zero if we don't recombine coefficients
    nz_ind = coeffs != 0
    ind, coeffs = torch.arange(len(coeffs))[nz_ind], coeffs[nz_ind]
    return (ind, coeffs, value)


def get_proportional_sum_constraints(
    X_columns: list[str],
    numerator_names: list[str],
    denominator_names: list[str],
    lower: float,
    upper: float,
) -> list[T_CONSTRAINT]:
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
    X_columns: list[str], subset_names: list[str]
) -> tuple[list[int], Tensor]:
    """Returns indices and coefficients such that `X[indices].dot(coeffs) == X[indices].sum()`,
    where indices are the indices of subset_names in X_columns.

    Args:
        X_columns: The column (variable) names.
        subset_names: The subset of variable names whose sum to compute.

    Returns:
        A tuple of `indices` (list of ints) and `coeffs` (Tensor) with which to
        compute the subset sum.
    """
    indices = [X_columns.index(name) for name in subset_names]
    coeffs = torch.zeros(len(X_columns))
    coeffs[indices] = 1
    return indices, coeffs


MORTAR_REFERENCE_POINT = torch.tensor([-400.0, 1000.0, 5000.0], dtype=torch.double)
CONCRETE_REFERENCE_POINT = torch.tensor([-200.0, 1000.0, 5000.0], dtype=torch.double)


def get_reference_point(optimization_mode: str = "concrete") -> Tensor:
    """Returns a reference point for Pareto frontier computation.

    The reference point specifies minimum acceptable values for each objective
    (GWP, 1-day strength, 28-day strength). Solutions that do not dominate
    this point are excluded from the Pareto frontier.

    Args:
        optimization_mode: ``"concrete"`` (default) or ``"mortar"``.

    Returns:
        A 3-element Tensor ``[-GWP_threshold, 1-day_threshold, 28-day_threshold]``.
    """
    if optimization_mode == "mortar":
        return MORTAR_REFERENCE_POINT.clone()
    return CONCRETE_REFERENCE_POINT.clone()


def get_day_zero_data(X: Tensor, bounds: Tensor | None, n: int = 128):
    """Computes a tensor of n sobol points that satisfy the bounds, appended with a
    zeros tensor. Useful to condition the strength GP to be zero at day zero.

    Args:
        X: The input tensor.
        bounds: The bounds of the input tensor. If None, will be inferred from X.
        n: The number of sobol points to generate.

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


def unique_elements(x: list) -> list:
    """Returns unique elements of x in the same order as their first
    occurrence in the input list.

    Args:
        x: A list of elements (possibly with duplicates).

    Returns:
        A list containing the unique elements in first-occurrence order.
    """
    return list(dict.fromkeys(x))


def reduce_to_optimization_space(
    bounds: Tensor,
    equality_constraints: list[T_CONSTRAINT],
    inequality_constraints: list[T_CONSTRAINT],
    fixed_features: dict[int, float],
) -> tuple[Tensor, list[T_CONSTRAINT], list[T_CONSTRAINT]]:
    """Removes fixed-feature dimensions from bounds and remaps constraint indices.

    When certain input features are fixed (e.g. via ``FixedFeatureModel``),
    the optimisation lives in a reduced-dimensional space.  This function
    projects bounds and linear constraints into that reduced space by:

    1. Dropping the fixed columns from ``bounds``.
    2. For each constraint ``coeffs @ X[indices] (>= or ==) value``,
       absorbing the fixed features' contributions into the constant
       and re-indexing the remaining entries.

    Args:
        bounds: ``2 x d`` bounds tensor in the full space.
        equality_constraints: List of ``(indices, coeffs, value)`` tuples
            in the full space.
        inequality_constraints: List of ``(indices, coeffs, value)`` tuples
            in the full space.
        fixed_features: Mapping from column index (in the full ``d``-dim
            space) to its fixed value.

    Returns:
        A 3-tuple ``(reduced_bounds, reduced_eq, reduced_ineq)`` in the
        ``(d - len(fixed_features))``-dimensional optimisation space.
    """
    if not fixed_features:
        return bounds, equality_constraints, inequality_constraints

    d = bounds.shape[-1]
    fixed_set = set(fixed_features.keys())

    keep = [i for i in range(d) if i not in fixed_set]
    old_to_new = {old: new for new, old in enumerate(keep)}

    reduced_bounds = bounds[:, keep]

    def _remap(constraint: T_CONSTRAINT) -> T_CONSTRAINT:
        indices, coeffs, value = constraint
        new_indices: list[int] = []
        new_coeffs: list[float] = []
        new_value = float(value)
        for idx_t, coeff_t in zip(indices, coeffs):
            idx = int(idx_t.item())
            coeff = float(coeff_t.item())
            if idx in fixed_set:
                new_value -= coeff * fixed_features[idx]
            else:
                new_indices.append(old_to_new[idx])
                new_coeffs.append(coeff)
        return (
            torch.tensor(new_indices, dtype=indices.dtype),
            torch.tensor(new_coeffs, dtype=coeffs.dtype),
            new_value,
        )

    reduced_eq = [_remap(c) for c in equality_constraints]
    reduced_ineq = [_remap(c) for c in inequality_constraints]

    return reduced_bounds, reduced_eq, reduced_ineq


def predict_pareto(
    model_list: ModelList,
    pareto_dims: list[int],
    ref_point: Tensor,
    bounds: Tensor,
    equality_constraints: list[T_CONSTRAINT],
    inequality_constraints: list[T_CONSTRAINT],
    num_candidates: int = 4096,
) -> tuple[Tensor, Tensor, Tensor]:
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
        A 3-tuple of Tensors containing the predicted Pareto-optimal inputs, outputs and
        their predictive uncertainties, i.e. predictive standard deviations.
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
    # sort by first dimension to enable easier plotting
    indices = Y[..., 0].argsort()
    X, Y, Ystd = X[indices], Y[indices], Ystd[indices]
    return X, Y, Ystd
