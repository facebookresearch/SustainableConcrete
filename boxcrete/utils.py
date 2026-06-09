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
]
DEFAULT_X_COLUMNS = [
    "Cement (kg/m3)",
    "Fly Ash (kg/m3)",
    "Slag (kg/m3)",
    "Water (kg/m3)",
    "HRWR (kg/m3)",
    "Fine Aggregate (kg/m3)",
    "Coarse Aggregates (kg/m3)",
    "Material Source",
    "Temp (C)",
    "Time",  # last dimension is assumed to be time
]
DEFAULT_Y_COLUMNS = ["GWP", "Strength (Mean)"]
SLUMP_Y_COLUMNS = ["GWP", "Strength (Mean)", "Slump (in)"]
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
            Y: ``n x m``-dim Tensor of outputs (``m = len(Y_columns)``),
                where ``Y[:, 0]`` is GWP, ``Y[:, 1]`` is mean strength, and
                optionally ``Y[:, 2]`` is slump when using ``SLUMP_Y_COLUMNS``.
            Ystd: ``n x m``-dim Tensor of empirical standard deviations of ``Y``.
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
        """The ``n x m``-dim output data ``Y`` (``m = len(Y_columns)``).
        ``Y[:, 0]`` is GWP, ``Y[:, 1]`` is strength, and optionally
        ``Y[:, 2]`` is slump.
        """
        return self._Y

    @property
    def Ystd(self) -> Tensor:
        """The ``n x m``-dim empirical standard deviation of the outputs.
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

    def _time_independent_data(
        self, y_col_idx: int, default_variance: float = 1e-2
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Returns unique-composition data for a time-independent output.

        Removes duplicate rows due to multiple time measurements and strips
        the time dimension from X and bounds.

        Args:
            y_col_idx: Column index into ``Y`` / ``Yvar`` to extract.
            default_variance: Fallback variance when ``Ystd`` doesn't have
                enough columns to cover ``y_col_idx``.

        Returns:
            A 4-tuple ``(X, Y, Yvar, X_bounds)`` for unique compositions.
        """
        unique_indices = self.unique_composition_indices
        X = self.X[unique_indices, :-1]
        Y = self.Y[unique_indices, y_col_idx].unsqueeze(-1)
        if self._Ystd.shape[-1] > y_col_idx:
            Yvar = self.Yvar[unique_indices, y_col_idx].unsqueeze(-1)
        else:
            Yvar = torch.full_like(Y, fill_value=default_variance)
        X_bounds = None
        if self.bounds is not None:
            X_bounds = self.bounds[:, :-1]
        return X, Y, Yvar, X_bounds

    @property
    def slump_data(self) -> tuple[Tensor, Tensor, Tensor, Tensor | None] | None:
        """Returns the data with which to fit a slump model.

        Slump is time-independent (like GWP), so duplicates from multiple
        time measurements are removed. Rows where Slump is NaN (e.g. mortar
        mixes without slump measurements) are excluded.

        Returns:
            A 4-tuple of Tensors containing 1) the ``n_valid x (d - 1)``
            unique compositions X *without* time, 2) the corresponding
            ``n_valid x 1``-dim Slump values Y, 3) the ``n_valid x 1``-dim
            Slump variances Yvar, and 4) the ``2 x (d - 1)``-dim bounds on X.
            Returns ``None`` if Slump is not present in Y_columns.
        """
        if "Slump (in)" not in self._Y_columns:
            return None
        slump_idx = self._Y_columns.index("Slump (in)")
        X, Y, Yvar, X_bounds = self._time_independent_data(
            slump_idx, default_variance=1e-2
        )
        # Filter out NaN slump values (mortar mixes without slump measurements)
        valid = ~Y.squeeze(-1).isnan()
        if not valid.any():
            return None
        return X[valid], Y[valid], Yvar[valid], X_bounds

    @property
    def gwp_data(self) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Returns the data with which to fit a GWP model.

        Returns:
            A 4-tuple of Tensors containing 1) the `n_unique x (d - 1)` unique
            compositions X *without* time since GWP does not depend on `time`, 2) the
            corresponding `n_unique x 1`-dim GWP values Y, 3) the `n_unique x 1`-dim
            GWP variances Yvar, and the `2 x (d - 1)`-dim bounds on X.
        """
        X, Y, Yvar, X_bounds = self._time_independent_data(0, default_variance=1e-3)
        if X_bounds is not None:
            if (X.amin(dim=0) < X_bounds[0, :]).any() or (
                X.amax(dim=0) > X_bounds[1, :]
            ).any():
                logger.warning(  # pragma: no cover
                    "Bounds do not hold in training data: "
                    f"{X_bounds[0, :], X.amin(dim=0)=}"
                    f"{X_bounds[1, :], X.amax(dim=0)=}"
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
            raise ValueError(
                "subselect_batch_names: this dataset was loaded without batch-"
                "name indices. Re-load with "
                "``load_concrete_strength(..., "
                "process_batch_names_from_mix_name=True)`` "
                "or pass an explicit ``batch_name_to_indices`` dict."
            )

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

    Notes on dataset conventions (these aren't enforced here, but are good
    to know if you write code that consumes the loaded `SustainableConcreteDataset`):

    - **`Strength (Std)` is the SAMPLE standard deviation** (`ddof=1`) of the
      strength replicates `Strength1/2/3 (psi)` — not the population stdev.
      Verified empirically against ~647 of 727 rows in `data/boxcrete_data.csv`.
    - **`# of measurements` is curated**, not always equal to
      `count(non-null Strength1, Strength2, Strength3)`. A handful of rows
      encode information about which replicates were valid (e.g. sentinel
      zeros from cylinders that failed at handling). The loader uses this
      column to weight observation noise variances accordingly.
    - **`GWP` is essentially deterministic** in `(composition, Material Source)`:
      the regression coefficients in `DEFAULT_GWP_COEFFICIENTS` reproduce
      the column to R² ≈ 1 per Material Source class (see
      `test_gwp_linearity` in `test/test_utils.py`).
    - **HRWR** is recorded in `kg/m³`. To convert to the field-conventional
      `oz/cwt of binder`, use:
          ``oz/cwt = HRWR (kg/m³) / Binder (kg/m³) × 1533.3 / ρ``
      where `Binder = Cement + Fly Ash + Slag` and ρ is the assumed HRWR
      liquid density (g/mL). A reasonable global default is ρ = 1.10 g/mL
      (typical polycarboxylate-based product).

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
    # Slump is optional — don't drop rows just because Slump is missing,
    # since those rows still have valid strength/GWP data.
    optional_columns = {"Slump (in)"}
    required_columns = [
        c for c in X_columns + Y_columns + Ystd_columns if c not in optional_columns
    ]
    required_columns = np.array(required_columns)
    is_missing = torch.tensor(df[required_columns].to_numpy()).isnan()
    n_missing = is_missing.sum(dim=0)
    missing_col_ind = n_missing > 0
    if missing_col_ind.any():
        logger.info(f"There are {missing_col_ind.sum()} columns with missing entries:")
        logger.info(f"{missing_col_ind=}")
        logger.info(f"{required_columns=}")
        logger.info(f"{n_missing=}")
        for name, missing in zip(
            required_columns[missing_col_ind], n_missing[missing_col_ind]
        ):
            logger.info("  - %s has %s missing entries.", name, missing.item())
        logger.info("Removing missing rows with missing entries from data.")
        missing_row_ind = [i for i in range(len(df)) if is_missing[i].any()]
        logger.info(f"  -Rows indices to be removed: {missing_row_ind=}")
        df = df.drop(missing_row_ind)
        logger.info(
            "  -Number of missing values after deletion (Should be zero): "
            f"{torch.tensor(df[required_columns].to_numpy()).isnan().sum()}"
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
        # Add small constant std for Slump if present
        if "Slump (in)" in Y_columns:
            Ystd = torch.cat(
                (Ystd, torch.full_like(Y[:, 0], 1e-1).unsqueeze(-1)),
                dim=-1,
            )
    else:
        raise NotImplementedError(
            "Multiple Ystd columns are not currently supported. The deployed "
            "model uses a single Ystd column ('Strength (Std)'); pass "
            "``Ystd_columns=['Strength (Std)']`` (the default), or open an "
            "issue describing the multi-Ystd shape you need."
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
    # mortar mode).
    bounds = torch.tensor(
        [bounds_dict.get(col, (0, 0)) for col in X_columns],
        dtype=torch.float64,
    ).T
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
    """Constrains the HRWR / binder ratio.

    Args:
        X_columns: Column names of the input features.
        lower: Lower bound on the HRWR / binder ratio.
        upper: Upper bound on the HRWR / binder ratio.

    Returns:
        A list of inequality constraint tuples.
    """
    return get_proportional_sum_constraints(
        X_columns=X_columns,
        numerator_names=["HRWR (kg/m3)"],
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
        numerator_names: The subset of variable names whose sum to use as
            the numerator.
        denominator_names: The subset of variable names whose sum to use as
            the denominator.
        lower: The lower limit of the fractional constraint.
        upper: The upper limit of the fractional constraint.

    Returns:
        A list of tuples of the form `(indices, coefficients, constant)` that represents
        the proportional sum constraint in its linear representation.
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
    """Returns indices and coefficients such that
    `X[indices].dot(coeffs) == X[indices].sum()`,
    where indices are the indices of subset_names in X_columns.

    Args:
        X_columns: The column (variable) names.
        subset_names: The subset of variable names whose sum to compute.

    Returns:
        A tuple of `indices` (list of ints) and `coeffs` (Tensor) with which to
        compute the subset sum.
    """
    indices = [X_columns.index(name) for name in subset_names]
    coeffs = torch.zeros(len(X_columns), dtype=torch.float64)
    coeffs[indices] = 1
    return indices, coeffs


MORTAR_REFERENCE_POINT = torch.tensor([-400.0, 1000.0, 5000.0], dtype=torch.double)
CONCRETE_REFERENCE_POINT = torch.tensor([-200.0, 1000.0, 5000.0], dtype=torch.double)

# Cost reference point thresholds ($/m³) in NATURAL units (positive).
# Negated internally by get_reference_point to match the -cost convention.
# Permissive values (well above typical $100-120/m³) to retain the full mix range.
CONCRETE_COST_THRESHOLD = 250.0
MORTAR_COST_THRESHOLD = 300.0

# Representative ingredient costs (USD/kg) as (mean, std) tuples in NATURAL
# units (positive values). The negation for joint maximization (minimize cost
# → maximize -cost) is applied internally by `fit_cost_model`.
# Sources: USGS 2023 Mineral Commodity Summaries, RS Means, PCA reports.
#
# Conversion example (Cement):
#   Range: $100-150/ton → midpoint $125/ton ÷ 1000 kg/ton = $0.125/kg ≈ 0.12
#   Std: range $50/ton ÷ 4 (≈ ±2σ) = $12.5/ton ÷ 1000 = $0.0125/kg ≈ 0.015
DEFAULT_COST_COEFFICIENTS: dict[str, tuple[float, float]] = {
    "Cement (kg/m3)": (0.12, 0.015),  # $100-150/ton regional variation
    "Fly Ash (kg/m3)": (0.04, 0.010),  # $30-60/ton; supply-dependent byproduct
    "Slag (kg/m3)": (0.09, 0.015),  # $70-120/ton; transport-dependent
    "Water (kg/m3)": (0.002, 0.001),  # Municipal rates, negligible
    "HRWR (kg/m3)": (3.00, 0.90),  # $2000-5000/ton; brand/supplier variation
    "Fine Aggregate (kg/m3)": (0.02, 0.006),  # $10-30/ton; transport-heavy
    "Coarse Aggregates (kg/m3)": (0.015, 0.005),  # $10-25/ton; transport-heavy
}

# GWP emission factors per Material Source class, as (mean, std) tuples in
# NATURAL units (positive values = kg CO₂ per kg of ingredient). The negation
# for joint maximization (minimize GWP → maximize -GWP) is applied internally
# by `fit_gwp_model`.
# Derived via per-class least-squares regression on training data (which
# stores -GWP). Magnitudes here are the absolute emission factors.
DEFAULT_GWP_COEFFICIENTS = {
    0: {  # Material Source 0
        "Cement (kg/m3)": (0.762610, 0.000365),
        "Fly Ash (kg/m3)": (0.029601, 0.000432),
        "Slag (kg/m3)": (0.085926, 0.000310),
        "Water (kg/m3)": (-0.001765, 0.001114),
        "HRWR (kg/m3)": (3.184692, 0.016001),
        "Fine Aggregate (kg/m3)": (0.002788, 0.000097),
        "Coarse Aggregates (kg/m3)": (0.003910, 0.000112),
    },
    1: {  # Material Source 1
        "Cement (kg/m3)": (0.773814, 0.007240),
        "Fly Ash (kg/m3)": (0.035681, 0.005542),
        "Slag (kg/m3)": (0.092849, 0.006469),
        "Water (kg/m3)": (0.003864, 0.019144),
        # HRWR GWP: consistent with Source 0 (3.18 vs 3.15 kg CO₂/kg).
        # See comment above for EPD references.
        "HRWR (kg/m3)": (3.151231, 0.498391),
        "Fine Aggregate (kg/m3)": (0.002513, 0.003486),
        "Coarse Aggregates (kg/m3)": (-0.000039, 0.002870),
    },
}


def get_reference_point(
    optimization_mode: str = "concrete", include_cost: bool = False
) -> Tensor:
    """Returns a reference point for Pareto frontier computation.

    The reference point specifies minimum acceptable values for each objective
    (GWP, 1-day strength, 28-day strength, and optionally cost). Solutions that
    do not dominate this point are excluded from the Pareto frontier.

    Args:
        optimization_mode: ``"concrete"`` (default) or ``"mortar"``.
        include_cost: If True, appends a permissive cost threshold to the
            reference point.

    Returns:
        A Tensor with 3 elements ``[-GWP, 1-day, 28-day]`` or 4 elements
        ``[-GWP, 1-day, 28-day, -Cost]`` when ``include_cost=True``.
    """
    if optimization_mode == "mortar":
        ref = MORTAR_REFERENCE_POINT.clone()
        if include_cost:
            # Negate: threshold is in natural units, model predicts -cost.
            ref = torch.cat(
                [ref, torch.tensor([-MORTAR_COST_THRESHOLD], dtype=ref.dtype)]
            )
    elif optimization_mode == "concrete":
        ref = CONCRETE_REFERENCE_POINT.clone()
        if include_cost:
            # Negate: threshold is in natural units, model predicts -cost.
            ref = torch.cat(
                [ref, torch.tensor([-CONCRETE_COST_THRESHOLD], dtype=ref.dtype)]
            )
    else:
        raise ValueError(
            "get_reference_point: optimization_mode must be 'concrete' or "
            f"'mortar'; got {optimization_mode!r}."
        )
    return ref


def make_linear_coefficients(
    X_columns: list[str],
    coefficients: dict[str, tuple[float, float]],
) -> tuple[Tensor, Tensor]:
    """Builds aligned coefficient tensors from a column-name-keyed dictionary.

    Maps a dictionary of ``{column_name: (mean, std)}`` pairs to a pair of
    tensors aligned with ``X_columns``. Columns not present in the dictionary
    receive zero mean and zero variance.

    Args:
        X_columns: Column names of the input features (without Time).
        coefficients: Mapping from column name to ``(mean, std)`` tuples.

    Returns:
        A 2-tuple ``(means, variances)`` of ``(d,)``-dim Tensors.
    """
    means = torch.zeros(len(X_columns), dtype=torch.double)
    variances = torch.zeros(len(X_columns), dtype=torch.double)
    for i, col in enumerate(X_columns):
        if col in coefficients:
            mean, std = coefficients[col]
            means[i] = mean
            variances[i] = std**2
    return means, variances


def get_day_zero_data(X: Tensor, n: int = 128):
    """Generates pseudo-observations at time=0 for conditioning the GP to predict
    zero strength at day zero.

    Uses the unique compositions from the training data (without time) to ensure
    the constraint is enforced at all observed mix designs. If the number of unique
    compositions exceeds n, a random subset is selected.

    Args:
        X: The input tensor (n_train x d), where the last column is time.
        n: Maximum number of pseudo-observations. If there are fewer unique
            compositions than n, all unique compositions are used.

    Returns:
        A tuple (X_0, Y_0, Yvar_0) of pseudo-observations at time=0.
    """
    # Use unique observed compositions (without time)
    unique_comps = torch.unique(X[:, :-1], dim=0)
    n_unique = unique_comps.shape[0]

    if n_unique <= n:
        # Use all unique compositions
        X_comps = unique_comps
    else:  # pragma: no cover
        # Only triggered when training data has >n unique compositions;
        # the strength dataset has 647 unique mixes < 128 default n.
        # Random subset of unique compositions.
        perm = torch.randperm(n_unique)[:n]
        X_comps = unique_comps[perm]

    n_out = X_comps.shape[0]
    # Append time=0
    X_0 = torch.cat((X_comps, torch.zeros(n_out, 1, dtype=X.dtype)), dim=-1)
    Y_0 = torch.zeros(n_out, 1, dtype=X.dtype)
    Yvar_0 = torch.full((n_out, 1), 1e-4, dtype=X.dtype)
    return X_0, Y_0, Yvar_0


def derive_bounds_from_X(X: Tensor, *, eps: float = 1e-8) -> Tensor:
    """Derive a ``[2, d]`` bounds tensor from per-dim min/max of ``X``,
    safely handling zero-width columns.

    Both :mod:`boxcrete.slump_model` and :mod:`boxcrete.strength_model`
    need to fall back to data-derived bounds when the caller doesn't
    supply explicit bounds. A zero-width column (e.g., a categorical
    material-source indicator that takes a single value across the fit
    subset, or any column containing a single value) causes ``Normalize``
    to produce NaNs — this helper widens such columns to
    ``[min, min + 1.0]`` so downstream normalisation stays finite.

    NaN inputs are rejected loudly: ``amin``/``amax`` propagate NaN,
    ``(NaN - NaN) < eps`` is False, and the helper would otherwise
    silently return NaN bounds — exactly the failure mode this helper
    is supposed to insulate ``Normalize`` from.

    Args:
        X: ``[n, d]`` input tensor.
        eps: width below which a column is treated as zero-width.

    Returns:
        ``[2, d]`` tensor stacked as ``[lower, upper]``.

    Raises:
        ValueError: if ``X`` contains NaN or infinite values, or has
            no rows.
    """
    if X.shape[0] < 1:
        raise ValueError(
            "derive_bounds_from_X: X must have at least one row; got X.shape="
            f"{tuple(X.shape)}."
        )
    nan_mask = torch.isnan(X).any(dim=0)
    inf_mask = torch.isinf(X).any(dim=0)
    bad_mask = nan_mask | inf_mask
    if bad_mask.any():
        bad_cols = bad_mask.nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            "derive_bounds_from_X: X contains NaN or infinite values in "
            f"columns {bad_cols}. Clean the input before deriving bounds, or "
            "pass an explicit ``X_bounds`` tensor."
        )
    x_min = X.amin(dim=0)
    x_max = X.amax(dim=0)
    zero_width = (x_max - x_min) < eps
    x_max = torch.where(zero_width, x_min + 1.0, x_max)
    return torch.stack([x_min, x_max], dim=0)


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
            different than those used to train the model to explore
            "what-if" scenarios.
        inequality_constraints: Inequality constraints. Similar to the
            bounds, these can be different than those used to train the
            model to explore "what-if" scenarios.
        num_candidates: The number of random inputs to generate in order
            to approximate the Pareto frontier. The higher the number of
            candidates, the more accurate.

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
