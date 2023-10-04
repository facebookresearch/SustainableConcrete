# utils module with data loaders and more.
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

_TOTAL_BINDER_NAMES = ["Cement", "Fly Ash", "Slag"]
_PASTE_CONTENT_NAMES = _TOTAL_BINDER_NAMES + ["Water"]
_BINDER_PLUS_AGGREGATE = _TOTAL_BINDER_NAMES + ["Fine Aggregate"]
_TOTAL_MASS_NAMES = _PASTE_CONTENT_NAMES + [
    "HRWR",
    "Coarse Aggregate",
    "Fine Aggregate",
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
        batch_name_to_indices: Optional[Dict[str, List[int]]] = None,
    ):
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
        self._batch_name_to_indices = batch_name_to_indices

    @property
    def X(self) -> Tensor:
        return self._X

    @property
    def Y(self) -> Tensor:
        return self._Y

    @property
    def Ystd(self) -> Tensor:
        return self._Ystd

    @property
    def Yvar(self) -> Tensor:
        return self.Ystd.square()

    @property
    def strength_data(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        X_bounds = get_mortar_bounds(self.X_columns)
        return self.X, self.Y[:, [1]], self.Yvar[:, [1]], X_bounds

    def strength_data_by_time(self, time: float) -> Tuple[Tensor, Tensor, Tensor]:
        X, Y, Yvar, _ = self.strength_data
        row_ind = torch.where(X[:, -1] == time)[0]
        return X[row_ind], Y[row_ind], Yvar[row_ind]

    def subselect_batch_names(self, names: List[str]) -> SustainableConcreteDataset:
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
    def unique_compositions(self) -> Tuple[Tensor, Tensor]:
        c = self.X[:, :-1]
        c_unique, rev = c.unique(dim=0, sorted=False, return_inverse=True)
        return c_unique, rev

    @property
    def unique_composition_indices(self) -> List[int]:
        c, rev = self.unique_compositions
        rev = [r.item() for r in rev]  # converting to a list of python ints
        # indices of first occurances of unique compositions
        unique_indices = [rev.index(i) for i in range(len(c))]
        # sorting in ascending order, to be identical to collection order
        unique_indices.sort()
        return unique_indices

    @property
    def gwp_data(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # removes duplicates due to multiple measurements in time, which is irrelevant for gwp
        unique_indices = self.unique_composition_indices
        X = self.X[unique_indices, :-1]
        Y = self.Y[unique_indices, 0].unsqueeze(-1)
        Yvar = self.Yvar[unique_indices, 0].unsqueeze(-1)
        X_bounds = get_mortar_bounds(self.X_columns[:-1])  # without time dimension
        if (X.min(dim=0).values < X_bounds[0, :]).any() or (
            X.max(dim=0).values > X_bounds[1, :]
        ).any():
            # raise Exception(
            print(
                "Bounds do not hold in training data: "
                f"{X_bounds[0, :], X.amin(dim=0) = }"
                f"{X_bounds[1, :], X.amax(dim=0) = }"
            )
        return X, Y, Yvar, X_bounds

    @property
    def X_columns(self):
        return self._X_columns

    @property
    def Y_columns(self):
        return self._Y_columns

    @property
    def Ystd_columns(self):
        return self._Ystd_columns


def load_concrete_strength(
    data_path: str = "data/concrete_strength.csv",
    verbose: bool = _VERBOSE,
    batch_names: Optional[List[str]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
):
    # loading csv into dataframe
    df = pd.read_csv(data_path, delimiter=",")

    used_columns = [
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
    df = df[used_columns]

    # dropping any mix id that is not in batch names
    if batch_names is not None:  # TODO: make this safe! "contains" only works if the batch names are unique strings, not numbers
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


def get_mortar_bounds(X_columns, verbose: bool = _VERBOSE) -> Tensor:
    """Returns bounds of columns in X for mortart mixes."""
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
        X_columns (_type_): Names of columns in the input dataset.
        min_wb (float, optional): Minimum water-binder ratio. Defaults to 0.35.
        verbose (bool, optional): Whether to print actions. Defaults to _VERBOSE.

    Returns:
        Tuple[List, List]: A tuple of equality and inequality constraints.
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


def get_concrete_constraints(X_columns, verbose: bool = _VERBOSE):
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


def get_mass_constraints(X_columns: List[str], lower: float, upper: float) -> List:
    return get_sum_constraints(
        X_columns=X_columns, subset_names=_TOTAL_MASS_NAMES, lower=lower, upper=upper
    )


def get_binder_constraints(X_columns: List[str], lower: float, upper: float) -> List:
    return get_sum_constraints(
        X_columns=X_columns, subset_names=_TOTAL_BINDER_NAMES, lower=lower, upper=upper
    )


def get_paste_constraints(X_columns: List[str], lower: float, upper: float):
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


def get_water_constraints(X_columns: List[str], lower: float, upper: float):
    # Constraint: lower < (Water) / (Total Binder) < upper
    # i.e. a proportional sum constraint
    return get_proportional_sum_constraints(
        X_columns=X_columns,
        numerator_names=["Water"],
        denominator_names=_TOTAL_BINDER_NAMES,
        lower=lower,
        upper=upper,
    )


def get_hrwr_constraints(X_columns: List[str], lower: float, upper: float):
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
) -> List:
    lower_constraint = get_sum_equality_constraint(X_columns, subset_names, value=lower)
    upper_constraint = get_sum_equality_constraint(X_columns, subset_names, value=upper)
    # rephrasing the upper as a lower bound
    upper_constraint = (upper_constraint[0], -upper_constraint[1], -upper_constraint[2])
    return [lower_constraint, upper_constraint]


def get_sum_equality_constraint(
    X_columns: List[str], subset_names: List[str], value: float
) -> Tuple[Tensor, Tensor, float]:
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
):
    """Converts a constraint on a fraction of two subset sums into a linear form,
    i.e. if the constraint is of the form

        lower < (sum of numerator_names) / (sum of denominator_names) < upper,

    then (numerator) < upper * (denominator) and so

        upper * (denominator) - (numerator) > 0, and
        (numerator) - lower * (denominator) > 0.
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


def get_reference_point():
    # gwp = -430.0  # based on existing minimum in the data (pure cement)
    gwp = -150.0  # chosen to hone in on the greener and strong region
    strength_day_1 = 1000
    # strength_day_7 = 3000
    strength_day_28 = 5000
    return torch.tensor([gwp, strength_day_1, strength_day_28], dtype=torch.double)


def get_day_zero_data(bounds: Tensor, n: int = 128):
    """Computes a tensor of n sobol points that satisfy the bounds, appended with a
    zeros tensor. Useful to condition the strength GP to be zero at day zero.
    """
    d = bounds.shape[-1]
    sobol_engine = torch.quasirandom.SobolEngine(dimension=(d - 1))  # excluding time
    X_0 = sobol_engine.draw(n)
    X_0 = torch.cat((X_0, torch.zeros(n, 1)), dim=-1)  # append time (zero)
    a, b = bounds[0], bounds[1]
    X_0 = (b - a) * X_0 + a  # scaling according to bounds
    Y_0 = torch.zeros(n, 1)  #  zero strength
    Yvar_0 = torch.full((n, 1), 1e-4)  #  with large certainty
    return X_0, Y_0, Yvar_0


# NOTE: copied over from map saas implementation.
# TODO: move this to OSS.
class LogTransformedInterval(Interval):
    """Modification of the GPyTorch interval class.

    The Interval class in GPyTorch will map the parameter to the range [0, 1] before
    applying the inverse transform. We don't want to do this when using log as an
    inverse transform. This class will skip this step and apply the log transform
    directly to the parameter values so we can optimize log(parameter) under the bound
    constraints log(lower) <= log(parameter) <= log(upper).
    """

    def __init__(self, lower_bound, upper_bound, initial_value=None):
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
    num_candidates: int = 1024,
):
    X = sample_q_batches_from_polytope(
        n=num_candidates,
        q=1,
        bounds=bounds,
        n_burnin=10000,
        thinning=2,  # don't actually need to thin for this problem
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
