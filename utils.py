# utils module with data loaders and more.
from typing import List, Tuple

import pandas as pd
import torch
from torch import Tensor

_TOTAL_BINDER_NAMES = ["Cement", "Fly Ash", "Slag"]
_PASTE_CONTENT_NAMES = _TOTAL_BINDER_NAMES + ["Water"]
_TOTAL_MASS_NAMES = _PASTE_CONTENT_NAMES + ["HRWR", "Coarse Aggregate", "Fine Aggregate"]
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
    ):
        if X_columns[-1] != "Time":
            raise ValueError(f"Last dimension of X assumed to be time, but is {X_columns[-1]}.")

        # making sure we are not overwriting these
        self._X_columns = X_columns
        self._Y_columns = Y_columns
        self._Ystd_columns = Ystd_columns
        self._X = X
        self._Y = Y
        self._Ystd = Ystd

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def Ystd(self):
        return self._Ystd

    @property
    def Yvar(self):
        return self.Ystd.square()

    @property
    def strength_data(self):
        X_bounds = get_mortar_bounds(self.X_columns)
        return self.X, self.Y[:, [1]], self.Yvar[:, [1]], X_bounds

    @property
    def unique_compositions(self):
        c = self.X[:, :-1]
        c_unique, rev = c.unique(dim=0, sorted=False, return_inverse=True)
        return c_unique, rev

    @property
    def unique_composition_indices(self):
        c, rev = self.unique_compositions
        rev = [r.item() for r in rev]  # converting to a list of python ints
        # indices of first occurances of unique compositions
        unique_indices = [rev.index(i) for i in range(len(c))]
        # sorting in ascending order, to be identical to collection order
        unique_indices.sort()
        return unique_indices

    @property
    def gwp_data(self):
        # removes duplicates due to multiple measurements in time, which is irrelevant for gwp
        unique_indices = self.unique_composition_indices
        X = self.X[unique_indices, :-1]
        Y = self.Y[unique_indices, 0].unsqueeze(-1)
        Yvar = self.Yvar[unique_indices, 0].unsqueeze(-1)
        X_bounds = get_mortar_bounds(self.X_columns[:-1])  # without time dimension
        if (X.min(dim=0).values < X_bounds[0, :]).any() or (X.max(dim=0).values > X_bounds[1, :]).any():
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
    data_path: str = "data/concrete_strength.csv", verbose: bool = _VERBOSE, dtype=torch.float32
):
    # loading csv into dataframe
    df = pd.read_csv(data_path, delimiter=",")

    if verbose:
        print(f"The data has {len(df)} rows and {len(df.columns)} columns, which are:")
        for column in df.columns.to_list():
            print("\t-", column)
        print()

    data_index = 3
    data_columns = df.columns[data_index:]
    n_missing = torch.tensor(df[data_columns].to_numpy()).isnan().sum(dim=0)
    missing_ind = n_missing > 0
    if verbose and missing_ind.any():
        print(f"There are {missing_ind.sum()} columns with missing entries:")
        for name, missing in zip(data_columns[missing_ind], n_missing[missing_ind]):
            print("\t-", name, "has", missing.item(), "missing entries.")
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
    X = torch.tensor(df[X_columns].to_numpy(), dtype=dtype)
    Y = torch.tensor(df[Y_columns].to_numpy(), dtype=dtype)

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
                torch.tensor(df[Ystd_columns].to_numpy(), dtype=dtype),
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

    n_measurements = torch.tensor(df["# of measurements"].to_numpy(), dtype=dtype)
    Ystd[:, 1] /= n_measurements.sqrt()
    return SustainableConcreteDataset(
        X=X, Y=Y, Ystd=Ystd, X_columns=X_columns, Y_columns=Y_columns, Ystd_columns=Ystd_columns
    )


def get_mortar_bounds(X_columns, verbose: bool = _VERBOSE) -> Tensor:
    """Returns bounds of columns in X for mortart mixes."""
    bounds_dict = {
        "Cement": (150, 350), # these are now in grams, as opposed to the original concrete bounds
        "Fly Ash": (0, 175),
        "Slag": (0, 225),
        "Fine Aggregate": (0, 1700), # will be fixed to 1375 with equality constraint
        "Time": (0, 28),  # up to 28 days
    }

    total_binder = 500.0
    bounds_dict.update(
        {
            "Water": (0.2 * total_binder, 0.5 * total_binder),
            "HRWR": (0, 0.1 * total_binder),  # we are not optimizing this, but need this to fit the model
        }
    )
    bounds = torch.tensor([bounds_dict[col] for col in X_columns]).T
    if verbose:
        print("The lower and upper bounds for the respective variables are set to:")
        for col, bound in zip(X_columns, bounds.T):
            print(f"\t- {col}: [{bound[0].item()}, {bound[1].item()}]")
        print()
    return bounds


def get_mortar_constraints(X_columns, verbose: bool = _VERBOSE) -> Tuple[List, List]:
    # inequality constraints
    equality_dict = {
        "Total Binder": 500.0,
        # "Fine Aggregate": 1375.0,
    }
    # inequality_dict = {
    #     "Water": (0.2, 0.5),  # as a proportion of total binder
    # }
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
        get_sum_equality_constraint(
            X_columns=X_columns,
            subset_names=_TOTAL_BINDER_NAMES,
            value=equality_dict["Total Binder"],
        ),
        # get_sum_equality_constraint(
        #     X_columns=X_columns,
        #     subset_names=["Fine Aggregate"],
        #     value=equality_dict["Fine Aggregate"],
        # )
    ]
    # inequality_constraints = [
        # as long as binder is constant, the water constraint is just a bound
        # *get_water_constraints(X_columns, *inequality_dict["Water"]),
    # ]
    return equality_constraints # , inequality_constraints


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
    upper_constraint[1] = -upper_constraint[1]  # rephrasing the upper as a lower bound
    upper_constraint[2] = -upper_constraint[2]
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
    """Converts a constraint on a fraction of two subset sums into a linear form, i.e. if the constraint is of the form

        lower < (sum of numerator_names) / (sum of denominator_names) < upper,

    then (numerator) < upper * (denominator) and so

        upper * (denominator) - (numerator) > 0, and
        (numerator) - lower * (denominator) > 0.
    """
    _, num_coeffs = get_subset_sum_tensors(X_columns=X_columns, subset_names=numerator_names)
    _, den_coeffs = get_subset_sum_tensors(X_columns=X_columns, subset_names=denominator_names)

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


def get_subset_sum_tensors(X_columns: List[str], subset_names: List[str]) -> Tuple[Tensor, Tensor]:
    indices = [X_columns.index(name) for name in subset_names]
    coeffs = torch.zeros(len(X_columns))
    coeffs[indices] = 1
    return indices, coeffs


def get_reference_point():
    gwp = -430.0  # based on existing minimum in the data (pure cement)
    strength_day_1 = 1000
    # strength_day_7 = 3000
    strength_day_28 = 5000
    return torch.tensor([gwp, strength_day_1, strength_day_28])


def get_day_zero_data(bounds: Tensor, n: int = 128):
    """Computes a tensor n sobol points that satisfy the bounds, appended with a
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
