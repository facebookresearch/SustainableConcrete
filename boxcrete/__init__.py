#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from boxcrete.concrete_model import SustainableConcreteModel
from boxcrete.features import (
    AppendDerivedFeatures,
    F5_ALLLOG_FEATURES,
    FEATURE_BUILDERS,
    GATE_TAU,
    IDX,
    append_engineered_features_callable,
    augmented_bounds,
    max_scale_Y,
)
from boxcrete.kernels import (
    TimeGatedKernel,
    additive_time_kernel,
    ard_matern_with_within_group_prior,
    build_strength_kernel_for_aug_dim,
    make_gated_strength_kernel_builder,
)
from boxcrete.likelihoods import GatedGaussianLikelihood, PartialFixedNoiseLikelihood
from boxcrete.model_utils import FixedFeatureModel, LinearModel
from boxcrete.plotting import (
    compute_loo_cv,
    plot_calibration,
    plot_feature_importance,
    plot_strength_curve,
)
from boxcrete.priors import WithinGroupShrinkagePrior, within_group_prior
from boxcrete.slump_model import fit_slump_gp
from boxcrete.strength_model import fit_strength_gp, load_pretrained_strength_gp
from boxcrete.strength_model_legacy import get_strength_gp_input_transform
from boxcrete.units import (
    convert_slump,
    convert_strength,
    DEFAULT_UNIT_SYSTEM,
    SLUMP_DISPLAY_SCALE,
    STRENGTH_DISPLAY_SCALE,
    slump_label,
    strength_label,
    UnitSystem,
)
from boxcrete.utils import (
    CONCRETE_BOUNDS_DICT,
    CONCRETE_CONSTRAINTS,
    CONCRETE_REFERENCE_POINT,
    DATA_PATH,
    DEFAULT_BOUNDS_DICT,
    DEFAULT_COST_COEFFICIENTS,
    DEFAULT_GWP_COEFFICIENTS,
    DEFAULT_X_COLUMNS,
    DEFAULT_Y_COLUMNS,
    DEFAULT_YSTD_COLUMNS,
    get_bounds,
    get_constraints,
    get_day_zero_data,
    get_reference_point,
    load_concrete_strength,
    make_linear_coefficients,
    MORTAR_BOUNDS_DICT,
    MORTAR_CONSTRAINTS,
    MORTAR_REFERENCE_POINT,
    predict_pareto,
    reduce_to_optimization_space,
    SLUMP_Y_COLUMNS,
    SustainableConcreteDataset,
)

__all__ = [
    "AppendDerivedFeatures",
    "CONCRETE_BOUNDS_DICT",
    "CONCRETE_CONSTRAINTS",
    "CONCRETE_REFERENCE_POINT",
    "DATA_PATH",
    "DEFAULT_BOUNDS_DICT",
    "DEFAULT_COST_COEFFICIENTS",
    "DEFAULT_GWP_COEFFICIENTS",
    "DEFAULT_UNIT_SYSTEM",
    "DEFAULT_X_COLUMNS",
    "DEFAULT_Y_COLUMNS",
    "DEFAULT_YSTD_COLUMNS",
    "F5_ALLLOG_FEATURES",
    "FEATURE_BUILDERS",
    "FixedFeatureModel",
    "GATE_TAU",
    "GatedGaussianLikelihood",
    "IDX",
    "LinearModel",
    "MORTAR_BOUNDS_DICT",
    "MORTAR_CONSTRAINTS",
    "MORTAR_REFERENCE_POINT",
    "PartialFixedNoiseLikelihood",
    "SLUMP_DISPLAY_SCALE",
    "SLUMP_Y_COLUMNS",
    "STRENGTH_DISPLAY_SCALE",
    "SustainableConcreteDataset",
    "SustainableConcreteModel",
    "TimeGatedKernel",
    "UnitSystem",
    "WithinGroupShrinkagePrior",
    "additive_time_kernel",
    "append_engineered_features_callable",
    "ard_matern_with_within_group_prior",
    "augmented_bounds",
    "build_strength_kernel_for_aug_dim",
    "compute_loo_cv",
    "convert_slump",
    "convert_strength",
    "fit_slump_gp",
    "fit_strength_gp",
    "get_bounds",
    "get_constraints",
    "get_day_zero_data",
    "get_reference_point",
    "get_strength_gp_input_transform",
    "load_concrete_strength",
    "load_pretrained_strength_gp",
    "make_gated_strength_kernel_builder",
    "make_linear_coefficients",
    "max_scale_Y",
    "plot_calibration",
    "plot_feature_importance",
    "plot_strength_curve",
    "predict_pareto",
    "reduce_to_optimization_space",
    "slump_label",
    "strength_label",
    "within_group_prior",
]
