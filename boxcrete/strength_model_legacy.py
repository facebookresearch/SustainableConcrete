# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Legacy V1 strength-GP input transform (instructional fixture).

The V1 architecture (single Matern + day-zero anchor pseudo-observations
+ scalar noise) is retained alongside the V2 production architecture
for instructional purposes — it makes it easy for new readers to
compare the deployed V2 strength GP's input-transform composition
against the prior-generation V1 baseline. (The full V1 fit factory
is not part of this release.)

This module hosts the V1-specific input transform:

  * :func:`get_strength_gp_input_transform` — V1's ``log(time + 1)``
    + ``Normalize`` chained input transform.

The HRWR/binder feature transform :class:`AppendDerivedFeatures` (also
used by V1 and by the production slump GP) lives in
:mod:`boxcrete.features` because it is conceptually a feature transform,
not a strength-V1 artifact.

New code should not consume this module — use ``boxcrete.fit_strength_gp``
(V2) or :mod:`boxcrete.slump_model` instead.
"""

from __future__ import annotations

import torch
from botorch.models.transforms.input import (
    AffineInputTransform,
    ChainedInputTransform,
    Log10,
    Normalize,
)
from torch import Tensor


def get_strength_gp_input_transform(
    d: int, bounds: Tensor | None
) -> ChainedInputTransform:
    """Chains a log(time + 1) and Normalize transform on d dimensional input data,
    with the provided bounds.

    Args:
        d: The input dimensionality.
        bounds: `2 x d` tensor of lower and upper bounds for each dimension.

    Returns:
        A ChainedInputTransform that log-transforms the time dimension and subsequently
        normalizes all dimensions to the unit hyper-cube.
    """
    time_index = [d - 1]
    tf1 = AffineInputTransform(  # adds one to time dimension before taking log
        d,
        coefficient=torch.ones(1, dtype=torch.float64),
        offset=torch.ones(1, dtype=torch.float64),
        indices=time_index,
        reverse=True,
    )
    tf2 = Log10(
        indices=time_index
    )  # taking log of time dimension for better extrapolation
    if bounds is not None:
        transformed_bounds = tf2(tf1(bounds))
        tf3 = Normalize(
            d, bounds=transformed_bounds
        )  # normalizing after log(t + 1) transform
    else:
        tf3 = Normalize(d)  # normalizing after log(t + 1) transform
    return ChainedInputTransform(tf1=tf1, tf2=tf2, tf3=tf3)


__all__ = [
    "get_strength_gp_input_transform",
]
