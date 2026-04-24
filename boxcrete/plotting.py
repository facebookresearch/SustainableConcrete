#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Plotting utilities for visualizing concrete strength curves and model predictions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch
from matplotlib.ticker import MultipleLocator
from torch import Tensor

if TYPE_CHECKING:
    from boxcrete.models import SustainableConcreteModel


def plot_strength_curve(
    model: SustainableConcreteModel,
    compositions: Tensor,
    plot_uncertainties: bool = True,
    observed_data: Tensor | None = None,
    observed_times: Tensor | None = None,
    title: str = "Predicted Mix",
    t_start: float = 0.0,
    t_stop: float = 28.0,
    num_t: int = 1024,
    nsigma: int = 2,
    xlim: tuple[float, float] = (0, 28),
    ylim: tuple[float, float] = (-500, 16000),
    figsize: tuple[float, float] = (4, 4),
    dpi: int = 600,
    create_fig: bool = True,
    colors: list[str] | None = None,
) -> plt.Figure:
    """Plots predicted compressive strength curves as a function of curing time.

    Given a fitted SustainableConcreteModel and one or more concrete compositions
    (without the time dimension), this function generates strength curve predictions
    with optional uncertainty bands.

    Args:
        model: A fitted SustainableConcreteModel with both `strength_model`
            and `gwp_model` attributes.
        compositions: A `(n_mixes x d)`-dim Tensor of concrete compositions
            (without the time dimension). Each row is a different mix to plot.
        plot_uncertainties: Whether to plot uncertainty bands around predictions.
        observed_data: Optional `(n_obs,)`-dim Tensor of observed strength values
            to overlay on the plot.
        observed_times: Optional `(n_obs,)`-dim Tensor of observed time points
            corresponding to `observed_data`.
        title: Title prefix for the plot. GWP values are appended automatically.
        t_start: Start time (days) for the strength curve.
        t_stop: End time (days) for the strength curve.
        num_t: Number of time points to evaluate.
        nsigma: Number of standard deviations for the uncertainty band.
        xlim: X-axis limits (days).
        ylim: Y-axis limits (strength in psi).
        figsize: Figure size in inches (width, height).
        dpi: Figure resolution.
        create_fig: Whether to create a new figure or plot on the current axes.
        colors: Optional list of colors for each composition. If None, uses
            matplotlib's Tableau color cycle.

    Returns:
        The matplotlib Figure object containing the plot.
    """
    if compositions.dim() == 1:
        compositions = compositions.unsqueeze(0)

    gwp_pred = model.gwp_model.posterior(compositions).mean
    plot_times = torch.linspace(t_start, t_stop, num_t)
    tableau = colors or list(mcolors.TABLEAU_COLORS.values())

    if create_fig:
        fig = plt.figure(dpi=dpi, figsize=figsize)
    else:
        fig = plt.gcf()

    for i in range(compositions.shape[0]):
        X_i = compositions[[i]]
        X_w_time = torch.cat(
            (X_i.expand(num_t, X_i.shape[-1]), plot_times.unsqueeze(-1)), dim=-1
        )

        curve_post = model.strength_model.posterior(X_w_time)
        curve_mean = curve_post.mean.detach().squeeze()
        curve_std = curve_post.variance.sqrt().detach().squeeze()

        color = tableau[i % len(tableau)]
        plt.plot(plot_times, curve_mean, color=color)

        if plot_uncertainties:
            plt.fill_between(
                plot_times,
                curve_mean - nsigma * curve_std,
                curve_mean + nsigma * curve_std,
                alpha=0.2,
                label="Predicted",
                color=color,
            )

        if observed_data is not None and observed_times is not None:
            plt.plot(observed_times, observed_data, "o", label="Observations", c=color)
            plt.legend()

    # Configure axes (once, after all curves are plotted)
    gwp_str = ", ".join([str(round(abs(val.item()), 2)) for val in gwp_pred])
    plt.title(
        f"{title} with GWP = {gwp_str} " + r"kg CO$_2$/m$^3$",
        fontsize=9,
    )
    ax = plt.gca()
    plt.xscale("linear")
    special_times = [0, 1, 3, 5, 14, 28]
    special_labels = ["0", "1", "3", "5", "14", "28"]
    plt.xticks(special_times, special_labels)
    plt.grid(visible=True)
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_xlabel("Curing Age (days)", fontsize=9)
    ax.set_ylabel("Compressive Strength (psi)", fontsize=9)
    ax.yaxis.set_major_locator(MultipleLocator(4000))
    ax.yaxis.set_minor_locator(MultipleLocator(2000))
    for spine in ax.spines.values():
        spine.set_linewidth(1)
    ax.tick_params(which="major", width=1, length=8)
    ax.tick_params(which="minor", width=1, length=4)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)

    return fig


def plot_slump_calibration(
    observed: Tensor,
    predicted_mean: Tensor,
    predicted_std: Tensor | None = None,
    nsigma: int = 2,
    title: str = "Slump Calibration",
    xlabel: str = "Observed Slump (in)",
    ylabel: str = "Predicted Slump (in)",
    figsize: tuple[float, float] = (5, 5),
    dpi: int = 150,
) -> plt.Figure:
    """Plots a calibration scatter plot of predicted vs. observed slump.

    Args:
        observed: ``(n,)``-dim Tensor of observed slump values.
        predicted_mean: ``(n,)``-dim Tensor of predicted slump means.
        predicted_std: Optional ``(n,)``-dim Tensor of predicted standard deviations
            for error bars.
        nsigma: Number of standard deviations for error bars.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        figsize: Figure size in inches.
        dpi: Figure resolution.

    Returns:
        The matplotlib Figure object.
    """
    obs = observed.detach().cpu()
    pred = predicted_mean.detach().cpu()

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    if predicted_std is not None:
        err = nsigma * predicted_std.detach().cpu()
        ax.errorbar(obs, pred, yerr=err, fmt="o", alpha=0.6, markersize=4, capsize=2)
    else:
        ax.scatter(obs, pred, alpha=0.6, s=20)

    # 1:1 diagonal
    lo = min(obs.min().item(), pred.min().item())
    hi = max(obs.max().item(), pred.max().item())
    margin = 0.05 * (hi - lo)
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin], "k--", alpha=0.5)

    # Metrics
    residuals = pred - obs
    ss_res = (residuals**2).sum().item()
    ss_tot = ((obs - obs.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = (ss_res / len(obs)) ** 0.5

    ax.annotate(
        f"R\u00b2 = {r2:.3f}\nRMSE = {rmse:.2f}",
        xy=(0.05, 0.92),
        xycoords="axes fraction",
        fontsize=9,
        verticalalignment="top",
    )

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    return fig


def plot_feature_importance(
    lengthscales: Tensor,
    feature_names: list[str],
    title: str = "Feature Importance (ARD Lengthscales)",
    figsize: tuple[float, float] = (6, 4),
    dpi: int = 150,
) -> plt.Figure:
    """Plots a horizontal bar chart of ARD lengthscale-derived feature importance.

    Importance is computed as ``1 / lengthscale`` (normalized to sum to 1).

    Args:
        lengthscales: ``(d,)``-dim Tensor of ARD lengthscales.
        feature_names: List of feature names corresponding to each lengthscale.
        title: Plot title.
        figsize: Figure size in inches.
        dpi: Figure resolution.

    Returns:
        The matplotlib Figure object.
    """
    ls = lengthscales.detach().cpu().float()
    importance = 1.0 / ls
    importance = importance / importance.sum()

    # Sort by importance
    order = importance.argsort(descending=True)
    importance = importance[order]
    names = [feature_names[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y_pos = range(len(names))
    ax.barh(y_pos, importance.numpy(), align="center")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig
