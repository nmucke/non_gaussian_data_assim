"""Plotting metric (CRPS, RMSE) time-series """

from typing import Mapping, Optional, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_timeseries(
    metrics: list[dict] | dict,
    show_fig: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot time-series of rmse(+spread) and crps"""

    if isinstance(metrics, list):
        multiplot = True
        titles = [
            "All states",
            "State 0",
            "State 1",
        ]
    elif isinstance(metrics, dict):
        multiplot = False
        metrics = [metrics]

    maxval = 0
    for i, metric_dict in enumerate(metrics):
        maxc = jnp.max(metric_dict["rmse_time"])
        maxval = maxc if maxc > maxval else maxval
    maxval = np.ceil(maxval * 2) / 2

    # -- Max-val for axis
    num_states = len(metrics)
    fig, axs = plt.subplots(
        nrows=1, ncols=num_states, figsize=(6 * num_states, 4), sharey=False
    )

    # --- Safe-guard if only one state is passed
    axs = [axs] if num_states == 1 else axs

    for i, metric_dict in enumerate(metrics):
        ax = axs[i] if multiplot else axs
        crps_time = metric_dict["crps_time"]
        rmse_time = metric_dict["rmse_time"]
        spread_time = metric_dict["spread_time"]
        crps_tot = metric_dict["crps"]
        rmse_tot = metric_dict["rmse"]

        # --- Textbox with total error
        stats = (
            # f"State {i}\nN = {z.size}\n"
            f"CRPS = {crps_tot:.3f}\n"
            f"RMSE  = {rmse_tot:.3f}"
        )
        ax.text(
            0.95,
            0.95,
            stats,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontsize=10,
            bbox={
                "boxstyle": "round,pad=0.3",
                "alpha": 0.95,
                "facecolor": "white",
                "edgecolor": "k",
            },
        )

        # --- Plot time-series
        steps_post = np.arange(len(crps_time))
        ax.plot(
            steps_post,
            spread_time,
            linewidth=2,
            label="Ens.Spread",
            color="tab:orange",
            alpha=0.25,
        )
        ax.plot(steps_post, crps_time, linewidth=1, label="CRPS", color="tab:blue")
        ax.plot(steps_post, rmse_time, linewidth=1, label="RMSE", color="tab:orange")

        # --- Asteatics
        ax.set_ylim(0, maxval)
        ax.set_xlabel("Model time-step")
        ax.legend(frameon=False, loc="upper left")
        ax.grid(alpha=0.25)

        if multiplot:
            ax.set_title(titles[i])

        if i == 0:
            ax.set_ylabel("Score")

    if show_fig:
        plt.show()

    return fig, ax
