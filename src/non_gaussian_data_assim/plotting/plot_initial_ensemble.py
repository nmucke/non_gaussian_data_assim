"""Plotting Initial- Conditions """

from typing import Mapping, Optional, Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

## Plot initial field
#   - I.C. (after spin-up)
#   - Ensemble Perturbations (mean)
#   - Breeding statistics
# fast (large-scale) atmosphereslow  || slow (small-scale) ocean


def plot_initial_fields(
    initial_ensemble: jnp.ndarray,
    truth_t0: jnp.ndarray,
    x_before_spinup: Optional[jnp.ndarray] = None,
    best_guess_profile: Optional[jnp.ndarray] = None,
    bv_dict: Optional[dict[str, jnp.ndarray]] = None,
) -> None:

    TOT_STATES = initial_ensemble.shape[2]
    TIMESTEP = 0

    nrows = 2 if bv_dict is not None else 1

    fig, axs = plt.subplots(
        ncols=TOT_STATES, nrows=nrows, figsize=(9 * TOT_STATES, 4 * nrows)
    )
    for state in range(TOT_STATES):
        ax = axs[0, state]

        # # --- DATA SELECTION ------------
        # ic = ic_ref[0, state, :]
        # ic_spinup = true_sol[0, TIMESTEP, state, :]
        # prior_ensemble = posterior_ensemble[:, TIMESTEP, state, :]
        if x_before_spinup is not None:
            ax.plot(
                x_before_spinup,
                linewidth=1,
                alpha=0.2,
                c="k",
                label="Truth (before SpinUp)",
                zorder=0,
            )

        # -- Plot truth
        ax.plot(truth_t0, linewidth=2, alpha=0.8, c="k", label="Truth t0", zorder=1)

        for i in range(initial_ensemble.shape[0]):
            label = "Ensemble" if i == 0 else None
            ax.plot(
                initial_ensemble[i],
                c="tab:orange",
                linewidth=0.5,
                alpha=0.2,
                label=label,
                zorder=2,
            )

        if best_guess_profile is not None:
            ax.plot(
                best_guess_profile[0, state],
                c="tab:blue",
                label="best-guess",
                linewidth=3,
                alpha=0.95,
                zorder=1,
            )

        ax.set_title(f"State {state}  |  Time-step 0")
        ax.set_xlabel("State Dimension")
    axs[1].legend()

    # ----- Plot Breeeding Stats
    if bv_dict is not None:

        # Extract fields
        bv_cyc_final = bv_dict["bv1_states"]
        bv_norms = bv_dict["bvp_norm"]

        ax_bvpert = axs[1, 0]

        # -- Plot breeding perturbation at end of last cycle
        colors = ["k", "tab:orange"]
        for state in range(TOT_STATES):
            for ens_id in range(bv_cyc_final.shape[0]):
                ax_bvpert.plot(
                    bv_cyc_final[ens_id, -1, state, :],
                    color=colors[state],
                    alpha=0.1 + 0.1 * state,
                )

            ymin, ymax = ax_bvpert.get_ylim()
            bound = max(abs(ymin), abs(ymax))
            ax_bvpert.set_ylim(-bound, bound)

        # ---- Plot evolution of BV norm over breeding "spin-up"
    #     bv_cylces = cfg.case.initial_ensemble.ens_perturbation.breeding_cycles
    #     steps_per_cyle = (
    #         cfg.case.initial_ensemble.ens_perturbation.outer_steps_per_cycle
    #         * cfg.model_integration_steps
    #     )
    #     tot_steps = steps_per_cyle * bv_cylces

    # def create_xaxis_for_bv_norm(bv_cylces, steps_per_cyle):
    #     n = steps_per_cyle
    #     num_blocks = bv_cylces
    #     eps = 10 * np.spacing(num_blocks * n)
    #     k = np.arange(num_blocks)[:, None]
    #     j = np.arange(n + 1)[None, :]
    #     x_blocks = k * n + j + k * eps
    #     return x_blocks.ravel()

    # x = create_xaxis_for_bv_norm(bv_cylces, steps_per_cyle)

    ax_bvnorm = axs[1, 1]
    bv_pert_ensmean = jnp.mean(bv_norms, axis=0)
    ax_bvnorm.plot(bv_pert_ensmean)
    ax_bvnorm.set_ylabel("|| BV perturbation ||")
    ax_bvnorm.set_xlabel("Inner Model steps")
    ax_bvnorm.set_title("Evolution of avg. ||BV|| over all BV-cycles")
    # ax.set_xticks(range(0, tot_steps, bv_cylces))
    ax_bvnorm.grid(True)
