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
    ensemble: jnp.ndarray,
    truth_t0: jnp.ndarray,
    x_before_spinup: Optional[jnp.ndarray] = None,
    best_guess_profile: Optional[jnp.ndarray] = None,
    bv_dict: Optional[dict[str, jnp.ndarray]] = None,
) -> None:

    TOT_STATES = ensemble.shape[2]
    TIMESTEP = 0

    # --- Select Initial Time-Step of Ensemble and Truth
    assert (
        ensemble.ndim == 4
    ), "Shape of ensemble must be (ensemble, time, StateNr, StateDim)"
    ensemble = ensemble[:, TIMESTEP, ...]
    assert truth_t0.ndim == 4, "Shape of truth must be (1, time, StateNr, StateDim)"
    truth_t0 = truth_t0[0, TIMESTEP, ...]

    if x_before_spinup is not None:
        assert (
            x_before_spinup.ndim == 2
        ), "Before-SpinUp Profile must have shape 2 (StateNr, StateDim)"
    if best_guess_profile is not None:
        assert (
            best_guess_profile.ndim == 2
        ), "best_guess_profile must have shape 2 (StateNr, StateDim)"

    nrows = 1 if bv_dict is None else 2

    fig, axs = plt.subplots(
        ncols=TOT_STATES, nrows=nrows, figsize=(9 * TOT_STATES, 4 * nrows)
    )

    for state in range(TOT_STATES):
        ax = axs[state] if bv_dict is None else axs[0, state]

        # -- Plot Initial state BEFORE spin-up
        if x_before_spinup is not None:
            tit0 = "Truth (before SpinUp)" if state == 1 else None
            ax.plot(
                x_before_spinup[state],
                linewidth=1,
                alpha=0.2,
                c="k",
                label=tit0,
                zorder=0,
            )

        # -- Plot truth
        ax.plot(
            truth_t0[state], linewidth=2, alpha=0.8, c="k", label="Truth t0", zorder=1
        )

        # -- Plot Initial Best-guess profile
        if best_guess_profile is not None:
            ax.plot(
                best_guess_profile[state],
                c="tab:blue",
                label="best-guess",
                linewidth=1.5,
                alpha=0.5,
                zorder=1,
            )

        # -- Plot Ensemble
        for i in range(ensemble.shape[0]):
            label = "Ensemble" if i == 0 else None
            ax.plot(
                ensemble[i, state, :],
                c="tab:orange",
                linewidth=0.5,
                alpha=0.2,
                label=label,
                zorder=0,
            )

        # -- Plot Initial Best-guess profile
        if best_guess_profile is not None:
            ax.plot(
                best_guess_profile[state],
                c="tab:blue",
                label="best-guess",
                linewidth=2,
                alpha=0.5,
                zorder=1,
            )

        ax.set_title(f"State {state}  |  Time-step 0")
        ax.set_xlabel("State Dimension")
        if state == 0:
            ax.legend()

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
                label = f"BV perts state {state}" if ens_id == 0 else None
                ax_bvpert.plot(
                    bv_cyc_final[ens_id, -1, state, :],
                    color=colors[state],
                    alpha=0.1 + 0.1 * state,
                    label=label,
                )

            ymin, ymax = ax_bvpert.get_ylim()
            bound = max(abs(ymin), abs(ymax))
            ax_bvpert.set_ylim(-bound, bound)
        ax_bvpert.legend()
        ax_bvpert.set_xlabel("State Dimension")
        ax_bvpert.set_title("BV at the end of the last cycle")
        # ---- Plot evolution of BV norm over breeding "spin-up"
        #     bv_cylces = cfg.case.ensemble.ens_perturbation.breeding_cycles
        #     steps_per_cyle = (
        #         cfg.case.ensemble.ens_perturbation.outer_steps_per_cycle
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

        fig.tight_layout()
