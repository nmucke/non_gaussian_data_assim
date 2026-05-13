"""Compare the ensemble DA methods against the analytical Kalman filter.

The system here is *linear and Gaussian by construction*:

    x_{k+1} = M x_k + w_k,        w_k ~ N(0, Q)      (model / process error)
    y_k     = H x_k + v_k,        v_k ~ N(0, R)      (identity observations, H = I)

so the exact filtering posterior is available in closed form (the Kalman
filter). We

  1. roll out a "truth" trajectory of the system,
  2. generate noisy observations through an identity observation operator,
  3. assimilate them with the analytical Kalman filter that is handed *every*
     true parameter (M, Q, H, R, prior mean/cov), and
  4. assimilate the same observations with each ensemble method in this repo,

and then score every step.

The point of this script: most of the diagnostics we use in practice
(``scripts/main.py``) do **not** see the true posterior -- RMSE / MAE / MAPE
against the truth, CRPS, ensemble spread, the chi-squared innovation
consistency check. Here we *do* have the true posterior (the analytical Kalman
filter), so we can also compute the "gold-standard" posterior-divergence
metrics. Plotting the cheap, truth-only metrics alongside the gold-standard
ones lets you judge whether the cheap ones are reasonable surrogates when the
true posterior is unavailable.

Run (defaults to the 2D config):
    python scripts/analytical/main_analytical_methods.py
    python scripts/analytical/main_analytical_methods.py --config-name analytical_7d
    python scripts/analytical/main_analytical_methods.py ensemble_size=2000 data_assimilation_steps=50

The state dimension is whatever ``prior_mean`` has length of, so the script
works for any N. Metrics are computed over the full N-dimensional state;
the plots show only the first two state dimensions to stay readable.

Configs live in ``configs/analytical/`` (e.g. ``analytical_2d.yaml``,
``analytical_7d.yaml``). Shared helpers (Kalman filter, metrics, plotting)
live in ``_common.py``.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig

from _common import (
    METRIC_TITLES,
    TIME_MEAN_METRICS,
    compute_metrics,
    cov_ellipse,
    ensemble_mean_cov,
    get_state_dim,
    load_system_matrices,
    run_da_method,
    run_truth_and_kf,
    sample_analytical_posterior,
    summary_value,
)


# --------------------------------------------------------------------------- #
#                                  Run                                        #
# --------------------------------------------------------------------------- #


@hydra_main(config_path="../../configs/analytical", config_name="analytical_2d", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    state_dim = get_state_dim(cfg)
    n_ens = cfg.ensemble_size
    n_steps = cfg.data_assimilation_steps
    key = jax.random.PRNGKey(cfg.seed)

    forward_model = instantiate(cfg.forward_model)
    obs_operator = instantiate(cfg.obs_operator)
    num_obs = obs_operator.num_obs

    M_eff, Q, R, m0, P0, H = load_system_matrices(cfg, state_dim, num_obs, obs_operator)

    da_methods = {
        name: instantiate(c, R=R, obs_operator=obs_operator, forward_operator=forward_model)
        for name, c in cfg.da_methods.items()
    }

    print(f"\nLinear-Gaussian DA experiment | state_dim={state_dim}, "
          f"ensemble_size={n_ens}, steps={n_steps}, methods={list(da_methods)}")
    print(f"M (effective per outer step):\n{np.asarray(M_eff)}")
    print(f"Q:\n{np.asarray(Q)}\nR:\n{np.asarray(R)}")
    print(f"prior mean: {np.asarray(m0)}  prior cov diag: {np.diag(np.asarray(P0))}\n")

    # --- Truth + observations + analytical KF (generated ONCE, shared across methods)
    key, truth_key = jax.random.split(key)
    truth_arr, obs_arr, kf_mean_arr, kf_cov_arr = run_truth_and_kf(
        truth_key, n_steps, state_dim, M_eff, Q, R, H, m0, P0, obs_operator
    )

    # --- Prior ensemble (shared across methods) ---------------------------- #
    key, prior_key = jax.random.split(key)
    prior_ensemble = jax.random.multivariate_normal(
        prior_key, m0, P0, shape=(n_ens,)
    ).reshape(n_ens, 1, state_dim)

    # --- Run each DA method ------------------------------------------------ #
    ens_arr, pred_obs_arr = {}, {}
    for name, model in da_methods.items():
        key, sub = jax.random.split(key)
        ens, pred_obs = run_da_method(sub, model, forward_model, obs_operator,
                                      prior_ensemble, obs_arr, Q, n_steps, state_dim)
        ens_arr[name] = ens
        pred_obs_arr[name] = pred_obs

    # Sample the analytical posterior at every step (for the non-parametric KL).
    key, sub = jax.random.split(key)
    analytical_samples_arr = sample_analytical_posterior(
        sub, n_steps, kf_mean_arr, kf_cov_arr, n_ens
    )

    metrics = {
        name: compute_metrics(state_dim, n_steps, ens_arr[name], pred_obs_arr[name],
                              truth_arr, obs_arr, R, kf_mean_arr, kf_cov_arr,
                              analytical_samples_arr)
        for name in da_methods
    }

    print("\n==============  FINAL-STEP POSTERIOR vs ANALYTICAL KALMAN  ==============")
    print(f"Analytical posterior mean : {np.asarray(kf_mean_arr[-1])}")
    print(f"Analytical posterior cov  :\n{np.asarray(kf_cov_arr[-1])}\n")
    _print_summary(da_methods, metrics)
    _plot(state_dim, n_steps, n_ens, da_methods,
          truth_arr, obs_arr, kf_mean_arr, kf_cov_arr,
          ens_arr, analytical_samples_arr, metrics, R)


# --------------------------------------------------------------------------- #
#                              Summary printing                               #
# --------------------------------------------------------------------------- #

TRUTH_ONLY_METRICS = ["rmse", "mae", "mape", "crps", "spread", "chi_squared"]
POSTERIOR_METRICS = ["gaussian_kl", "hist_kl", "mean_err", "cov_err_fro"]


def _print_summary(da_methods, metrics) -> None:
    cols = ["rmse", "crps", "spread", "chi_squared",
            "gaussian_kl", "hist_kl", "mean_err", "cov_err_fro"]
    w = 17
    header = (f"{'method':<16}"
              + "".join(f"{(c + ('*' if c in TIME_MEAN_METRICS else '')):>{w}}" for c in cols))
    print(header)
    print("-" * len(header))
    for name in da_methods:
        vals = [summary_value(metrics[name][c], c) for c in cols]
        print(f"{name:<16}" + "".join(f"{v:>{w}.4f}" for v in vals))
    print("* = time-mean (others are final-step values)")
    print("(truth-only metrics: rmse, crps, spread, chi_squared | "
          "posterior-aware: gaussian_kl, hist_kl, mean_err, cov_err_fro)")
    print("=" * len(header) + "\n")


# --------------------------------------------------------------------------- #
#                                 Plotting                                    #
# --------------------------------------------------------------------------- #

_PALETTE = ["tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:brown", "tab:olive"]


def _color(name, da_methods):
    return _PALETTE[list(da_methods).index(name) % len(_PALETTE)]


def _plot(state_dim, n_steps, n_ens, da_methods,
          truth_arr, obs_arr, kf_mean_arr, kf_cov_arr,
          ens_arr, analytical_samples_arr, metrics, R) -> None:
    # Metrics are computed on the full N-d state. The plots show only the first
    # two state dimensions to stay readable when state_dim is large.
    n_plot_dims = min(state_dim, 2)

    steps = np.arange(n_steps + 1)
    truth = np.asarray(truth_arr).reshape(n_steps + 1, state_dim)
    obs = np.asarray(obs_arr) if n_steps > 0 else None
    kf_mean = np.asarray(kf_mean_arr)
    kf_std = np.sqrt(np.clip(np.asarray(jnp.diagonal(kf_cov_arr, axis1=1, axis2=2)), 0.0, None))
    final_ens = {n: np.asarray(ens_arr[n][:, -1]).reshape(n_ens, -1) for n in da_methods}
    m_kf_f, P_kf_f = np.asarray(kf_mean_arr[-1]), np.asarray(kf_cov_arr[-1])
    ana_f = np.asarray(analytical_samples_arr[-1])

    # ===== Figure 1: state trajectories (first 2 dims) ==================== #
    fig, axes = plt.subplots(n_plot_dims, 1, figsize=(11, 3.0 * n_plot_dims), squeeze=False)
    for d in range(n_plot_dims):
        ax = axes[d, 0]
        ax.plot(steps, truth[:, d], "k-", lw=2, label="truth")
        if obs is not None and d < obs.shape[1]:
            ax.plot(steps[1:], obs[:, d], "kx", ms=6, label="observations")
        ax.plot(steps, kf_mean[:, d], color="tab:red", lw=2, label="Kalman (analytical)")
        ax.fill_between(steps, kf_mean[:, d] - 2 * kf_std[:, d],
                        kf_mean[:, d] + 2 * kf_std[:, d], color="tab:red", alpha=0.15,
                        label="Kalman ±2σ")
        for name in da_methods:
            mean = np.asarray(ens_arr[name][:, :, 0, d]).mean(axis=0)
            ax.plot(steps, mean, "--", color=_color(name, da_methods), label=name)
        ax.set_ylabel(f"state[{d}]")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(ncol=3, fontsize=8)
    axes[-1, 0].set_xlabel("assimilation step")
    fig.suptitle(f"State trajectories: truth, analytical Kalman filter, ensemble methods  "
                 f"(showing first {n_plot_dims} of {state_dim} state dims)")
    fig.tight_layout()

    # ===== Figure 2: per-method trajectory with ±2σ uncertainty bands ===== #
    nm = len(da_methods)
    fig, axes = plt.subplots(nm, n_plot_dims, figsize=(6.0 * n_plot_dims, 2.6 * nm),
                             squeeze=False, sharex=True)
    for r, name in enumerate(da_methods):
        ens = np.asarray(ens_arr[name][:, :, 0, :])
        method_mean = ens.mean(axis=0)
        method_std = ens.std(axis=0)
        col = _color(name, da_methods)
        for d in range(n_plot_dims):
            ax = axes[r, d]
            ax.plot(steps, truth[:, d], "k-", lw=1.6, label="truth")
            ax.plot(steps, kf_mean[:, d], color="tab:red", lw=1.6,
                    label="Kalman (analytical)")
            ax.fill_between(steps, kf_mean[:, d] - 2 * kf_std[:, d],
                            kf_mean[:, d] + 2 * kf_std[:, d],
                            color="tab:red", alpha=0.1, label="Kalman ±2σ")
            ax.plot(steps, method_mean[:, d], color=col, lw=1.6, label=name)
            ax.fill_between(steps, method_mean[:, d] - 2 * method_std[:, d],
                            method_mean[:, d] + 2 * method_std[:, d],
                            color=col, alpha=0.3, label=f"{name} ±2σ")
            if r == 0:
                ax.set_title(f"state[{d}]")
            if r == nm - 1:
                ax.set_xlabel("assimilation step")
            if d == 0:
                ax.set_ylabel(name)
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(f"Per-method trajectory with ±2σ uncertainty bands  "
                 f"(Kalman shaded at α=0.1, method shaded at α=0.3; "
                 f"first {n_plot_dims} of {state_dim} state dims)")
    fig.tight_layout()

    # ===== Figure 3: all metrics per assimilation step ==================== #
    panels = TRUTH_ONLY_METRICS + POSTERIOR_METRICS
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    kf_spread = np.sqrt(np.mean(
        np.asarray(jnp.diagonal(kf_cov_arr, axis1=1, axis2=2)), axis=1
    ))
    for ax, key in zip(axes.flat, panels):
        ax.axis("on")
        truth_only = key in TRUTH_ONLY_METRICS
        for name in da_methods:
            x = steps[1:] if key == "chi_squared" else steps
            ax.plot(x, metrics[name][key], "-o", ms=3, color=_color(name, da_methods), label=name)
        if key == "chi_squared":
            ax.axhline(1.0, color="k", ls=":", lw=1)
        if key == "spread":
            ax.plot(steps, kf_spread, color="tab:red", lw=2, label="Kalman (analytical)")
            ax.legend(fontsize=8)
        if key in POSTERIOR_METRICS:
            ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(METRIC_TITLES.get(key, key)
                     + ("" if truth_only else "  [needs true posterior]"),
                     fontsize=9, color=("black" if truth_only else "tab:red"))
        ax.set_xlabel("assimilation step")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Diagnostics per assimilation step — truth-only (black titles) vs "
                 "posterior-aware metrics (red titles)")
    fig.tight_layout()

    # ===== Figure 4: cheap metric vs gold-standard KL (scatter) =========== #
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), squeeze=False)
    for ax, key in zip(axes.flat, TRUTH_ONLY_METRICS):
        for name in da_methods:
            gkl = metrics[name]["gaussian_kl"]
            yv = gkl[1:] if key == "chi_squared" else gkl
            ax.scatter(metrics[name][key], yv, s=25, color=_color(name, da_methods),
                       label=name, alpha=0.8)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel(METRIC_TITLES.get(key, key))
        ax.set_ylabel("Gaussian KL(method ‖ analytical)")
        ax.set_title(f"{METRIC_TITLES.get(key, key)}  vs  true-posterior KL", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Is the cheap (truth-only) metric a good surrogate for the "
                 "true-posterior KL?  (each point = one method at one step)")
    fig.tight_layout()

    # ===== Figure 5: normalized-innovation histograms ===================== #
    fig, axes = plt.subplots(1, len(da_methods), figsize=(4.6 * len(da_methods), 3.8),
                             squeeze=False, sharex=True, sharey=True)
    g = np.linspace(-4, 4, 200)
    std_normal = np.exp(-0.5 * g ** 2) / np.sqrt(2 * np.pi)
    for ax, name in zip(axes[0], da_methods):
        z = metrics[name]["z"].reshape(-1)
        ax.hist(z, bins=31, density=True, color=_color(name, da_methods), alpha=0.6)
        ax.plot(g, std_normal, "k-", lw=2, label="N(0,1)")
        ax.set_title(f"{name}: z (mean={z.mean():.2f}, var={z.var():.2f})", fontsize=9)
        ax.set_xlabel("z")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Innovation whitening check — z should be ~N(0,1) if forecast spread + R are consistent")
    fig.tight_layout()

    # ===== Figure 6: final-step posterior marginals (first 2 dims) ======== #
    fig, axes = plt.subplots(1, n_plot_dims, figsize=(5.0 * n_plot_dims, 4.0), squeeze=False)
    for d in range(n_plot_dims):
        ax = axes[0, d]
        mu, sigma = float(m_kf_f[d]), float(np.sqrt(P_kf_f[d, d]))
        gx = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        ax.plot(gx, np.exp(-0.5 * ((gx - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
                color="tab:red", lw=2.5, label="analytical Kalman")
        for name in da_methods:
            ax.hist(final_ens[name][:, d], bins=40, density=True, histtype="step",
                    lw=1.5, color=_color(name, da_methods), label=name)
        ax.set_title(f"posterior marginal — state[{d}]")
        ax.set_xlabel(f"state[{d}]")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Final-step posterior marginals vs analytical Kalman  "
                 f"(showing first {n_plot_dims} of {state_dim} state dims)")
    fig.tight_layout()

    # ===== Figure 7: joint posterior in the first 2 dims ================== #
    if n_plot_dims >= 2:
        m_kf_2d = m_kf_f[:2]
        P_kf_2d = P_kf_f[:2, :2]
        nm = len(da_methods)
        fig, axes = plt.subplots(1, nm, figsize=(5.0 * nm, 4.8), squeeze=False,
                                 sharex=True, sharey=True)
        for ax, name in zip(axes[0], da_methods):
            ax.scatter(ana_f[:, 0], ana_f[:, 1], s=6, alpha=0.15, color="tab:red",
                       label="analytical samples")
            ax.scatter(final_ens[name][:, 0], final_ens[name][:, 1], s=6, alpha=0.15,
                       color=_color(name, da_methods), label=f"{name} samples")
            mu_p_full, sig_p_full = ensemble_mean_cov(jnp.asarray(final_ens[name]))
            mu_p_2d = np.asarray(mu_p_full)[:2]
            sig_p_2d = np.asarray(sig_p_full)[:2, :2]
            for n_std, ls in ((1.0, "-"), (2.0, "--")):
                cov_ellipse(ax, m_kf_2d, P_kf_2d, n_std, edgecolor="tab:red", lw=2, ls=ls)
                cov_ellipse(ax, mu_p_2d, sig_p_2d, n_std,
                            edgecolor=_color(name, da_methods), lw=2, ls=ls)
            ax.set_title(name)
            ax.set_xlabel("state[0]")
            ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel("state[1]")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Final-step joint posterior in state[0]-state[1]: ensemble vs "
                     f"analytical Kalman (solid = 1σ, dashed = 2σ ellipses; "
                     f"state_dim={state_dim})")
        fig.tight_layout()

    # ===== Figure 8: observation-space diagnostics ======================= #
    obs_noise_std = float(np.sqrt(np.mean(np.diag(np.asarray(R)))))
    obs_panels = [
        "innov_rmse", "innov_mae", "innov_bias", "pred_obs_spread",
        "pred_obs_crps", "chi_squared", "coverage_2sigma",
    ]
    ncols = 4
    nrows = int(np.ceil(len(obs_panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    obs_steps = steps[1:]
    for ax, key in zip(axes.flat, obs_panels):
        ax.axis("on")
        for name in da_methods:
            ax.plot(obs_steps, metrics[name][key], "-o", ms=3,
                    color=_color(name, da_methods), label=name)
        if key in ("innov_rmse", "innov_mae", "pred_obs_spread"):
            ax.axhline(obs_noise_std, color="k", ls=":", lw=1,
                       label=f"obs-noise std = {obs_noise_std:.2f}")
        if key == "innov_bias":
            ax.axhline(0.0, color="k", ls=":", lw=1, label="unbiased")
        if key == "chi_squared":
            ax.axhline(1.0, color="k", ls=":", lw=1, label="ideal = 1")
        if key == "coverage_2sigma":
            ax.axhline(0.95, color="k", ls=":", lw=1, label="ideal ≈ 0.95")
            ax.set_ylim(-0.05, 1.05)
        ax.set_title(METRIC_TITLES.get(key, key), fontsize=9)
        ax.set_xlabel("assimilation step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Observation-space diagnostics: predicted observations vs noisy observations")
    fig.tight_layout()

    # ===== Figure 9: obs-only metric vs its true-posterior analog ========= #
    pairings = [
        ("innov_rmse", "mean_err"),
        ("innov_mae", "mean_err"),
        ("innov_bias", "state_bias"),
        ("pred_obs_spread", "spread"),
        ("pred_obs_crps", "crps"),
        ("chi_squared", "state_mahalanobis"),
        ("coverage_2sigma", "state_coverage_2sigma"),
    ]
    ncols = 4
    nrows = int(np.ceil(len(pairings) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (obs_key, state_key) in zip(axes.flat, pairings):
        ax.axis("on")
        for name in da_methods:
            x = np.asarray(metrics[name][obs_key])
            y = np.asarray(metrics[name][state_key])
            if y.shape[0] == x.shape[0] + 1:
                y = y[1:]
            ax.scatter(x, y, s=25, color=_color(name, da_methods), label=name, alpha=0.8)
        ax.set_xlabel(METRIC_TITLES.get(obs_key, obs_key) + "\n(obs-only)", fontsize=8)
        ax.set_ylabel(METRIC_TITLES.get(state_key, state_key) + "\n(full-state, true posterior)",
                      fontsize=8)
        ax.set_title(f"{obs_key}  vs  {state_key}", fontsize=9)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Reliability of obs-only metrics: each obs-only diagnostic plotted "
                 "against its full-state, true-posterior analog\n"
                 "(monotone clouds ⇒ the obs-only metric is a trustworthy proxy for "
                 "the full-information metric)")
    fig.tight_layout()

    # ===== Figure 10: each obs-only metric vs true-posterior KL =========== #
    obs_metrics = ["innov_rmse", "innov_mae", "innov_bias", "pred_obs_spread",
                   "pred_obs_crps", "chi_squared", "coverage_2sigma"]
    ncols = 4
    nrows = int(np.ceil(len(obs_metrics) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, key in zip(axes.flat, obs_metrics):
        ax.axis("on")
        for name in da_methods:
            x = np.asarray(metrics[name][key])
            y = np.asarray(metrics[name]["gaussian_kl"])
            ax.scatter(x, y[1:], s=25, color=_color(name, da_methods), label=name, alpha=0.8)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel(METRIC_TITLES.get(key, key) + "\n(obs-only)", fontsize=8)
        ax.set_ylabel("Gaussian KL(method ‖ analytical)", fontsize=8)
        ax.set_title(f"{key}  vs  Gaussian KL", fontsize=9)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Obs-only metrics vs the gold-standard true-posterior KL  "
                 "(each point = one method at one step)")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
