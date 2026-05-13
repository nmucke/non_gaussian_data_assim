"""Compare a single DA method against the analytical Kalman filter, sweeping ensemble size.

Companion to ``main_analytical_methods.py``. The system is the same linear-Gaussian
setup -- the analytical Kalman filter is the gold-standard posterior -- but
instead of comparing several DA methods at one ensemble size, we run *one* DA
method (default EnKF; edit ``da_method`` in the YAML to switch) at every
ensemble size in ``cfg.ensemble_sizes`` and look at how the diagnostics change
with N.

Run:
    python scripts/analytical/main_analytical_ensemble_size.py
    python scripts/analytical/main_analytical_ensemble_size.py --config-name ensemble_7d
    python scripts/analytical/main_analytical_ensemble_size.py 'ensemble_sizes=[10,100,1000,10000]'

Configs live in ``configs/analytical/`` (``ensemble_2d.yaml``, ``ensemble_7d.yaml``).
Shared helpers (Kalman filter, metrics, plotting) live in ``_common.py``.
"""

import jax
import jax.numpy as jnp
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
import numpy as np
from hydra import main as hydra_main
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from _common import (
    METRIC_TITLES,
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
#                                  Main                                       #
# --------------------------------------------------------------------------- #


@hydra_main(config_path="../../configs/analytical", config_name="ensemble_2d", version_base=None)  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)  # allow injecting `ensemble_size` per run

    state_dim = get_state_dim(cfg)
    sizes = sorted(int(N) for N in cfg.ensemble_sizes)
    n_steps = cfg.data_assimilation_steps
    key = jax.random.PRNGKey(cfg.seed)

    forward_model = instantiate(cfg.forward_model)
    obs_operator = instantiate(cfg.obs_operator)
    num_obs = obs_operator.num_obs

    M_eff, Q, R, m0, P0, H = load_system_matrices(cfg, state_dim, num_obs, obs_operator)

    method_name = cfg.da_method._target_.split(".")[-1]
    print(f"\nLinear-Gaussian ensemble-size sweep | state_dim={state_dim}, "
          f"method={method_name}, sizes={sizes}, steps={n_steps}")

    # --- Truth + observations + analytical KF (generated ONCE, shared across N) #
    key, truth_key = jax.random.split(key)
    truth_arr, obs_arr, kf_mean_arr, kf_cov_arr = run_truth_and_kf(
        truth_key, n_steps, state_dim, M_eff, Q, R, H, m0, P0, obs_operator
    )

    # --- Run the DA method at each ensemble size --------------------------- #
    ens_arr_by_N, pred_obs_arr_by_N = {}, {}
    for N in sizes:
        print(f"  ensemble_size = {N} ...")
        key, prior_key, run_key = jax.random.split(key, 3)
        prior_ensemble = jax.random.multivariate_normal(
            prior_key, m0, P0, shape=(N,)
        ).reshape(N, 1, state_dim)
        cfg.ensemble_size = N  # resolves any ${ensemble_size} in cfg.da_method
        da_model = instantiate(cfg.da_method, R=R, obs_operator=obs_operator,
                               forward_operator=forward_model)
        ens, pred_obs = run_da_method(run_key, da_model, forward_model, obs_operator,
                                      prior_ensemble, obs_arr, Q, n_steps, state_dim)
        ens_arr_by_N[N] = ens
        pred_obs_arr_by_N[N] = pred_obs

    # Sample analytical posterior at every step (fixed size = max(N) for the
    # non-parametric KL).
    key, sub = jax.random.split(key)
    analytical_samples_arr = sample_analytical_posterior(
        sub, n_steps, kf_mean_arr, kf_cov_arr, max(sizes)
    )

    metrics = {
        N: compute_metrics(state_dim, n_steps, ens_arr_by_N[N], pred_obs_arr_by_N[N],
                           truth_arr, obs_arr, R, kf_mean_arr, kf_cov_arr,
                           analytical_samples_arr)
        for N in sizes
    }

    _print_summary(sizes, metrics, method_name)
    _plot(state_dim, n_steps, sizes, method_name,
          truth_arr, obs_arr, kf_mean_arr, kf_cov_arr,
          ens_arr_by_N, analytical_samples_arr, metrics, R)


# --------------------------------------------------------------------------- #
#                              Summary printing                               #
# --------------------------------------------------------------------------- #

PER_STEP_METRICS = [
    "rmse", "mae", "mape", "crps", "spread", "chi_squared",
    "gaussian_kl", "hist_kl", "mean_err", "cov_err_fro",
]
OBS_METRICS = [
    "innov_rmse", "innov_mae", "innov_bias", "pred_obs_spread",
    "pred_obs_crps", "chi_squared", "coverage_2sigma",
]
SUMMARY_METRICS = [
    "rmse", "crps", "spread", "chi_squared",
    "gaussian_kl", "hist_kl", "mean_err", "cov_err_fro",
    "innov_rmse", "pred_obs_spread", "pred_obs_crps", "coverage_2sigma",
]


def _print_summary(sizes, metrics, method_name) -> None:
    print(f"\n=========  FINAL-STEP METRICS by ensemble size  ({method_name})  =========")
    cols = SUMMARY_METRICS
    w = 16
    header = f"{'N':<6}" + "".join(f"{(c + ('*' if c == 'chi_squared' else '')):>{w}}" for c in cols)
    print(header)
    print("-" * len(header))
    for N in sizes:
        vals = [summary_value(metrics[N][c], c) for c in cols]
        print(f"{N:<6}" + "".join(f"{v:>{w}.4f}" for v in vals))
    print("* = time-mean (others are final-step values)")
    print("=" * len(header) + "\n")


# --------------------------------------------------------------------------- #
#                                 Plotting                                    #
# --------------------------------------------------------------------------- #


def _size_colors(sizes):
    norm = mcolors.LogNorm(vmin=min(sizes), vmax=max(sizes)) \
        if min(sizes) > 0 and max(sizes) > min(sizes) \
        else mcolors.Normalize(vmin=min(sizes), vmax=max(sizes))
    cmap = mcm.viridis
    return {N: cmap(norm(N)) for N in sizes}


def _plot(state_dim, n_steps, sizes, method_name,
          truth_arr, obs_arr, kf_mean_arr, kf_cov_arr,
          ens_arr_by_N, analytical_samples_arr, metrics, R) -> None:
    n_plot_dims = min(state_dim, 2)
    steps = np.arange(n_steps + 1)
    truth = np.asarray(truth_arr).reshape(n_steps + 1, state_dim)
    obs = np.asarray(obs_arr) if n_steps > 0 else None
    kf_mean = np.asarray(kf_mean_arr)
    kf_std = np.sqrt(np.clip(np.asarray(jnp.diagonal(kf_cov_arr, axis1=1, axis2=2)), 0.0, None))
    colors = _size_colors(sizes)

    # ===== Figure 1: per-N trajectory with truth + KF + method ±2σ ======== #
    fig, axes = plt.subplots(len(sizes), n_plot_dims,
                             figsize=(6.0 * n_plot_dims, 2.6 * len(sizes)),
                             squeeze=False, sharex=True)
    for r, N in enumerate(sizes):
        ens = np.asarray(ens_arr_by_N[N][:, :, 0, :])
        mean_N = ens.mean(axis=0)
        std_N = ens.std(axis=0)
        for d in range(n_plot_dims):
            ax = axes[r, d]
            ax.plot(steps, truth[:, d], "k-", lw=1.6, label="truth")
            if obs is not None and d < obs.shape[1]:
                ax.plot(steps[1:], obs[:, d], "kx", ms=5, alpha=0.5)
            ax.plot(steps, kf_mean[:, d], color="tab:red", lw=1.6, label="Kalman (analytical)")
            ax.fill_between(steps, kf_mean[:, d] - 2 * kf_std[:, d],
                            kf_mean[:, d] + 2 * kf_std[:, d],
                            color="tab:red", alpha=0.1, label="Kalman ±2σ")
            ax.plot(steps, mean_N[:, d], color=colors[N], lw=1.6, label=f"N={N}")
            ax.fill_between(steps, mean_N[:, d] - 2 * std_N[:, d],
                            mean_N[:, d] + 2 * std_N[:, d],
                            color=colors[N], alpha=0.3, label=f"N={N} ±2σ")
            if r == 0:
                ax.set_title(f"state[{d}]")
            if r == len(sizes) - 1:
                ax.set_xlabel("assimilation step")
            if d == 0:
                ax.set_ylabel(f"N = {N}")
            ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(f"Per-ensemble-size trajectory with ±2σ uncertainty bands "
                 f"({method_name}; Kalman α=0.1, method α=0.3; "
                 f"first {n_plot_dims} of {state_dim} state dims)")
    fig.tight_layout()

    # ===== Figure 2: per-step metrics, lines = ensemble sizes ============= #
    panels = PER_STEP_METRICS
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    kf_spread = np.sqrt(np.mean(np.asarray(jnp.diagonal(kf_cov_arr, axis1=1, axis2=2)), axis=1))
    for ax, key in zip(axes.flat, panels):
        ax.axis("on")
        for N in sizes:
            x = steps[1:] if key == "chi_squared" else steps
            ax.plot(x, metrics[N][key], "-o", ms=3, color=colors[N], label=f"N={N}")
        if key == "chi_squared":
            ax.axhline(1.0, color="k", ls=":", lw=1)
        if key == "spread":
            ax.plot(steps, kf_spread, color="tab:red", lw=2, label="Kalman (analytical)")
        if key in ("gaussian_kl", "hist_kl", "mean_err", "cov_err_fro"):
            ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(METRIC_TITLES.get(key, key), fontsize=9)
        ax.set_xlabel("assimilation step")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Per-step diagnostics, lines = ensemble size  ({method_name})")
    fig.tight_layout()

    # ===== Figure 3: metric value vs ensemble size (aggregated over time) ==
    panels = SUMMARY_METRICS
    ncols = 4
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    sizes_np = np.asarray(sizes)
    for ax, key in zip(axes.flat, panels):
        ax.axis("on")
        ys = np.array([summary_value(metrics[N][key], key) for N in sizes])
        ax.plot(sizes_np, ys, "-o", color="tab:blue")
        for N, y in zip(sizes_np, ys):
            ax.scatter([N], [y], s=45, color=colors[int(N)], zorder=3)
        ax.set_xscale("log")
        if key == "chi_squared":
            ax.axhline(1.0, color="k", ls=":", lw=1, label="ideal = 1")
            ax.legend(fontsize=7)
        if key == "coverage_2sigma":
            ax.axhline(0.95, color="k", ls=":", lw=1, label="ideal ≈ 0.95")
            ax.legend(fontsize=7)
            ax.set_ylim(-0.05, 1.05)
        suffix = "  (time-mean)" if key == "chi_squared" else "  (final step)"
        ax.set_title(METRIC_TITLES.get(key, key) + suffix, fontsize=9)
        ax.set_xlabel("ensemble size  N")
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(f"Metric value vs ensemble size  ({method_name})  -- "
                 "log-scaled x-axis; point colors match Figure 1/2.")
    fig.tight_layout()

    # ===== Figure 4: observation-space diagnostics per step ================ #
    obs_noise_std = float(np.sqrt(np.mean(np.diag(np.asarray(R)))))
    ncols = 4
    nrows = int(np.ceil(len(OBS_METRICS) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    obs_steps = steps[1:]
    for ax, key in zip(axes.flat, OBS_METRICS):
        ax.axis("on")
        for N in sizes:
            ax.plot(obs_steps, metrics[N][key], "-o", ms=3, color=colors[N], label=f"N={N}")
        if key in ("innov_rmse", "innov_mae", "pred_obs_spread"):
            ax.axhline(obs_noise_std, color="k", ls=":", lw=1,
                       label=f"obs-noise std = {obs_noise_std:.2f}")
        if key == "innov_bias":
            ax.axhline(0.0, color="k", ls=":", lw=1)
        if key == "chi_squared":
            ax.axhline(1.0, color="k", ls=":", lw=1)
        if key == "coverage_2sigma":
            ax.axhline(0.95, color="k", ls=":", lw=1)
            ax.set_ylim(-0.05, 1.05)
        ax.set_title(METRIC_TITLES.get(key, key), fontsize=9)
        ax.set_xlabel("assimilation step")
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Observation-space diagnostics per step  ({method_name})")
    fig.tight_layout()

    # ===== Figure 5: reliability scatter (obs-only vs true-posterior) ====== #
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
        for N in sizes:
            x = np.asarray(metrics[N][obs_key])
            y = np.asarray(metrics[N][state_key])
            if y.shape[0] == x.shape[0] + 1:
                y = y[1:]
            ax.scatter(x, y, s=22, color=colors[N], label=f"N={N}", alpha=0.8)
        ax.set_xlabel(METRIC_TITLES.get(obs_key, obs_key) + " (obs-only)", fontsize=8)
        ax.set_ylabel(METRIC_TITLES.get(state_key, state_key) + " (full-state)", fontsize=8)
        ax.set_title(f"{obs_key}  vs  {state_key}", fontsize=9)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Reliability of obs-only metrics across ensemble sizes  ({method_name})\n"
                 f"(each point = one assimilation step at one N; monotone clouds ⇒ trustworthy proxy)")
    fig.tight_layout()

    # ===== Figure 5b: each obs-only metric vs true-posterior KL =========== #
    obs_metrics_kl = ["innov_rmse", "innov_mae", "innov_bias", "pred_obs_spread",
                      "pred_obs_crps", "chi_squared", "coverage_2sigma"]
    ncols = 4
    nrows = int(np.ceil(len(obs_metrics_kl) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, key in zip(axes.flat, obs_metrics_kl):
        ax.axis("on")
        for N in sizes:
            x = np.asarray(metrics[N][key])
            y = np.asarray(metrics[N]["gaussian_kl"])
            ax.scatter(x, y[1:], s=22, color=colors[N], label=f"N={N}", alpha=0.8)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel(METRIC_TITLES.get(key, key) + " (obs-only)", fontsize=8)
        ax.set_ylabel("Gaussian KL(method ‖ analytical)", fontsize=8)
        ax.set_title(f"{key}  vs  Gaussian KL", fontsize=9)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(f"Obs-only metrics vs the gold-standard true-posterior KL  "
                 f"({method_name}; each point = one step at one N)")
    fig.tight_layout()

    # ===== Figure 6: final-step posterior marginals, color = N ============ #
    m_kf_f, P_kf_f = np.asarray(kf_mean_arr[-1]), np.asarray(kf_cov_arr[-1])
    fig, axes = plt.subplots(1, n_plot_dims, figsize=(5.0 * n_plot_dims, 4.0), squeeze=False)
    for d in range(n_plot_dims):
        ax = axes[0, d]
        mu, sigma = float(m_kf_f[d]), float(np.sqrt(P_kf_f[d, d]))
        gx = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 300)
        ax.plot(gx, np.exp(-0.5 * ((gx - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)),
                color="tab:red", lw=2.5, label="analytical Kalman")
        for N in sizes:
            v = np.asarray(ens_arr_by_N[N][:, -1, 0, d])
            ax.hist(v, bins=30, density=True, histtype="step", lw=1.5,
                    color=colors[N], label=f"N={N}")
        ax.set_title(f"posterior marginal — state[{d}] (final step)")
        ax.set_xlabel(f"state[{d}]")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Final-step posterior marginals across ensemble sizes  ({method_name})")
    fig.tight_layout()

    # ===== Figure 7 (2D only): final-step joint posterior, one col per N == #
    if n_plot_dims >= 2:
        ana_f = np.asarray(analytical_samples_arr[-1])
        m_kf_2d = m_kf_f[:2]
        P_kf_2d = P_kf_f[:2, :2]
        ncol = len(sizes)
        fig, axes = plt.subplots(1, ncol, figsize=(4.5 * ncol, 4.5), squeeze=False,
                                 sharex=True, sharey=True)
        for ax, N in zip(axes[0], sizes):
            ens_f = np.asarray(ens_arr_by_N[N][:, -1, 0, :])
            ax.scatter(ana_f[:, 0], ana_f[:, 1], s=4, alpha=0.12, color="tab:red",
                       label="analytical samples")
            ax.scatter(ens_f[:, 0], ens_f[:, 1], s=8, alpha=0.5, color=colors[N],
                       label=f"N={N}")
            mu_p, sig_p = ensemble_mean_cov(jnp.asarray(ens_f))
            mu_p_2d = np.asarray(mu_p)[:2]
            sig_p_2d = np.asarray(sig_p)[:2, :2]
            for n_std, ls in ((1.0, "-"), (2.0, "--")):
                cov_ellipse(ax, m_kf_2d, P_kf_2d, n_std, edgecolor="tab:red", lw=2, ls=ls)
                cov_ellipse(ax, mu_p_2d, sig_p_2d, n_std, edgecolor=colors[N], lw=2, ls=ls)
            ax.set_title(f"N = {N}")
            ax.set_xlabel("state[0]")
            ax.grid(alpha=0.3)
        axes[0, 0].set_ylabel("state[1]")
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"Final-step joint posterior state[0]-state[1] vs analytical "
                     f"Kalman  ({method_name}; solid=1σ, dashed=2σ; state_dim={state_dim})")
        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
