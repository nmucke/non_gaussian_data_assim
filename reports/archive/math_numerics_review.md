# Mathematics & Numerics Review — non_gaussian_data_assim

Scope: the mathematical content of all DA methods (`da_methods/`), forward models (`forward_models/`), localization/kernels, ensemble generation (perturbations, breeding, spinup), observation handling, and all metric modules. Software-engineering issues live in `reports/code_review.md`; a few items appear in both because they are both.

Things that were checked and found **correct** (so you don't have to re-derive them): the L96 RHS index convention; the KS wavenumber scaling `k = 2πm/L` via `rfftfreq(n, d=L/(2πn))`; the KS linear operator `k² − k⁴` and the conservative form `−½∂ₓ(u²)` of the nonlinearity with 2/3 dealiasing; the coupled-KS per-state operators (including the halved atmospheric biharmonic term, matching the docstring); the SVGD/PFF update structure `φ(xᵢ) = (1/N) Σⱼ [k(xⱼ,xᵢ) s(xⱼ) + ∇_{xⱼ} k(xⱼ,xᵢ)]` (indexing through the double vmap verified); the fair-CRPS sort identity in `ensemble_metrics._crps_pointwise`; the whitened-innovation / chi² construction `z = L⁻¹d` with `S = UUᵀ + R`; the systematic resampler; the analytical Kalman filter with Joseph-lite symmetrization; and the AGMF bridging rule `w ← αw̃ + (1−α)/N` with `α = N_eff/N` (this matches Stordal et al. 2011).

---

## 1. Clear errors and mistakes

### 1.1 Observations are assimilated at the wrong time (highest priority)

`generate_observations` (`observations/observation_utils.py:37`) observes the truth at inner-step index `m·i + 1` (`m = model_integration_steps`), while the assimilation loop applies `observations[i]` to the forecast at inner-step index `m·(i+1)`. The observation used at each analysis is therefore `m − 1` inner steps **stale** — 4 steps (0.04 time units) for Lorenz 63 defaults, 31 steps (~2 time units) for the coupled KS case, 49 for the repo-level default `m=50`.

Consequences:

- Every filter solves the wrong inverse problem: it fits the current forecast to an observation of the *past* truth. For chaotic systems this behaves like an unmodeled representativeness error that grows with `m` and with the leading Lyapunov exponent; RMSE/CRPS comparisons across `m` values are not measuring what they appear to.
- The information-metrics block of `main.py:360` extracts model states at `m·(i+1)` and pairs them with observations taken at `m·i + 1` — X and Y in the mutual-information estimates are not simultaneous, so `I(Z;X)`, `I(Z;Y|X)`, and the derived "assimilation efficiency" are computed between time-shifted variables.
- The chi² innovation diagnostic pairs are also misaligned (see 1.8), so the primary spread-consistency check cannot flag the problem.

Fix: define the convention once — "`observations[i]` is the truth at analysis time `m·(i+1)` plus noise" — and set `obs_idx = m * (i + 1)` (the commented-out line in the function is *also* wrong; it adds a spurious `+1`). `scripts/main_simple.py` already implements the correct convention and documents it; make the library match. Add a unit test with `m > 1` on the linear model, where the misalignment shows up as a deterministic bias.

### 1.2 The Burridge–Knopoff RK4 integrator is not RK4

`forward_models/bk_rsf_1d.py` (both `rk4_bk_1d_step` and `rk4_bk_1d_ensemble`): the stage states accumulate instead of restarting from the initial state. After K1 the state is `x + ½dt·k₁` (correct for K2), but after K2 it becomes `x + ½dt·k₁ + ½dt·k₂` — K3 must be evaluated at `x + ½dt·k₂`, and K4 at `x + dt·k₃`, not at the accumulated states. The final combination `x + dt/6 (k₁ + 2k₂ + 2k₃ + k₄)` is applied to stage slopes that don't satisfy the RK4 order conditions, so the scheme is a nonstandard method of reduced order (≤2), not RK4. With the stiff `log(v+1)` terms in this friction model, this can also change stability behavior. Fix each stage to recompute from `theta_in/u_in/v_in`:

```python
theta_2 = theta_in + 0.5 * dt * k11   # not theta_00 accumulated
```

(While fixing it, port the model to the `BaseForwardModel`/JAX interface — it is the only earthquake-relevant model in a repo whose companion paper is about earthquake forecasting, and it currently can't be used by any filter.)

### 1.3 AGMF weight update is not the AGMF weight update

`gaussian_mixture.gaussian_mixt` (called from `agmf.py:116`) computes, per member i,

`w̃ᵢ ∝ w_prev,i · N(y_pert,0 − H xᵢ^a ; 0, R)`

Three deviations from the adaptive Gaussian mixture filter (Stordal et al. 2011; Hoteit et al. 2008):

1. **Wrong particles**: weights must be computed from the *forecast* (prior) members `xᵢ^f`. Using the already-Kalman-updated posterior members double-counts the observation — the particles have already been pulled toward `y`, so likelihood ratios between them are flattened and degeneracy is systematically underestimated (`N_eff` too high, `α` too high, resampling too rare).
2. **Wrong observation**: `obs_vect[:, 0]` is the *first member's perturbed* observation, not the actual observation `y`. This injects one realization of observation noise into every member's weight — a pure error, since the weight update in a mixture filter uses the unperturbed `y`.
3. **Wrong innovation covariance**: the mixture-component likelihood is `N(y; H xᵢ^f, H Σ Hᵀ + R)` where `Σ` is the (bandwidth-scaled) mixture covariance — not `R` alone. With `R` only, weights are far too peaked whenever forecast spread ≳ obs noise.

Additional problems in the same routine: `1/sqrt((2π)^p det R)` overflows/underflows for realistic observation counts (Lorenz 96: p=25; coupled KS: p≈91) — compute log-weights and normalize with logsumexp (as `particle_filter.py` already does, correctly); the O(N) Python loop re-inverts `R` every iteration.

### 1.4 Covariance inflation: inconsistent convention, and it never inflates the ensemble

- EnKF multiplies the prior covariance by `λ` (`enkf.py:90`); AGMF by `λ²` (`agmf.py:84`); PFF by `λ` (`pff.py:194`). With the shared config key `inflation_factor: 1.2`, EnKF gets 1.2× variance while AGMF gets 1.44× — cross-method comparisons with "the same inflation" are not the same inflation. Pick one convention (λ on standard deviation, λ² on covariance is the common one) and apply it everywhere.
- More fundamentally, in the EnKF the inflated covariance is used **only inside the Kalman gain**; the ensemble anomalies themselves are never inflated. Standard multiplicative inflation is `xᵢ ← x̄ + λ(xᵢ − x̄)` applied to the forecast ensemble; that changes both the gain *and* the posterior spread. As implemented, inflation actually *increases* the gain (pulling members toward observations, reducing posterior spread) without ever adding the spread that inflation exists to add — close to the opposite of the intent. The AGMF has the same issue, and additionally its resampling branch recomputes `cov_posterior` with `λ²` but the ensemble again never sees it.
- Effect: chronic underdispersion that grows with λ, visible as chi² > 1 — but the chi² diagnostic is currently computed from misaligned pairs (1.8), so it won't cleanly show this either.

### 1.5 The localization taper is not positive semi-definite

`localization.distance_based_localization` builds `mask = exp(−d²/r²)` and then hard-truncates it: `mask = where(d ≤ 3r, mask, 0)`. A truncated Gaussian is **not** a valid correlation function, so the Schur (elementwise) product with the sample covariance is not guaranteed PSD — the localized "covariance" can have negative eigenvalues. Downstream this can produce a gain that increases error variance, non-Cholesky-able matrices, and (in the PFF) an indefinite `prior_cov_inv`, i.e. a prior "score" pointing the wrong way in some subspace. The standard fix is the Gaspari–Cohn 5th-order piecewise-rational taper: compactly supported *and* PSD by construction, effectively the same shape. This is a drop-in replacement of ~10 lines.

Two secondary points: (i) `r_influ` enters as `exp(−d²/r²)` — note this makes `r` the e-folding scale, not the Gaspari–Cohn cutoff radius; document whichever definition you keep, since "localization_distance=10" means different things under each. (ii) For `num_states > 1` the same spatial mask is tiled across all state-pair blocks — reasonable, but consider allowing a different (usually broader) radius for cross-state blocks in the coupled KS system, where atmosphere–ocean covariances have different correlation lengths by construction.

### 1.6 PFF kernel bandwidth: dimensionally wrong for multi-state systems, arbitrary otherwise

`pff.py:204`: `distance_weight_matrix = jnp.eye(self.state_dim) * jnp.pi`.

- The kernel operates on full flattened states of size `dofs = num_states · state_dim`, so for the coupled KS case the quadratic form `(x−y)ᵀ A (x−y)` has a shape mismatch and crashes — this is exactly the `KNOWN_FAILURES` entry in `tests/test_main.py`. It should be `eye(dofs)`.
- A fixed bandwidth `π·I` has no scale awareness at all: for L63 (state values ~O(10)) squared distances are O(100), so `k ≈ exp(−π·100) ≈ 0` — all kernel interactions vanish and the flow degenerates to independent gradient ascent of each particle on the posterior; for tightly clustered ensembles the opposite happens. The commented-out line above it, `inv(alpha * prior_cov)`, is the Hu & van Leeuwen (2020) choice and should be restored (with the localized, regularized prior covariance), or use a median heuristic `A = I / med²(pairwise dists)` rescaled by `1/dofs`. The `alpha` constructor argument is currently dead.

### 1.7 PFF prior score with rank-deficient covariance

`prior_cov = jnp.cov(x_s.T)` with `dofs ≫ N` (KS: 512 dofs, 20–50 members) has rank ≤ N−1; `jnp.linalg.inv` of it is numerically explosive. `prior_cov_regularization` exists but defaults to `None`, and the L63 config omits it. Since the Gaussian-prior PFF *requires* an invertible prior covariance, regularization (or a pseudo-inverse / factored form `P⁻¹ ≈ U(UᵀU)⁻²Uᵀ` in ensemble space) should be the default, not an option. Also, localization is applied to `P` and then the full `dofs×dofs` matrix is inverted — O(dofs³) per analysis step.

### 1.8 Innovation diagnostics use the pre-forecast ensemble

`main.py:225`: `HXf = da_model.obs_operator(prior_current)` where `prior_current` is the *previous analysis* (before the model propagation), not the forecast valid at the observation time. The chi² statistic `dᵀS⁻¹d / p` is therefore built from a mean and covariance valid `m` inner steps before the observation. Together with 1.1, both legs of the innovation are time-shifted (by different amounts). After fixing 1.1, take the forecast ensemble at the analysis time — it is already computed inside `_assimilate_data` (`forecast_ensemble[:, -1]`); expose it rather than re-deriving.

### 1.9 Best-guess construction rescales the state to the perturbation scale

`InitialEnsembleGenerator.sample` (`ensemble_generation/initial_ensemble.py:67-70`):

```python
center_raw = true_state + noise
center_raw = center_raw / jnp.std(center_raw, ddof=1) * best_guess_pert_scale
```

`best_guess_pert_scale` is passed `natural_variability_truth` from `main.py`. The comment says "rescale perturbed truth s.t. it has same variability as actual truth", but the code rescales the **whole state** (not the perturbation, not anomalies about the state mean) so that its total std equals the natural-variability scalar. For fields with nonzero mean or with amplitude different from their temporal variability, this teleports the best guess off the attractor by an arbitrary factor; the subsequent spin-up then relaxes it back to *some* attractor state, but the "best guess close to the truth" property is lost in an uncontrolled way. Presumably intended: `center = true_state + noise` where `noise` has std equal to (a fraction of) the natural variability. Related: `natural_variability = jnp.std(rolled, ddof=1)` in `utils/spinup.py:47` reduces over *all* axes — ensemble, time, states, and space — which for the coupled KS system mixes the fast atmosphere and slow ocean scales into a single scalar; compute it per state.

### 1.10 AGMF resampling jitter is rank-one noise

`agmf.py:145-148`: `epsc` is one scalar per member, and `noise = sqrt(diag(P))[:, None] * epsc[None, :]`, so each resampled member is perturbed along the *same* direction (the diagonal-std vector), scaled by a single N(0, 0.1²) draw. Proper mixture resampling draws fresh noise per component, e.g. `N(0, h²P)` (or at minimum elementwise `N(0, h²·diag(P))`); the bandwidth 0.1 is hard-coded. Also two RNG-hygiene bugs in the same block: `rng_key` is consumed by `jax.random.choice` and then split again (key reuse), and the weights are not reset to uniform after resampling (they should be — otherwise the next step's `w_prev` double-counts the observation that was just absorbed by resampling).

### 1.11 EnKF observation perturbations: uncentered (minor, known bias)

The stochastic EnKF (`enkf.py:94-97`) draws perturbations with `multivariate_normal` but does not subtract their ensemble mean. For small N this adds a spurious O(1/√N) shift to the analysis mean and makes the effective perturbation covariance differ from `R`. Centering the perturbations (and optionally rescaling them to have exactly sample covariance R) is a one-line improvement. (This is the classical Burgers/van Leeuwen/Evensen caveat; deterministic square-root filters avoid it entirely — see §3.)

### 1.12 Information metrics: estimator bias makes the headline numbers unreliable

`metrics/information_metrics.py` + `main.py:339-396`: entropies are estimated from histograms with `n_bins=30` per axis, on sample sizes equal to `data_assimilation_steps × num_obs` pairs — but the *joint* entropies use 30² = 900 cells (2-D) and 30³ = 27,000 cells (3-D) . With the configured runs (e.g. L63: 250 steps × 3 obs = 750 samples; kuramoto smoke runs: far fewer), the triple-joint histogram is overwhelmingly empty and `H(X,Y,Z)` is biased low by an amount comparable to the MI values being reported; interaction information `I(X;Y;Z)` (a difference of seven such terms) is dominated by this bias. The negative `data_efficiency` printed in test runs is symptomatic. Recommendations, in increasing order of effort: (a) drastically reduce bins for joint terms (scale as `~N^(1/(2+dim))` or Freedman–Diaconis); (b) apply Miller–Madow or jackknife bias correction; (c) replace histogram MI with a KSG (Kraskov) k-NN estimator, which handles 750 samples in 3-D comfortably. Also document that all entropies are in "normalized log(n_bins)" units — the mixing of normalized and unnormalized quantities (`i_z_y * h_z − i_xyz`) is consistent but very easy to get wrong when editing; consider carrying nats throughout and normalizing only at the end.

### 1.13 Small definite errors

- `time_integrators.get_backward_euler`: Newton tolerance `1e-10` is unreachable in float32, so the `while_loop` almost always runs all 10 iterations; and the dense `jacfwd` Jacobian + `linalg.solve` is O(d³) per step per member — unusable for KS-sized states. Fine as a demo; document as such, or implement Jacobian-free Newton–Krylov if implicit stepping is actually needed.
- `Lorenz63Model` default `beta=2.6666666` — write `8.0/3.0`.
- `forward_models/base.py.__call__` with `is_ensemble=False` and `include_initial_state=not return_model_integration_steps`: the two flags interact so that the non-trajectory path returns `last_step` (include flag irrelevant) — harmless but confusing; simplify.
- `metrics/trajectory_metrics.MAPE`: `|pred−truth| / (|truth| + 1e-6)` on near-zero-mean chaotic fields (KS, L96 anomalies) produces the huge values seen in test output (MAPE ≈ 48). MAPE is not meaningful for these systems; drop it or replace with a normalized RMSE (by climatological std).

---

## 2. Improvements (no new features)

1. **One documented time convention.** After fixing 1.1/1.8, write the convention (what index `observations[i]` refers to; where analyses sit in saved trajectories) in one docstring and reference it from `generate_observations`, `da_rollout`, and `main.py`. Most of the misalignments in this repo are convention drift, not math misunderstanding.
2. **Gaspari–Cohn localization** (replaces the truncated Gaussian, same feature): PSD, compactly supported, standard in the literature your paper will be compared against.
3. **Anomaly-space EnKF algebra.** Both Kalman filters form the full `P ∈ R^{dofs×dofs}` and localize it. Equivalent gain: `K = X'(HX')ᵀ [HX'(HX')ᵀ + (N−1)R]⁻¹` with anomalies `X'` — O(dofs·N·p) instead of O(dofs²·N), and B-localization can be applied as obs-space tapering of `X'(HX')ᵀ`. This matters for the 2×1024 coupled case and any future 2-D model.
4. **Log-domain weights in AGMF** (mirror `particle_filter.py`'s logsumexp implementation), fixing 1.3's overflow at the same time as its statistics.
5. **PFF pseudo-time integration control**: monitor `‖φ‖` (the RHS magnitude) and stop when the flow stagnates instead of a fixed `num_pseudo_time_steps`; the configs currently hand-tune 5,000 steps for L63 vs. 10 for kuramoto, which strongly suggests neither is calibrated. An adaptive criterion also removes `step_size` from the per-case tuning burden.
6. **ETDRK4 for the KS models.** The current ETDRK1 (exponential Euler) is first-order in the nonlinear term; Kassam–Trefethen ETDRK4 with contour-integral coefficient evaluation is the standard for KS and is ~30 lines. At `dt=0.05–0.0625` the first-order splitting error is likely visible in climate statistics used for `natural_variability` and spinup.
7. **Exact coupling in the coupled KS integrator.** The `(O−A)` coupling is linear and diagonal in Fourier space per wavenumber pair — it can be folded into a 2×2 matrix exponential per wavenumber (still ETDRK1 for the Burgers term). Removes a stiffness constraint when `alpha_oa/tau` is increased.
8. **Consistent RNG discipline.** `randsample` (NumPy global RNG) should go; AGMF key reuse (1.10); `_powerlaw_rednoise_periodic_1d` synthesizes a non-Hermitian spectrum and takes `.real` (fine because it renormalizes, but generating Hermitian coefficients directly via `irfft` — as `CoupledKuramotoPseudo1DProfile` already does — is cleaner and half the work).
9. **Spread–skill comparison factor.** When comparing `ensemble_spread` to RMSE of the ensemble mean, the consistent target is `spread ≈ RMSE·√((N+1)/N)`; worth encoding in the plotting/diagnostics so small-N cases aren't misread as underdispersed.
10. **Breeding**: all bred vectors are rescaled independently toward the leading Lyapunov direction, so for long breeding cycles the ensemble collapses onto ±one direction — consider paired ± perturbations (already common practice) or Gram–Schmidt/EOF diversification of the bred set before centering the ensemble (this is arguably a feature; minimally, document the collapse behavior). Also `SelectedStateL2Norm` rescales the *coupled* perturbation by the atmosphere-only norm — intended per the docstring, but note the ocean perturbation amplitude is then unconstrained.
11. **Chi² should gate on N vs p.** `NormalizedInnovations` builds `S` from an N-member anomaly matrix; for `p ≥ N` the ensemble term is rank-deficient (fine, R regularizes it) but the chi² expectation is no longer exactly 1 — a note or a small-sample correction would prevent over-interpretation.

---

## 3. Suggested extensions (methods & mathematics)

Ordered roughly by (value ÷ effort) for a non-Gaussian-DA comparison library:

1. **Deterministic square-root EnKFs: ETKF and LETKF.** The stochastic EnKF's perturbed-obs sampling noise (1.11) is a confound in every comparison at small N. An ETKF is ~40 lines in the existing structure; the LETKF (domain-localized, embarrassingly parallel over grid points) is the natural strong baseline for KS/coupled-KS and removes the need for the PSD covariance tapering entirely (R-localization instead).
2. **Adaptive inflation.** Anderson (2009) spatially-varying Bayesian inflation, or the simpler Desroziers-statistics-based scalar tuning. This removes the hand-tuned `inflation_factor` per case and directly addresses the underdispersion machinery in 1.4.
3. **Local particle filter (Poterjoy 2016) and/or Ensemble Transform Particle Filter (Reich 2013).** The repo's stated theme is non-Gaussian filters, but the only PF is the bootstrap SIR, which the docstring itself concedes degenerates in high dimensions. The ETPF (optimal-transport resampling) and the local PF are the two standard ways to make particle methods competitive on L96/KS-scale problems, and they slot into the existing `_analysis_step` interface.
4. **Tempered/annealed likelihood assimilation.** A shared bridge between your methods: split the likelihood into K tempered steps (`R → R/βₖ`); the EnKF, AGMF, and PFF all improve, and it gives the PFF a principled pseudo-time schedule (its flow is exactly a tempering when discretized).
5. **PFF upgrades from the recent literature**: matrix-valued localized kernels (Hu & van Leeuwen 2020) — the localization structure is already in the repo; higher-order flow discretization; mini-batching of observations for the likelihood score. If ambitions extend further: replace the Gaussian prior score with a score estimated by denoising score matching on the forecast ensemble — a genuinely non-Gaussian prior, and a distinctive contribution.
6. **Gaussian mixture EnKF hybrids.** The AGMF's α-bridging is one point on the EnKF↔PF spectrum; implementing the Stordal-style hybrid properly (after fixing 1.3) plus e.g. the adaptive Gaussian-mixture EnKF of Hoteit/Luo gives a family of bridge filters your paper can compare on equal footing.
7. **Model error / stochastic dynamics in the main pipeline.** Only the analytical harness adds process noise; the twin experiments are perfect-model. Add (a) additive process noise per outer step (already the pattern in `_common.run_da_method`), and (b) an imperfect-model mode (perturbed parameters for the truth vs. the filter model, e.g. forcing 8.0 vs 7.9 in L96). Non-Gaussian filters differentiate themselves most under model error.
8. **Rank-based calibration metrics**: rank histograms / reliability, the energy score and variogram score (multivariate generalizations of CRPS) — cheap given the existing metric framework, and much more informative than MAPE for non-Gaussian posteriors. The analytical harness could validate them against the exact posterior the same way it validates chi².
9. **Fixed-lag ensemble smoothing.** The trajectory storage is already in place; augmenting the state with lagged copies (or implementing EnKS updates on the stored trajectory) is cheap and highly relevant for the geophysical use case.
10. **A 2-D or two-scale testbed.** The two-scale Lorenz 96 (fast/slow) or a barotropic QG channel would complete the hierarchy between L96 and coupled KS, and exercise localization in a setting where its PSD-ness and radius actually bind.
11. **Convergence harness against the analytical posterior.** `scripts/analytical/` is a genuinely nice asset. Extend it to (a) sweep `m > 1` to lock in the timing convention (regression test for 1.1), (b) verify weak convergence rates in N (EnKF bias O(1/N), PF variance O(1/N)), and (c) run the AGMF with the corrected weight update against the exact posterior on a bimodal (mixture-prior) linear problem — the one setting where its non-Gaussianity is provably measurable.
