# PFF Deep Review & SDE-Sampler Plan

Scope: `da_methods/pff.py` and the modules it composes with — `kernels.py`, `jax_utils.py`, `time_integrators.py`, and the observation-operator interface it consumes. Three parts:

1. Issues in the current implementation (correctness → design → efficiency).
2. A refactoring proposal that makes kernels, flow equations (RHS), and prior scores pluggable.
3. Research notes + a concrete plan for an SDE-based sampler in a new file, sharing components with the ODE PFF.

What is **already correct** and worth preserving: the SVGD update structure is right. `get_pairwise_interaction_fn` produces `result[i, j] = pair_fn(x[j], x[i])`, the divergence kernel is `∇_{x_j} k(x_j, x_i)`, and `rhs_fn` computes `φ(xᵢ) = (1/N) Σⱼ [k(xⱼ,xᵢ) s(xⱼ) + ∇_{xⱼ} k(xⱼ,xᵢ)]` — I verified the indexing through both vmaps. The separation into score functions / kernels / pairwise machinery / steppers is also the right skeleton; the problem is that each seam is only half-open.

---

## 1. Issues in the current implementation

### 1.1 Correctness

**(a) Kernel bandwidth matrix: wrong dimension, magic constant, dead parameter.**
`pff.py:204`:

```python
distance_weight_matrix = jnp.eye(self.state_dim) * jnp.pi
```

- The kernel operates on flattened states of dimension `dofs = num_states * state_dim`, so this crashes for any multi-state model (`coupled_kuramoto` — the `KNOWN_FAILURES` entry in `tests/test_main.py`). Must be `dofs`.
- `π` as a bandwidth is arbitrary and scale-blind. For Lorenz 63 (state magnitude ~O(10)), typical squared distances are O(10²–10³), so `k = exp(−π·O(100)) ≈ 0`: all particle interactions and repulsion terms vanish, and the "flow" degenerates into N independent gradient ascents on the posterior — the ensemble collapses toward the MAP point. This is probably why the L63 config needs 5000 pseudo-steps.
- The principled choice is one line above, commented out: `inv(alpha * prior_cov)` — this is the choice in Hu & van Leeuwen (2021, QJRMS). The `alpha` constructor argument (and `DEFAULT_ALPHA`, `DEFAULT_B_D`) exist solely to feed that dead line.

**(b) The nonlinear-observation likelihood score is shape-broken.**
`get_likelihood_score_fn_with_non_linear_obs_operator` computes

```python
obs_gradient @ obs_cov_inv @ (obs_operator._obs_operator(x_s) - obs_vect)
```

`NonlinearObservationOperator.grad_obs_operator` uses `jax.grad`, which (i) requires scalar output and (ii) returns shape `[dofs]`. Then `[dofs] @ [p, p]` only contracts if `dofs == p` — for the sine operator (`dofs=2, p=1`) this is a hard shape error. The correct expression is `J(x)ᵀ R⁻¹ (h(x) − y)` with the Jacobian `J = jax.jacobian(h)` of shape `[p, dofs]`. The linear branch works only because `LinearObservationOperator.grad_obs_operator` happens to return `Hᵀ` (`[dofs, p]`) — i.e. the two operators' "gradient" methods return different objects, and the score code is written for the linear one. Fix the interface: `grad_obs_operator` should uniformly return the Jacobian `[p, dofs]`, and the score should be `jac.T @ obs_cov_inv @ residual`. Also stop calling the private `_obs_operator` from outside the class.

**(c) Dead scalar-covariance guard after the inverse.**
`pff.py:198-201`: `prior_cov_inv = jnp.linalg.inv(prior_cov)` runs *before* `if len(prior_cov.shape) == 0: prior_cov = prior_cov.reshape(1, 1)`. A 0-d covariance fails at `inv` first, and even if it didn't, `prior_cov_inv` is never recomputed. Move the guard up or delete it.

**(d) Rank-deficient prior covariance is the default.**
`jnp.cov` of an `[N, dofs]` ensemble with `dofs ≫ N` (KS: 512 dofs, 20–50 members) has rank ≤ N−1; `jnp.linalg.inv` of it is meaningless noise amplification. `prior_cov_regularization` exists but defaults to `None` (and the L63 config omits it). Since the Gaussian prior score *requires* an invertible covariance, regularization must be the default, not an option — or the score should be computed in ensemble space (see §2.1). Related: the localization mask applied first is a truncated Gaussian, which is not PSD (see the main math report §1.5), so the "covariance" being inverted can be indefinite — the prior score then *pushes particles away* from the prior mean in the negative directions.

**(e) `return_pff_trajectory=True` breaks the base-class contract.**
`_analysis_step` then returns a 4-D `[N, T_pseudo, num_states, state_dim]` array, which `_assimilate_data` will happily pass on or fail to concatenate. Fine as a debug tool, but it should be a separate method (`flow_trajectory(...)`) rather than a mode of the analysis step.

**(f) Small stuff.** The comment on `pff.py:187` says `[dofs, ensemble]` for an `[ensemble, dofs]` array. Type annotations on all the factory functions claim `-> np.ndarray` while returning callables. The docstring of `get_rhs_fn` says `kernel_matrix_fn` maps to `[ensemble, ensemble, 1]`, but the scalar kernel yields `[ensemble, ensemble]` and `rhs_fn` adds the `None` axis itself — one of the two should change. `get_pairwise_kernel_scalar_fn` in `kernels.py` indexes `K[0, 0]` on a 0-d array and would crash if ever called (it never is).

### 1.2 Flexibility blockers (given your stated goals)

**(a) Hand-derived kernel gradients are the main obstacle to "different kernels".**
`kernels.py` implements the Gaussian kernel *and* its analytic gradient as separate, string-dispatched factories (`get_kernel_fn`, `get_kernel_matrix_fn`, `get_divergence_kernel_fn` — three parallel `if kernel_type == "scalar"` ladders). Every new kernel currently requires deriving and hand-coding its gradient and touching three dispatch sites. In JAX this is unnecessary: given `k(x, y)`, `jax.grad(k, argnums=0)` *is* the divergence-kernel term. A kernel should be one object with a `value` and an auto-derived `grad_x` (analytic override optional as a performance escape hatch). This single change makes IMQ, Matérn, rational-quadratic, or sum-of-kernels a two-line addition each.

**(b) The RHS is hard-wired to the KL/Stein flow.**
`get_rhs_fn` bakes in the SVGD form `k·s + ∇k`. Different flow derivations change this structurally, not just parametrically:

- *KL, Stein geometry* (current): `φ(xᵢ) = E_j[k s + ∇k]` — the only flow where the dependence on the unknown particle density `q` integrates away by parts. This is why it's the default everywhere.
- *Other f-divergences (Hellinger, χ², α-divergences)*: the Wasserstein gradient-flow velocity is `v = −∇ f′(q/π)`, which **requires an estimate of the current density `q`** (or the ratio `q/π`) — via KDE on the particles or a density-ratio estimator. The generalized-Wasserstein-gradient framework (GWG, NeurIPS 2023) formalizes exactly this family. So the RHS abstraction must allow flows that consume a density estimate, not just a score.
- *Fisher–Rao / birth–death flows*: change particle *weights* (or kill/duplicate particles) rather than positions; a hybrid Wasserstein–Fisher–Rao flow returns `(velocity, dlog_weight)`. The interface should anticipate a weight channel even if the first implementation keeps weights uniform.
- *Preconditioned/matrix-kernel flows* (Hu & van Leeuwen 2021; localized mapping PF, NPG 2026): kernel value becomes a `[dofs, dofs]`-matrix (in practice diagonal or low-rank), so the einsum in the RHS changes shape.

Conclusion: `get_rhs_fn` should become a family of `FlowField` implementations behind one interface (§2.2), not one function with flags.

**(c) The prior score is hard-wired Gaussian.**
`get_prior_score_fn` assumes `−P⁻¹(x − μ)`. For a *non-Gaussian* DA library this is the most interesting seam: mixture-of-Gaussians score (connects to the AGMF), KDE score on the forecast ensemble, or a learned score. It's already a standalone function — it just needs to be a constructor argument instead of being built inline in `_analysis_step`.

**(d) No tempering hook.** Annealed SVGD and all homotopy/SDE variants need a likelihood weight `β(s)` ramping 0→1 in pseudo-time. Currently the score functions close over fixed arrays and the stepper has no notion of pseudo-time. Making the RHS accept `(x, s)` (or making score factories accept `β`) unlocks annealed SVGD, homotopy flows, and the tempered SDE samplers below with one mechanism.

### 1.3 Efficiency

- **Per-call recompilation.** `_analysis_step` calls `jax.jit` on freshly created closures (`posterior_score_vmap`, both kernel functions, `rollout_fn`) on *every* analysis step, and the jit cache is keyed by function identity — so every DA cycle recompiles the entire flow. Since `obs_vect`, `prior_mean`, `prior_cov` change per step, the right structure is one jitted function of those arrays, built once in `__init__` (or `functools.cache`d), with the per-step arrays passed as arguments. The `compile_rhs = True` local flag is dead weight.
- **`inv` everywhere.** `jnp.linalg.inv(prior_cov)` (O(dofs³) per analysis step, numerically worse than a factorization) and `jnp.linalg.inv(self.R)` (recomputed per step although `R` is fixed — hoist to `__init__`, or store a Cholesky factor and use triangular solves inside the score).
- **O(N²·dofs) divergence tensor.** `divergence_kernel_fn` materializes `[N, N, dofs]`. Fine at current sizes (50²·512 ≈ 5 MB), but for the Gaussian kernel the whole update reduces to matrix algebra: `Σⱼ ∇ⱼk(xⱼ,xᵢ) = A (K 1 ∘ xᵢ − K X)` — a `[N,N] @ [N,dofs]` matmul. Worth having as the analytic fast path on the Gaussian kernel while the generic vmap path serves arbitrary kernels.
- **Fixed pseudo-time budget.** `num_pseudo_time_steps` is hand-tuned per case (5000 for L63, 10 for kuramoto, 100 for coupled) with no convergence measure. Add a monitor: run `lax.while_loop` until `mean‖φ‖` drops below a tolerance or a step cap; also expose an AdaGrad-style per-particle step scaling (standard for SVGD, removes most of the `step_size` tuning).

---

## 2. Proposed architecture

Create a subpackage that both the ODE filter and the SDE filter (Part 3) import. Suggested layout:

```
src/non_gaussian_data_assim/flows/
    __init__.py
    scores.py        # prior / likelihood / posterior score models, tempering
    kernels.py       # Kernel class + bandwidth policies (absorbs top-level kernels.py)
    fields.py        # FlowField implementations (deterministic RHS terms)
    integrators.py   # convergence monitor + (Part 3) stochastic steppers
    gaussian_prior.py# ensemble -> (mean, cov, factor) preamble shared by filters
```

### 2.1 `scores.py`

```python
class ScoreModel(Protocol):
    def __call__(self, x: Array, beta: float = 1.0) -> Array: ...   # [dofs] -> [dofs]

@dataclass
class GaussianPriorScore:            # -P^{-1}(x - mu); holds a Cholesky factor, not an inverse
    mean: Array
    chol: Array                      # from the regularized, localized covariance

class EnsemblePriorScore:            # ensemble-space form: avoids dofs×dofs inverse entirely,
    ...                              # score = -X'(X'^T X')^{-2} X'^T (x - mu) * (N-1)  (+ ridge)

@dataclass
class GaussianLikelihoodScore:       # -J(x)^T R^{-1} (h(x) - y); J via jax.jacobian for the
    obs_operator: ...                # nonlinear case, H^T for the linear case (fixes issue 1.1b)
    R_chol: Array
    obs: Array

def posterior_score(prior: ScoreModel, lik: ScoreModel) -> ScoreModel:
    # (x, beta) -> prior(x) + beta * lik(x)      <- the tempering hook
```

`gaussian_prior.py` centralizes today's preamble (reshape → localize → inflate → regularize → factorize) as one function `build_gaussian_prior(ensemble, *, localization, inflation, regularization) -> GaussianPrior(mean, cov, chol)`, shared by PFF, the SDE sampler, and — later — EnKF/AGMF.

### 2.2 `kernels.py`

```python
@dataclass
class Kernel:
    def value(self, x, y) -> Array: ...          # scalar (or [dofs] diag for matrix kernels)
    def grad_x(self, x, y) -> Array:             # default: jax.grad(self.value, argnums=0)
        ...

class GaussianKernel(Kernel):        # value = exp(-0.5 (x-y)^T A (x-y)); analytic grad override
class IMQKernel(Kernel):             # (c + ||x-y||_A^2)^{-1/2} — heavier tails, better for outliers
class DiagonalMatrixKernel(Kernel):  # per-dof bandwidth => Hu & van Leeuwen-style localization
```

Bandwidth policies as small callables `(prior: GaussianPrior, particles) -> A`:
`FixedBandwidth(A)`, `MedianHeuristic()` (A = I / median²(pairwise dist) · 1/dofs), `PriorCovBandwidth(alpha)` (restores the commented-out `inv(alpha·prior_cov)` line — the Hu & van Leeuwen choice), recomputed either once per analysis or per pseudo-time step (make that a flag; adaptive bandwidth formally breaks the gradient-flow structure but works well in practice — document it).

The existing top-level `kernels.py` and `jax_utils.get_pairwise_interaction_fn` fold into this module; the pairwise vmap helper stays as-is (it's good).

### 2.3 `fields.py` — the pluggable RHS

```python
class FlowField(Protocol):
    def velocity(self, particles: Array, score: ScoreModel, s: float) -> Array:
        """[N, dofs] -> [N, dofs]; s = pseudo-time in [0, 1] for tempered flows."""

class SteinKLFlow(FlowField):        # current behavior: (1/N) Σ_j [k s_j + ∇_j k]
    kernel: Kernel

class AnnealedSteinFlow(FlowField):  # SteinKLFlow with beta(s) schedule passed into the score

class GWGFlow(FlowField):            # generalized Wasserstein gradient flows for f-divergences
    kernel: Kernel                   # (Hellinger, chi^2, alpha): needs a density/ratio estimate,
    density: DensityEstimator        # so it takes a KDE component — this is where "Hellinger
    f: ConvexFunction                # instead of KL" plugs in (see Cheng et al., NeurIPS 2023)
```

Design decisions worth making explicit:

- `velocity` receives the *whole particle set* (interactions are the point), the score model, and pseudo-time. Everything else (kernel, density estimator, divergence choice) is constructor state. This keeps the integrator ignorant of the flow type.
- Reserve a richer return type for later: `velocity_and_dlogw(particles, ...) -> (v, dlogw)` with a default `dlogw = 0`, so Fisher–Rao / birth–death terms can be added without breaking the interface. The filter applies weight changes by resampling (machinery already exists in `particle_filter.py`).
- `ParticleFlowFilter.__init__` then takes `flow: FlowField`, `prior_score_builder`, and integrator settings; `kernel_type: str` remains as a convenience that builds the default `SteinKLFlow(GaussianKernel(PriorCovBandwidth()))`. Hydra composes this naturally with nested `_target_` blocks, matching how the rest of the repo already instantiates components.

### 2.4 `integrators.py`

Reuse `time_integrators.get_stepper` for the deterministic flow. Add:

- `ConvergenceMonitor(tol, max_steps)` → `lax.while_loop` wrapper around the stepper, returning `(x_final, n_steps, final_residual)`; expose `n_steps`/residual as a diagnostic so per-case pseudo-time tuning disappears.
- Optional AdaGrad step scaling (per-particle running average of `‖φ‖²`), the de-facto standard in SVGD implementations.
- (Part 3) stochastic steppers — see below.

### 2.5 Migration order (each step leaves the repo green)

1. Fix the outright bugs in place: bandwidth `eye(dofs)`, Jacobian-based nonlinear likelihood score, reshape-before-inv, hoist `R` inverse/factor, jit-once-in-`__init__`. Add a unit test: PFF on the 2-D linear-Gaussian case vs. the analytical KF posterior (harness exists in `scripts/analytical/_common.py`), plus a shape test with `num_states=2` (kills the two `KNOWN_FAILURES`).
2. Introduce `flows/` with `Kernel` (auto-grad) + `GaussianPrior` + score models; rewrite `pff.py` to compose them, keeping constructor args backward-compatible. Delete the string-dispatch triplets in the old `kernels.py`.
3. Add `FlowField` with `SteinKLFlow` as the only implementation; move `get_rhs_fn` logic into it. Add the convergence monitor.
4. Add `MedianHeuristic`/`PriorCovBandwidth`, `IMQKernel`, `AnnealedSteinFlow` — each is now ~20 lines and independently testable.
5. (Stretch) `GWGFlow` + KDE density estimator for Hellinger/f-divergence flows.

---

## 3. SDE-based sampler: research notes and plan

### 3.1 What the literature offers (quick survey)

Four families are relevant, in increasing order of specialization to this repo:

1. **Overdamped Langevin / ULA.** `dX = ∇log π(X) ds + √2 dW` has π as its invariant measure; Euler–Maruyama discretization (ULA) has O(Δs) bias, removable with a Metropolis correction (MALA). Uses *exactly* the posterior score the PFF already builds — the score models of §2.1 are shared verbatim. Unlike the PFF, plain Langevin needs burn-in toward stationarity rather than transporting prior samples in finite time.

2. **Interacting / preconditioned Langevin: EKS and ALDI.** The Ensemble Kalman Sampler ([Garbuno-Iñigo, Hoffmann, Li & Stuart, SIAM JADS 2020](https://epubs.siam.org/doi/10.1137/19M1251655)) preconditions the Langevin drift with the ensemble covariance `C(X)`; ALDI ([Garbuno-Iñigo, Nüsken & Reich, 2020](https://arxiv.org/pdf/1912.02859)) adds the finite-N correction term and achieves *affine invariance*:
   `dXᵢ = C(X) ∇log π(Xᵢ) ds + (d+1)/N (Xᵢ − x̄) ds + √2 C(X)^{1/2} dWᵢ`,
   with the key trick that `C^{1/2}` never needs forming: `C^{1/2} dW = (1/√N) X′ dξᵢ` with anomalies `X′ ∈ R^{dofs×N}` and `ξᵢ ∈ R^N` — no matrix square roots, works in high dimension, and there is a **gradient-free variant** that replaces `C∇log π` with EnKF-style cross-covariances (a natural bridge to this repo's EnKF).

3. **Stochastic Particle Flow Filter (SPFF).** Van Leeuwen's group adds Gaussian noise to the SVGD dynamics with an amplitude/covariance derived so the interacting system targets the posterior exactly; the resulting filter is **unbiased in ensemble spread at any ensemble size** ([EGU 2024 abstract](https://ui.adsabs.harvard.edu/abs/2024EGUGA..2614208Y/abstract), [AMS 2023](https://ui.adsabs.harvard.edu/abs/2023AMS...10321356V/abstract)) — directly addressing the deterministic PFF's known low-spread bias at small N. This is the most on-theme variant for this repo (same authorship lineage as the implemented PFF; see also [Hu & van Leeuwen 2021](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.4028) and the [localized mapping particle filter, NPG 2026](https://npg.copernicus.org/articles/33/33/2026/)).

4. **Homotopy / tempered stochastic flows.** The Daum–Huang tradition derives particle motion from a log-homotopy `π_s ∝ prior · likelihood^{β(s)}`, `β: 0→1`, in the Fokker–Planck equation — yielding either an ODE or an SDE ([parameterized family of stochastic particle flow filters](https://arxiv.org/pdf/2103.09676); [Gaussian particle flow importance sampling](https://arxiv.org/pdf/1406.3183)). Practically: run any Langevin scheme above with a tempered score `s(x, β(s))` and you get finite-pseudo-time prior→posterior transport like the PFF, but stochastic — no burn-in problem, and the ODE/SDE pair share the tempering schedule. Related background: [particle filters for high-dimensional geoscience applications (review)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3551), [Fisher–Rao gradient flows for derivative-free inference](https://arxiv.org/pdf/2406.17263).

Design implication: **the drift of every one of these SDEs is built from the same three ingredients the ODE PFF already has** — a posterior score (possibly tempered), optionally an ensemble covariance/preconditioner, optionally a kernel. Only the noise term and the time-stepping are new. That is exactly the sharing boundary.

### 3.2 Plan for the new file

**New file: `src/non_gaussian_data_assim/da_methods/sde_pff.py`** (working name `StochasticParticleFlowFilter`). Everything listed under "shared" lives in `flows/` per §2 and is imported by both filters.

**Shared components (used by both `pff.py` and `sde_pff.py`):**

| Component | Module | Used by ODE | Used by SDE |
|---|---|---|---|
| `build_gaussian_prior` (reshape/localize/inflate/regularize/factorize) | `flows/gaussian_prior.py` | ✓ | ✓ |
| `GaussianPriorScore`, `GaussianLikelihoodScore`, `posterior_score(β)` | `flows/scores.py` | ✓ | ✓ (drift) |
| `Kernel` + bandwidth policies | `flows/kernels.py` | ✓ | SPFF scheme only |
| `SteinKLFlow.velocity` | `flows/fields.py` | ✓ (the RHS) | SPFF drift term |
| `ConvergenceMonitor`, step scaling | `flows/integrators.py` | ✓ | ✓ (residual diagnostic) |
| Tempering schedules `β(s)` (linear, cosine, adaptive) | `flows/scores.py` | annealed SVGD | tempered Langevin |

**New components (SDE-only):**

1. `flows/integrators.py` additions:
   ```python
   def get_euler_maruyama(dt, drift_fn, noise_fn):
       # stepper(x, key, s) = x + dt * drift_fn(x, s) + sqrt(dt) * noise_fn(x, key)
   def stochastic_rollout(stepper, num_steps):
       # lax.scan over pre-split keys, carrying (x, s); mirrors time_integrators.rollout
   ```
   Note the deterministic steppers in `time_integrators.py` take `x` only; the stochastic ones need `(x, key, s)`. Keep them in `flows/integrators.py` rather than widening the old signatures — the forward models keep using the old module untouched.

2. `sde_pff.py`: `class StochasticParticleFlowFilter(BaseDataAssimilationMethod)` with a `scheme` option, all sharing `_analysis_step` scaffolding with today's PFF (reshape, `build_gaussian_prior`, score construction, integrate, reshape back). `rng_key` is already threaded through `_analysis_step` by the base class, so no interface change is needed. Schemes, in implementation order:
   - **`ula`** (baseline, ~15 lines): drift `= posterior_score(x, β(s))`, noise `= √2 ξ`. Validates the plumbing.
   - **`aldi`**: drift `= C(X) s(xᵢ, β) + (d+1)/N (xᵢ − x̄)`, noise `= √(2/N) X′ ξᵢ`. Affine invariant, no `dofs×dofs` factorization (anomaly trick), strongest theoretical package for the ill-conditioned high-dim cases. Optional `gradient_free=True` flag replacing `C∇log lik` with ensemble cross-covariances (EnKF-like drift).
   - **`stochastic_svgd`** (the SPFF): deterministic drift = `SteinKLFlow.velocity` (shared object), plus the noise term with the theory-prescribed covariance from the van Leeuwen SPFF papers (to be transcribed from the paper when implementing; structurally it is kernel-smoothed noise, so it reuses the pairwise-kernel machinery).
   - Every scheme accepts `tempering: β(s)` — with `β` ramping 0→1 over the pseudo-time window, the sampler transports prior particles to the posterior in finite time (homotopy mode); with `β ≡ 1` it is a stationary sampler needing burn-in. Default: linear ramp + a short `β=1` tail.
3. `configs/da_method/sde_pff.yaml` mirroring `pff.yaml` (`scheme`, `num_pseudo_time_steps`, `step_size`, `tempering`, shared localization/inflation/regularization keys), plus `da_method_overrides.sde_pff` entries in the case files.
4. Optional phase 2: MALA accept/reject per particle (removes ULA discretization bias; needs the posterior log-density, which the Gaussian-prior + Gaussian-likelihood models can also provide — add `log_density` alongside `score` in `ScoreModel` when this lands).

**Validation plan** (all reuse existing infrastructure):

- *Exactness*: linear-Gaussian 2-D/7-D configs vs. the analytical Kalman posterior (`scripts/analytical/`) — ULA/ALDI posterior mean, covariance, Gaussian-KL within tolerance. Run all schemes through `main_analytical_methods.py` alongside EnKF/PFF.
- *Stationarity*: initialize particles as exact posterior samples, run the SDE with `β ≡ 1`; ensemble statistics must not drift (catches wrong noise scaling — the classic √2 / √dt mistakes).
- *The SPFF selling point*: small-N spread-bias experiment — deterministic PFF vs. `stochastic_svgd` at N ∈ {5, 10, 20} on L63; the stochastic version's spread/chi² should stay unbiased where the deterministic one collapses. This doubles as the headline figure for adding the method.
- *Smoke*: add `sde_pff` to the `DA_METHODS` list in `tests/test_main.py`.

**Suggested sequencing**: do §2.5 steps 1–3 first (bug fixes + `flows/` extraction) — the SDE file then costs roughly a day of work for `ula` + `aldi` + tests, because its drift is entirely reused code; `stochastic_svgd` follows once the SPFF noise term is transcribed from the paper.

### Sources

- [Interacting Langevin Diffusions: Gradient Structure and Ensemble Kalman Sampler (Garbuno-Iñigo, Hoffmann, Li, Stuart)](https://epubs.siam.org/doi/10.1137/19M1251655)
- [Affine Invariant Interacting Langevin Dynamics for Bayesian Inference (ALDI; Garbuno-Iñigo, Nüsken, Reich)](https://arxiv.org/pdf/1912.02859)
- [Unbiased fully nonlinear data assimilation: the Stochastic Particle Flow Filter (EGU 2024)](https://ui.adsabs.harvard.edu/abs/2024EGUGA..2614208Y/abstract)
- [A Stochastic Particle Flow Filter for Unbiased Nonlinear High-Dimensional Data Assimilation (AMS 2023)](https://ui.adsabs.harvard.edu/abs/2023AMS...10321356V/abstract)
- [A particle flow filter for high-dimensional system applications (Hu & van Leeuwen, QJRMS 2021)](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/qj.4028)
- [Localization in the mapping particle filter (NPG 2026)](https://npg.copernicus.org/articles/33/33/2026/)
- [A New Parameterized Family of Stochastic Particle Flow Filters](https://arxiv.org/pdf/2103.09676)
- [Approximations of the Optimal Importance Density using Gaussian Particle Flow Importance Sampling](https://arxiv.org/pdf/1406.3183)
- [Particle-based Variational Inference with Generalized Wasserstein Gradient Flow (NeurIPS 2023)](https://arxiv.org/abs/2310.16516)
- [Annealed Stein Variational Gradient Descent](https://arxiv.org/pdf/2101.09815)
- [Particle filters for high-dimensional geoscience applications: A review (van Leeuwen et al., QJRMS 2019)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3551)
- [Efficient, Multimodal, and Derivative-Free Bayesian Inference With Fisher–Rao Gradient Flows](https://arxiv.org/pdf/2406.17263)
- [Sampling via Gradient Flows in the Space of Probability Measures](https://arxiv.org/pdf/2310.03597)
