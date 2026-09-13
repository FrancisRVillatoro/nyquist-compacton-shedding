# WP5 — Verification, uncertainty, conditioning, and numerical robustness

## Executive conclusion

WP5 audits the numerical claims used in WP1--WP4 at the level expected for a the associated article submission. The main conclusions are:

1. The Floquet eigenvalues are insensitive to Ritz seed and subspace dimension at the `1e-10--1e-12` level in the two principal midpoint cases.
2. Domain-size effects are `O(1e-5)` or smaller in the tested range and are less than `2e-4` of the instability margin `rho_F-1`.
3. Even deliberately loose nonlinear-solver tolerances move `rho_F` by at most `1.9e-4`, less than `0.34%` of the instability margin. Tight tolerances reduce this variation by roughly two orders of magnitude.
4. Tangent/adjoint implementations satisfy adjoint identities at `1e-14--1e-15`; directional finite differences reach `3e-5--1e-4`, consistent with the fact that `|u|u` is `C^1` but not `C^2` at zero.
5. The nonlinear Nyquist profile coefficients `V_h` are converged to floating-point precision under Fourier-domain and resolution variation; profile-equation residuals are `2.5e-13--3.0e-13`.
6. Packet center, amplitude, and profile fits are extraordinarily insensitive to fitting range and reasonable local fit windows. The deterministic trajectory fit error is much smaller than the residual model/calibration error in the velocity law.
7. After correcting the very small amplitude mismatch between natural and independently prepared WP4 waves, their speeds agree to `1.2e-7--2.6e-7` relative, substantially better than the raw `2.6e-5--8.8e-5` comparison.
8. The critical hyperviscosity roots are robust under local fit-window/polynomial changes; the previously quoted half-grid uncertainty `±0.00125` is conservative.
9. The leading Floquet modes are strongly non-normal: representative eigenvalue condition numbers are about 254 (midpoint, `h=.01,kappa=5`), 45 (midpoint, `h=.02,kappa=20`) and 94 (GL4, `h=.01,kappa=5`). This is a real feature of the dynamics, not numerical uncertainty, and justifies reporting both modal growth and transient amplification.

No audit result challenges the edge--Floquet/Nyquist-shedding mechanism. Several results strengthen the quantitative claims and refine how uncertainties should be reported.

---

## 1. What “uncertainty” means in this paper

The simulations are deterministic. We therefore distinguish three notions rather than reporting statistical confidence intervals as though the data were noisy experiments:

- **solver uncertainty:** variation with nonlinear tolerance, Ritz subspace, seed, domain, and numerical resolution;
- **post-processing uncertainty:** variation with fit window, fit interval, snapshot resampling, and regression;
- **model discrepancy:** the remaining difference between a converged measured quantity and an asymptotic/calibrated prediction.

Bootstrap intervals below quantify sensitivity to snapshot selection and post-processing; they are not physical sampling uncertainties.

---

## 2. Floquet eigensolver reproducibility

For the two principal midpoint numerical compactons, the leading multiplier was recomputed for block dimensions `nb=6,8,10` and four independent random seeds.

- `h=.01, kappa=5`: peak-to-peak variation in `rho_F` = `7.77e-11`.
- `h=.02, kappa=20`: peak-to-peak variation = `3.99e-12`.

The largest leading-eigenpair residuals were `1.51e-10` and `1.38e-11`, respectively. Thus randomness in the initial Krylov/Ritz subspace is irrelevant on the scale of the instability margins (`rho_F-1 ≈ 5.7e-2` and `1.31e-1`).

See `ritz_seed_subspace_audit.csv`.

---

## 3. Nonlinear-solver tolerance audit

A representative fixed-phase calculation was repeated with stage Newton tolerances from `1e-10` to `2e-13` and outer fixed-point targets from `1e-8` to the `1e-10` numerical floor.

Including the deliberately loose runs, the peak-to-peak changes in `rho_F` are

- `1.87e-4` for `h=.01,kappa=5`, only `3.3e-3` of the instability margin;
- `3.61e-5` for `h=.02,kappa=20`, only `2.8e-4` of the instability margin.

At the tighter tolerances, the `h=.01,kappa=5` values differ by only `1.45e-6`. The audit confirms that the sign of `rho_F-1` is separated from the nonlinear-solver floor by many orders of magnitude.

See `solver_tolerance_audit.csv`.

---

## 4. Domain-size audit

At fixed forward-edge phase `phi_R=1/4`, the complete traveling-wave/Floquet calculation was repeated for periodic boxes between `L=14` and `L=20`.

- `h=.01,kappa=5`: total spread in `rho_F` = `6.34e-6`.
- `h=.02,kappa=20`: total spread = `2.51e-5`; already between `L=16` and `L=20` the difference is only about `6e-9`.

Even the worst spread is below `2e-4` of the relevant instability margin. The `L=14` case leaves only a small zero-background buffer around the compacton and is intentionally conservative.

See `domain_audit_h001.csv` and `domain_audit_h002.csv`.

---

## 5. Tangent and adjoint verification

WP3 already verified the midpoint, GL4, and GL6 tangent maps by directional finite differences and adjoint identities. WP5 repeated the check for two independent spatial stencils (de Frutos and Padé-6) under GL4.

- minimum directional-derivative errors: `2.86e-5` and `4.02e-5`;
- adjoint identity errors: `4.16e-14` and `4.13e-15`.

The directional-difference error does not converge indefinitely as `delta^2` because `Phi(u)=|u|u` is only `C^1` at background nodes where `U_*=0`. The discrete analytic tangent remains validated to far better accuracy than needed to distinguish `rho_F>1`.

See `spatial_tangent_adjoint_summary.csv` and the WP3 tangent verification archive.

---

## 6. Nyquist-profile solver convergence

All four stencil-specific profile equations were recomputed at `h=.02` while varying:

- Fourier points: 8192, 16384, 32768;
- envelope-domain half-length: 60, 90, 120 cells.

The velocity coefficients are unchanged to floating-point precision:

- Ismail: peak-to-peak `V_h` variation `4.4e-16`;
- de Frutos: `7.1e-15`;
- Padé-6: `1.4e-14`;
- Padé-8: `7.1e-15`.

The profile-equation residuals stay below `3.0e-13`. The reported FWHM has the expected grid-quantization dependence and should not be quoted to more than about two decimal places unless interpolated.

See `profile_solver_convergence.csv` and `profile_solver_convergence_summary.csv`.

---

## 7. Packet-fit and velocity uncertainty

### Archived midpoint packets

The clean de Frutos packets were refitted using half-windows 10, 14, 20, and 30 cells and multiple trimmed time intervals. The extracted amplitudes and center velocities are essentially invariant. The maximum velocity variation under time-range trimming is

- `1.7e-9` for the `h=.01,kappa=5` packet;
- `1.7e-7` for `h=.02,kappa=20`.

Window dependence is smaller still. Bootstrap resampling of snapshots consequently gives extremely narrow post-processing intervals; the limiting uncertainty is the calibrated temporal velocity law, not the trajectory regression.

Using the revised WP3 midpoint law, the out-of-sample relative discrepancies are approximately

- `7.4e-6` for the archived `h=.01,kappa=5` packet;
- `5.2e-7` for `h=.02,kappa=20`.

These small residuals are best reported as deterministic model discrepancies, rather than claimed to vanish within a statistical confidence interval.

### Natural GL4 packets

The two clean natural GL4 packets have relative velocity discrepancies of

- `7.8e-8` and
- `5.7e-8`

against the independently calibrated GL4 velocity law. Snapshot and fit-range effects are below `1e-7` in velocity.

See `packet_uncertainty_summary_with_calibration.csv`, `fit_range_sensitivity_summary.csv`, and the individual bootstrap tables.

---

## 8. Fit-window audit for Ismail and Padé-8 packets

The alternative-stencil natural packets were refitted over several local half-windows.

The median fitted amplitude and velocity are extremely stable. Peak-to-peak velocity variations are

- `1.15e-7` and `3.62e-7` for the two Ismail packets;
- below `4e-10` for the two Padé-8 packets.

For very large windows, the **maximum** residual can rise because the window begins to include neighboring radiation, especially for the dense Ismail train. The median core residual remains essentially unchanged. This supports the use of a local packet window and demonstrates why a global-window norm would be a poor capture diagnostic in a multi-packet radiation field.

See `wp4_window_audit_summary.csv` and `wp5_fit_window_robustness.png`.

---

## 9. Amplitude-matched natural-versus-prepared test

WP4 originally compared natural packets with independently prepared packets whose fitted amplitudes differed from the natural median by `O(1e-5)` in `A/h^2`. Because `v` is nearly linear in amplitude, that tiny mismatch accounted for almost all of the reported `2.6e-5--8.8e-5` relative speed difference.

Rescaling the prepared-wave velocity to the **same fitted amplitude** gives relative differences

- Ismail I: `2.04e-7`;
- Ismail II: `2.59e-7`;
- Padé-8 I: `2.06e-7`;
- Padé-8 II: `1.22e-7`.

This is a stronger result than WP4 originally reported. It confirms that, at fixed amplitude, the natural packet and independently initialized solitary wave follow the same fully discrete propagation branch to sub-ppm relative accuracy.

The remaining `O(1e-7)` discrepancy is deterministic and larger than the formal regression standard error for the very long Padé-8 tracks; it should therefore be retained as a small systematic model/extraction discrepancy, not hidden by statistical error bars.

See `wp4_amplitude_matched_natural_prepared.csv` and `wp5_amplitude_matched_spatial_validation.png`.

---

## 10. Snapshot storage precision

A representative clean Ismail packet was evolved and fitted directly in float64 and then refitted after round-trip conversion of the **same state** through float32, matching the storage format used for the WP4 spacetime archives.

The float32 storage changes

- `A/h^2` by only `1.7e-8` relative;
- the fitted center by `3.0e-9` cells;
- the profile residual by about `1e-8` absolute relative to its `1.2e-4` value.

Thus float32 snapshot storage is adequate for the quoted WP4 packet diagnostics. Production/reproducibility metadata should nevertheless state the storage precision explicitly.

See `float32_storage_audit.csv`.

---

## 11. Hyperviscosity threshold uncertainty

The critical roots from WP2 were refitted with different local point counts and both linear and quadratic models.

For `h=.01,kappa=5`, local fits give a spread of only about `2.2e-5` around `epsilon_c≈0.36866`.

For `h=.02,kappa=20`, fits restricted to the local complex branch give a spread of about `1.4e-4` around `epsilon_c≈0.19098`. Broad fits that cross the branch-transition region are intentionally poor and should not be used to estimate the root.

Therefore the WP2 quoted `±0.00125` half-grid uncertainty is conservative by roughly one to two orders of magnitude.

See `hyperviscosity_threshold_fit_sensitivity.csv` and `wp5_threshold_fit_sensitivity.png`.

---

## 12. Nonnormality and eigenvalue conditioning

Unit-normalized right/left eigenvectors give condition numbers

\[
\kappa_\lambda=\frac{\|r\|\,\|\ell\|}{|\ell^*r|}
\]

of approximately

- 254 for midpoint, `h=.01,kappa=5`;
- 45 for midpoint, `h=.02,kappa=20`;
- 94 for GL4, `h=.01,kappa=5` (complex dominant pair).

The corresponding unit-vector overlaps are only `3.94e-3`, `2.21e-2`, and `1.07e-2`. These values confirm substantial non-normality. In the archived midpoint cases, the previously computed one-cell gain for perturbations localized near the forward edge exceeds `rho_F` by factors about 3.45 and 2.86.

This does not weaken the modal mechanism: `rho_F>1` is robust. It explains why transient amplification, left/right eigenvectors, and careful projection onto the full unstable subspace matter for onset and control.

See `floquet_nonnormality_conditioning.csv` and `wp5_floquet_condition_numbers.png`.

---

## 13. Cost reporting

WP3 timings already use repeated one-cell runs. Depending on implementation, the observed min-to-max timing range is roughly 2--22% of the median. Therefore the paper should report **median wall times** and avoid more than two significant digits in timing-based speedup claims.

With banded linear algebra, the asymptotic work remains `O(N*kappa)` per cell crossing for midpoint and GL4, with a larger stage-block constant for GL4 but much smaller `kappa` needed at fixed accuracy. The WP3 conclusion that high-order Gauss integration can be cheaper at fixed spectral accuracy is unaffected by timing variability.

See `cost_repeatability_summary.csv`.

---

## 14. Recommended uncertainty statements for the article main paper

The main paper should avoid pseudo-statistical notation where the dominant uncertainty is deterministic. Recommended language:

- Floquet multipliers: give the numerical value plus eigenpair residual and a sentence stating the domain/solver envelope.
- Hyperviscosity thresholds: retain the conservative bracket from the continuation mesh, optionally quote the much tighter local-fit sensitivity in the supplement.
- Packet amplitudes/velocities: quote the fitted value and either a compact post-processing envelope or the peak-to-peak variation under fit-window/range changes.
- Velocity-law validation: quote the **observed relative model discrepancy** (`7.4e-6`, `5.2e-7`, `7.8e-8`, ...), not a misleading p-value or stochastic confidence level.
- Cost: quote medians and coarse speedup factors only.

---

## Main-text versus supplement placement

### Main text

A compact verification subsection or paragraph should include:

1. eigenpair residuals and domain/tolerance separation from `rho_F=1`;
2. the velocity-validation figure with uncertainty/model-discrepancy bars;
3. the amplitude-matched natural/prepared spatial-stencil comparison;
4. one sentence on nonnormal condition numbers;
5. a concise table of representative uncertainty envelopes.

### Supplement

Move the full Ritz-seed audit, Newton-tolerance table, domain sweeps, profile-resolution table, fit-window/range tables, bootstrap samples, threshold-fit variants, storage-precision audit, and cost-repeatability table to the supplement/repository.

---

## Final WP5 verdict

The strongest claims of WP1--WP4 survive every verification performed. Numerical uncertainty is far smaller than the effects being interpreted. The most important quantitative refinement from WP5 is the amplitude-matched WP4 comparison, which improves natural/prepared speed agreement to `O(1e-7)` relative. The main conceptual refinement is that the Floquet problem is substantially non-normal, so left/right conditioning and transient gain should be reported explicitly alongside the unstable spectral radius.
