# WP1 block 2 — fixed-phase map and nonlinear phase-latency validation

## Fixed-phase map

The full 5x5 Floquet survey was recomputed at fixed right-edge phase `phi_R=0.25`, using `kappa=h/(c dt)`.
All 25 sampled numerical compactons remain unstable, with
`1.039215873 <= rho_F <= 1.132794723`.
The branch classification is unchanged at all 25/25 points (14 real flips and 11 complex pairs).
Relative to the original map, whose edge phase varied with h, the maximum absolute change in rho is `8.181e-04` and the maximum relative change is `7.736e-04`; the median absolute change is `1.694e-05`.

The maximum fixed-point residual is `7.851e-09`, the maximum leading-eigenpair residual is `2.627e-09`, and at least `99.380%` of the leading-mode energy lies in the two edge windows.

## Direct nonlinear latency test

For each principal real-multiplier case, the phases with the largest and smallest adjoint projection of the analytic-compacton mismatch were evolved with the original nonlinear scheme.
In the linear regime, `q_n` follows `|q0| rho_F^n`. Exponential fits have R2 between `0.998301` and `0.999978`.

At the common onset threshold `q_on=0.01`:

- `(h,kappa)=(0.01,5)`: observed min-minus-max delay = `43` cells (`0.430` time units), predicted = `42.333` cells (`0.4233`), error = `0.667` cells.
- `(h,kappa)=(0.02,20)`: observed delay = `23` cells (`0.460`), predicted = `23.497` cells (`0.4699`), error = `-0.497` cells.

The same agreement holds across `q_on=0.005, 0.008, 0.010, 0.012`. Subtracting the small constant fixed-point residual changes the q trajectories by at most the values tabulated in `phase_onset_residual_correction_check.csv` and does not change any headline latency.

## Scope of the conclusion

The test validates the causal relation between subcell phase, unstable-mode seed, and the latency of entry into the Floquet-dominated regime. It does not imply that the entire nonlinear shedding time is determined by `q0` and `rho_F` alone. In the kappa=5 case, other components of the full analytic mismatch affect the first nonlinear turnover after q reaches approximately 0.01--0.02; the kappa=20 case remains close to the isolated strong-unstable excursion for longer.
