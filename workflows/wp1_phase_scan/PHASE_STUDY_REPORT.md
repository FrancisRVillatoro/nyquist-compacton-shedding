# Phase-resolved Floquet study for the article-strengthened manuscript

## Purpose

The existing 5x5 map varied the mesh spacing `h` while keeping the physical compacton location fixed. Since the right-edge phase

\[
\phi_R = (x_R/h) \bmod 1
\]

then changes with `h`, mesh dependence and edge-phase dependence were potentially confounded. This study separates them directly.

The conventional Courant number is `nu = c dt/h`. Throughout this study we use

\[
\kappa = h/(c\,dt)=\nu^{-1},
\]

the number of timesteps per cell crossing for integer `kappa`.

## Computation

For each prescribed right-edge phase `phi_R`, a fully discrete numerical traveling compacton was recomputed as a fixed point of the one-cell co-moving map. The full tangent map and its adjoint were then formed matrix-free. The dominant right and left Ritz pairs were computed, with residuals reported in the CSV files. For the two principal real-multiplier cases, the sampled analytic compacton mismatch was projected onto the biorthogonally normalized unstable eigendirection to obtain the initial coordinate `q0`.

Sixteen equally spaced phases were used in each of three cases:

- `(h,kappa)=(0.01,5)`, principal real flip branch;
- `(h,kappa)=(0.02,20)`, principal real flip branch;
- `(h,kappa)=(0.01,20)`, representative complex-conjugate branch.

## Main findings

### 1. Floquet instability is essentially phase-independent

- `(0.01,5)`: `rho_F = 1.056446 ... 1.057449`, relative peak-to-peak variation `9.49e-4`.
- `(0.02,20)`: `rho_F = 1.130948 ... 1.130962`, relative peak-to-peak variation `1.27e-5`.
- `(0.01,20)`: `rho_F = 1.093176 ... 1.093186`, relative peak-to-peak variation `9.15e-6`.

Thus the subcell phase does not create or remove the instability in these representative branches.

### 2. The spectral branch type is phase-robust

All 16 phases remain on the real negative branch for both principal cases. All 16 phases remain on the complex-conjugate branch for `(h,kappa)=(0.01,20)`. Hence the real/complex changes seen in the broader `(h,kappa)` survey are not caused by accidental phase changes in these examples.

### 3. Edge localization remains extremely strong

At least 99.918% of the dominant-mode energy lies in the two edge windows for `(0.01,5)`, at least 99.966% for `(0.02,20)`, and at least 99.939% for the complex branch.

### 4. Phase strongly controls the seed, not the growth rate

For the mismatch between the sampled analytic compacton and the numerical traveling compacton,

- `(0.01,5)`: `|q0|` varies by a factor 10.73;
- `(0.02,20)`: `|q0|` varies by a factor 18.02.

Using the previously resolved first-turnover coordinates, the linear latency estimate varies from 0.610 to 1.034 in physical time for `(0.01,5)` and from 0.681 to 1.151 for `(0.02,20)`.

The natural interpretation is therefore:

\[
\text{phase controls seeding and onset latency, whereas }(h,\kappa)
\text{ control the Floquet growth law.}
\]

## Consequences for the paper

1. Replace the nonstandard label `CFL=h/dt` by `kappa=h/(c dt)` and call it the inverse Courant number or timesteps per cell crossing.
2. Add a phase-resolved figure and report that the dominant Floquet growth is phase-robust.
3. State separately that phase strongly modulates the projection of the analytic initial condition onto the unstable mode.
4. Recompute the full 5x5 `(h,kappa)` map at one fixed phase to remove the remaining formal confounding completely.
5. Add a nonlinear latency test at selected high-seed and low-seed phases.

## Numerical quality

Across the phase study, the largest dominant right-eigenpair residual is `1.62e-9`; fixed-point residuals are below `4.15e-10`; and the dominant modes retain greater than 99.9% edge localization in all three scans.
