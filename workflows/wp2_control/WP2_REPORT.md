# WP2 — Floquet-guided causal intervention and control

## Scope

WP2 tests causality rather than correlation.  The central question is whether
moving the edge--Floquet spectrum through the unit circle suppresses the
forward Nyquist solitary-wave shedding while leaving the resolved compacton
essentially unchanged.

## 1. Why an orbit-preserving continuation is required

Ordinary fourth-order hyperviscosity adds `epsilon D4 U`, with
`D4 = E^-2 - 4 E^-1 + 6 - 4 E + E^2`.  It dissipates the quadratic invariant,
so a nonzero compacton cannot be an exactly periodic traveling fixed point of
constant amplitude.  A conventional Floquet problem is therefore unavailable.

For the spectral continuation we instead add

`epsilon D4 (W - Wstar_k)`

at midpoint substep `k`, where `Wstar_k` is the midpoint of the undamped
numerical traveling orbit.  This control leaves the orbit fixed exactly and
has precisely the same linear hyperviscous action on perturbations.  The
ordinary total-state scheme is then run separately at the predicted threshold.

## 2. Spectral thresholds

For `h=0.01, kappa=5`, the real negative edge branch crosses the unit circle at

`epsilon_c = 0.368749 +/- 0.001250`.

For `h=0.02, kappa=20`, the leading real mode is stabilized first; a
complex-conjugate edge pair controls the final crossing at

`epsilon_c = 0.190972 +/- 0.001250`.

The local linear regressions have R^2 values 0.999939 and 0.999982.  The
corresponding continuum coefficients `epsilon_c h^4` are
`3.6875e-09` and `3.0556e-08`.  Thus the dimensionless epsilon is
not a measure of core distortion: the actual coefficient multiplying u_xxxx
is extremely small.

The Nyquist damping accumulated over one cell crossing is
`delta_N = 120 epsilon h`, giving `0.4425` and
`0.4583`.  Their closeness is suggestive but is not claimed as a
universal law from only two cases.

## 3. Ordinary hyperviscosity gives the predicted nonlinear suppression

At the critical values, the maximum forward high-pass packet amplitude is
reduced from `0.116563` to
`0.000146373` for `h=.01,kappa=5`, and from
`0.130872` to
`0.000103395` for `h=.02,kappa=20`.
The suppression factors are `796.3` and
`1265.7`.

The earlier value `epsilon=1e-5` is several orders of magnitude below the
spectral threshold and does not suppress the phenomenon.  In the second run it
can even alter the detailed micro-radiation in a nonmonotone way; this is
expected for a nonlinear unstable transient and is not evidence of
stabilization.

At threshold, the maximum relative compacton peak changes are
`7.38e-10` and
`4.40e-09`; mass drift is at
roundoff and the best-fit compacton errors at the final times are
`1.58e-07` and
`1.14e-07`.  Hyperviscosity therefore
removes the grid-scale instability without appreciably damping the resolved
core over the diagnostic interval.

## 4. Eigenmode-seeded tests

A pure edge eigenmode of normalized high-pass amplitude 0.005 was evolved
below, near, and above the threshold.  Below threshold it grows or reaches a
nonlinear finite-amplitude state; at the threshold it is nearly neutral; above
threshold it decays strongly.  These tests remove ambiguity due to the broad
initial compacton mismatch.

## 5. Mode-selective intervention

For a real unstable eigenpair, the rank-one control

`P_alpha = (I - alpha r l^T) P`

moves the selected multiplier exactly to `(1-alpha) lambda`, with
`alpha_c = 1-1/|lambda|`.  It suppresses the forward packet strongly in the
`h=.01,kappa=5` case.

The `h=.02,kappa=20` case contains a leading real mode and a secondary complex
unstable pair.  Removing only the real mode does **not** stabilize the dynamics;
indeed, the remaining pair can dominate.  A real rank-three biorthogonal
projector onto the full unstable subspace reduces the maximum forward packet
from `0.146606`
to `0.00246873`,
a factor of approximately
`59.4`.
Full removal drives the unstable coordinates to roundoff.

This is an important correction to a scalar narrative: the causal object is the
complete unstable edge subspace.

## 6. Negative control: one-time projection removal

Deleting the leading unstable component only at t=0 does not prevent shedding.
Other edge components and nonnormal coupling regenerate it.  This negative
result is retained because it distinguishes continuous spectral stabilization
from a fortuitous change of initial data.

## 7. article-level conclusion

WP2 supplies four independent causal checks:

1. a controlled Floquet crossing through rho=1;
2. growth/neutrality/decay of a seeded eigenmode on the corresponding sides;
3. three-orders-of-magnitude suppression by ordinary hyperviscosity at the
   predicted coefficient with negligible core distortion;
4. suppression by continuous projection of the complete unstable subspace.

The result is considerably stronger than a correlation between a multiplier
and a packet: changing the spectrum changes the nonlinear outcome in the
predicted direction.

## Main-paper recommendation

Use the Floquet-continuation figure and the ordinary-hyperviscosity suppression
figure in the main paper.  Put eigenmode seeding, rank-one/rank-three control,
one-time removal, detailed spectral residuals, and all tabulated runs in the
supplement.  This keeps the main paper compact while preserving the full causal
case in the reproducibility archive.
