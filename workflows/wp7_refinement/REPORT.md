# WP7 — Fully discrete mesh refinement of naturally shed Nyquist packets

## Question

How does the physical size of the naturally generated grid-scale artifact change under mesh refinement at fixed inverse Courant number, while the numerical traveling compacton remains Floquet unstable at grid scale?

## Controlled refinement protocol


**Qualification.** The inverse Courant number is fixed at `kappa=5`, so `dt` scales linearly with `h`. This experiment therefore characterizes a fully discrete refinement path; it is not a simultaneous time-refined semidiscrete limit.

The central de Frutos spatial discretization and implicit midpoint time integration were used at fixed inverse Courant number

\[
\kappa=\frac{h}{c\Delta t}=5,
\qquad c=1.
\]

To remove both subcell-edge-phase and Nyquist-parity confounding, the grid spacings were chosen as

\[
h=\frac{4\pi}{M},
\]

with **even** integer `M`, so the compacton support spans exactly `M` grid cells. The forward edge was placed at phase \(\phi_R=1/4\); since `M` is an even integer, the rear edge has the same phase and the same relative Nyquist parity. Seven clean refinement levels were obtained:

`M = 628, 838, 1006, 1256, 1676, 2094, 2514`, corresponding to

\[
h\approx0.0200101,0.0149957,0.0124914,0.0100051,0.00749783,0.00600113,0.00499856.
\]

The analytic sampled compacton was evolved by repeated application of the **one-cell co-moving map** \(\mathcal P_{\rm NL}=S^{-1}\Phi_{\Delta t}^{\kappa}\). This is exactly the laboratory-frame evolution modulo an integer translation after every cell crossing and prevents irrelevant periodic wrap-around of the translating compacton.

A clean packet was accepted only when a fixed-shape fit to the independently computed de Frutos Nyquist profile persisted over multiple saved states. The reported tracks use 60 clean snapshots. Median profile residuals are between \(5.7\times10^{-4}\) and \(1.8\times10^{-3}\).

## Main results

The normalized amplitudes of the first persistent clean packets remain order unity:

\[
0.06595\le A/h^2\le0.11746.
\]

A log-log regression gives

\[
A\propto h^{2.185\pm0.161},
\]

while the slope of \(A/h^2\) is

\[
0.185\pm0.161,
\]

whose 95% regression interval contains zero. Thus there is no resolved monotone trend in the normalized amplitude over a factor four in \(h\), whereas the physical amplitude tends rapidly to zero.

The independently computed Nyquist family has approximately constant FWHM in cell coordinates,

\[
W_{\rm cells}\simeq5.449,
\]

so the physical width is exactly proportional to \(h\) over this refinement family:

\[
W_x\simeq5.449h.
\]

Consequently an \(A=O(h^2)\), cell-scale packet predicts

\[
\|u_p\|_\infty=O(h^2),\qquad
\|u_p\|_1=O(h^3),\qquad
\|u_p\|_2=O(h^{5/2}).
\]

Direct local norms from the natural packets give fitted exponents

\[
\|u_p\|_\infty\propto h^{2.187\pm0.148},
\]
\[
\|u_p\|_1\propto h^{3.154\pm0.134},
\]
\[
\|u_p\|_2\propto h^{2.683\pm0.131}.
\]

The deviations from the ideal powers reflect the 20% level branch-to-branch scatter in \(A/h^2\); all physical norms nevertheless decay rapidly and the theoretical exponents lie within the finite-sample regression uncertainty at the level relevant to this deterministic study.

The clean-packet formation cell is highly non-monotone (148 to 1304 cell crossings), confirming that nonlinear latency is not a useful continuum-convergence observable. The packet **size and physical norms**, not the formation time, provide the clean refinement statement.

## Reconciliation with Floquet instability

WP1 already shows that the de Frutos one-cell Floquet spectral radius stays outside the unit circle throughout the corresponding fixed-phase range. WP7 therefore resolves the apparent paradox:

- the **grid-scale dynamical instability** can have an O(1) multiplier per cell crossing;
- the edge injects an O(h^2) physical perturbation;
- the selected coherent packet has O(1) amplitude only after normalization by h^2 and O(1) width only in **cell coordinates**;
- hence its physical amplitude and continuum norms vanish as h -> 0.

There is no contradiction between discrete Floquet instability and the observed decay of the physical numerical error along this fixed-`kappa` refinement path.

## Domain check

The two finest cases used a periodic box of approximately 32 units rather than 40 for computational efficiency. At `M=1676` (h≈0.0074978), rerunning the identical experiment at L≈32 and L≈40 changes the normalized packet amplitude and velocity only at the 1e-4 relative level, with the first clean cell and profile residual unchanged.

## Negative coarse-grid observation

At the additional coarse level `M=420` (h≈0.02992), no clean own-family packet was isolated within 650 cell crossings. This is not used in the refinement fit and is consistent with the already established fact that Floquet growth alone does not determine nonlinear packet-isolation latency.
