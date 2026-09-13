# WP3 — Generality with respect to time discretization

## Executive conclusion

WP3 separates the effects of the spatial discretization from those of the time integrator.  The cubic-B-spline Petrov--Galerkin semidiscretization was held fixed and was combined with three Gauss collocation methods:

- implicit midpoint (one-stage Gauss, order two),
- two-stage Gauss--Legendre (GL4, order four), and
- three-stage Gauss--Legendre (GL6, order six).

The central conclusion is unambiguous:

\[
\boxed{\text{the edge--Nyquist instability is not created by implicit midpoint.}}
\]

GL4 and GL6 converge to the same unstable time-continuous/semidiscrete limit.  Time integration nevertheless matters quantitatively: at coarse timestep it changes the growth rate, can change the dominant Floquet branch from a real flip multiplier to a complex pair, and produces a method-specific correction to the velocity of the emitted Nyquist solitary waves.

WP3 also found and corrected an important issue in the audited v2 manuscript.  The provisional midpoint velocity correction

\[
1-9.05\chi^2+362.3\chi^4+\cdots
\]

is not supported by the direct fully discrete convergence study.  It must be replaced by the calibrated midpoint law reported below.  The revised law improves the two independent archived-packet velocity tests from relative discrepancies of \(1.16\times10^{-3}\) and \(6.8\times10^{-5}\) to \(6.7\times10^{-6}\) and \(2.2\times10^{-6}\), respectively.

---

## 1. Numerical design

The semidiscrete equation is

\[
\mathcal A\dot U+\mathcal L\Phi(U)=0,
\qquad
\mathcal L=\mathcal B+\mathcal C,
\qquad
\Phi(U)=|U|U.
\]

All spectral comparisons use the same spatial operator, periodic domain \(L=16\), compacton speed \(c=1\), no artificial dissipation, and fixed forward-edge phase \(\phi_R=1/4\).  The timestep is parameterized by

\[
\kappa=\frac{h}{c\Delta t},
\]

the number of timesteps per compacton cell crossing.  For each \((h,\kappa)\) and each integrator, a relative periodic numerical compacton was computed and the exact tangent one-cell map was formed.  Tangent and adjoint actions were implemented analytically at the discrete-algorithm level.

For the free Nyquist-wave tests, the independently computed semidiscrete profile \(F_h\) was propagated at several amplitudes and timesteps.  Its exact semidiscrete speed is

\[
v_{\rm sd}=\frac{A}{h^2}V_h.
\]

Thus the temporal velocity error can be measured without extrapolating a reference trajectory.

---

## 2. Semidiscrete Floquet limit and integrator dependence

Independent high-order reference values were defined as the medians of GL4 at \(\kappa=40,80\) and GL6 at \(\kappa=10,20\):

| \(h\) | reference \(\rho_F\) | half range of reference ensemble |
|---:|---:|---:|
| 0.01 | 1.0958938054 | \(8.94\times10^{-7}\) |
| 0.02 | 1.1342141857 | \(3.00\times10^{-7}\) |

The ensemble spread is many orders of magnitude smaller than \(\rho_F-1\).  Hence the time-refined semidiscrete numerical compacton is unequivocally unstable for both meshes.

### Principal cases

| \(h\) | \(\kappa\) | integrator | dominant multiplier | \(\rho_F\) | branch |
|---:|---:|---|---:|---:|---|
| 0.01 | 5 | midpoint | \(-1.056633\) | 1.056633 | real flip |
| 0.01 | 5 | GL4 | \(-1.085169\pm0.151297i\) | 1.095665 | complex |
| 0.01 | 5 | GL6 | \(-1.085385\pm0.151592i\) | 1.095920 | complex |
| 0.02 | 20 | midpoint | \(-1.130947\) | 1.130947 | real flip |
| 0.02 | 20 | GL4 | \(-1.134215\) | 1.134215 | real flip |
| 0.02 | 20 | GL6 | \(-1.134214\) | 1.134214 | real flip |

The first row set is especially instructive.  At \(h=0.01,\kappa=5\), midpoint not only underestimates the growth rate; it selects a different dominant branch.  GL4 and GL6 agree that the semidiscrete dominant instability is oscillatory.  Therefore a coarse second-order integrator can partially mask and qualitatively alter the instability while not removing it.

The numerical compacton profiles themselves remain extremely close.  In the principal runs,

\[
\|U_*^{\rm MP}-U_*^{\rm GL4}\|_\infty
=1.29\times10^{-8}
\quad(h=0.01,\kappa=5),
\]

and

\[
=9.57\times10^{-8}
\quad(h=0.02,\kappa=20).
\]

Thus the spectral changes are not caused by a macroscopic change of the base compacton.

### Temporal convergence of the Floquet spectrum

Using the time-refined GL4/GL6 limit as reference, midpoint gives observed orders \(2.04\) and \(2.02\) for \(h=0.01\) and \(0.02\).  GL4 gives approximately \(3.69\) and \(4.30\) before the nonlinear fixed-point and Ritz tolerances set the numerical floor.  GL6 independently reaches the same limiting spectrum; at \(h=0.02\), already at \(\kappa=5\), its spectral-radius error is below \(2\times10^{-6}\).

A Richardson extrapolation gives an independent check.  Using the formal
orders two, four, and six on the finest available pairs yields

| $h$ | midpoint extrapolation | GL4 extrapolation | GL6 extrapolation |
|---:|---:|---:|---:|
| 0.01 | 1.0958959393 | 1.0958939823 | 1.0958943138 |
| 0.02 | 1.1342157133 | 1.1342144332 | 1.1342142081 |

The GL4--GL6 discrepancies are only $3.3\times10^{-7}$ and
$2.3\times10^{-7}$.  Even the independently extrapolated midpoint values
agree within about $2\times10^{-6}$.  This three-method agreement is the
high-accuracy reference integration of the semidiscrete relative orbit used in
WP3.

These results support the interpretation

\[
\boxed{
\text{spatial semidiscretization: existence of the instability};\qquad
\text{time integrator: distortion of its multiplier and branch.}}
\]

---

## 3. Temporal convergence of free Nyquist solitary waves

For a normalized amplitude \(A/h^2=0.1\), log--log fits of the velocity error give:

| \(h\) | midpoint | GL4 | GL6 |
|---:|---:|---:|---:|
| 0.01 | 1.9957 | 3.9512 | 5.9310 |
| 0.02 | 1.9935 | 3.9511 | 5.9310 |

The fitted orders agree with the formal orders two, four, and six.  The collapse between the two spatial meshes is excellent.  The largest matched-parameter difference between \(h=0.01\) and \(0.02\) is \(7.61\times10^{-6}\) for the midpoint/GL4 data and \(3.46\times10^{-8}\) for GL6; the median GL6 difference is \(4.35\times10^{-11}\).

This establishes that the natural temporal similarity parameter is

\[
\chi=\frac{A\Delta t}{h^3}=\frac{A/h^2}{\kappa}.
\]

---

## 4. Method-dependent fully discrete velocity laws

Direct propagation and profile fitting give the following calibrated asymptotic laws over the stated ranges.

### Implicit midpoint

For \(\chi\le 0.036\),

\[
\boxed{
\frac{v_{\rm MP}}{v_{\rm sd}}-1
=-11.659289\,\chi^2
+452.119772\,\chi^4
-2.10497\times10^4\,\chi^6
+O(\chi^8).}
\]

The fit RMS is \(5.0\times10^{-7}\).  The leading coefficient obtained independently on the two meshes differs by only \(1.2\times10^{-3}\).

### Two-stage Gauss--Legendre (GL4)

For \(\chi\le 0.050\),

\[
\boxed{
\frac{v_{\rm GL4}}{v_{\rm sd}}-1
=-44.079876\,\chi^4
+719.228502\,\chi^6
+8148.209713\,\chi^8
+O(\chi^{10}).}
\]

The fit RMS is \(1.0\times10^{-8}\).  The leading \(\chi^4\) coefficient from the two meshes agrees within about \(2.0\times10^{-2}\).

### Three-stage Gauss--Legendre (GL6)

For \(\chi\le0.050\),

\[
\boxed{
\frac{v_{\rm GL6}}{v_{\rm sd}}-1
=-116.356187\,\chi^6
+1724.135317\,\chi^8
+O(\chi^{10}).}
\]

The fit RMS is \(1.0\times10^{-10}\).  Separate fits give leading coefficients \(-116.3773\) and \(-116.3351\) for \(h=0.01\) and \(0.02\).

The leading powers \(\chi^2,\chi^4,\chi^6\) are robust and coincide with the method orders.  The higher coefficients are deterministic calibration coefficients over the quoted ranges; they should not be interpreted as certified global asymptotic constants without a separate functional-analytic derivation.

---

## 5. Independent nonlinear validation of the velocity corrections

The calibration data above use deliberately initialized free Nyquist waves.  Four independently generated packets provide out-of-sample tests.

| packet | baseline relative discrepancy | corrected relative discrepancy |
|---|---:|---:|
| archived midpoint, \(h=0.01,\kappa=5\) | \(-1.16\times10^{-3}\) | \(+6.65\times10^{-6}\) |
| archived midpoint, \(h=0.02,\kappa=20\) | \(-6.80\times10^{-5}\) | \(+2.18\times10^{-6}\) |
| natural GL4 packet I | \(-5.92\times10^{-5}\) | \(+7.82\times10^{-8}\) |
| natural GL4 packet II | \(-8.01\times10^{-7}\) | \(+5.68\times10^{-8}\) |

For the midpoint rows, “baseline” is the provisional law used in manuscript v2.  For the GL4 rows, it is the uncorrected semidiscrete law.  The GL4 packets were generated spontaneously by evolution of the analytic compacton; they were not part of the free-wave calibration.

The two clean GL4 packets have

\[
A/h^2=0.1710967,\qquad v=3.5349106,
\]

and

\[
A/h^2=0.0590858,\qquad v=1.2208014.
\]

Their median fixed-profile residuals are \(1.2\times10^{-5}\) and \(2.4\times10^{-6}\).  The same independently computed semidiscrete profile \(F_h\) therefore describes packets emitted under both midpoint and GL4.  The only detectable change in their clean propagation is the method-specific temporal velocity correction.

Fit-window and fit-range variants change the corrected predictions by at most a few \(10^{-6}\) relative for the archived midpoint packets and a few \(10^{-7}\) for the first GL4 packet.  Full regression/bootstrap uncertainties are deferred to WP5.

---

## 6. Accuracy-normalized cost

A wall-clock benchmark was performed for one compacton cell crossing at \(h=0.02\), \(N=800\), on the same software/hardware environment.  These timings characterize the supplied reference implementation and are not hardware-independent complexity constants.

Representative results are:

| method | \(\kappa\) | time/cell (s) | \(|\rho_F-\rho_{F,\rm ref}|\) | free-wave velocity error |
|---|---:|---:|---:|---:|
| midpoint, banded | 40 | 0.0311 | \(7.93\times10^{-4}\) | \(7.28\times10^{-5}\) |
| midpoint, banded | 80 | 0.0622 | \(1.97\times10^{-4}\) | -- |
| GL4, banded | 3 | 0.0114 | -- | \(5.34\times10^{-5}\) |
| GL4, banded | 5 | 0.0207 | \(2.88\times10^{-4}\) | \(7.01\times10^{-6}\) |
| GL4, banded | 10 | 0.0381 | \(1.66\times10^{-5}\) | \(4.40\times10^{-7}\) |
| GL6, generic sparse | 2 | 0.0316 | \(1.61\times10^{-4}\) | \(1.75\times10^{-6}\) |
| GL6, generic sparse | 5 | 0.0585 | \(1.87\times10^{-6}\) | \(7.40\times10^{-9}\) |

Despite its larger stage system, GL4 reaches a prescribed accuracy at a much smaller \(\kappa\).  With banded solvers for both methods, GL4 at \(\kappa=5\) is about three times faster than midpoint at \(\kappa=80\) at comparable Floquet-radius accuracy.  GL4 at \(\kappa=10\) is about 1.6 times faster than midpoint at \(\kappa=80\) while giving an order-of-magnitude smaller spectral error.  GL6 is not yet band-optimized; nevertheless, at \(\kappa=5\) it has nearly the same cost as banded midpoint at \(\kappa=80\) and about two orders of magnitude smaller spectral error.

Thus higher order is not merely a verification luxury here: it can be computationally advantageous for resolving the edge spectrum and packet velocity.

---

## 7. Verification of tangent and adjoint implementations

For all three integrators, the one-cell tangent action was compared with central finite differences of the nonlinear map.  The minimum relative discrepancies were approximately

\[
3.2\times10^{-5}\quad(\text{midpoint}),\qquad
1.2\times10^{-4}\quad(\text{GL4}),\qquad
1.3\times10^{-4}\quad(\text{GL6}),
\]

before roundoff amplification.  The convergence is limited by the \(C^1\), non-\(C^2\) character of \(|u|u\) at grid nodes where the background vanishes.

The discrete adjoint identities were satisfied to

\[
\max \frac{|\langle Pz,w\rangle-\langle z,P^*w\rangle|}
{|\langle Pz,w\rangle|+|\langle z,P^*w\rangle|}
<2.8\times10^{-14}.
\]

These checks will be integrated into the broader verification matrix in WP5.

---

## 8. Exploratory GL6 nonlinear run

A short natural GL6 run at \(h=0.01,\kappa=5\) developed a strong forward high-frequency field, reaching \(A_{\rm HP}/h^2\simeq0.202\) by \(t=6\), but it remained a mixed shedding transient and did not yield an isolated packet with local profile residual below 2% within that interval.  It is therefore not used as a quantitative packet benchmark.  This negative result is retained in the reproducibility package.

The appropriate conclusion is not that every time integrator produces identical packet timing or multiplicity.  Rather:

- the unstable semidiscrete edge spectrum is integrator-independent in the time-refined limit;
- the free nonlinear Nyquist-wave family is the same spatial family;
- the detailed nonlinear shedding sequence is sensitive to temporal discretization;
- at least one independent high-order integrator (GL4) spontaneously generates clean packets captured by that same family.

---

## 9. Consequences for the article manuscript

WP3 changes the manuscript in four substantive ways.

1. **Generality.**  The edge instability and Nyquist-wave family are not artifacts of implicit midpoint.
2. **Qualification.**  Coarse time integration can change the dominant Floquet branch and the observed shedding chronology.
3. **Correction.**  The midpoint velocity law in audited v2 must be replaced by the calibrated \(-11.6593\chi^2+\cdots\) law.
4. **Practical guidance.**  GL4/GL6 recover the semidiscrete spectrum and packet speed at much smaller \(\kappa\) and can be more efficient at fixed accuracy.

A compact main-paper presentation should contain:

- one three-integrator Floquet-convergence figure;
- one three-integrator velocity-order/correction figure;
- the four-row independent packet-validation table;
- a concise accuracy-normalized cost paragraph;
- the natural GL4 packet trajectories or profiles.

The full tableaux, all spectral values, directional-derivative checks, fixed-point differences, fit sensitivities, and exploratory GL6 transient should remain in the supplement/repository.

---

## 10. Final WP3 verdict

\[
\boxed{
\begin{aligned}
&\text{The spatial discretization creates an unstable edge--Nyquist mode;}\\
&\text{the time integrator controls how faithfully that mode is represented;}\\
&\text{the emitted free packet belongs to a common semidiscrete solitary family;}\\
&\text{its finite-}\Delta t\text{ velocity correction is method-dependent and order-consistent.}
\end{aligned}}
\]

WP3 therefore closes the principal “midpoint artifact” objection and materially strengthens the paper for *the associated article*.
