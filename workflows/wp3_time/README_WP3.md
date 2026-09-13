# WP3 reproducibility package

This directory contains the calculations used to compare implicit midpoint,
two-stage Gauss--Legendre (GL4), and three-stage Gauss--Legendre (GL6) for the
same cubic-B-spline Petrov--Galerkin compacton semidiscretization.

## Principal outputs

- `WP3_REPORT.md`: scientific report and interpretation.
- `WP3_MANUSCRIPT_INSERT.tex`: concise main-paper insertion.
- `WP3_SUPPLEMENT_INSERT.tex`: implementation verification and extra details.
- `WP3_summary.csv`: headline numerical quantities.
- `time_integrator_spectrum_reference_comparison.csv`: all Floquet comparisons.
- `free_wave_three_integrator_benchmark.csv`: all free-wave runs.
- `velocity_correction_fits_three_integrators.csv`: calibrated corrections.
- `independent_packet_velocity_validation.csv`: out-of-sample packet tests.
- `accuracy_normalized_cost.csv`: prototype wall-clock/accuracy comparison.
- `tangent_directional_verification.csv`: tangent/adjoint implementation test.

## Core scripts

- `wp3_integrators.py`: midpoint and GL4 nonlinear/tangent/adjoint maps.
- `wp3_gl6.py`: GL6 nonlinear/tangent/adjoint maps.
- `wp3_gl4_banded.py`: banded GL4 production solver.
- `run_spectral_convergence.py`, `run_wp3_case.py`: compacton/Floquet runs.
- `run_free_wave_benchmark.py`, `run_free_wave_h002.py`,
  `run_free_wave_gl6.py`: free-wave convergence runs.
- `run_gl4_natural_compacton.py`: natural GL4 shedding experiment.
- `finalize_wp3.py`: tables and figures used in the report.

The exploratory GL6 natural run is included for transparency but is not used in
the quantitative packet-validation claims.

## Additional implementation files

- `wp3_midpoint_banded.py`: cyclic-pentadiagonal midpoint solver used for the fair cost comparison.
- `gl4_natural_packet_profile_snapshots.csv` and
  `wp3_gl4_natural_packet_profiles.png`: fixed-profile capture of two naturally emitted GL4 packets.
- `gl6_reference_spectrum.csv`: independent GL6 Floquet checks.
- `free_wave_gl6_benchmark.csv`: sixth-order free-wave verification.
