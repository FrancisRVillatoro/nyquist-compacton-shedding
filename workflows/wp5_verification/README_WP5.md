# WP5 verification/uncertainty package

This directory contains deterministic audits for the article compacton/Nyquist-shedding manuscript.

Key outputs:
- `WP5_REPORT.md`: scientific interpretation.
- `WP5_summary.csv`: headline numerical audit metrics.
- `packet_uncertainty_summary_with_calibration.csv`: packet amplitude/velocity uncertainty and model discrepancy.
- `ritz_seed_subspace_audit.csv`: random-seed and block-size reproducibility of the Floquet solver.
- `solver_tolerance_audit.csv`: nonlinear-solver tolerance sensitivity.
- `domain_audit_h001.csv`, `domain_audit_h002.csv`: fixed-phase box-size audit.
- `profile_solver_convergence*.csv`: Fourier profile convergence.
- `floquet_nonnormality_conditioning.csv`: left/right conditioning.
- `hyperviscosity_threshold_fit_sensitivity.csv`: critical-root fit robustness.
- `wp4_window_audit*.csv`: packet fit-window sensitivity.
- `fit_range_sensitivity*.csv`: clean-track range sensitivity.
- `float32_storage_audit.csv`: representative storage-precision check.
- `cost_repeatability_summary.csv`: timing range from repeated WP3 benchmark runs.
- `WP5_MANUSCRIPT_INSERT.tex`, `WP5_SUPPLEMENT_INSERT.tex`: proposed text blocks.

All bootstrap RNGs use seed 20260904. Bootstrap intervals quantify numerical/post-processing sensitivity, not physical statistical noise.
