# Reproducing WP4

## Environment

Tested with Python 3.13, NumPy, SciPy, pandas, and Matplotlib.

## Files

- `wp4_spatial_methods.py`: four sparse spatial systems, GL4 flow and tangent maps, midpoint tangent map, fixed-point and Ritz routines, and Nyquist profile solver.
- `wp4_banded.py`: fast cyclic-pentadiagonal midpoint solver for long nonlinear runs.
- `run_floquet_case.py`: command-line GL4 fixed-point/Floquet calculation.
- `finalize_wp4_tables.py`: creates all summary tables and performs profile/capture analysis.
- `make_wp4_figures.py`: recreates all final figures from the raw and derived data.
- `run_prepared_wave_validation.py`: independent free-wave validation.

## Typical commands

```bash
python run_floquet_case.py --method pade6 --h 0.02 --kappa 10 --L 20 --phase 0.25
python finalize_wp4_tables.py
python make_wp4_figures.py
```

The four natural-run NPZ files contain `x`, saved times `t`, solution snapshots `U`, the initial center `x0`, `h`, and `kappa`.  All final numerical values used in the manuscript are also exported to CSV so the figures do not require rerunning the expensive fixed-point/Floquet calculations.

## Precision caveat

The Ismail and Padé-8 traveling fixed points are more ill-conditioned than the de Frutos and Padé-6 points in the current Newton--Krylov implementation.  Their residuals are `O(10^-8)`, whereas their Floquet distances from the unit circle are `O(10^-2)` and their eigenpair residuals are `O(10^-10)--O(10^-12)`.  These values are reported explicitly rather than hidden.
