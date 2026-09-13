#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python test_banded_solver.py
python derive_edge_coefficients.py > edge_coefficients_symbolic.txt
# Expensive natural and Floquet runs are normally reused from the archived NPZ/CSV files.
# Uncomment to regenerate them:
# for m in ismail defrutos pade6 pade8; do python run_natural_simulations.py --method "$m"; done
# for m in ismail defrutos pade6 pade8; do python run_floquet_case.py --method "$m" --h .02 --kappa 10 --L 20 --phase .25; done
python run_prepared_wave_validation.py
python finalize_wp4_tables.py
python make_wp4_figures.py
