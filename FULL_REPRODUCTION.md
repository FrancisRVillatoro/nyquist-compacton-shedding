> `python reproduce.py full --wp all` prints the complete command plan. Add `--execute` to run it.

# Full numerical reproduction guide

Run from the repository root. The workflows are ordered because later verification steps use earlier deterministic outputs.

## WP1 — phase, fixed-phase map, and nonlinear latency

1. `workflows/wp1_phase_scan/phase_floquet_scan.py`
2. `workflows/wp1_fixed_phase/fixed_phase_map.py`
3. `workflows/wp1_fixed_phase/phase_onset_validation.py`
4. `workflows/wp1_fixed_phase/analyze_fixed_phase_onset.py`

The production 5x5 map may be parallelized using `run_fixed_case.py`; do not change `phi=0.25` for the fixed-phase map.

## WP2 — causal hyperviscous and modal control

The reusable numerical implementation is in `wp2_control_helper.py` and `wp2_fast_control.py`. Rebuild the two base orbits with `solve_base`, continue the edge spectrum with `hyper_block_ritz`, then run `wp2_seed_run.py`, `wp2_total_hyper.py`, and `wp2_modal_control.py`. The archived `raw_data/*.csv` are the exact trajectories used in the report.

## WP3 — temporal discretization

First run `build_nyquist_profiles.py`. Then reproduce Floquet convergence with `run_wp3_case.py`, `run_spectral_convergence.py`, and `run_gl6_reference.py`; free-wave propagation with `run_free_wave_benchmark.py` and `run_free_wave_gl6.py`; natural GL4 shedding with `run_gl4_natural_compacton.py`; and implementation verification with `tangent_directional_verification.py`. Finish with `finalize_wp3.py`.

## WP4 — spatial discretization

`cd workflows/wp4_space && bash run_all_wp4.sh`. Uncomment the expensive natural/Floquet lines in that script to regenerate the NPZ checkpoints rather than reuse them.

## WP5 — verification and uncertainty

Run the `wp5_*audit.py` scripts and `wp5_uncertainty_analysis.py`, followed by `make_wp5_figures.py`. Bootstrap sampling uses the fixed RNG seed 20260904.

## Determinism

All explicit random starts used for Ritz or bootstrap calculations have fixed seeds in the source. Differences caused by BLAS/LAPACK implementations should remain far below the instability margins summarized in WP5.


## WP8 — closure controls

Run `workflows/wp8_closure/wp8_hyper_spectrum_strict.py` for the strict edge-Floquet damping sweep. The direct nonlinear damping controls, noninteger-kappa run, moving-frame controls, classical K(2,2) negative control, and prepared finite-c0 packets are listed by `python reproduce.py full --wp 8`. Finish with `make_wp8_figures.py`. The strict dense spectrum is the version used by the final manuscript.


## WP7 — fully discrete mesh refinement

The retained checkpoints and summary tables are in `workflows/wp7_refinement/`. A complete regeneration uses `run_wp7_comoving_case.py` at the seven clean even-`M` levels and the coarse negative case, followed by `analyze_wp7_refinement.py`, `domain_check_fit.py`, and `make_wp7_figures.py`. The inverse Courant number is fixed at `kappa=5`; this is a fully discrete refinement path.
