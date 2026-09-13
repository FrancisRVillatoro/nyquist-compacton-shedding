# Reproducibility tiers

Three tiers are provided so that a referee can verify the work without first rerunning the entire expensive campaign.

## Tier 1 — integrity and headline checks (seconds)

```bash
python reproduce.py check
```

This verifies SHA-256 integrity for tracked files, validates the machine-readable headline numbers against the CSV outputs, checks exact edge-defect coefficients, and compiles all Python scripts.

## Tier 2 — regenerate paper-level diagnostic figures (seconds to minutes)

```bash
python reproduce.py figures
```

This recreates a compact set of cross-WP figures from the archived derived data into `figures_regenerated/`.

## Tier 3 — numerical smoke tests (minutes)

```bash
python reproduce.py smoke
```

This runs the pytest suite, including deterministic algebraic checks and small numerical-kernel tests.

## Full numerical campaign

```bash
python reproduce.py full --wp all
# dry-run above; add --execute to launch the full expensive campaign
```

The full campaign is intentionally not run in CI because it includes Newton--Krylov fixed points, block-Ritz/adjoint iterations, multi-parameter scans, natural shedding simulations and bootstrap post-processing. `FULL_REPRODUCTION.md` lists the exact workflow order and expected output files. Existing deterministic checkpoints are retained so that every main claim can be inspected without repeating the expensive runs.

The full plan includes the fixed-`kappa` WP7 mesh-refinement workflow and the strict WP8 closure calculations.
