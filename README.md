# Nyquist solitary-wave shedding from numerical compactons: software and data archive

Reproducibility repository for the associated article manuscript by Rubén Garralón-López, Francisco Rus, and Francisco R. Villatoro.

The central computational result is that moving low-regularity compacton edges excite a stencil-dependent Nyquist sector; the resulting numerical traveling compactons possess edge-localized Floquet instabilities, nonlinear unstable excursions and coherent Nyquist solitary-wave shedding. The repository contains the numerical workflows, derived data, verification checks, and closure controls supporting that mechanism across multiple time integrators and spatial discretizations.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python reproduce.py check
python reproduce.py figures
python reproduce.py smoke
```

## Repository map

- `workflows/wp1_phase_scan/`: subcell phase-resolved Floquet scan.
- `workflows/wp1_fixed_phase/`: fixed-phase parameter map and nonlinear latency validation.
- `workflows/wp2_control/`: hyperviscous continuation, seeded eigenmode and unstable-subspace causal controls.
- `workflows/wp3_time/`: midpoint/GL4/GL6 temporal-discretization comparison and semidiscrete references.
- `workflows/wp4_space/`: Ismail--Taha, de Frutos, Padé-6 and Padé-8 spatial-discretization comparison.
- `workflows/wp5_verification/`: tolerances, domain, Ritz, tangent/adjoint, uncertainty and nonnormality audits.
- `workflows/wp7_refinement/`: fixed-`kappa` fully discrete mesh refinement of naturally shed packets.
- `workflows/wp8_closure/`: strict threshold spectrum and final closure controls.
- `data/derived/historical/`: small derived products from the two historical simulations.
- `data/raw/`: location of the two historical `.mat` files in the full archival release.
- `metadata/`: software versions, source provenance, claim registry and SHA-256 manifests.
- `tests/`: fast deterministic checks suitable for CI.

See `REPRODUCIBILITY.md` for the three verification tiers and `FULL_REPRODUCTION.md` for the expensive numerical campaign.

## Important terminology

The quantity `kappa = h/(c*dt)` is the **inverse Courant number** / number of time steps per cell crossing. Historical filenames may still contain the label `CFL5` or `CFL20`; those names are preserved only for provenance.

## Raw data

The two historical MATLAB files are kept in a separate archive. Exact SHA-256 hashes are recorded in `metadata/RAW_DATA_MANIFEST.csv`.

## Version 1.0.0

This tree is the frozen source/derived-data release used for the associated study. It includes the WP1--WP5 workflows, the fixed-`kappa` WP7 refinement workflow, and the strict WP8 closure calculations. The public repository is:

https://github.com/FrancisRVillatoro/nyquist-compacton-shedding

The software release will be archived through the Zenodo--GitHub integration. The two immutable historical MATLAB inputs are deposited separately as a cross-linked Zenodo dataset so that raw data and software retain clear licensing and citation metadata.

## License

- **Software/source code:** MIT License; see `LICENSE`.
- **Derived numerical data, numerical tables, workflow-generated figures, and associated data documentation:** Creative Commons Attribution 4.0 International (CC BY 4.0); see `LICENSE-DATA.md`.
- **Historical raw MATLAB inputs:** CC BY 4.0 in the companion Zenodo dataset record.

See `CONTRIBUTORS.md` for the CRediT contribution statement.
