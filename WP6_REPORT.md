# WP6 — Integrated reproducibility package

WP6 consolidates WP1--WP5, WP7, and WP8 into a repository-ready computational archive. The package provides deterministic workflow code, derived data, retained numerical checkpoints, raw-data hashes, environment files, CI smoke tests, a machine-readable headline-result registry, a three-tier reproduction interface, and GitHub/Zenodo release metadata.

## Integrity design

The source repository and raw historical data are separated. The two historical MATLAB files are immutable inputs identified by exact byte size and SHA-256. All tracked repository files are covered by `metadata/FILES.sha256`. Source WP archives are also hashed in `metadata/SOURCE_ARCHIVES.sha256` to preserve provenance from the research campaign.

## Referee-facing reproduction

A referee can run `python reproduce.py check` in seconds, `python reproduce.py figures` to regenerate cross-WP headline plots, and `python reproduce.py smoke` for deterministic numerical/algebraic tests. Expensive scans are documented separately and the exact checkpoints used in the article are retained.

## Public-release status

The repository is ready for a GitHub import, but no public license, GitHub URL or Zenodo DOI has been invented. Those are author decisions/actions and are explicitly marked as pending.
