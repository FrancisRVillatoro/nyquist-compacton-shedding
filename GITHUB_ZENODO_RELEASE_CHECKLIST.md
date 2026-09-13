# GitHub / Zenodo release checklist

1. Select software and data licenses; replace `LICENSE_PENDING.md`.
2. Create the neutral GitHub repository `nyquist-compacton-shedding`.
3. Upload/push this frozen source tree; do not add the two raw `.mat` files to the Git repository.
4. Run the smoke workflow and verify the frozen manifest.
5. Add the two raw `.mat` files to the Zenodo archival release or to a cross-linked companion data record. Their immutable sizes and SHA-256 hashes are in `metadata/RAW_DATA_MANIFEST.csv`.
6. Change the release-candidate version in `CITATION.cff` from `1.0.0-rc1` to `1.0.0`.
7. Enable the GitHub--Zenodo integration before creating GitHub release `v1.0.0`.
8. Mint the version DOI and concept DOI; insert the public repository URL and DOI into `README.md`, `CITATION.cff`, `DATA_AVAILABILITY.md`, and the manuscript citation.
9. Recompute the frozen manifest only if any released file changes.
10. Tag the exact release used for submission as `v1.0.0`.
