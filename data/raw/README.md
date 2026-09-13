# Raw historical simulations

The source repository intentionally does not duplicate the two large historical MATLAB files. Their exact names, byte sizes, and SHA-256 hashes are recorded in `metadata/RAW_DATA_MANIFEST.csv`.

For the archival release, place these files in this directory:

- `CFL05_L250-T200-dx0.01-dt0.002-Vel1-Co0-E0.mat`
- `CFL20_L250-T200-dx0.02-dt0.001-Vel1-Co0-E0.mat`

A separate raw-data archive is distributed alongside this repository. When a Zenodo DOI is minted, this README should be updated with the DOI and `scripts/fetch_raw_data.py` may be configured with the final download URLs.
