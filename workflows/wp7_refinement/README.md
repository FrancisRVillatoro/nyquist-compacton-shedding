# WP7 — fully discrete mesh refinement of naturally shed Nyquist packets

This workflow contains the fixed-parameter mesh-refinement experiment for naturally generated de Frutos Nyquist packets.

## Controlled parameters

- equation: absolute `|K|(2,2)`, `c=1`, `c0=0`;
- spatial discretization: de Frutos cubic-B-spline/Petrov--Galerkin operator;
- time integrator: implicit midpoint;
- inverse Courant number: `kappa=h/(c*dt)=5` fixed throughout;
- mesh: `h=4*pi/M` with even `M`;
- forward and rear edge phases: `phi_R=phi_L=1/4`;
- evolution: co-moving one-cell map `S^{-1} Phi_dt^kappa`.

Because `kappa` is fixed, this is a **fully discrete refinement path**, not a simultaneous semidiscrete time-refinement limit.

## Headline outputs

- `wp7_refinement_summary.csv`: seven clean refinement levels and one coarse negative case;
- `wp7_scaling_fits.csv`: deterministic log--log fits;
- `wp7_clean_packet_tracks.csv`: persistent packet tracks;
- `wp7_candidate_scan.csv`: scanned packet candidates;
- `wp7_profile_family.csv`: de Frutos Nyquist-profile quantities used in the fits;
- `wp7_domain_check.csv`: `L≈32` versus `L≈40` domain check.

## Main conclusion

Along the tested fixed-`kappa` refinement path, the normalized packet amplitude `A/h^2` remains bounded and order one, while physical width is proportional to `h`. The measured physical amplitude and local norms decrease strongly with mesh refinement, consistently with the `h^2`, `h^3`, and `h^(5/2)` scalings implied by a cell-scale packet with bounded `A/h^2`.
