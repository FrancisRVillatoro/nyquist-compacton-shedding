# WP8 closure controls

This workflow contains the final closure calculations added after the main WP1--WP5 and WP7 campaigns: a strict edge-Floquet damping sweep, noninteger inverse-Courant control, moving-frame controls, the classical K(2,2) negative control, and prepared-wave propagation at finite c0.

The production manuscript uses the strict spectrum from `wp8_hyper_spectrum_strict.py`. The direct dense diagonalization forms the N=800 tangent operator explicitly and reduces the selected edge-eigenpair residuals to approximately 1e-14. The nonlinear damping trajectories retain the original damping values used in the closure experiment; `make_wp8_figures.py` combines them with the strict spectrum.

All paths are repository-relative. Derived outputs used by the article are retained under `derived/`.
