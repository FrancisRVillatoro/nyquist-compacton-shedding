#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys, shlex
R=Path(__file__).resolve().parents[1]
PY=sys.executable

def steps(wp):
    W=R/'workflows'
    d={
    '1':[
      ([PY,'phase_floquet_scan.py'],'wp1_phase_scan'),
      ([PY,'fixed_phase_map.py','--phi','0.25'],'wp1_fixed_phase'),
      ([PY,'phase_onset_validation.py'],'wp1_fixed_phase'),
      ([PY,'analyze_fixed_phase_onset.py'],'wp1_fixed_phase'),
    ],
    '2':[
      ([PY,'wp2_rebuild_inputs_and_spectrum.py'],'wp2_control'),
      ([PY,'wp2_seed_run.py'],'wp2_control'),
      ([PY,'wp2_total_hyper.py'],'wp2_control'),
      ([PY,'wp2_modal_control.py'],'wp2_control'),
    ],
    '3':[
      ([PY,'build_nyquist_profiles.py'],'wp3_time'),
      ([PY,'run_spectral_convergence.py'],'wp3_time'),
      ([PY,'run_gl6_reference.py'],'wp3_time'),
      ([PY,'run_free_wave_benchmark.py'],'wp3_time'),
      ([PY,'run_free_wave_h002.py'],'wp3_time'),
      ([PY,'run_free_wave_gl6.py'],'wp3_time'),
      ([PY,'run_gl4_natural_compacton.py'],'wp3_time'),
      ([PY,'gl4_natural_profile_snapshots.py'],'wp3_time'),
      ([PY,'benchmark_integrator_cost.py'],'wp3_time'),
      ([PY,'tangent_directional_verification.py'],'wp3_time'),
      ([PY,'finalize_wp3.py'],'wp3_time'),
    ],
    '4':[
      ([PY,'test_banded_solver.py'],'wp4_space'),
      ([PY,'derive_edge_coefficients.py'],'wp4_space'),
      *[([PY,'run_natural_simulations.py','--method',m],'wp4_space') for m in ['ismail','defrutos','pade6','pade8']],
      *[([PY,'run_floquet_case.py','--method',m,'--h','.02','--kappa','10','--L','20','--phase','.25'],'wp4_space') for m in ['ismail','defrutos','pade6','pade8']],
      ([PY,'run_prepared_wave_validation.py'],'wp4_space'),
      ([PY,'finalize_wp4_tables.py'],'wp4_space'),
      ([PY,'make_wp4_figures.py'],'wp4_space'),
    ],
    '5':[
      ([PY,'wp5_profile_convergence.py'],'wp5_verification'),
      ([PY,'wp5_domain_audit.py'],'wp5_verification'),
      ([PY,'wp5_ritz_seed_audit.py'],'wp5_verification'),
      ([PY,'wp5_tangent_spatial_audit.py'],'wp5_verification'),
      ([PY,'wp5_solver_tolerance_audit_fast.py'],'wp5_verification'),
      ([PY,'wp5_float_storage_audit.py'],'wp5_verification'),
      ([PY,'wp5_wp4_window_audit.py'],'wp5_verification'),
      ([PY,'wp5_fit_range_sensitivity.py'],'wp5_verification'),
      ([PY,'wp5_uncertainty_analysis.py'],'wp5_verification'),
      ([PY,'wp5_nonnormality.py'],'wp5_verification'),
      ([PY,'make_wp5_figures.py'],'wp5_verification'),
    ],
    '7':[
      ([PY,'run_wp7_comoving_case.py','--Mwidth','628','--kappa','5','--cells','650'],'wp7_refinement'),
      ([PY,'run_wp7_comoving_case.py','--Mwidth','838','--kappa','5','--cells','650'],'wp7_refinement'),
      ([PY,'run_wp7_comoving_case.py','--Mwidth','1006','--kappa','5','--cells','1450'],'wp7_refinement'),
      ([PY,'run_wp7_comoving_case.py','--Mwidth','1256','--kappa','5','--cells','650'],'wp7_refinement'),
      ([PY,'run_wp7_comoving_case.py','--Mwidth','1676','--kappa','5','--cells','650'],'wp7_refinement'),
      ([PY,'run_wp7_comoving_case.py','--Mwidth','2094','--kappa','5','--cells','650'],'wp7_refinement'),
      ([PY,'run_wp7_comoving_case.py','--Mwidth','2514','--kappa','5','--cells','1250'],'wp7_refinement'),
      ([PY,'run_wp7_comoving_case.py','--Mwidth','420','--kappa','5','--cells','650'],'wp7_refinement'),
      ([PY,'analyze_wp7_refinement.py'],'wp7_refinement'),
      ([PY,'domain_check_fit.py'],'wp7_refinement'),
      ([PY,'make_wp7_figures.py'],'wp7_refinement'),
    ],
    '8':[
      ([PY,'wp8_hyper_spectrum_strict.py'],'wp8_closure'),
      ([PY,'wp8_hyper_nonlinear_single.py','0.50','300'],'wp8_closure'),
      ([PY,'wp8_hyper_nonlinear_single.py','0.80','300'],'wp8_closure'),
      ([PY,'wp8_hyper_nonlinear_single.py','0.90','300'],'wp8_closure'),
      ([PY,'wp8_hyper_nonlinear_single.py','0.95','700'],'wp8_closure'),
      ([PY,'wp8_hyper_nonlinear_single.py','1.00','300'],'wp8_closure'),
      ([PY,'wp8_hyper_nonlinear_single.py','1.05','300'],'wp8_closure'),
      ([PY,'wp8_hyper_nonlinear_single.py','1.20','300'],'wp8_closure'),
      ([PY,'wp8_controls.py','baseline'],'wp8_closure'),
      ([PY,'wp8_controls.py','c0_half'],'wp8_closure'),
      ([PY,'wp8_continue_c0half.py'],'wp8_closure'),
      ([PY,'wp8_controls.py','c0_equal_c'],'wp8_closure'),
      ([PY,'wp8_controls.py','kappa20p5'],'wp8_closure'),
      ([PY,'wp8_K22_faildiag.py'],'wp8_closure'),
      ([PY,'wp8_prepared_c0.py','0.0'],'wp8_closure'),
      ([PY,'wp8_prepared_c0.py','0.05'],'wp8_closure'),
      ([PY,'wp8_prepared_c0.py','0.10'],'wp8_closure'),
      ([PY,'wp8_prepared_c0.py','0.20'],'wp8_closure'),
      ([PY,'wp8_prepared_c0.py','0.50'],'wp8_closure'),
      ([PY,'make_wp8_figures.py'],'wp8_closure'),
    ]}
    return [(cmd,W/cwd) for cmd,cwd in d[wp]]

def main():
    p=argparse.ArgumentParser();p.add_argument('--wp',choices=['1','2','3','4','5','7','8','all'],default='all');p.add_argument('--execute',action='store_true');a=p.parse_args()
    wps=['1','2','3','4','5','7','8'] if a.wp=='all' else [a.wp]
    allsteps=[]
    for wp in wps: allsteps.extend([(wp,*x) for x in steps(wp)])
    for i,(wp,cmd,cwd) in enumerate(allsteps,1):
        print(f'[{i:02d}/{len(allsteps):02d}] WP{wp}  (cd {cwd.relative_to(R)})  '+shlex.join(cmd),flush=True)
        if a.execute: subprocess.run(cmd,cwd=cwd,check=True)
    if not a.execute: print('\nDry run only. Re-run with --execute to launch the expensive campaign.')
if __name__=='__main__':main()
