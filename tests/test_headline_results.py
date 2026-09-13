from pathlib import Path
import json, pandas as pd, numpy as np
R=Path(__file__).resolve().parents[1]
E=json.loads((R/'metadata/HEADLINE_RESULTS.json').read_text())

def test_wp1_map():
    q=pd.read_csv(R/'workflows/wp1_fixed_phase/WP1_block2_summary.csv').iloc[0]
    assert int(q.fixed_map_n_unstable)==E['wp1']['fixed_phase_unstable']==25
    assert np.isclose(q.fixed_map_rho_min,E['wp1']['rho_min'],rtol=0,atol=1e-12)
    assert np.isclose(q.fixed_map_rho_max,E['wp1']['rho_max'],rtol=0,atol=1e-12)

def test_wp2_suppression():
    q=pd.read_csv(R/'workflows/wp2_control/WP2_causal_suppression_summary.csv').set_index('case')
    assert q.loc['h001_k5','suppression_at_critical']>700
    assert q.loc['h002_k20','suppression_at_critical']>1000

def test_wp3_reference_and_orders():
    q=pd.read_csv(R/'workflows/wp3_time/WP3_summary.csv').set_index('quantity').value
    assert np.isclose(q['rho reference h=0.01'],E['wp3']['rho_sd_h001'],atol=1e-12)
    assert 1.9<q['midpoint observed velocity order']<2.1
    assert 3.8<q['GL4 observed velocity order']<4.1
    assert 5.7<q['GL6 observed velocity order']<6.2

def test_wp4_all_stencils_unstable():
    q=pd.read_csv(R/'workflows/wp4_space/WP4_summary.csv')
    assert len(q)==4
    assert (q.GL4_rhoF_h002_k10>1).all()

def test_wp5_amplitude_matched():
    q=pd.read_csv(R/'workflows/wp5_verification/wp4_amplitude_matched_natural_prepared.csv')
    col='natural_vs_rescaled_prepared_relative_speed_difference'
    assert q[col].abs().max()<3e-7


def test_wp7_refinement():
    q=pd.read_csv(R/'workflows/wp7_refinement/wp7_refinement_summary.csv')
    g=q[q.status=='clean']
    assert len(g)==E['wp7']['n_clean_levels']==7
    assert np.isclose(g.A_over_h2_median.min(),E['wp7']['A_over_h2_min'],atol=1e-12)
    assert np.isclose(g.A_over_h2_median.max(),E['wp7']['A_over_h2_max'],atol=1e-12)
    f=pd.read_csv(R/'workflows/wp7_refinement/wp7_scaling_fits.csv').set_index('quantity')
    assert np.isclose(f.loc['FWHM_x','observed_exponent'],E['wp7']['width_exponent'],atol=1e-12)


def test_wp8_strict_closure():
    q=pd.read_csv(R/'workflows/wp8_closure/derived/wp8_hyperviscosity_spectrum_strict.csv')
    assert q.residual.max()<1e-12
    assert q.iloc[3].rho>1 and q.iloc[5].rho<1
    f=pd.read_csv(R/'workflows/wp8_closure/derived/wp8_onset_scaling_fit.csv')
    y=f.onset_cell_0p005.to_numpy(float); yp=f.fit_onset.to_numpy(float)
    r2=1-np.sum((y-yp)**2)/np.sum((y-y.mean())**2)
    assert np.isclose(r2,E['wp8']['onset_fit_R2'],atol=1e-12)
    root=pd.read_csv(R/'workflows/wp8_closure/derived/wp8_hyperviscosity_strict_root.csv').iloc[0]
    assert np.isclose(root.epsilon_c_strict,E['wp8']['epsilon_c_strict'],atol=1e-12)
