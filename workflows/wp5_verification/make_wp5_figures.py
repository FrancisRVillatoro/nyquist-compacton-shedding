from pathlib import Path
import pandas as pd,numpy as np,matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
u=pd.read_csv(OUT/'packet_uncertainty_summary_with_calibration.csv');g=u[u.method.isin(['midpoint','gl4'])].copy();labels=g.dataset.tolist();y=(g.delta_v/g.predicted_velocity).values;lo=(g.delta_total_95_low/g.predicted_velocity).values;hi=(g.delta_total_95_high/g.predicted_velocity).values;yerr=np.vstack([y-lo,hi-y])
fig,ax=plt.subplots(figsize=(8,5));x=np.arange(len(g));ax.errorbar(x,y,yerr=yerr,fmt='o',capsize=4);ax.axhline(0,linewidth=1);ax.set_xticks(x);ax.set_xticklabels(labels,rotation=20,ha='right');ax.set_ylabel('relative velocity discrepancy');fig.tight_layout();fig.savefig(OUT/'wp5_velocity_validation_uncertainty.png',dpi=180);plt.close(fig)
w=pd.read_csv(OUT/'wp4_amplitude_matched_natural_prepared.csv');fig,ax=plt.subplots(figsize=(7.5,4.8));x=np.arange(len(w));ax.scatter(x,w.natural_vs_rescaled_prepared_relative_speed_difference);ax.axhline(0,linewidth=1);ax.set_xticks(x);ax.set_xticklabels([f'{m} {p}' for m,p in zip(w.method,w.packet)]);ax.set_ylabel('relative speed difference after amplitude matching');fig.tight_layout();fig.savefig(OUT/'wp5_amplitude_matched_spatial_validation.png',dpi=180);plt.close(fig)
m=pd.read_csv(OUT/'verification_margin_matrix.csv');q=m[m.category!='profile solver'].copy();fig,ax=plt.subplots(figsize=(8,5));x=np.arange(len(q));ax.semilogy(x,q.ratio_to_instability_margin,marker='o',linestyle='none');ax.set_xticks(x);ax.set_xticklabels([f'{c}\n{case}' for c,case in zip(q.category,q.case)],rotation=20,ha='right');ax.set_ylabel('numerical variation / instability margin');fig.tight_layout();fig.savefig(OUT/'wp5_floquet_verification_margins.png',dpi=180);plt.close(fig)
n=pd.read_csv(OUT/'floquet_nonnormality_conditioning.csv');fig,ax=plt.subplots(figsize=(7.2,4.8));x=np.arange(len(n));ax.bar(x,n.eigenvalue_condition_number);ax.set_xticks(x);ax.set_xticklabels([f'{r.method}\nh={r.h:g}, k={r.kappa:g}' for _,r in n.iterrows()]);ax.set_ylabel('eigenvalue condition number');fig.tight_layout();fig.savefig(OUT/'wp5_floquet_condition_numbers.png',dpi=180);plt.close(fig)
h=pd.read_csv(OUT/'hyperviscosity_threshold_fit_sensitivity.csv');fig,ax=plt.subplots(figsize=(7.5,4.8));offset=0
for case,gc in h.groupby('case'):
 ax.scatter(np.arange(len(gc)),gc.epsilon_c_fit,label=case)
ax.set_xlabel('local fit variant');ax.set_ylabel('estimated critical epsilon');ax.legend();fig.tight_layout();fig.savefig(OUT/'wp5_threshold_fit_sensitivity.png',dpi=180);plt.close(fig)
# fit-window sensitivity, use median residual and velocity span
s=pd.read_csv(OUT/'wp4_window_audit_summary.csv');fig,ax=plt.subplots(figsize=(8,5))
for (m,p),gc in s.groupby(['method','packet']):ax.plot(gc.W,gc.profile_relL2_median,marker='o',label=f'{m} {p}')
ax.set_yscale('log');ax.set_xlabel('fit half-window (cells)');ax.set_ylabel('median profile residual');ax.legend();fig.tight_layout();fig.savefig(OUT/'wp5_fit_window_robustness.png',dpi=180);plt.close(fig)
print('done')
