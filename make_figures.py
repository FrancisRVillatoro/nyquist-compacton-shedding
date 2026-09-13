from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'figures_regenerated';OUT.mkdir(exist_ok=True)

# WP1 fixed-phase map
f=pd.read_csv(ROOT/'workflows/wp1_fixed_phase/fixed_phase_floquet_map.csv')
hs=sorted(f.h.unique());ks=sorted(f.kappa.unique())
Z=f.pivot(index='kappa',columns='h',values='rho_F').loc[ks,hs].values
fig,ax=plt.subplots(figsize=(7.4,4.8));im=ax.imshow(Z,origin='lower',aspect='auto');fig.colorbar(im,ax=ax,label=r'$\rho_F$')
ax.set_xticks(range(len(hs)),[f'{x:g}' for x in hs]);ax.set_yticks(range(len(ks)),[f'{x:g}' for x in ks]);ax.set_xlabel('h');ax.set_ylabel(r'$\kappa$')
fig.tight_layout();fig.savefig(OUT/'wp1_fixed_phase_map.png',dpi=180);plt.close(fig)

# WP2 hyperviscosity continuation
q=pd.read_csv(ROOT/'workflows/wp2_control/orbit_preserving_hyperviscosity_spectrum.csv')
fig,ax=plt.subplots(figsize=(7.4,4.8))
casecol='case' if 'case' in q.columns else None
if casecol:
    for name,g in q.groupby(casecol):
        y='rho_F' if 'rho_F' in g.columns else ('rho' if 'rho' in g.columns else None)
        x='epsilon'
        if y: ax.plot(g[x],g[y],marker='o',label=name)
else:
    y='rho_F' if 'rho_F' in q.columns else 'rho'
    ax.plot(q.epsilon,q[y],marker='o')
ax.axhline(1,linewidth=1);ax.set_xlabel(r'$\varepsilon$');ax.set_ylabel(r'$\rho_F$');ax.legend() if casecol else None
fig.tight_layout();fig.savefig(OUT/'wp2_hyperviscosity_continuation.png',dpi=180);plt.close(fig)

# WP3 temporal comparison
q=pd.read_csv(ROOT/'workflows/wp3_time/time_integrator_spectrum_reference_comparison.csv')
# Flexible columns: plot every row by integrator when available.
fig,ax=plt.subplots(figsize=(7.5,4.8))
if {'method','kappa','rho_F'}.issubset(q.columns):
    for name,g in q.groupby('method'): ax.plot(g.kappa,g.rho_F,marker='o',label=name)
elif {'integrator','kappa','rho_F'}.issubset(q.columns):
    for name,g in q.groupby('integrator'): ax.plot(g.kappa,g.rho_F,marker='o',label=name)
else:
    # fallback to archived final image data registry via WP3 summary
    s=pd.read_csv(ROOT/'workflows/wp3_time/WP3_summary.csv'); vals=s[s.quantity.str.contains('rho reference')]
    ax.scatter([.01,.02],vals.value);ax.set_xlabel('h')
ax.set_ylabel(r'$\rho_F$');ax.legend() if len(ax.lines)>0 else None
fig.tight_layout();fig.savefig(OUT/'wp3_temporal_floquet.png',dpi=180);plt.close(fig)

# WP4 stencil comparison
q=pd.read_csv(ROOT/'workflows/wp4_space/WP4_summary.csv')
fig,ax=plt.subplots(figsize=(7.4,4.8));x=np.arange(len(q));ax.bar(x,q.GL4_rhoF_h002_k10);ax.axhline(1,linewidth=1);ax.set_xticks(x,q.label,rotation=15,ha='right');ax.set_ylabel(r'$\rho_F$ (GL4, h=0.02, $\kappa=10$)');fig.tight_layout();fig.savefig(OUT/'wp4_spatial_floquet.png',dpi=180);plt.close(fig)

# WP5 verification margin
q=pd.read_csv(ROOT/'workflows/wp5_verification/verification_margin_matrix.csv')
q=q[q.category!='profile solver'].copy();fig,ax=plt.subplots(figsize=(8,4.8));x=np.arange(len(q));ax.semilogy(x,q.ratio_to_instability_margin,marker='o',linestyle='none');ax.set_xticks(x,[f'{a}\n{b}' for a,b in zip(q.category,q.case)],rotation=20,ha='right');ax.set_ylabel('numerical variation / instability margin');fig.tight_layout();fig.savefig(OUT/'wp5_verification_margins.png',dpi=180);plt.close(fig)

print(f'Regenerated 5 figures in {OUT}')
