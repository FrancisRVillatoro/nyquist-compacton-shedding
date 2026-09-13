from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent
s=pd.read_csv(OUT/'wp7_refinement_summary.csv');g=s[s.status=='clean'].sort_values('h')
# Reference laws anchored at geometric-mean h to median normalized constant.
h=g.h.values
# amplitude
fig,ax=plt.subplots(figsize=(6.8,4.8));ax.loglog(h,g.A_median,'o-',label='first clean natural packet')
c=np.exp(np.mean(np.log(g.A_median/h**2)));hr=np.geomspace(h.min(),h.max(),200);ax.loglog(hr,c*hr**2,'--',label=r'$\propto h^2$')
ax.set_xlabel(r'$h$');ax.set_ylabel(r'physical packet amplitude $A$');ax.legend();fig.tight_layout();fig.savefig(OUT/'wp7_amplitude_scaling.png',dpi=200);plt.close(fig)
# normalized amplitude
fig,ax=plt.subplots(figsize=(6.8,4.8));ax.semilogx(h,g.A_over_h2_median,'o-');ax.axhline(g.A_over_h2_median.median(),linestyle='--',linewidth=1,label='median')
ax.set_xlabel(r'$h$');ax.set_ylabel(r'$A/h^2$');ax.legend();fig.tight_layout();fig.savefig(OUT/'wp7_normalized_amplitude.png',dpi=200);plt.close(fig)
# width
fig,ax=plt.subplots(figsize=(6.8,4.8));ax.loglog(h,g.FWHM_x,'o-',label='physical FWHM');cw=np.median(g.FWHM_x/h);ax.loglog(hr,cw*hr,'--',label=r'$\propto h$');ax.set_xlabel(r'$h$');ax.set_ylabel('physical FWHM');ax.legend();fig.tight_layout();fig.savefig(OUT/'wp7_width_scaling.png',dpi=200);plt.close(fig)
# norms
fig,ax=plt.subplots(figsize=(6.8,4.8));ax.loglog(h,g.observed_L2_local,'o-',label=r'local $L^2$ norm');c2=np.exp(np.mean(np.log(g.observed_L2_local/h**2.5)));ax.loglog(hr,c2*hr**2.5,'--',label=r'$\propto h^{5/2}$');ax.set_xlabel(r'$h$');ax.set_ylabel(r'local $L^2$ norm');ax.legend();fig.tight_layout();fig.savefig(OUT/'wp7_L2_scaling.png',dpi=200);plt.close(fig)
# normalized all three to expected powers and median
fig,ax=plt.subplots(figsize=(7.2,4.8))
qs=[('A_median',2.0,r'$A/h^2$'),('observed_L1_local',3.0,r'$\|u_p\|_1/h^3$'),('observed_L2_local',2.5,r'$\|u_p\|_2/h^{5/2}$')]
for col,powr,label in qs:
 y=g[col].values/h**powr;y=y/np.median(y);ax.semilogx(h,y,'o-',label=label+' / median')
ax.axhline(1,linewidth=1,linestyle='--');ax.set_xlabel(r'$h$');ax.set_ylabel('normalized scaled quantity');ax.legend();fig.tight_layout();fig.savefig(OUT/'wp7_scaled_norms_collapse.png',dpi=200);plt.close(fig)
# latency
fig,ax=plt.subplots(figsize=(6.8,4.8));ax.semilogx(h,g.first_clean_cell,'o-');ax.set_xlabel(r'$h$');ax.set_ylabel('first persistent clean-packet cell crossing');fig.tight_layout();fig.savefig(OUT/'wp7_first_clean_latency.png',dpi=200);plt.close(fig)
