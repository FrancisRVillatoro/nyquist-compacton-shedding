from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from wp4_spatial_methods import METHODS,compacton
methods=['ismail','defrutos','pade6','pade8']
labels={m:METHODS[m]['label'] for m in methods}

# 1. Profiles
plt.figure(figsize=(7.8,5.0))
for m in methods:
    d=np.load(ROOT/f'nyquist_profile_{m}_h002.npz')
    eta=np.asarray(d['eta']);F=np.asarray(d['F'])
    q=np.abs(eta)<=13
    plt.plot(eta[q],F[q],label=labels[m])
plt.xlabel(r'cell coordinate $\eta$')
plt.ylabel(r'normalized envelope $F_h(\eta)$')
plt.legend()
plt.tight_layout()
plt.savefig(ROOT/'wp4_nyquist_profiles.png',dpi=200)
plt.close()

prof=pd.read_csv(ROOT/'nyquist_profile_summary.csv')
p2=prof[np.isclose(prof.h,.02)].set_index('method').loc[methods]
x=np.arange(len(methods))
plt.figure(figsize=(7.5,4.8))
plt.bar(x,p2.V_h.values)
plt.xticks(x,[labels[m] for m in methods],rotation=15)
plt.ylabel(r'amplitude--velocity coefficient $V_h$')
plt.tight_layout()
plt.savefig(ROOT/'wp4_velocity_coefficients.png',dpi=200)
plt.close()

plt.figure(figsize=(7.5,4.8))
plt.bar(x,p2.FWHM_cells.values)
plt.xticks(x,[labels[m] for m in methods],rotation=15)
plt.ylabel('FWHM of envelope (grid cells)')
plt.tight_layout()
plt.savefig(ROOT/'wp4_profile_widths.png',dpi=200)
plt.close()

coef=pd.read_csv(ROOT/'spatial_method_coefficients.csv').set_index('method').loc[methods]
plt.figure(figsize=(7.5,4.8))
plt.bar(x,coef.edge_DN_relative_to_deFrutos.values)
plt.xticks(x,[labels[m] for m in methods],rotation=15)
plt.ylabel('leading edge-seed coefficient / de Frutos value')
plt.tight_layout()
plt.savefig(ROOT/'wp4_edge_seed_coefficients.png',dpi=200)
plt.close()

# 2. Floquet comparison
fl=pd.read_csv(ROOT/'spatial_floquet_summary.csv')
for integrator,h,kappa,fn in [
    ('GL4',.02,10,'wp4_spatial_floquet_GL4.png'),
    ('midpoint',.02,20,'wp4_spatial_floquet_midpoint.png')]:
    g=fl[(fl.integrator==integrator)&np.isclose(fl.h,h)&(fl.kappa==kappa)].set_index('method').loc[methods]
    plt.figure(figsize=(7.5,4.8))
    plt.plot(x,g.rho_F.values,marker='o')
    plt.axhline(1.0,linestyle='--',linewidth=1)
    plt.xticks(x,[labels[m] for m in methods],rotation=15)
    plt.ylabel(r'leading Floquet spectral radius $\rho_F$')
    plt.tight_layout()
    plt.savefig(ROOT/fn,dpi=200)
    plt.close()

# 3. Spacetime diagrams of |high-pass| in the moving edge frame
def hp(z):return (2*z-np.roll(z,1)-np.roll(z,-1))/4.0
spacetime_specs={
 'ismail':(3.5,15.0,650),
 'defrutos':(0.0,15.0,900),
 'pade6':(0.0,30.0,2000),
 'pade8':(15.0,26.0,2000),
}
for m,(ta,tb,nmax) in spacetime_specs.items():
    d=np.load(ROOT/f'natural_{m}_h002_k20.npz')
    xx=np.asarray(d['x']);tt=np.asarray(d['t']);UU=np.asarray(d['U']);x0=float(d['x0']);h=float(d['h']);L=xx[-1]+h
    ks=np.where((tt>=ta-1e-12)&(tt<=tb+1e-12))[0]
    nn=np.arange(0,nmax+1)
    Z=np.empty((len(ks),len(nn)),float)
    for ii,k in enumerate(ks):
        tq=float(tt[k]);e=np.asarray(UU[k],float)-compacton(xx,x0+tq,L);z=np.abs(hp(e))/h**2
        edge=x0+tq+2*np.pi;me=int(np.rint(edge/h))%len(xx)
        Z[ii]=z[(me+nn)%len(xx)]
    vmax=float(np.percentile(Z,99.5))
    plt.figure(figsize=(8.3,5.0))
    plt.imshow(np.minimum(Z,vmax),origin='lower',aspect='auto',extent=[0,nmax,tt[ks[0]],tt[ks[-1]]])
    plt.colorbar(label=r'$|H(U-u_c)|/h^2$')
    plt.xlabel('grid cells ahead of the right compacton edge')
    plt.ylabel('time')
    plt.tight_layout()
    plt.savefig(ROOT/f'wp4_spacetime_{m}.png',dpi=200)
    plt.close()

# 4. Representative profile captures
def plot_capture(method,packet,tq,fn,xlim):
    d=np.load(ROOT/f'natural_{method}_h002_k20.npz');xx=np.asarray(d['x']);tt=np.asarray(d['t']);UU=np.asarray(d['U']);x0=float(d['x0']);h=float(d['h']);L=xx[-1]+h
    fits=pd.read_csv(ROOT/'natural_packet_snapshot_fits.csv')
    q=fits[(fits.method==method)&(fits.packet==packet)&np.isclose(fits.t,tq)].iloc[0]
    k=int(np.argmin(abs(tt-tq)));e=np.asarray(UU[k],float)-compacton(xx,x0+float(tt[k]),L)
    X=float(q.center_index);W=25 if method=='pade8' else 12
    j0=int(np.rint(X));js=np.arange(j0-W,j0+W+1);A=float(q.A)
    y=e[js%len(xx)]*((-1.0)**js)/A
    pd0=np.load(ROOT/f'nyquist_profile_{method}_h002.npz');Fcs=CubicSpline(pd0['eta'],pd0['F'],extrapolate=False)
    eta=js-X;F=np.nan_to_num(Fcs(eta),nan=0.0)
    plt.figure(figsize=(7.7,4.8))
    plt.plot(eta,y,marker='o',markersize=3,label='natural emitted packet')
    plt.plot(eta,F,linewidth=2,label=f'{labels[method]} $F_h$')
    plt.xlim(*xlim)
    plt.xlabel(r'$j-X$')
    plt.ylabel('demodulated profile / fitted amplitude')
    plt.legend()
    plt.tight_layout()
    plt.savefig(ROOT/fn,dpi=200)
    plt.close()
plot_capture('ismail','I',6.0,'wp4_profile_capture_ismail.png',(-8,8))
plot_capture('pade8','I',20.0,'wp4_profile_capture_pade8.png',(-12,12))

# 5. Natural vs prepared velocities
nv=pd.read_csv(ROOT/'natural_vs_prepared_wave_validation.csv')
keys=[f"{labels[r.method]} {r.packet}" for r in nv.itertuples()]
xx=np.arange(len(nv))
plt.figure(figsize=(8.2,4.9))
plt.plot(xx,nv.natural_velocity,marker='o',linestyle='none',label='natural emitted packet')
plt.plot(xx,nv.prepared_velocity,marker='x',linestyle='none',label='independently prepared wave')
plt.plot(xx,nv.semidiscrete_velocity_Va,marker='s',linestyle='none',label=r'semidiscrete $V_h A/h^2$')
plt.xticks(xx,keys,rotation=18)
plt.ylabel('absolute packet velocity')
plt.legend()
plt.tight_layout()
plt.savefig(ROOT/'wp4_natural_prepared_velocity.png',dpi=200)
plt.close()

# 6. Cross-profile specificity
cr=pd.read_csv(ROOT/'cross_profile_specificity.csv')
plt.figure(figsize=(8.0,4.9))
for source,marker in [('ismail','o'),('pade8','s')]:
    g=cr[cr.source_method==source].set_index('template_method').loc[methods]
    plt.semilogy(x,g.profile_relL2.values,marker=marker,label=f"source: {labels[source]}")
plt.xticks(x,[labels[m] for m in methods],rotation=15)
plt.ylabel('relative profile-fit residual')
plt.legend()
plt.tight_layout()
plt.savefig(ROOT/'wp4_cross_profile_specificity.png',dpi=200)
plt.close()
print('Created WP4 figures')
