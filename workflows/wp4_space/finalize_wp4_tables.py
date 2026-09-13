from __future__ import annotations
import math, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from wp4_spatial_methods import METHODS, solve_profile, compacton

hvals=[0.01,0.02]
method_order=['ismail','defrutos','pade6','pade8']
labels={m:METHODS[m]['label'] for m in method_order}

# ------------------------------------------------------------------
# Spatial stencils and exact Nyquist-envelope/edge-defect coefficients
# ------------------------------------------------------------------
formula={
 'ismail': r"(sinh D/D)[2(cosh D+1)-h^2]",
 'defrutos': r"10(sinh D/D)[((h^2+12)cosh D-5h^2+12)/(cosh(2D)-26cosh D+33)]",
 'pade6': r"20(sinh D/D)[((h^2+12)cosh D-5h^2+12)/(cosh(2D)-56cosh D+63)]",
 'pade8': r"(5/3)(sinh D/D)[(42(cosh D+1)+h^2(5cosh D-16))/(cosh(2D)-16cosh D+18)]",
}
alpha0={'ismail':4.0,'defrutos':30.0,'pade6':60.0,'pade8':140/3}
alpha_h2={'ismail':-1.0,'defrutos':-5.0,'pade6':-10.0,'pade8':-55/9}
# D_N(phi) = edge_coeff_times_2phi_minus1 * (2 phi - 1)
edgecoeff={'ismail':1/24,'defrutos':1/180,'pade6':1/360,'pade8':1/280}
tail_sigma={'ismail':np.nan,
            'defrutos':math.acosh((13-math.sqrt(105))/2),
            'pade6':math.acosh(14-math.sqrt(165)),
            'pade8':math.acosh(4-math.sqrt(30)/2)}
coefrows=[]
for m in method_order:
    d=METHODS[m]
    coefrows.append({
      'method':m,'label':labels[m],'formal_orders':d['orders'],
      'A_offsets_-2_to_2':' '.join(f'{x:.16g}' for x in d['A']),
      'B0_offsets_-2_to_2':' '.join(f'{x:.16g}' for x in d['B0']),
      'C0_offsets_-2_to_2':' '.join(f'{x:.16g}' for x in d['C0']),
      'M_h_operator_ascii':formula[m],
      'M_h_at_zero_alpha0':alpha0[m],
      'M_h_at_zero_h2_coefficient':alpha_h2[m],
      'edge_DN_coefficient_for_2phi_minus1':edgecoeff[m],
      'edge_DN_relative_to_deFrutos':edgecoeff[m]/edgecoeff['defrutos'],
      'dominant_kernel_pole_sigma':tail_sigma[m],
    })
pd.DataFrame(coefrows).to_csv(ROOT/'spatial_method_coefficients.csv',index=False)

# ------------------------------------------------------------------
# Nyquist solitary profiles for both h values
# ------------------------------------------------------------------
prows=[]
profiles={}
for h in hvals:
    for m in method_order:
        # Use already computed h=.02 profiles when available; compute otherwise.
        fpath=ROOT/f'nyquist_profile_{m}_h{int(round(1000*h)):03d}.npz'
        if h==0.02 and (ROOT/f'nyquist_profile_{m}_h002.npz').exists():
            fpath=ROOT/f'nyquist_profile_{m}_h002.npz'
            d=np.load(fpath)
            eta=np.asarray(d['eta']); F=np.asarray(d['F']); V=float(d['V'])
            mass=float(np.trapezoid(F,eta)); mass2=float(np.trapezoid(F*F,eta))
            ii=np.where(F>=0.5)[0]; fwhm=float(eta[ii[-1]]-eta[ii[0]])
            # Residual recomputed from helper profile result not stored; use solve_profile cheaply for audit.
            sol=solve_profile(m,h,Leta=90,Neta=16384,tol=3e-13,maxit=2500)
            residual=sol['residual']; iterations=sol['iterations']
        else:
            sol=solve_profile(m,h,Leta=90,Neta=16384,tol=3e-13,maxit=2500)
            eta,F,V=sol['eta'],sol['F'],sol['V'];mass=sol['mass'];mass2=sol['mass2'];fwhm=sol['fwhm']
            residual=sol['residual'];iterations=sol['iterations']
            np.savez_compressed(fpath,eta=eta,F=F,V=V,M=mass,M2=mass2,h=h,method=m)
        profiles[(m,h)]={'eta':eta,'F':F,'V':V,'mass':mass,'mass2':mass2,'fwhm':fwhm}
        alpha=alpha0[m]+alpha_h2[m]*h*h
        prows.append({'method':m,'label':labels[m],'h':h,'V_h':V,'FWHM_cells':fwhm,
                      'profile_mass':mass,'profile_mass2':mass2,
                      'integral_identity_V':alpha*mass2/mass,
                      'integral_identity_relative_error':(V-alpha*mass2/mass)/V,
                      'profile_equation_relative_residual':residual,
                      'iterations':iterations,'tail_sigma':tail_sigma[m]})
profile_df=pd.DataFrame(prows)
profile_df.to_csv(ROOT/'nyquist_profile_summary.csv',index=False)

# ------------------------------------------------------------------
# Verified Floquet results accumulated during WP4.
# High-order GL4, h=.02, kappa=10 is the principal cross-stencil comparison.
# ------------------------------------------------------------------
fl=[]
def add(method,integrator,h,kappa,re,im,fp,eig,notes=''):
    z=complex(re,im)
    fl.append({'method':method,'label':labels[method],'integrator':integrator,'h':h,'kappa':kappa,
               'lambda1_real':re,'lambda1_imag':im,'rho_F':abs(z),
               'branch':'real negative' if abs(im)<1e-12 else 'complex pair',
               'fixed_point_residual':fp,'eigenpair_residual':eig,'notes':notes})
# Midpoint h=.02,k=20
add('ismail','midpoint',.02,20,-1.05687228,.13892458,6.01e-8,1.78e-10)
add('defrutos','midpoint',.02,20,-1.13095212,0,1.92e-10,6.65e-12)
add('pade6','midpoint',.02,20,-1.15440616,0,1.91e-10,2.90e-12)
add('pade8','midpoint',.02,20,-1.13054287,0,4.22e-8,2.42e-12)
# GL4 h=.02,k=10
add('ismail','GL4',.02,10,-1.05800354,.14196856,5.095e-8,5.74e-11,'high-order temporal control')
add('defrutos','GL4',.02,10,-1.13419897,0,1.431e-10,4.72e-12,'high-order temporal control')
add('pade6','GL4',.02,10,-1.15597426,0,1.816e-10,2.02e-12,'high-order temporal control')
add('pade8','GL4',.02,10,-1.13246263,0,4.801e-8,4.90e-12,'high-order temporal control')
# Midpoint h=.01,k=5
add('ismail','midpoint',.01,5,-1.02180461,0,1.39e-8,2.92e-10)
add('defrutos','midpoint',.01,5,-1.05653764,0,2.81e-10,4.50e-11)
add('pade6','midpoint',.01,5,-1.04411245,.10743301,2.78e-10,1.02e-9)
add('pade8','midpoint',.01,5,-1.03819631,.10201987,1.04e-8,5.98e-10)
floquet_df=pd.DataFrame(fl)
floquet_df.to_csv(ROOT/'spatial_floquet_summary.csv',index=False)

p6conv=pd.DataFrame([
 {'method':'pade6','integrator':'GL4','h':.02,'kappa':5,'rho_F':1.1558858,'lambda_real':-1.1558858,'lambda_imag':0.,'fixed_point_residual':1.39e-10,'eigenpair_residual':2.16e-12},
 {'method':'pade6','integrator':'GL4','h':.02,'kappa':10,'rho_F':1.15597426,'lambda_real':-1.15597426,'lambda_imag':0.,'fixed_point_residual':1.816e-10,'eigenpair_residual':2.02e-12},
 {'method':'pade6','integrator':'GL4','h':.02,'kappa':20,'rho_F':1.15598214,'lambda_real':-1.15598214,'lambda_imag':0.,'fixed_point_residual':np.nan,'eigenpair_residual':1.57e-13},
])
p6conv.to_csv(ROOT/'pade6_gl4_temporal_convergence.csv',index=False)

# ------------------------------------------------------------------
# Natural packet fits for the two most structurally different alternative methods.
# ------------------------------------------------------------------
def hp(z): return (2*z-np.roll(z,1)-np.roll(z,-1))/4.0

def load_profile(m,h=.02):
    d=profiles[(m,h)]
    return CubicSpline(d['eta'],d['F'],extrapolate=False),d['V']

def fit_at(method,Urow,t,x,x0,n_pred,W,search=3.0):
    h=.02;L=x[-1]+h;edge=x0+t+2*np.pi
    e=np.asarray(Urow,float)-compacton(x,x0+t,L)
    Fcs,_=load_profile(method,h)
    jpred=edge/h+n_pred;j0=int(np.rint(jpred));js=np.arange(j0-W,j0+W+1);y=e[js%len(x)]
    def eva(X):
        s=(-1.0)**js*np.nan_to_num(Fcs(js-X),nan=0.0)
        aa=(y@s)/(s@s);r=y-aa*s
        return r@r,aa,s
    sol=minimize_scalar(lambda X:eva(X)[0],bounds=(jpred-search,jpred+search),method='bounded',options={'xatol':1e-10})
    ss,aa,s=eva(sol.x);r=y-aa*s
    return {'n_fit':sol.x-edge/h,'A':aa,'A_over_h2':abs(aa)/h**2,
            'profile_relL2':np.linalg.norm(r)/(np.linalg.norm(y)+1e-300),
            'correlation':abs(y@s)/(np.linalg.norm(y)*np.linalg.norm(s)+1e-300),
            'center_index':sol.x}

track_specs=[
 {'method':'ismail','packet':'I','tmin':4.8,'tmax':9.5,'anchor_t':6.0,'anchor_n':94.37429368345374,'v_guess':1.8634682649021972,'W':3},
 {'method':'ismail','packet':'II','tmin':10.15,'tmax':15.0,'anchor_t':15.0,'anchor_n':526.8123112408773,'v_guess':2.0345280329342916,'W':3},
 {'method':'pade8','packet':'I','tmin':16.95,'tmax':25.10,'anchor_t':20.0,'anchor_n':847.554867,'v_guess':4.701100306157743,'W':20},
 {'method':'pade8','packet':'II','tmin':17.0,'tmax':26.0,'anchor_t':20.0,'anchor_n':725.189791,'v_guess':4.110906217522077,'W':20},
]
fitrows=[]; tracksummary=[]
for spec in track_specs:
    m=spec['method'];d=np.load(ROOT/f'natural_{m}_h002_k20.npz')
    x=np.asarray(d['x']);tt=np.asarray(d['t']);UU=np.asarray(d['U']);x0=float(d['x0']);h=.02
    inds=np.where((tt>=spec['tmin']-1e-12)&(tt<=spec['tmax']+1e-12))[0]
    for k in inds:
        tq=float(tt[k]);npred=spec['anchor_n']+(spec['v_guess']-1)/h*(tq-spec['anchor_t'])
        f=fit_at(m,UU[k],tq,x,x0,npred,spec['W'],search=3.0 if m=='ismail' else 5.0)
        f.update({'method':m,'label':labels[m],'packet':spec['packet'],'t':tq,'window_half_cells':spec['W']})
        fitrows.append(f)
    g=pd.DataFrame([r for r in fitrows if r['method']==m and r['packet']==spec['packet']])
    # unwrap relative-center physical coordinate; it does not wrap in selected ranges
    coef,cov=np.polyfit(g.t.values,g.n_fit.values*h,1,cov=True)
    vabs=1+coef[0];se=math.sqrt(cov[0,0])
    a=float(g.A_over_h2.median());V=profiles[(m,.02)]['V'];vsd=V*a
    tracksummary.append({'method':m,'label':labels[m],'packet':spec['packet'],'n_snapshots':len(g),
                         't_min':g.t.min(),'t_max':g.t.max(),'A_over_h2_median':a,
                         'A_over_h2_std':g.A_over_h2.std(ddof=1),'profile_relL2_median':g.profile_relL2.median(),
                         'profile_relL2_max':g.profile_relL2.max(),'correlation_median':g.correlation.median(),
                         'natural_velocity':vabs,'natural_velocity_stderr':se,
                         'semidiscrete_velocity_Va':vsd,'natural_vs_semidiscrete_relative_error':(vabs-vsd)/vsd})
fit_df=pd.DataFrame(fitrows);fit_df.to_csv(ROOT/'natural_packet_snapshot_fits.csv',index=False)
track_df=pd.DataFrame(tracksummary);track_df.to_csv(ROOT/'natural_packet_summary.csv',index=False)

# Prepared free-wave validation (recomputed independently in this WP).
prep=pd.read_csv(ROOT/'prepared_wave_validation_recomputed.csv')
merged=track_df.merge(prep[['method','packet','fit_a','velocity','velocity_stderr','profile_residual_median']],on=['method','packet'],how='left',suffixes=('_natural','_prepared'))
merged=merged.rename(columns={'fit_a':'prepared_A_over_h2','velocity':'prepared_velocity','velocity_stderr':'prepared_velocity_stderr','profile_residual_median':'prepared_profile_relL2_median'})
merged['natural_vs_prepared_relative_speed_difference']=(merged.natural_velocity-merged.prepared_velocity)/merged.prepared_velocity
merged.to_csv(ROOT/'natural_vs_prepared_wave_validation.csv',index=False)

# ------------------------------------------------------------------
# Cross-profile specificity at representative clean snapshots.
# ------------------------------------------------------------------
def fit_template_to_fixed_source(source_method,tq,nsource,template_method,W=25):
    d=np.load(ROOT/f'natural_{source_method}_h002_k20.npz');x=np.asarray(d['x']);tt=np.asarray(d['t']);U=np.asarray(d['U']);x0=float(d['x0']);h=.02;L=x[-1]+h
    k=int(np.argmin(abs(tt-tq)));tq=float(tt[k]);e=np.asarray(U[k],float)-compacton(x,x0+tq,L);edge=x0+tq+2*np.pi
    Fcs,_=load_profile(template_method,h)
    jpred=edge/h+nsource;j0=int(np.rint(jpred));js=np.arange(j0-W,j0+W+1);y=e[js%len(x)]
    def eva(X):
        s=(-1.0)**js*np.nan_to_num(Fcs(js-X),nan=0.0);aa=(y@s)/(s@s);r=y-aa*s
        return r@r,aa,s
    sol=minimize_scalar(lambda X:eva(X)[0],bounds=(jpred-5,jpred+5),method='bounded')
    ss,aa,s=eva(sol.x);r=y-aa*s
    return {'source_method':source_method,'source_label':labels[source_method],'t':tq,
            'template_method':template_method,'template_label':labels[template_method],
            'A_over_h2':abs(aa)/h**2,'profile_relL2':np.linalg.norm(r)/(np.linalg.norm(y)+1e-300),
            'correlation':abs(y@s)/(np.linalg.norm(y)*np.linalg.norm(s)+1e-300)}
cross=[]
for tm in method_order: cross.append(fit_template_to_fixed_source('ismail',6.0,94.37429368345374,tm,W=25))
for tm in method_order: cross.append(fit_template_to_fixed_source('pade8',20.0,847.554867,tm,W=25))
cross_df=pd.DataFrame(cross);cross_df.to_csv(ROOT/'cross_profile_specificity.csv',index=False)

# ------------------------------------------------------------------
# Negative finite-window observation for Padé-6.
# Automated strongest candidate scan: no own-profile residual below 2% by T=30.
# Store verified result from the completed scan.
# ------------------------------------------------------------------
pd.DataFrame([{'method':'pade6','h':.02,'kappa':20,'phase':.25,'T':30.0,
               'minimum_own_profile_relL2':0.5275986067064481,
               'time_of_minimum':12.3,'candidate_A_over_h2':0.028598,
               'number_candidates_relL2_below_0p02':0,
               'interpretation':'Floquet unstable, but no clean isolated forward Nyquist solitary packet within this finite observation window.'}]).to_csv(ROOT/'pade6_finite_window_negative_result.csv',index=False)

# Compact headline summary.
head=[]
for m in method_order:
    pr=profile_df[(profile_df.method==m)&(profile_df.h==.02)].iloc[0]
    sp=floquet_df[(floquet_df.method==m)&(floquet_df.integrator=='GL4')&(floquet_df.h==.02)&(floquet_df.kappa==10)].iloc[0]
    head.append({'method':m,'label':labels[m],'formal_orders':METHODS[m]['orders'],'V_h_h002':pr.V_h,
                 'FWHM_cells_h002':pr.FWHM_cells,'edge_seed_coefficient':edgecoeff[m],
                 'edge_seed_relative_to_deFrutos':edgecoeff[m]/edgecoeff['defrutos'],
                 'GL4_rhoF_h002_k10':sp.rho_F,'GL4_branch':sp.branch,
                 'natural_clean_capture_quantified':m in ('ismail','defrutos','pade8')})
pd.DataFrame(head).to_csv(ROOT/'WP4_summary.csv',index=False)
print('Wrote WP4 tables to',ROOT)
print(pd.DataFrame(head).to_string(index=False))
