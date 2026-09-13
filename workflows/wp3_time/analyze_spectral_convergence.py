from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent

d=pd.read_csv(ROOT/'time_integrator_spectrum_cases.csv')
# time-refined GL4 k=80 as reference for each h
rows=[]; prof=[]
for h in sorted(d.h.unique()):
    refrow=d[(np.isclose(d.h,h))&(d.method=='gl4')&(d.kappa==80)].iloc[0]
    rho_ref=refrow.rho_F
    lam_ref=complex(refrow.lambda1_real,abs(refrow.lambda1_imag))
    pref=ROOT/f'compacton_gl4_h{h:.3f}_k80.npz'
    Uref=np.load(pref)['U']
    x=np.load(pref)['x']; x0=float(np.load(pref)['x0']);N=len(Uref)
    for _,r in d[np.isclose(d.h,h)].iterrows():
        lam=complex(r.lambda1_real,abs(r.lambda1_imag))
        errrho=abs(r.rho_F-rho_ref)
        errlam=abs(lam-lam_ref)
        p=ROOT/f'compacton_{r.method}_h{h:.3f}_k{int(r.kappa)}.npz'
        U=np.load(p)['U']
        du=U-Uref
        # edge windows 50 cells each
        mR=int(np.rint((x0+2*np.pi)/h))%N;mL=int(np.rint((x0-2*np.pi)/h))%N
        idx=np.unique(np.r_[(mR+np.arange(-50,51))%N,(mL+np.arange(-50,51))%N])
        hp=lambda z:(2*z-np.roll(z,1)-np.roll(z,-1))/4
        rows.append({'h':h,'method':r.method,'kappa':int(r.kappa),'dt':r['dt'],
                     'rho_F':r.rho_F,'rho_reference_gl4_k80':rho_ref,
                     'rho_abs_error':errrho,'lambda_complex_error':errlam,
                     'compacton_Linf_error':np.max(np.abs(du)),
                     'compacton_relL2_error':np.linalg.norm(du)/np.linalg.norm(Uref),
                     'edge_L2_error_over_h2':np.linalg.norm(du[idx])/h**2,
                     'edge_HP_peak_error_over_h2':np.max(np.abs(hp(du)[idx]))/h**2,
                     'fixed_point_residual':r.fixed_point_residual,
                     'lambda1_residual':r.lambda1_residual,
                     'mode_type':r.mode_type})
    # empirical convergence slopes using selected asymptotic ranges
    for method in ['midpoint','gl4']:
        q=pd.DataFrame(rows)
        g=q[(np.isclose(q.h,h))&(q.method==method)&(q.rho_abs_error>0)].sort_values('kappa')
        if method=='midpoint': fit=g[g.kappa.isin([20,40,80])]
        else:
            # include values above numerical floor only
            fit=g[(g.kappa<=20)&(g.rho_abs_error>5e-7)]
        if len(fit)>=2:
            slope=np.polyfit(np.log(fit.kappa),np.log(fit.rho_abs_error),1)[0]
            prof.append({'h':h,'method':method,'quantity':'rho_F','observed_order':-slope,
                         'kappas_used':','.join(map(str,fit.kappa.astype(int))),
                         'reference':'GL4 kappa=80'})
        # profile error slope
        fitp=g[(g.kappa<=40)&(g.compacton_Linf_error>1e-12)]
        if method=='midpoint': fitp=fitp[fitp.kappa.isin([10,20,40])]
        else: fitp=fitp[fitp.kappa.isin([5,10,20])]
        if len(fitp)>=2:
            slope=np.polyfit(np.log(fitp.kappa),np.log(fitp.compacton_Linf_error),1)[0]
            prof.append({'h':h,'method':method,'quantity':'compacton_Linf','observed_order':-slope,
                         'kappas_used':','.join(map(str,fitp.kappa.astype(int))),
                         'reference':'GL4 kappa=80'})

err=pd.DataFrame(rows).sort_values(['h','method','kappa'])
err.to_csv(ROOT/'time_integrator_convergence_errors.csv',index=False)
pd.DataFrame(prof).to_csv(ROOT/'time_integrator_observed_orders.csv',index=False)

# Plots
plt.figure(figsize=(7.5,5))
for h in sorted(err.h.unique()):
 for method,marker in [('midpoint','o'),('gl4','s')]:
  g=err[(np.isclose(err.h,h))&(err.method==method)].sort_values('kappa')
  plt.loglog(g.kappa,g.rho_abs_error,marker=marker,label=f'{method}, h={h:g}')
plt.xlabel(r'$\kappa=h/\Delta t$')
plt.ylabel(r'$|\rho_F-\rho_{F,\mathrm{ref}}|$')
plt.legend(fontsize=8);plt.tight_layout();plt.savefig(ROOT/'wp3_floquet_time_convergence.png',dpi=200);plt.close()

plt.figure(figsize=(7.5,5))
for h in sorted(err.h.unique()):
 for method,marker in [('midpoint','o'),('gl4','s')]:
  g=err[(np.isclose(err.h,h))&(err.method==method)].sort_values('kappa')
  plt.loglog(g.kappa,g.compacton_Linf_error,marker=marker,label=f'{method}, h={h:g}')
plt.xlabel(r'$\kappa=h/\Delta t$')
plt.ylabel(r'$\|U_*-U_{*,\mathrm{ref}}\|_\infty$')
plt.legend(fontsize=8);plt.tight_layout();plt.savefig(ROOT/'wp3_compacton_time_convergence.png',dpi=200);plt.close()

plt.figure(figsize=(7.5,5))
for h in sorted(d.h.unique()):
 for method,marker in [('midpoint','o'),('gl4','s')]:
  g=d[(np.isclose(d.h,h))&(d.method==method)].sort_values('kappa')
  plt.plot(g.kappa,g.rho_F,marker=marker,label=f'{method}, h={h:g}')
plt.xlabel(r'$\kappa=h/\Delta t$');plt.ylabel(r'$\rho_F$');plt.legend(fontsize=8)
plt.tight_layout();plt.savefig(ROOT/'wp3_floquet_spectra_vs_kappa.png',dpi=200);plt.close()

print(err[['h','method','kappa','rho_F','rho_abs_error','compacton_Linf_error','edge_HP_peak_error_over_h2','mode_type']].to_string(index=False))
print('\nOrders:')
print(pd.DataFrame(prof).to_string(index=False))
