from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent

spec=pd.read_csv(ROOT/'time_integrator_spectrum_cases.csv')
g6s=pd.read_csv(ROOT/'gl6_reference_spectrum.csv')
fw=pd.read_csv(ROOT/'free_wave_time_integrator_benchmark_combined.csv')
g6w=pd.read_csv(ROOT/'free_wave_gl6_benchmark.csv')
cost=pd.read_csv(ROOT/'time_integrator_cost_benchmark.csv').drop_duplicates(['method_implementation','kappa'],keep='last')
tan=pd.read_csv(ROOT/'tangent_directional_verification.csv')
nat=pd.read_csv(ROOT/'gl4_natural_packet_velocity_summary.csv')

# ------------------------------------------------------------------
# 1. Independent high-order spectral reference and errors.
# ------------------------------------------------------------------
refs=[]
for h in [.01,.02]:
    vals=list(spec[(np.isclose(spec.h,h))&(spec.method=='gl4')&(spec.kappa.isin([40,80]))].rho_F)
    vals+=list(g6s[(np.isclose(g6s.h,h))&(g6s.kappa.isin([10,20]))].rho_F)
    refs.append({'h':h,'rho_reference':float(np.median(vals)),
                 'reference_min':float(np.min(vals)),'reference_max':float(np.max(vals)),
                 'reference_half_range':float((np.max(vals)-np.min(vals))/2),
                 'reference_std':float(np.std(vals,ddof=1))})
refs=pd.DataFrame(refs);refs.to_csv(ROOT/'time_integrator_semidiscrete_reference.csv',index=False)

spall=pd.concat([
    spec[['h','method','kappa','dt','rho_F','lambda1_real','lambda1_imag','lambda1_residual','fixed_point_residual','mode_type']].copy(),
    g6s[['h','method','kappa','dt','rho_F','lambda1_real','lambda1_imag','lambda1_residual','fixed_point_residual','mode_type']].copy()
],ignore_index=True)
spall=spall.merge(refs[['h','rho_reference','reference_half_range']],on='h')
spall['rho_abs_error']=abs(spall.rho_F-spall.rho_reference)
spall['rho_rel_error']=spall.rho_abs_error/spall.rho_reference
spall.sort_values(['h','method','kappa']).to_csv(ROOT/'time_integrator_spectrum_reference_comparison.csv',index=False)

# Principal comparisons.
principal=[]
for h,k in [(.01,5),(.02,20)]:
    for method in ['midpoint','gl4','gl6']:
        q=spall[(np.isclose(spall.h,h))&(spall.method==method)&(spall.kappa==k)].iloc[0]
        principal.append({'h':h,'kappa':k,'method':method,'rho_F':q.rho_F,
                          'lambda1_real':q.lambda1_real,'lambda1_imag':q.lambda1_imag,
                          'mode_type':q.mode_type,'rho_abs_error_to_reference':q.rho_abs_error,
                          'lambda_residual':q.lambda1_residual})
pd.DataFrame(principal).to_csv(ROOT/'principal_cross_integrator_spectra.csv',index=False)

# ------------------------------------------------------------------
# 2. Velocity order and correction laws.
# ------------------------------------------------------------------
FW=pd.concat([fw,g6w],ignore_index=True)
FW=FW.sort_values('runtime_s').drop_duplicates(['h','method','kappa','input_a'],keep='first')
FW.to_csv(ROOT/'free_wave_three_integrator_benchmark.csv',index=False)
orders=[]
for h in [.01,.02]:
    ranges={'midpoint':[5,8,10,20,40,80], 'gl4':[1,2,3,4,5,8,10,20], 'gl6':[1,2,3,4,5,8]}
    for method,ks in ranges.items():
        q=FW[(np.isclose(FW.h,h))&(FW.method==method)&np.isclose(FW.input_a,.1)&FW.kappa.isin(ks)]
        q=q.drop_duplicates('kappa').sort_values('kappa')
        q=q[abs(q.relative_velocity_error)>2e-10]
        slope,intercept=np.polyfit(np.log(q.kappa),np.log(abs(q.relative_velocity_error)),1)
        pred=intercept+slope*np.log(q.kappa)
        ssr=np.sum((np.log(abs(q.relative_velocity_error))-pred)**2)
        sst=np.sum((np.log(abs(q.relative_velocity_error))-np.log(abs(q.relative_velocity_error)).mean())**2)
        orders.append({'h':h,'method':method,'observed_order':float(-slope),'R2_loglog':float(1-ssr/sst),
                       'kappa_min':int(q.kappa.min()),'kappa_max':int(q.kappa.max()),'n':len(q)})
orders=pd.DataFrame(orders);orders.to_csv(ROOT/'free_wave_observed_temporal_orders.csv',index=False)

# Stable correction fits. Higher coefficients are calibration coefficients, not claimed universal beyond ranges.
fitdefs={
 'midpoint':([2,4,6],.0361),
 'gl4':([4,6,8],.0501),
 'gl6':([6,8],.0501),
}
fits=[]
for method,(powers,chimax) in fitdefs.items():
    q=FW[(FW.method==method)&(FW.chi<=chimax)]
    X=np.column_stack([q.chi.values**p for p in powers]);y=q.relative_velocity_error.values
    co=np.linalg.lstsq(X,y,rcond=None)[0];pr=X@co
    for p,c in zip(powers,co):
        fits.append({'method':method,'fit_scope':'combined_h','h':np.nan,'power':p,'coefficient':c,
                     'chi_max':q.chi.max(),'n_fit':len(q),'fit_RMSE':np.sqrt(np.mean((pr-y)**2)),
                     'fit_max_abs_error':np.max(abs(pr-y))})
    for h in [.01,.02]:
        z=q[np.isclose(q.h,h)];Xh=np.column_stack([z.chi.values**p for p in powers]);ch=np.linalg.lstsq(Xh,z.relative_velocity_error.values,rcond=None)[0]
        ph=Xh@ch
        for p,c in zip(powers,ch):
            fits.append({'method':method,'fit_scope':'per_h','h':h,'power':p,'coefficient':c,
                         'chi_max':z.chi.max(),'n_fit':len(z),'fit_RMSE':np.sqrt(np.mean((ph-z.relative_velocity_error.values)**2)),
                         'fit_max_abs_error':np.max(abs(ph-z.relative_velocity_error.values))})
fits=pd.DataFrame(fits);fits.to_csv(ROOT/'velocity_correction_fits_three_integrators.csv',index=False)

# h collapse for GL6.
m=g6w.pivot_table(index=['kappa','input_a'],columns='h',values='relative_velocity_error').dropna()
m['abs_h_difference']=abs(m[.01]-m[.02]);m.reset_index().to_csv(ROOT/'free_wave_gl6_h_collapse.csv',index=False)

# ------------------------------------------------------------------
# 3. Independent packet validation of the calibrated corrections.
# ------------------------------------------------------------------
def coeff(method):
    q=fits[(fits.method==method)&(fits.fit_scope=='combined_h')]
    return {int(r.power):r.coefficient for _,r in q.iterrows()}
cm=coeff('midpoint');cg=coeff('gl4')
pack=[]
# Archived midpoint packets.
for label,h,k,a,v,V,oldv in [
    ('archived midpoint packet, kappa=5',.01,5,.106388,2.186750,20.66153231806912,2.189291),
    ('archived midpoint packet, kappa=20',.02,20,.102562,2.118340,20.660521654243198,2.118484)]:
    chi=a/k;new=V*a*(1+sum(c*chi**p for p,c in cm.items()))
    pack.append({'dataset':label,'method':'midpoint','h':h,'kappa':k,'A_over_h2':a,'chi':chi,
                 'measured_velocity':v,'baseline_prediction':oldv,'baseline_relative_error':(v-oldv)/oldv,
                 'revised_prediction':new,'revised_relative_error':(v-new)/new})
# Natural GL4 packets, baseline = semidiscrete prediction.
for _,r in nat.iterrows():
    a=r.A_over_h2_median;k=5;chi=a/k;base=r.semidiscrete_velocity_prediction
    new=base*(1+sum(c*chi**p for p,c in cg.items()))
    pack.append({'dataset':r.track,'method':'gl4','h':.01,'kappa':k,'A_over_h2':a,'chi':chi,
                 'measured_velocity':r.absolute_velocity_fit,'baseline_prediction':base,
                 'baseline_relative_error':(r.absolute_velocity_fit-base)/base,
                 'revised_prediction':new,'revised_relative_error':(r.absolute_velocity_fit-new)/new,
                 'profile_residual_median':r.profile_residual_median})
pack=pd.DataFrame(pack);pack.to_csv(ROOT/'independent_packet_velocity_validation.csv',index=False)

# Fit-sensitivity uncertainty of independent predictions.
models=[]
orig=pd.read_csv(ROOT/'free_wave_time_integrator_benchmark_combined.csv')
for method,powers,limits in [('midpoint',[2,4,6],[.02,.025,.03,.036]),('gl4',[4,6,8],[.025,.03,.04,.05])]:
    for hs in ['all',.01,.02]:
        g=orig[orig.method==method]
        if hs!='all':g=g[np.isclose(g.h,hs)]
        for cmx in limits:
            q=g[g.chi<=cmx]
            if len(q)<len(powers)+2:continue
            X=np.column_stack([q.chi.values**p for p in powers]);c=np.linalg.lstsq(X,q.relative_velocity_error.values,rcond=None)[0]
            models.append((method,hs,cmx,powers,c))
unc=[]
for _,r in pack.iterrows():
    V=(20.66153231806912 if np.isclose(r.h,.01) else 20.660521654243198)
    vals=[]
    for method,hs,cmx,powers,c in models:
        if method!=r.method:continue
        vals.append(V*r.A_over_h2*(1+sum(cc*r.chi**p for p,cc in zip(powers,c))))
    unc.append({'dataset':r.dataset,'n_fit_variants':len(vals),'prediction_mean':np.mean(vals),'prediction_std':np.std(vals,ddof=1),
                'prediction_min':np.min(vals),'prediction_max':np.max(vals),
                'fit_variant_peak_to_peak_relative_to_measured':(np.max(vals)-np.min(vals))/r.measured_velocity})
pd.DataFrame(unc).to_csv(ROOT/'packet_prediction_fit_sensitivity.csv',index=False)

# ------------------------------------------------------------------
# 4. Accuracy-normalized cost.
# ------------------------------------------------------------------
rho_ref=float(refs[np.isclose(refs.h,.02)].rho_reference.iloc[0])
# map implementation labels to spectrum methods
entries=[]
for impl,k in [('midpoint_banded',20),('midpoint_banded',40),('midpoint_banded',80),
               ('gl4_banded',3),('gl4_banded',5),('gl4_banded',10),
               ('gl6_generic',2),('gl6_generic',5)]:
    ct=cost[(cost.method_implementation==impl)&(cost.kappa==k)].iloc[-1]
    method='midpoint' if impl.startswith('midpoint') else ('gl4' if impl.startswith('gl4') else 'gl6')
    sq=spall[(np.isclose(spall.h,.02))&(spall.method==method)&(spall.kappa==k)]
    rhoerr=float(sq.rho_abs_error.iloc[0]) if len(sq) else np.nan
    vq=FW[(np.isclose(FW.h,.02))&(FW.method==method)&(FW.kappa==k)&np.isclose(FW.input_a,.1)]
    verr=float(abs(vq.relative_velocity_error.iloc[0])) if len(vq) else np.nan
    entries.append({'implementation':impl,'method':method,'kappa':k,'one_cell_time_s':ct.median_one_cell_time_s,
                    'rho_abs_error':rhoerr,'free_wave_velocity_relative_error_a0p1':verr})
acc=pd.DataFrame(entries);acc.to_csv(ROOT/'accuracy_normalized_cost.csv',index=False)

# ------------------------------------------------------------------
# 5. Summaries.
# ------------------------------------------------------------------
summary=[
 {'quantity':'rho reference h=0.01','value':refs.loc[np.isclose(refs.h,.01),'rho_reference'].iloc[0],'note':'median of GL4 k=40,80 and GL6 k=10,20'},
 {'quantity':'rho reference h=0.02','value':refs.loc[np.isclose(refs.h,.02),'rho_reference'].iloc[0],'note':'median of GL4 k=40,80 and GL6 k=10,20'},
 {'quantity':'midpoint observed velocity order','value':orders[orders.method=='midpoint'].observed_order.mean(),'note':'mean over h=0.01,0.02'},
 {'quantity':'GL4 observed velocity order','value':orders[orders.method=='gl4'].observed_order.mean(),'note':'mean over h=0.01,0.02'},
 {'quantity':'GL6 observed velocity order','value':orders[orders.method=='gl6'].observed_order.mean(),'note':'mean over h=0.01,0.02'},
 {'quantity':'max GL6 h-collapse difference','value':m.abs_h_difference.max(),'note':'relative velocity error'},
 {'quantity':'natural GL4 packet I corrected velocity rel error','value':abs(pack[pack.dataset=='GL4 packet I'].revised_relative_error.iloc[0]),'note':'independent nonlinear validation'},
 {'quantity':'natural GL4 packet II corrected velocity rel error','value':abs(pack[pack.dataset=='GL4 packet II'].revised_relative_error.iloc[0]),'note':'independent nonlinear validation'},
 {'quantity':'archived midpoint kappa=5 corrected velocity rel error','value':abs(pack[(pack.method=='midpoint')&(pack.kappa==5)].revised_relative_error.iloc[0]),'note':'supersedes old correction'},
 {'quantity':'archived midpoint kappa=20 corrected velocity rel error','value':abs(pack[(pack.method=='midpoint')&(pack.kappa==20)].revised_relative_error.iloc[0]),'note':'supersedes old correction'},
 {'quantity':'best tangent central finite-difference error midpoint','value':tan[tan.method=='midpoint'].central_relative_error.min(),'note':'implementation check'},
 {'quantity':'best tangent central finite-difference error GL4','value':tan[tan.method=='gl4'].central_relative_error.min(),'note':'implementation check'},
 {'quantity':'best tangent central finite-difference error GL6','value':tan[tan.method=='gl6'].central_relative_error.min(),'note':'implementation check'},
 {'quantity':'max adjoint identity error','value':tan.adjoint_relative_error.max(),'note':'all three integrators'},
]
pd.DataFrame(summary).to_csv(ROOT/'WP3_summary.csv',index=False)

# ------------------------------------------------------------------
# Figures (default matplotlib colors/styles).
# ------------------------------------------------------------------
# Spectral convergence all three methods.
plt.figure(figsize=(8,5.2))
for h in [.01,.02]:
    for method,marker in [('midpoint','o'),('gl4','s'),('gl6','^')]:
        q=spall[(np.isclose(spall.h,h))&(spall.method==method)].sort_values('kappa')
        plt.plot(q.kappa,q.rho_F,marker=marker,label=f'{method}, h={h:g}')
for _,r in refs.iterrows():plt.axhline(r.rho_reference,linewidth=.7,linestyle='--')
plt.xlabel(r'$\kappa=h/(c\Delta t)$')
plt.ylabel(r'Floquet spectral radius $\rho_F$')
plt.legend(fontsize=8,ncol=2)
plt.tight_layout();plt.savefig(ROOT/'wp3_floquet_three_integrators.png',dpi=220);plt.close()

# Error to spectral reference.
plt.figure(figsize=(8,5.2))
for h in [.01,.02]:
    for method,marker in [('midpoint','o'),('gl4','s'),('gl6','^')]:
        q=spall[(np.isclose(spall.h,h))&(spall.method==method)&(spall.rho_abs_error>0)].sort_values('kappa')
        plt.loglog(q.kappa,q.rho_abs_error,marker=marker,label=f'{method}, h={h:g}')
plt.xlabel(r'$\kappa=h/(c\Delta t)$')
plt.ylabel(r'$|\rho_F-\rho_{F,\mathrm{ref}}|$')
plt.legend(fontsize=8,ncol=2)
plt.tight_layout();plt.savefig(ROOT/'wp3_floquet_reference_error_three_integrators.png',dpi=220);plt.close()

# Velocity temporal order.
plt.figure(figsize=(8,5.2))
for h in [.01,.02]:
    for method,marker in [('midpoint','o'),('gl4','s'),('gl6','^')]:
        q=FW[(np.isclose(FW.h,h))&(FW.method==method)&np.isclose(FW.input_a,.1)].drop_duplicates('kappa').sort_values('kappa')
        plt.loglog(q.kappa,abs(q.relative_velocity_error),marker=marker,label=f'{method}, h={h:g}')
plt.xlabel(r'$\kappa=h/(c\Delta t)$')
plt.ylabel(r'$|v/v_{\rm sd}-1|$')
plt.legend(fontsize=8,ncol=2)
plt.tight_layout();plt.savefig(ROOT/'wp3_velocity_order_three_integrators.png',dpi=220);plt.close()

# Correction collapse all methods.
plt.figure(figsize=(8,5.2))
for method,marker in [('midpoint','o'),('gl4','s'),('gl6','^')]:
    for h,openmarker in [(.01,False),(.02,True)]:
        q=FW[(FW.method==method)&np.isclose(FW.h,h)].sort_values('chi')
        if openmarker:plt.scatter(q.chi,q.relative_velocity_error,marker=marker,facecolors='none',label=f'{method}, h={h:g}')
        else:plt.scatter(q.chi,q.relative_velocity_error,marker=marker,label=f'{method}, h={h:g}')
plt.xlabel(r'$\chi=A\Delta t/h^3$')
plt.ylabel(r'$v/v_{\rm sd}-1$')
plt.legend(fontsize=8,ncol=2)
plt.tight_layout();plt.savefig(ROOT/'wp3_velocity_correction_three_integrators.png',dpi=220);plt.close()

# Leading scaled error.
plt.figure(figsize=(8,5.2))
for method,p,marker,chimax in [('midpoint',2,'o',.025),('gl4',4,'s',.04),('gl6',6,'^',.06)]:
    q=FW[(FW.method==method)&(FW.chi<=chimax)&(FW.chi>0)].sort_values('chi')
    plt.scatter(q.chi,q.relative_velocity_error/q.chi**p,marker=marker,label=rf'{method}: divide by $\chi^{{{p}}}$')
plt.xlabel(r'$\chi$')
plt.ylabel('leading-order scaled velocity error')
plt.legend(fontsize=9)
plt.tight_layout();plt.savefig(ROOT/'wp3_velocity_leading_scaled_three_integrators.png',dpi=220);plt.close()

# Independent packet validation: baseline versus revised.
plt.figure(figsize=(7.5,5))
x=np.arange(len(pack));w=.25
plt.bar(x-w,abs(pack.baseline_relative_error),width=w,label='baseline law')
plt.bar(x,abs(pack.revised_relative_error),width=w,label='calibrated fully discrete law')
plt.yscale('log');plt.xticks(x,[rf'{m}\n$\kappa$={int(k)}' for m,k in zip(pack.method,pack.kappa)])
plt.ylabel('absolute relative velocity error')
plt.legend();plt.tight_layout();plt.savefig(ROOT/'wp3_independent_packet_validation.png',dpi=220);plt.close()

# Cost vs spectral accuracy.
q=acc[np.isfinite(acc.rho_abs_error)]
plt.figure(figsize=(7.5,5))
for method,marker in [('midpoint','o'),('gl4','s'),('gl6','^')]:
    g=q[q.method==method].sort_values('one_cell_time_s')
    plt.loglog(g.one_cell_time_s,g.rho_abs_error,marker=marker,label=method)
    for _,r in g.iterrows():plt.annotate(rf'$\kappa={int(r.kappa)}$',(r.one_cell_time_s,r.rho_abs_error),fontsize=7)
plt.xlabel('measured wall time per cell crossing (s)')
plt.ylabel(r'$|\rho_F-\rho_{F,\mathrm{ref}}|$')
plt.legend();plt.tight_layout();plt.savefig(ROOT/'wp3_accuracy_normalized_cost_spectrum.png',dpi=220);plt.close()

# Cost vs velocity accuracy.
q=acc[np.isfinite(acc.free_wave_velocity_relative_error_a0p1)]
plt.figure(figsize=(7.5,5))
for method,marker in [('midpoint','o'),('gl4','s'),('gl6','^')]:
    g=q[q.method==method].sort_values('one_cell_time_s')
    plt.loglog(g.one_cell_time_s,g.free_wave_velocity_relative_error_a0p1,marker=marker,label=method)
    for _,r in g.iterrows():plt.annotate(rf'$\kappa={int(r.kappa)}$',(r.one_cell_time_s,r.free_wave_velocity_relative_error_a0p1),fontsize=7)
plt.xlabel('measured wall time per cell crossing (s)')
plt.ylabel(r'$|v/v_{\rm sd}-1|$ for $A/h^2=0.1$')
plt.legend();plt.tight_layout();plt.savefig(ROOT/'wp3_accuracy_normalized_cost_velocity.png',dpi=220);plt.close()

# Tangent verification.
plt.figure(figsize=(7.5,5))
for method,marker in [('midpoint','o'),('gl4','s'),('gl6','^')]:
    q=tan[tan.method==method].sort_values('delta')
    plt.loglog(q.delta,q.central_relative_error,marker=marker,label=method)
plt.xlabel(r'finite-difference increment $\delta$')
plt.ylabel('central-difference tangent relative error')
plt.legend();plt.tight_layout();plt.savefig(ROOT/'wp3_tangent_directional_verification.png',dpi=220);plt.close()

print('REFERENCES')
print(refs.to_string(index=False))
print('\nORDERS')
print(orders.to_string(index=False))
print('\nCORRECTION FITS')
print(fits[fits.fit_scope=='combined_h'].to_string(index=False))
print('\nPACKET VALIDATION')
print(pack.to_string(index=False))
print('\nCOST')
print(acc.to_string(index=False))
