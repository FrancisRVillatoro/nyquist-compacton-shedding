from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
ROOT=Path(__file__).parent
d1=pd.read_csv(ROOT/'free_wave_time_integrator_benchmark.csv')
d2=pd.read_csv(ROOT/'free_wave_time_integrator_benchmark_h002.csv')
d=pd.concat([d1,d2],ignore_index=True)
# Deduplicate exact repeated designs within h/method/k/a
D=d.sort_values('runtime_s').drop_duplicates(['h','method','kappa','input_a'],keep='first').copy()
rows=[]
for method in ['midpoint','gl4']:
 g=D[D.method==method]
 if method=='midpoint': powers=[2,4,6];fit=g[g.chi<=.04]
 else:powers=[4,6,8];fit=g[g.chi<=.055]
 X=np.column_stack([fit.chi.values**p for p in powers]);y=fit.relative_velocity_error.values
 coef=np.linalg.lstsq(X,y,rcond=None)[0];pred=X@coef;rmse=np.sqrt(np.mean((pred-y)**2));maxe=np.max(np.abs(pred-y))
 for p,c in zip(powers,coef):rows.append({'method':method,'power':p,'coefficient':c,'n_fit':len(fit),'chi_max':fit.chi.max(),'fit_RMSE':rmse,'fit_max_abs_error':maxe})
 D.loc[g.index,'combined_fit_prediction']=np.column_stack([g.chi.values**p for p in powers])@coef
 # Fit each h separately to compare
 for h in sorted(g.h.unique()):
  gh=fit[np.isclose(fit.h,h)];Xh=np.column_stack([gh.chi.values**p for p in powers]);ch=np.linalg.lstsq(Xh,gh.relative_velocity_error.values,rcond=None)[0]
  for p,c in zip(powers,ch):rows.append({'method':method,'h':h,'power':p,'coefficient':c,'n_fit':len(gh),'chi_max':gh.chi.max(),'fit_type':'per_h'})
C=pd.DataFrame(rows);C.to_csv(ROOT/'free_wave_velocity_correction_combined_coefficients.csv',index=False)
D.to_csv(ROOT/'free_wave_time_integrator_benchmark_combined.csv',index=False)
print(C.to_string(index=False))
# h collapse difference matched designs
m=D.pivot_table(index=['method','kappa','input_a'],columns='h',values='relative_velocity_error').dropna()
m['abs_h_difference']=abs(m[.01]-m[.02]);m.to_csv(ROOT/'free_wave_h_collapse.csv')
print('\nmax h-difference',m.abs_h_difference.max(),'median',m.abs_h_difference.median())

plt.figure(figsize=(7.5,5))
for method,marker in [('midpoint','o'),('gl4','s')]:
 for h,fill in [(.01,'full'),(.02,'none')]:
  g=D[(D.method==method)&np.isclose(D.h,h)].sort_values('chi')
  plt.scatter(g.chi,g.relative_velocity_error,marker=marker,facecolors=('none' if fill=='none' else None),label=f'{method}, h={h:g}')
  if h==.01:
   plt.plot(g.chi,g.combined_fit_prediction,label=f'{method} fit')
plt.xlabel(r'$\chi=A\Delta t/h^3$');plt.ylabel(r'$v/v_{\rm sd}-1$');plt.legend(fontsize=8);plt.tight_layout();plt.savefig(ROOT/'wp3_velocity_correction_collapse.png',dpi=220);plt.close()

plt.figure(figsize=(7.5,5))
for method,p,marker in [('midpoint',2,'o'),('gl4',4,'s')]:
 g=D[(D.method==method)&(D.chi<=.04)].sort_values('chi')
 plt.scatter(g.chi,g.relative_velocity_error/g.chi**p,marker=marker,label=method)
plt.xlabel(r'$\chi$');plt.ylabel('leading-order scaled error');plt.legend();plt.tight_layout();plt.savefig(ROOT/'wp3_velocity_correction_scaled.png',dpi=220);plt.close()
