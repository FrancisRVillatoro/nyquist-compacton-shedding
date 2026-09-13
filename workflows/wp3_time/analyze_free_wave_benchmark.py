from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
ROOT=Path(__file__).parent
d=pd.read_csv(ROOT/'free_wave_time_integrator_benchmark.csv')
rows=[]
# Remove duplicated a=.1,k5 if any; there is one only per method/k/input
for method in ['midpoint','gl4']:
 g=d[d.method==method].copy()
 # use modest chi range; all fits weighted equally
 if method=='midpoint':
  fit=g[g.chi<=0.05]
  powers=[2,4,6]
 else:
  fit=g[g.chi<=0.10]
  powers=[4,6,8]
 X=np.column_stack([fit.chi.values**p for p in powers])
 coef=np.linalg.lstsq(X,fit.relative_velocity_error.values,rcond=None)[0]
 pred=X@coef;rmse=np.sqrt(np.mean((pred-fit.relative_velocity_error.values)**2))
 for p,c in zip(powers,coef):rows.append({'method':method,'power':p,'coefficient':c,'fit_chi_max':fit.chi.max(),'n':len(fit),'rmse':rmse})
 d.loc[g.index,'polynomial_prediction']=np.column_stack([g.chi.values**p for p in powers])@coef
# simpler leading coefficient at small chi
for method,p,maxchi in [('midpoint',2,.025),('gl4',4,.04)]:
 g=d[(d.method==method)&(d.chi<=maxchi)&(d.chi>0)]
 c=np.dot(g.chi.values**p,g.relative_velocity_error.values)/np.dot(g.chi.values**p,g.chi.values**p)
 rows.append({'method':method,'power':p,'coefficient':c,'fit_chi_max':maxchi,'n':len(g),'rmse':np.sqrt(np.mean((c*g.chi.values**p-g.relative_velocity_error.values)**2)),'fit_type':'leading_only'})
coefdf=pd.DataFrame(rows);coefdf.to_csv(ROOT/'free_wave_velocity_correction_coefficients.csv',index=False)
d.to_csv(ROOT/'free_wave_time_integrator_benchmark_with_fit.csv',index=False)
print(coefdf.to_string(index=False))

plt.figure(figsize=(7.5,5))
for method,marker in [('midpoint','o'),('gl4','s')]:
 g=d[d.method==method].sort_values('chi')
 plt.plot(g.chi,g.relative_velocity_error,marker=marker,linestyle='none',label=method)
 # fit curve unique range
 gg=g.dropna(subset=['polynomial_prediction']).sort_values('chi')
 plt.plot(gg.chi,gg.polynomial_prediction,label=f'{method} fit')
plt.xlabel(r'$\chi=A\Delta t/h^3$');plt.ylabel(r'$v/v_{\rm sd}-1$');plt.legend();plt.tight_layout();plt.savefig(ROOT/'wp3_velocity_corrections.png',dpi=200);plt.close()

plt.figure(figsize=(7.5,5))
for method,p,marker in [('midpoint',2,'o'),('gl4',4,'s')]:
 g=d[(d.method==method)&(d.chi>0)].sort_values('chi')
 plt.plot(g.chi,g.relative_velocity_error/(g.chi**p),marker=marker,label=f'{method}: error/$\\chi^{p}$')
plt.xlabel(r'$\chi$');plt.ylabel('scaled velocity error');plt.legend();plt.tight_layout();plt.savefig(ROOT/'wp3_velocity_scaled_error.png',dpi=200);plt.close()
