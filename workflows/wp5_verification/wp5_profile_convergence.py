import sys,numpy as np,pandas as pd
from pathlib import Path
WP4=Path(__file__).resolve().parents[1]/'wp4_space';sys.path.insert(0,str(WP4))
from wp4_spatial_methods import solve_profile
OUT=Path(__file__).resolve().parent
rows=[]
for method in ['ismail','defrutos','pade6','pade8']:
 for Leta,Neta in [(90,8192),(90,16384),(90,32768),(60,16384),(120,16384)]:
  s=solve_profile(method,.02,Leta=Leta,Neta=Neta,tol=3e-13,maxit=3000)
  rows.append({'method':method,'h':.02,'Leta':Leta,'Neta':Neta,'V_h':s['V'],'FWHM_cells':s['fwhm'],'profile_residual':s['residual'],'iterations':s['iterations'],'last_error':s['last_error']})
  print(rows[-1],flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'profile_solver_convergence.csv',index=False)
# spread per method
sumdf=df.groupby('method').agg(V_min=('V_h','min'),V_max=('V_h','max'),V_ptp=('V_h',lambda x:x.max()-x.min()),FWHM_min=('FWHM_cells','min'),FWHM_max=('FWHM_cells','max'),resid_max=('profile_residual','max')).reset_index();sumdf.to_csv(OUT/'profile_solver_convergence_summary.csv',index=False)
print(sumdf.to_string(index=False))
