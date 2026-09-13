import sys,numpy as np,pandas as pd,time
from pathlib import Path
WP3=Path(__file__).resolve().parents[1]/'wp3_time';sys.path.insert(0,str(WP3))
from wp3_integrators import SpatialSystem,leading_ritz
OUT=Path(__file__).resolve().parent
rows=[]
for f in ['compacton_midpoint_h0.010_k5.npz','compacton_midpoint_h0.020_k20.npz']:
 d=np.load(WP3/f);h=float(d['h']);kappa=int(d['kappa']);x0=float(d['x0']);U=np.asarray(d['U']);S=SpatialSystem(len(U),h)
 for nb in [6,8,10]:
  for seed in [1,17,123,2026]:
   t0=time.time();sp=leading_ritz(S,U,h,kappa,'midpoint',x0,nb=nb,maxiter=500,tol=5e-9,seed=seed);z=sp['values'][0]
   rows.append({'h':h,'kappa':kappa,'nb':nb,'seed':seed,'rho_F':abs(z),'lambda_real':z.real,'lambda_imag':z.imag,'eigenpair_residual':sp['residuals'][0],'ritz_iterations':sp['iterations'],'runtime_s':time.time()-t0})
   print(rows[-1],flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'ritz_seed_subspace_audit.csv',index=False)
print(df.groupby(['h','kappa']).agg(rho_min=('rho_F','min'),rho_max=('rho_F','max'),rho_ptp=('rho_F',lambda x:x.max()-x.min()),rho_std=('rho_F','std'),resid_max=('eigenpair_residual','max')).to_string())
