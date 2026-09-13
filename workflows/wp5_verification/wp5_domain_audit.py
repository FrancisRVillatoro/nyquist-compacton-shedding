import sys,time,numpy as np,pandas as pd
from pathlib import Path
WP3=Path(__file__).resolve().parents[1]/'wp3_time';WP4=Path(__file__).resolve().parents[1]/'wp4_space'
sys.path[:0]=[str(WP3),str(WP4)]
from wp3_integrators import SpatialSystem,compacton,solve_traveling_compacton,leading_ritz
from wp4_spatial_methods import fixed_phase_x0
OUT=Path(__file__).resolve().parent
rows=[]
for h,kappa in [(.01,5),(.02,20)]:
 for L in [14.,16.,20.,24.]:
  N=int(round(L/h));x=np.arange(N)*h;x0=fixed_phase_x0(L,h,.25);S=SpatialSystem(N,h);Uc=compacton(x,x0,L)
  t0=time.time();Us,res,dt=solve_traveling_compacton(S,Uc,h,kappa,'midpoint',f_tol=2e-10,maxiter=10,inner_maxiter=20)
  sp=leading_ritz(S,Us,h,kappa,'midpoint',x0,nb=8,maxiter=500,tol=5e-9,seed=2468);z=sp['values'][0]
  rows.append({'h':h,'kappa':kappa,'L':L,'N':N,'fixed_point_residual':res,'rho_F':abs(z),'lambda_real':z.real,'lambda_imag':z.imag,'eigenpair_residual':sp['residuals'][0],'edge_fraction':sp['edge_frac'],'runtime_s':time.time()-t0})
  print(rows[-1],flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'fixed_phase_domain_audit.csv',index=False)
print('\nSUMMARY')
print(df.groupby(['h','kappa']).agg(rho_min=('rho_F','min'),rho_max=('rho_F','max'),rho_ptp=('rho_F',lambda x:x.max()-x.min()),eigres_max=('eigenpair_residual','max')).to_string())
