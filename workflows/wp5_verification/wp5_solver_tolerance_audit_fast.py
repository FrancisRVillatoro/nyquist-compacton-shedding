import sys,time,numpy as np,pandas as pd
from pathlib import Path
WP3=Path(__file__).resolve().parents[1]/'wp3_time';WP4=Path(__file__).resolve().parents[1]/'wp4_space'
sys.path[:0]=[str(WP3),str(WP4)]
from wp3_integrators import SpatialSystem,compacton,solve_traveling_compacton,leading_ritz
from wp4_spatial_methods import fixed_phase_x0
OUT=Path(__file__).resolve().parent
class AuditSystem(SpatialSystem):
 def __init__(self,N,h,stage_tol): super().__init__(N,h);self.stage_tol=stage_tol
 def midpoint_step(self,U,dt,store=False,newton_tol=None,maxit=None): return super().midpoint_step(U,dt,store=store,newton_tol=self.stage_tol,maxit=8)
rows=[]
for h,kappa in [(.01,5),(.02,20)]:
 L=14.;N=int(round(L/h));x=np.arange(N)*h;x0=fixed_phase_x0(L,h,.25);Uc=compacton(x,x0,L)
 for stage_tol,ftol in [(1e-10,1e-8),(1e-12,1e-10),(2e-13,2e-10)]:
  t0=time.time();S=AuditSystem(N,h,stage_tol);Us,res,dt=solve_traveling_compacton(S,Uc,h,kappa,'midpoint',f_tol=ftol,maxiter=9,inner_maxiter=18)
  sp=leading_ritz(S,Us,h,kappa,'midpoint',x0,nb=6,maxiter=380,tol=2e-8,seed=2026);z=sp['values'][0]
  rows.append(dict(h=h,kappa=kappa,stage_tol=stage_tol,outer_ftol=ftol,fixed_point_residual=res,rho_F=abs(z),lambda_real=z.real,lambda_imag=z.imag,eigenpair_residual=sp['residuals'][0],edge_fraction=sp['edge_frac'],runtime_s=time.time()-t0))
  print(rows[-1],flush=True)
df=pd.DataFrame(rows);df.to_csv(OUT/'solver_tolerance_audit_fast.csv',index=False)
print('\nSUMMARY');print(df.groupby(['h','kappa']).agg(rho_min=('rho_F','min'),rho_max=('rho_F','max'),rho_ptp=('rho_F',lambda x:x.max()-x.min()),eigres_max=('eigenpair_residual','max'),fpres_max=('fixed_point_residual','max')).to_string())
