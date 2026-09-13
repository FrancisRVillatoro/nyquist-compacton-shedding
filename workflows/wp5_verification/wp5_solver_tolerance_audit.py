import sys, time, numpy as np, pandas as pd
from pathlib import Path
WP3=Path(__file__).resolve().parents[1]/'wp3_time'; WP4=Path(__file__).resolve().parents[1]/'wp4_space'
sys.path.insert(0,str(WP3));sys.path.insert(0,str(WP4))
from wp3_integrators import SpatialSystem, compacton, solve_traveling_compacton, leading_ritz
from wp4_spatial_methods import fixed_phase_x0
OUT=Path(__file__).resolve().parent

class AuditSystem(SpatialSystem):
    def __init__(self,N,h,stage_tol,maxit=10):
        super().__init__(N,h); self.stage_tol=stage_tol; self.stage_maxit=maxit
    def midpoint_step(self,U,dt,store=False,newton_tol=None,maxit=None):
        return super().midpoint_step(U,dt,store=store,newton_tol=self.stage_tol,maxit=self.stage_maxit)

rows=[];staterows=[]
for h,kappa in [(.01,5),(.02,20)]:
    L=16.;N=int(round(L/h));x=np.arange(N)*h;x0=fixed_phase_x0(L,h,.25);Uc=compacton(x,x0,L)
    refU=None
    for stage_tol,ftol in [(1e-10,1e-8),(1e-12,1e-10),(2e-13,2e-10),(5e-14,1e-11)]:
        t0=time.time();sysm=AuditSystem(N,h,stage_tol,maxit=10)
        Us,res,dt=solve_traveling_compacton(sysm,Uc,h,kappa,'midpoint',f_tol=ftol,maxiter=12,inner_maxiter=25)
        if refU is None:refU=Us.copy()
        staterows.append({'h':h,'kappa':kappa,'stage_tol':stage_tol,'outer_ftol':ftol,'fixed_point_residual':res,'state_inf_diff_from_first':float(np.max(np.abs(Us-refU)))})
        for nb,seed in [(6,11),(8,22),(10,33)]:
            sp=leading_ritz(sysm,Us,h,kappa,'midpoint',x0,nb=nb,maxiter=700,tol=5e-9,seed=seed)
            z=sp['values'][0]
            rows.append({'h':h,'kappa':kappa,'stage_tol':stage_tol,'outer_ftol':ftol,'nb':nb,'seed':seed,
                         'fixed_point_residual':res,'lambda_real':z.real,'lambda_imag':z.imag,'rho_F':abs(z),
                         'eigenpair_residual':sp['residuals'][0],'edge_fraction':sp['edge_frac'],'ritz_iterations':sp['iterations'],
                         'runtime_s':time.time()-t0})
        print('done',h,kappa,stage_tol,ftol,res,rows[-1]['rho_F'],flush=True)
pd.DataFrame(rows).to_csv(OUT/'solver_ritz_tolerance_audit.csv',index=False)
pd.DataFrame(staterows).to_csv(OUT/'fixed_point_tolerance_audit.csv',index=False)
print(pd.DataFrame(rows).groupby(['h','kappa']).agg(rho_min=('rho_F','min'),rho_max=('rho_F','max'),rho_std=('rho_F','std'),eigres_max=('eigenpair_residual','max'),fpres_max=('fixed_point_residual','max')).to_string())
