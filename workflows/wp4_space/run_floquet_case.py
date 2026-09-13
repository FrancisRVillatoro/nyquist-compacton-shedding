import sys,time,json,argparse,numpy as np,pandas as pd
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from wp4_spatial_methods import *
pa=argparse.ArgumentParser();pa.add_argument('--method');pa.add_argument('--h',type=float);pa.add_argument('--kappa',type=int,default=10);pa.add_argument('--L',type=float,default=20.0);pa.add_argument('--phase',type=float,default=.25);args=pa.parse_args()
method=args.method;h=args.h;kappa=args.kappa;Ldom=args.L;phase=args.phase
N=int(round(Ldom/h));x=np.arange(N)*h;x0=fixed_phase_x0(Ldom,h,phase)
sysm=SpatialSystem(method,N,h);Uc=compacton(x,x0,Ldom)
t0=time.time();Us,res,dt=solve_tw(sysm,Uc,h,kappa,ftol=2e-10,maxiter=14)
print('fixed',method,h,kappa,'N',N,'res',res,'sec',time.time()-t0,flush=True)
t1=time.time();spc=leading_ritz(sysm,Us,h,kappa,x0,nb=10,maxiter=800,tol=5e-9,seed=4400+int(100*h)+kappa)
print('ritz',spc['values'][0],abs(spc['values'][0]),'res',spc['residuals'][0],'edge',spc['edge_frac'],'sec',time.time()-t1,flush=True)
out=Path(__file__).parent
np.savez_compressed(out/f'compacton_{method}_h{h:.3f}_k{kappa}.npz',x=x,x0=x0,Us=Us,Uc=Uc,leading_vector=spc['vector'],values=spc['values'],residuals=spc['residuals'])
row={'method':method,'h':h,'kappa':kappa,'dt':dt,'Ldom':Ldom,'N':N,'phase':phase,'fixed_point_residual':res,
     'lambda1_real':spc['values'][0].real,'lambda1_imag':spc['values'][0].imag,'rhoF':abs(spc['values'][0]),'lambda1_residual':spc['residuals'][0],
     'lambda2_real':spc['values'][1].real,'lambda2_imag':spc['values'][1].imag,'lambda2_abs':abs(spc['values'][1]),'lambda2_residual':spc['residuals'][1],
     'edge_energy_fraction':spc['edge_frac'],'right_fraction':spc['right_frac'],'left_fraction':spc['left_frac'],'ritz_iterations':spc['iterations']}
pd.DataFrame([row]).to_csv(out/f'floquet_{method}_h{h:.3f}_k{kappa}.csv',index=False)
