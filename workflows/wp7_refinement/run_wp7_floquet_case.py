from __future__ import annotations
import argparse,sys,time
from pathlib import Path
import numpy as np
ROOT4=Path(__file__).resolve().parents[1]/'wp4_space';sys.path.insert(0,str(ROOT4))
from wp4_spatial_methods import SpatialSystem,compacton,fixed_phase_x0,solve_tw_mp,leading_ritz_mp
p=argparse.ArgumentParser();p.add_argument('--Mwidth',type=int,required=True);p.add_argument('--kappa',type=int,default=5);p.add_argument('--phase',type=float,default=.25);p.add_argument('--Ltarget',type=float,default=16.0);p.add_argument('--outdir',default=str(Path(__file__).resolve().parent));a=p.parse_args()
h=4*np.pi/a.Mwidth;N=int(round(a.Ltarget/h));N += N%2;L=N*h;x=np.arange(N)*h;x0=fixed_phase_x0(L,h,a.phase);sysm=SpatialSystem('defrutos',N,h);Uc=compacton(x,x0,L)
t0=time.time();Us,res,dt=solve_tw_mp(sysm,Uc,h,a.kappa,ftol=2e-10,maxiter=14);rr=leading_ritz_mp(sysm,Us,h,a.kappa,x0,nb=8,maxiter=500,tol=5e-9,seed=1234);lam=rr['values'][0]
out=Path(a.outdir)/f'floquet_M{a.Mwidth:04d}.npz';np.savez(out,h=h,Mwidth=a.Mwidth,N=N,L=L,x0=x0,kappa=a.kappa,phase=a.phase,Us=Us,lambda_real=lam.real,lambda_imag=lam.imag,rho=abs(lam),fixed_point_residual=res,eigen_residual=rr['residuals'][0],edge_frac=rr['edge_frac'])
print(f'{a.Mwidth},{h:.14g},{abs(lam):.12g},{lam.real:.12g},{lam.imag:.12g},{res:.3e},{rr["residuals"][0]:.3e},{rr["edge_frac"]:.9f},{time.time()-t0:.2f}',flush=True)
