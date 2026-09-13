import sys,json,argparse,time,math
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase_floquet_scan import (build_ops,compacton,solve_traveling_compacton,one_cell_map,apply_P,block_ritz)

p=argparse.ArgumentParser(); p.add_argument('--h',type=float,required=True); p.add_argument('--kappa',type=float,required=True); p.add_argument('--phi',type=float,required=True); p.add_argument('--out',required=True); p.add_argument('--maxiter',type=int,default=500); a=p.parse_args()
h=a.h;kappa=a.kappa;phi=a.phi;Ldom=16.0;dt=h/kappa;N=int(round(Ldom/h));x=np.arange(N)*h
A,L=build_ops(N,h);nR=int(np.rint((Ldom/2+2*np.pi)/h));xR=h*(nR+phi);x0=xR-2*np.pi;Uc=compacton(x,x0,Ldom)
t0=time.time();Us,fp=solve_traveling_compacton(Uc,A,L,dt,int(kappa));_,fac=one_cell_map(Us,A,L,dt,int(kappa),record=True)
qL=(x0-2*np.pi)/h;nL=int(np.floor(qL));phiL=qL-nL;mR=nR%N;mL=int(np.rint(qL))%N;win=np.arange(-40,41);idxR=(mR+win)%N;idxL=(mL+win)%N
rng=np.random.default_rng(8101);sR=np.zeros(N);sL=np.zeros(N);sR[idxR]=rng.normal(size=len(idxR));sL[idxL]=rng.normal(size=len(idxL))
ew,ev,Q,nit,_=block_ritz(lambda Z:apply_P(Z,fac),N,[sR,sL],nb=10,maxiter=a.maxiter)
lam=ew[0];r=Q@ev[:,0];res=float(np.linalg.norm(apply_P(r,fac)-lam*r)/np.linalg.norm(r));e=np.abs(r)**2;et=e.sum();er=e[idxR].sum()/et;el=e[idxL].sum()/et
row={'case':f'h{h:g}_k{kappa:g}','h':h,'kappa':kappa,'dt':dt,'phi_right':phi,'phi_left':phiL,'x0':x0,'N':N,'Ldom':Ldom,'fixed_point_residual':fp,'lambda_real':float(lam.real),'lambda_imag':float(lam.imag),'rho_F':float(abs(lam)),'arg_lambda':float(np.angle(lam)),'mode_type':'real flip' if abs(lam.imag)<1e-7 and lam.real<0 else 'complex pair','right_eigen_residual':res,'right_edge_energy_fraction':float(er),'left_edge_energy_fraction':float(el),'two_edge_energy_fraction':float(er+el),'right_ritz_iterations':nit,'runtime_s':time.time()-t0}
Path(a.out).write_text(json.dumps(row,indent=2));print(json.dumps(row))
