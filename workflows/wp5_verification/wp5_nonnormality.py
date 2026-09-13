import sys,numpy as np,pandas as pd
from pathlib import Path
WP3=Path(__file__).resolve().parents[1]/'wp3_time';WP4=Path(__file__).resolve().parents[1]/'wp4_space';sys.path[:0]=[str(WP3),str(WP4)]
from wp3_integrators import SpatialSystem,compacton,solve_traveling_compacton
from wp4_spatial_methods import fixed_phase_x0
OUT=Path(__file__).resolve().parent

def subspace_eigs(apply,N,idxR,idxL,nb=8,iters=500,seed=11):
 rng=np.random.default_rng(seed);Q=rng.normal(size=(N,nb))*1e-5;Q[idxR,0]+=rng.normal(size=len(idxR));Q[idxL,1]+=rng.normal(size=len(idxL));Q,_=np.linalg.qr(Q)
 for _ in range(iters):Q,_=np.linalg.qr(apply(Q))
 AQ=apply(Q);H=Q.conj().T@AQ;ew,ev=np.linalg.eig(H);o=np.argsort(np.abs(ew))[::-1];return ew[o],Q@ev[:,o]

rows=[]
for h,kappa,method in [(.01,5,'midpoint'),(.02,20,'midpoint'),(.01,5,'gl4'),(.02,20,'gl4')]:
 L=16.;N=int(round(L/h));x=np.arange(N)*h;x0=fixed_phase_x0(L,h,.25);S=SpatialSystem(N,h);Uc=compacton(x,x0,L)
 Us,res,dt=solve_traveling_compacton(S,Uc,h,kappa,method,f_tol=2e-10,maxiter=10,inner_maxiter=20)
 _,data=S.cell_map(Us,dt,kappa,method=method,store=True)
 def P(Z):return S.tangent_cell(Z,dt,data,method)
 def PT(Z):return S.tangent_cell_adj(Z,dt,data,method)
 mR=int(np.rint((x0+2*np.pi)/h))%N;mL=int(np.rint((x0-2*np.pi)/h))%N;idxR=(mR+np.arange(-35,36))%N;idxL=(mL+np.arange(-35,36))%N
 ew,R=subspace_eigs(P,N,idxR,idxL,nb=8,iters=450,seed=11)
 ewL,Lv=subspace_eigs(PT,N,idxR,idxL,nb=8,iters=450,seed=22)
 lam=ew[0];r=R[:,0];r=r/np.linalg.norm(r)
 # left eig of P^T should match conjugate(lam) for complex representation
 j=int(np.argmin(np.abs(ewL-np.conj(lam))))
 l=Lv[:,j];l=l/np.linalg.norm(l)
 ov=abs(np.vdot(l,r));cond=1/ov
 rr=np.linalg.norm(P(r)-lam*r);lr=np.linalg.norm(PT(l)-np.conj(lam)*l)
 rows.append({'h':h,'kappa':kappa,'method':method,'lambda_real':lam.real,'lambda_imag':lam.imag,'rho_F':abs(lam),'left_lambda_real':ewL[j].real,'left_lambda_imag':ewL[j].imag,'right_residual':rr,'left_residual':lr,'unit_left_right_overlap':ov,'eigenvalue_condition_number':cond,'fixed_point_residual':res})
 print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'floquet_nonnormality_conditioning.csv',index=False)
