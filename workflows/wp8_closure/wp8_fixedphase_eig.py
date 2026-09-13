import sys,numpy as np, pandas as pd,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'workflows/wp3_time'))
from wp3_integrators import SpatialSystem,compacton,solve_traveling_compacton,leading_ritz
h=.02;kappa=20;L=16.;N=int(round(L/h));x=np.arange(N)*h;phi=.25;target=L/2
m=int(round((target+2*np.pi)/h-phi));x0=(m+phi)*h-2*np.pi
sysm=SpatialSystem(N,h);Uc=compacton(x,x0,L);t=time.time();Us,res,dt=solve_traveling_compacton(sysm,Uc,h,kappa,'midpoint',f_tol=2e-10,maxiter=12,inner_maxiter=24)
print('base',res,'sec',time.time()-t,'x0',x0,'phi',((x0+2*np.pi)/h)%1,flush=True)
t=time.time();r=leading_ritz(sysm,Us,h,kappa,'midpoint',x0,nb=12,maxiter=600,tol=2e-8,seed=20260910);print('ritz',r['values'][:4],r['residuals'][:4],r['edge_frac'],'sec',time.time()-t,flush=True)
v=r['leading_vector'];mR=int(np.rint((x0+2*np.pi)/h))%N
# phase coordinate relative to exact right edge in cells, unwrap nearest
edge=(x0+2*np.pi)/h
jj=np.arange(mR-20,mR+21);coord=jj-edge
vv=v[jj%N]
hp=(2*v-np.roll(v,1)-np.roll(v,-1))/4
hh=hp[jj%N]
# alternating demodulated signed vector at nodes
alt=(-1.)**jj*vv
althp=(-1.)**jj*hh
OUT=Path(__file__).resolve().parent/'derived'; OUT.mkdir(exist_ok=True)
pd.DataFrame({'j':jj,'coord_from_edge_cells':coord,'v_real':np.real(vv),'v_imag':np.imag(vv),'demod_v_real':np.real(alt),'demod_v_imag':np.imag(alt),'hp_abs':np.abs(hh),'demod_hp_real':np.real(althp)}).to_csv(OUT/'wp8_fixedphase_eigenvector_right_edge.csv',index=False)
np.savez_compressed(OUT/'wp8_fixedphase_eigenvector.npz',x=x,x0=x0,Uc=Uc,Us=Us,eigs=r['values'],residuals=r['residuals'],v=v,h=h,kappa=kappa,L=L)
