from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
ROOT4=Path(__file__).resolve().parents[1]/'wp4_space';sys.path.insert(0,str(ROOT4))
from wp4_banded import coeffs, midpoint_step
from wp4_spatial_methods import compacton, fixed_phase_x0
p=argparse.ArgumentParser();p.add_argument('--Mwidth',type=int,required=True);p.add_argument('--kappa',type=int,default=5);p.add_argument('--phase',type=float,default=.25);p.add_argument('--Ltarget',type=float,default=40.0);p.add_argument('--cells',type=int,default=650);p.add_argument('--save-every',type=int,default=2);p.add_argument('--outdir',default=str(Path(__file__).resolve().parent));a=p.parse_args()
h=4*np.pi/a.Mwidth;N=int(round(a.Ltarget/h));N += N%2;L=N*h;x=np.arange(N)*h;x0=fixed_phase_x0(L,h,a.phase);dt=h/a.kappa
ca,cl=coeffs('defrutos',h);U=compacton(x,x0,L)
ns=a.cells//a.save_every+1;snaps=np.empty((ns,N),np.float32);cc=np.empty(ns,int);snaps[0]=U;cc[0]=0;ii=1
t0=time.time();
for cell in range(1,a.cells+1):
    for _ in range(a.kappa): U=midpoint_step(U,dt,ca,cl,newton=3,tol=2e-13)
    U=np.roll(U,-1)
    if cell%a.save_every==0:
        snaps[ii]=U;cc[ii]=cell;ii+=1
out=Path(a.outdir)/f'comoving_defrutos_M{a.Mwidth:04d}_h{h:.8f}_k{a.kappa}.npz'
np.savez_compressed(out,x=x,cells=cc[:ii],t=cc[:ii]*h,U=snaps[:ii],x0=x0,h=h,kappa=a.kappa,phase=a.phase,Mwidth=a.Mwidth,L=L)
print(f'SAVED {out} runtime={time.time()-t0:.3f}s N={N} L={L:.8f} shape={snaps[:ii].shape}',flush=True)
