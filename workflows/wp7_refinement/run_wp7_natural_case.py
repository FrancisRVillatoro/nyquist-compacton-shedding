from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

ROOT4=Path(__file__).resolve().parents[1]/'wp4_space'
sys.path.insert(0,str(ROOT4))
from wp4_banded import simulate
from wp4_spatial_methods import fixed_phase_x0

p=argparse.ArgumentParser()
p.add_argument('--Mwidth',type=int,required=True,help='number of grid cells across compacton support 4*pi/h; use even integer')
p.add_argument('--kappa',type=int,default=20)
p.add_argument('--phase',type=float,default=.25)
p.add_argument('--Ltarget',type=float,default=40.0)
p.add_argument('--Tcells',type=float,default=800.0)
p.add_argument('--save-cells',type=float,default=2.0)
p.add_argument('--outdir',default=str(Path(__file__).resolve().parent))
a=p.parse_args()

h=4*np.pi/a.Mwidth
# Choose an even periodic N close to Ltarget/h. Keep Nyquist parity compatible.
N=int(round(a.Ltarget/h))
if N%2: N+=1
L=N*h
x0=fixed_phase_x0(L,h,a.phase)
T=a.Tcells*h
save_dt=a.save_cells*h
print(f'Mwidth={a.Mwidth} h={h:.12g} N={N} L={L:.12g} T={T:.12g} dt={h/a.kappa:.12g} x0={x0:.12g}',flush=True)
t0=time.time()
x,t,U=simulate('defrutos',h,a.kappa,L,T,x0,save_dt)
out=Path(a.outdir)/f'natural_defrutos_M{a.Mwidth:04d}_h{h:.8f}_k{a.kappa}.npz'
np.savez_compressed(out,x=x,t=t,U=U,x0=x0,h=h,kappa=a.kappa,phase=a.phase,Mwidth=a.Mwidth,L=L,Tcells=a.Tcells,save_cells=a.save_cells)
print(f'SAVED {out} runtime={time.time()-t0:.3f}s shape={U.shape}',flush=True)
