from __future__ import annotations
import sys,time,argparse
from pathlib import Path
import numpy as np
ROOT4=Path(__file__).resolve().parents[1]/'wp4_space';sys.path.insert(0,str(ROOT4))
from wp4_banded import coeffs,midpoint_step
p=argparse.ArgumentParser();p.add_argument('--file',required=True);p.add_argument('--to-cells',type=int,required=True);p.add_argument('--save-every',type=int,default=2);a=p.parse_args()
z=np.load(a.file);x=z['x'];oldcells=z['cells'];oldU=z['U'];h=float(z['h']);kappa=int(z['kappa']);M=int(z['Mwidth']);dt=h/kappa;ca,cl=coeffs('defrutos',h);U=np.asarray(oldU[-1],float);start=int(oldcells[-1]);newcells=list(oldcells.astype(int));snaps=[np.asarray(u,np.float32) for u in oldU]
t0=time.time()
for cell in range(start+1,a.to_cells+1):
 for _ in range(kappa):U=midpoint_step(U,dt,ca,cl,newton=3,tol=2e-13)
 U=np.roll(U,-1)
 if cell%a.save_every==0:newcells.append(cell);snaps.append(U.astype(np.float32))
out=Path(a.file)
np.savez_compressed(out,x=x,cells=np.array(newcells),t=np.array(newcells)*h,U=np.stack(snaps),x0=z['x0'],h=h,kappa=kappa,phase=z['phase'],Mwidth=M,L=z['L'])
print('extended',out,'from',start,'to',a.to_cells,'runtime',time.time()-t0,'shape',len(snaps))
