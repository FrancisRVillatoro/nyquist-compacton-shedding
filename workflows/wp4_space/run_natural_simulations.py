import argparse,sys,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
from wp4_banded import simulate
from wp4_spatial_methods import fixed_phase_x0
p=argparse.ArgumentParser()
p.add_argument('--method',choices=['ismail','defrutos','pade6','pade8'],required=True)
p.add_argument('--h',type=float,default=.02)
p.add_argument('--kappa',type=int,default=20)
p.add_argument('--L',type=float,default=80.)
p.add_argument('--T',type=float,default=None)
p.add_argument('--phase',type=float,default=.25)
p.add_argument('--save-dt',type=float,default=.05)
a=p.parse_args()
T=a.T if a.T is not None else (30. if a.method in ('pade6','pade8') else 15.)
x0=fixed_phase_x0(a.L,a.h,a.phase)
x,t,U=simulate(a.method,a.h,a.kappa,a.L,T,x0,a.save_dt)
out=Path(__file__).parent/f'natural_{a.method}_h{int(round(1000*a.h)):03d}_k{a.kappa}.npz'
np.savez_compressed(out,x=x,t=t,U=U,x0=x0,h=a.h,kappa=a.kappa,phase=a.phase)
print(out)
