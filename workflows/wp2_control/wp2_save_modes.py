import sys,numpy as np
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from wp2_control_helper import solve_base,base_orbit,hyper_block_ritz,highpass
OUT=Path(__file__).resolve().parent
# CFL5 real branch
b=solve_base(.01,5);orb=base_orbit(b);Q=None
for e in [0,.1,.2,.3,.3687,.42]:
 Q,rr,_=hyper_block_ritz(b,e,Q=Q,nb=16,niter=400 if e>=.3 else 100,orbit=orb)
 edge=sorted([z for z in rr if z['edgefrac']>.95],key=lambda z:abs(z['eig']),reverse=True)
 z=edge[0];v=np.real(z['vec']);v/=np.max(np.abs(highpass(v)[b.idxR]))
 if e in [.3,.3687,.42]:
  np.savez_compressed(OUT/f'h001_k5_mode_eps_{e:.4f}.npz',x=b.x,x0=b.x0,v=v,eig=z['eig'],rho=abs(z['eig']),res=z['res'])
  print('save5',e,z['eig'],z['res'])
# CFL20 complex branch continuation
b=solve_base(.02,20);orb=base_orbit(b);Q=None
for e,nit in [(0,100),(.05,100),(.1,100),(.14,100),(.16,300),(.19,500),(.22,80)]:
 Q,rr,_=hyper_block_ritz(b,e,Q=Q,nb=18,niter=nit,orbit=orb)
 edge=sorted([z for z in rr if z['edgefrac']>.95],key=lambda z:abs(z['eig']),reverse=True)
 # choose complex pair branch where available, else top
 comp=[z for z in edge if abs(np.imag(z['eig']))>.05]
 z=comp[0] if comp else edge[0]
 v=z['vec'];vr=np.real(v);vi=np.imag(v)
 scale=max(np.max(np.abs(highpass(vr)[b.idxR])),np.max(np.abs(highpass(vi)[b.idxR])))
 vr/=scale;vi/=scale
 if e in [.16,.19,.22]:
  np.savez_compressed(OUT/f'h002_k20_mode_eps_{e:.4f}.npz',x=b.x,x0=b.x0,vr=vr,vi=vi,eig=z['eig'],rho=abs(z['eig']),res=z['res'])
  print('save20',e,z['eig'],z['res'])
