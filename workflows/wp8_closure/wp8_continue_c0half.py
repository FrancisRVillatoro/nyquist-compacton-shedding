import sys,time,numpy as np,pandas as pd
from wp8_controls import step,compacton,hp,fit_at
from pathlib import Path
OUT=Path(__file__).resolve().parent/'derived'; OUT.mkdir(exist_ok=True)
D=np.load(OUT/'wp8_c0_half_snapshots.npz');x=D['x'];U=D['U'][-1].astype(float);h=float(D['h']);dt=float(D['dt']);c0=float(D['c0']);c=float(D['c']);x0=float(D['x0']);L=len(x)*h
start=float(D['t'][-1]); T=50.; nsteps=int(round((T-start)/dt)); stride=int(round(.02/dt)); rows=[];mx=0
for n in range(nsteps+1):
 if n%stride==0:
  t=start+n*dt;ctr=x0+(c-c0)*t;uc=compacton(x,ctr,L,c);du=U-uc;z=hp(du)/h**2;edge=ctr+2*np.pi;rel=(x-edge+L/2)%L-L/2;mask=(rel>.08)&(rel<L/2-.5);idx=np.where(mask)[0];a=float(np.max(np.abs(z[idx])));mx=max(mx,a);rows.append(dict(time=t,max_ahead_HP_over_h2=a))
 if n<nsteps:U=step(U,dt,h,c0,'abs')
 if n and n%(nsteps//4)==0:print('progress',n,nsteps,flush=True)
pd.DataFrame(rows).to_csv(OUT/'wp8_c0_half_trace_25_50.csv',index=False)
print('max',mx,'last',rows[-1])
