import sys,numpy as np,pandas as pd
from wp8_controls import step,compacton,hp
h=.02;L=16.;N=int(round(L/h));x=np.arange(N)*h;c=1.;c0=0.;kappa=80.;dt=h/kappa;T=4.;phi=.25;target=15.
m=int(round((target+2*np.pi)/h-phi));x0=(m+phi)*h-2*np.pi;U=compacton(x,x0,L,c);rows=[]
fail=None
for n in range(int(round(T/dt))+1):
 if n%40==0:
  t=n*dt;uc=compacton(x,x0+c*t,L,c);du=U-uc;z=hp(du)/h**2
  rows.append(dict(step=n,time=t,maxabs=float(np.max(np.abs(U))),minU=float(np.min(U)),maxU=float(np.max(U)),hp_global=float(np.max(np.abs(z)))))
 if n<int(round(T/dt)):
  try:U=step(U,dt,h,c0,'square')
  except Exception as e:
   fail=(n,t,type(e).__name__,str(e));break
from pathlib import Path
OUT=Path(__file__).resolve().parent/'derived'; OUT.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(OUT/'wp8_K22_kappa80_failure_trace.csv',index=False)
print('failure',fail);print('last',rows[-1])
