import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from wp3_integrators import SpatialSystem,compacton
from wp3_gl6 import gl6_cell
h=.01;kappa=5;L=20.;phi=.318530717959;target=6.5;N=int(round(L/h));x=np.arange(N)*h
m=int(round((target+2*np.pi)/h-phi));x0=(m+phi)*h-2*np.pi
sysm=SpatialSystem(N,h);dt=h/kappa;nmax=600;save_every=5
statep=ROOT/'natural_gl6_checkpoint.npz';snapp=ROOT/'natural_gl6_snapshots.npy';metap=ROOT/'natural_gl6_meta.json'
if not statep.exists():
 U=compacton(x,x0,L);start=0
 snap=np.lib.format.open_memmap(snapp,mode='w+',dtype='float64',shape=(nmax//save_every+1,N));snap[0]=U;snap.flush()
 metap.write_text(json.dumps({'h':h,'kappa':kappa,'L':L,'N':N,'x0':x0,'phi_right':phi,'nmax':nmax,'save_every':save_every},indent=2))
 np.savez(statep,U=U,cell=start)
else:
 d=np.load(statep);U=d['U'];start=int(d['cell']);snap=np.lib.format.open_memmap(snapp,mode='r+')
end=min(start+200,nmax);t0=time.time()
for n in range(start+1,end+1):
 U=gl6_cell(sysm,U,dt,kappa)
 if n%save_every==0:snap[n//save_every]=U
snap.flush();np.savez(statep,U=U,cell=end)
print('advanced',start,'to',end,'elapsed',time.time()-t0,flush=True)
