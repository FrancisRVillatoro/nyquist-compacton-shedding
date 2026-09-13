from pathlib import Path
import sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from wp3_integrators import SpatialSystem
from wp3_gl6 import gl6_cell,tangent_cell as gl6_tangent_cell,tangent_cell_adj as gl6_tangent_adj

h=.02;kappa=5;N=800
sysm=SpatialSystem(N,h);dt=h/kappa
rows=[]
for method in ['midpoint','gl4','gl6']:
 U=np.load(ROOT/f'compacton_{method}_h{h:.3f}_k{kappa}.npz')['U']
 if method in ['midpoint','gl4']:
  P0,data=sysm.cell_map(U,dt,kappa,method=method,store=True)
  tf=lambda z:sysm.tangent_cell(z,dt,data,method)
  ta=lambda z:sysm.tangent_cell_adj(z,dt,data,method)
  nl=lambda V:sysm.cell_map(V,dt,kappa,method=method)
 else:
  P0,data=gl6_cell(sysm,U,dt,kappa,store=True)
  tf=lambda z:gl6_tangent_cell(sysm,z,dt,data)
  ta=lambda z:gl6_tangent_adj(sysm,z,dt,data)
  nl=lambda V:gl6_cell(sysm,V,dt,kappa)
 rng=np.random.default_rng(450+len(method))
 # edge-localized smooth random direction
 x0=float(np.load(ROOT/f'compacton_{method}_h{h:.3f}_k{kappa}.npz')['x0']) if method!='gl6' else float(np.load(ROOT/f'compacton_gl4_h{h:.3f}_k80.npz')['x0'])
 mR=int(round((x0+2*np.pi)/h))%N;mL=int(round((x0-2*np.pi)/h))%N
 z=np.zeros(N)
 for m in [mR,mL]:
  nn=np.arange(-40,41);q=rng.normal(size=len(nn))*np.exp(-(nn/18)**2);z[(m+nn)%N]+=q
 z/=np.linalg.norm(z)
 Dz=tf(z);normD=np.linalg.norm(Dz)
 u=rng.normal(size=N);v=rng.normal(size=N)
 aderr=abs(np.dot(tf(u),v)-np.dot(u,ta(v)))/(abs(np.dot(tf(u),v))+abs(np.dot(u,ta(v)))+1e-300)
 for delta in np.logspace(-3,-10,8):
  fwd=(nl(U+delta*z)-P0)/delta
  cen=(nl(U+delta*z)-nl(U-delta*z))/(2*delta)
  rows.append({'method':method,'h':h,'kappa':kappa,'delta':delta,
               'forward_relative_error':np.linalg.norm(fwd-Dz)/normD,
               'central_relative_error':np.linalg.norm(cen-Dz)/normD,
               'adjoint_relative_error':aderr})
 print(method,'adjoint',aderr,flush=True)
df=pd.DataFrame(rows);df.to_csv(ROOT/'tangent_directional_verification.csv',index=False)
print(df.to_string(index=False))
