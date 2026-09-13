from pathlib import Path
import sys,time
import numpy as np,pandas as pd
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from wp3_integrators import SpatialSystem
from wp3_gl4_banded import gl4_cell_banded
from wp3_gl6 import gl6_cell

h=.02
ref=np.load(ROOT/'compacton_gl4_h0.020_k80.npz')
U=ref['U'];N=len(U);sysm=SpatialSystem(N,h)
configs=[
 ('midpoint_generic',80),('midpoint_generic',40),
 ('gl4_generic',10),('gl4_generic',5),
 ('gl4_banded',10),('gl4_banded',5),
 ('gl6_generic',5),('gl6_generic',2),
]
rows=[]
for name,k in configs:
 dt=h/k
 def f(V):
  if name=='midpoint_generic':return sysm.cell_map(V,dt,k,method='midpoint')
  if name=='gl4_generic':return sysm.cell_map(V,dt,k,method='gl4')
  if name=='gl4_banded':return gl4_cell_banded(sysm,V,dt,k,maxit=4,tol=2e-12)
  if name=='gl6_generic':return gl6_cell(sysm,V,dt,k)
 # warmup
 V=f(U.copy())
 times=[];errs=[]
 for rep in range(5):
  t=time.perf_counter();V=f(U.copy());times.append(time.perf_counter()-t)
  errs.append(np.linalg.norm(V-U,np.inf))
 rows.append({'method_implementation':name,'h':h,'N':N,'kappa':k,'dt':dt,
              'median_one_cell_time_s':np.median(times),'min_one_cell_time_s':np.min(times),
              'max_one_cell_time_s':np.max(times),'map_fixed_point_residual_on_ref':np.median(errs)})
 print(rows[-1],flush=True)
df=pd.DataFrame(rows);df.to_csv(ROOT/'time_integrator_cost_benchmark.csv',index=False)
print(df.to_string(index=False))
