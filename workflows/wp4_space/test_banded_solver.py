import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from wp4_spatial_methods import METHODS,OFFS
from wp4_banded import cyclic_penta_solve
rng=np.random.default_rng(2026)
for method,m in METHODS.items():
    N=127;h=.02;dt=.001
    dphi=np.abs(rng.normal(size=N))+.2
    cl=m['B0']/h+m['C0']/h**3
    rows=[];cols=[];vals=[]
    for k,a,c in zip(OFFS,m['A'],cl):
        j=np.arange(N);i=(j-k)%N
        rows.extend(i);cols.extend(j);vals.extend((2/dt*a+c*dphi[j]).tolist())
    J=sp.csc_matrix((vals,(rows,cols)),shape=(N,N))
    rhs=rng.normal(size=N)
    xb=cyclic_penta_solve(dphi,rhs,dt,m['A'],cl)
    xs=spla.spsolve(J,rhs)
    rel=np.linalg.norm(xb-xs)/np.linalg.norm(xs)
    print(method,rel)
    assert rel<1e-11
