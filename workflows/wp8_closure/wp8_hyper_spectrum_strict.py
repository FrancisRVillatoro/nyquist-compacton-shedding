#!/usr/bin/env python3
from pathlib import Path
import sys,csv
import numpy as np
import scipy.linalg as la
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'workflows/wp2_control'))
from wp2_control_helper import solve_base,base_orbit,hyper_tangent_factors,apply_factors_shift
OUT=Path(__file__).resolve().parent/'derived'; OUT.mkdir(exist_ok=True)
RUN_EC=0.190972
EPS=[r*RUN_EC for r in (0.5,0.8,0.9,0.95,1.0,1.05,1.2)]
base=solve_base(.02,20,16.0); orbit=base_orbit(base); I=np.eye(len(base.Us))
def dominant_edge(eps):
    P=apply_factors_shift(hyper_tangent_factors(base,eps,orbit),I)
    vals,vecs=la.eig(P,left=False,right=True,check_finite=False)
    cand=[]
    for j,lam in enumerate(vals):
        v=vecs[:,j]
        ef=float(np.sum(np.abs(v[base.idxEdge])**2)/np.sum(np.abs(v)**2))
        if ef>.99:
            res=float(np.linalg.norm(P@v-lam*v)/np.linalg.norm(v))
            cand.append((abs(lam),lam,res,ef))
    return max(cand,key=lambda z:z[0])
root_pts=[]
for eps in (0.19100,0.19102): root_pts.append((eps,*dominant_edge(eps)))
e1,r1=root_pts[0][0],root_pts[0][1]; e2,r2=root_pts[1][0],root_pts[1][1]
eps_c=e1+(1-r1)*(e2-e1)/(r2-r1)
rows=[]
for eps in EPS:
    rho,lam,res,ef=dominant_edge(eps)
    rows.append(dict(epsilon=eps,epsilon_over_eps_c=eps/eps_c,rho=rho,real=lam.real,imag=lam.imag,residual=res,edge_fraction=ef))
with open(OUT/'wp8_hyperviscosity_spectrum_strict.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
with open(OUT/'wp8_hyperviscosity_strict_root.csv','w',newline='') as f:
    w=csv.writer(f);w.writerow(['epsilon_c','eps_lo','rho_lo','eps_hi','rho_hi']);w.writerow([eps_c,e1,r1,e2,r2])
print('epsilon_c',eps_c,'max residual',max(r['residual'] for r in rows))
