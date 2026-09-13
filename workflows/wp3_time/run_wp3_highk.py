import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from wp3_integrators import *

def x0_fixed(L,h,phi=.25):
    m=int(np.rint((L/2+2*np.pi)/h-phi));return (m+phi)*h-2*np.pi

def append(row):
    p=ROOT/'time_integrator_spectrum_cases.csv'
    d=pd.read_csv(p) if p.exists() else pd.DataFrame()
    if len(d): d=d[~((np.isclose(d.h,row['h']))&(d.kappa==row['kappa'])&(d.method==row['method']))]
    d=pd.concat([d,pd.DataFrame([row])],ignore_index=True).sort_values(['h','method','kappa'])
    d.to_csv(p,index=False)

def run(h,k,method,L=16.):
    x0=x0_fixed(L,h);N=int(round(L/h));x=np.arange(N)*h;s=SpatialSystem(N,h)
    candidates=[]
    for meth in ([method,'midpoint' if method=='gl4' else 'gl4']):
      for q in ROOT.glob(f'compacton_{meth}_h{h:.3f}_k*.npz'):
        kk=int(q.stem.split('_k')[-1]);candidates.append((abs(kk-k),q))
    U=np.load(sorted(candidates,key=lambda z:z[0])[0][1])['U'] if candidates else compacton(x,x0,L)
    dt=h/k;t=time.time();res=float(np.linalg.norm(s.cell_map(U,dt,k,method)-U,np.inf));ts=time.time()-t
    if res>5e-9:
      U,res,dt=solve_traveling_compacton(s,U,h,k,method,f_tol=1e-9,maxiter=4,inner_maxiter=10)
      ts=time.time()-t
    np.savez_compressed(ROOT/f'compacton_{method}_h{h:.3f}_k{k}.npz',x=x,U=U,x0=x0,h=h,kappa=k,Ldom=L,residual=res)
    t=time.time();rr=leading_ritz(s,U,h,k,method,x0,nb=6,maxiter=380,tol=1e-8,seed=7000+k+int(h*10000));te=time.time()-t
    vals=rr['values'];rres=rr['residuals']
    row={'h':h,'Ldom':L,'x0':x0,'phi_right':((x0+2*np.pi)/h)%1,'method':method,'kappa':k,'dt':dt,'N':N,
      'fixed_point_residual':res,'solve_runtime_s':ts,'spectrum_runtime_s':te,'ritz_iterations':rr['iterations'],
      'edge_energy_fraction':rr['edge_frac'],'right_edge_energy_fraction':rr['right_frac'],'left_edge_energy_fraction':rr['left_frac']}
    for j in range(4):
      row[f'lambda{j+1}_real']=float(vals[j].real);row[f'lambda{j+1}_imag']=float(vals[j].imag);row[f'lambda{j+1}_abs']=float(abs(vals[j]));row[f'lambda{j+1}_residual']=float(rres[j])
    row['rho_F']=row['lambda1_abs'];row['mode_type']='real' if abs(row['lambda1_imag'])<1e-8 else 'complex'
    append(row);print(json.dumps(row),flush=True)

if __name__=='__main__':run(float(sys.argv[1]),int(sys.argv[2]),sys.argv[3])
