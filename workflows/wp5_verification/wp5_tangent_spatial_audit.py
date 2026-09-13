import sys,time,numpy as np,pandas as pd
from pathlib import Path
WP4=Path(__file__).resolve().parents[1]/'wp4_space';sys.path.insert(0,str(WP4))
from wp4_spatial_methods import SpatialSystem,compacton,fixed_phase_x0,solve_tw,leading_ritz
OUT=Path(__file__).resolve().parent
rows=[];adj=[]
h=.02;kappa=10;L=16.;N=int(round(L/h));x=np.arange(N)*h;x0=fixed_phase_x0(L,h,.25)
for method in ['ismail','defrutos','pade6','pade8']:
 S=SpatialSystem(method,N,h);Uc=compacton(x,x0,L);Us,res,dt=solve_tw(S,Uc,h,kappa,ftol=2e-10,maxiter=12)
 _,data=S.cell_map(Us,dt,kappa,True)
 rng=np.random.default_rng(100+len(method));z=rng.normal(size=N);z/=np.linalg.norm(z);w=rng.normal(size=N);w/=np.linalg.norm(w)
 Pz=S.tangent_cell(z,dt,data);PTw=S.tangent_cell_adj(w,dt,data)
 adjerr=abs(np.dot(Pz,w)-np.dot(z,PTw))/(abs(np.dot(Pz,w))+abs(np.dot(z,PTw))+1e-300)
 adj.append({'method':method,'h':h,'kappa':kappa,'fixed_point_residual':res,'adjoint_identity_relative_error':adjerr})
 for delta in [1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5,3e-6,1e-6,3e-7,1e-7]:
  # relative perturbation h^2*delta so deltas correspond to natural edge-scale coordinates
  dd=h*h*delta
  fp=S.cell_map(Us+dd*z,dt,kappa);fm=S.cell_map(Us-dd*z,dt,kappa)
  fd=(fp-fm)/(2*dd)
  er=np.linalg.norm(fd-Pz)/(np.linalg.norm(Pz)+1e-300)
  rows.append({'method':method,'h':h,'kappa':kappa,'delta_scaled':delta,'physical_delta':dd,'directional_relative_error':er})
 print(method,'fp',res,'adj',adjerr,'min fd',min(r['directional_relative_error'] for r in rows if r['method']==method),flush=True)
pd.DataFrame(rows).to_csv(OUT/'spatial_tangent_directional_audit.csv',index=False)
pd.DataFrame(adj).to_csv(OUT/'spatial_adjoint_identity_audit.csv',index=False)
print(pd.DataFrame(adj).to_string(index=False))
