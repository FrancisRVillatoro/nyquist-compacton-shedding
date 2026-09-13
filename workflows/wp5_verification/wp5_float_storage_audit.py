import sys,numpy as np,pandas as pd,time
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
WP4=Path(__file__).resolve().parents[1]/'wp4_space';sys.path.insert(0,str(WP4))
from wp4_banded import coeffs,midpoint_step
from wp4_spatial_methods import fixed_phase_x0,compacton
OUT=Path(__file__).resolve().parent
h=.02;kappa=20;dt=h/kappa;L=80.;N=int(round(L/h));x=np.arange(N)*h;x0=fixed_phase_x0(L,h,.25)
cases=[('ismail',6.0,94.37429368345374,25),('pade8',20.0,847.554867,25)]
rows=[]
for method,T,nrel,W in cases:
 ca,cl=coeffs(method,h);U=compacton(x,x0,L);nsteps=int(round(T/dt));t0=time.time()
 for n in range(nsteps):U=midpoint_step(U,dt,ca,cl,newton=3,tol=2e-13)
 p=np.load(WP4/f'nyquist_profile_{method}_h002.npz');Fcs=CubicSpline(p['eta'],p['F'],extrapolate=False)
 def fit(V):
  e=np.asarray(V,float)-compacton(x,x0+T,L);edge=x0+T+2*np.pi;jpred=edge/h+nrel;j0=int(np.rint(jpred));js=np.arange(j0-W,j0+W+1);y=e[js%N]
  def eva(X):
   s=(-1.0)**js*np.nan_to_num(Fcs(js-X),nan=0.0);a=(y@s)/(s@s);r=y-a*s;return r@r,a,s
  sol=minimize_scalar(lambda X:eva(X)[0],bounds=(jpred-5,jpred+5),method='bounded',options={'xatol':1e-11});ss,a,s=eva(sol.x);r=y-a*s
  return sol.x-edge/h,abs(a)/h**2,np.linalg.norm(r)/(np.linalg.norm(y)+1e-300)
 fd=fit(U);fs=fit(U.astype(np.float32).astype(float))
 rows.append({'method':method,'t':T,'double_nfit':fd[0],'float32_nfit':fs[0],'nfit_difference_cells':fs[0]-fd[0],'double_A_over_h2':fd[1],'float32_A_over_h2':fs[1],'A_relative_difference':(fs[1]-fd[1])/fd[1],'double_relL2':fd[2],'float32_relL2':fs[2],'runtime_s':time.time()-t0})
 print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(OUT/'float32_storage_audit.csv',index=False)
