import sys, numpy as np, pandas as pd, time
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
sys.path.insert(0,str(Path(__file__).resolve().parent))
from wp4_banded import coeffs, midpoint_step
P=Path(__file__).resolve().parent
CASES=[
 ('ismail','I',0.6525448125,2.0),('ismail','II',0.7125395990,2.0),
 ('pade8','I',0.1471261963,1.5),('pade8','II',0.1286200691,1.5)]
h=.02;kappa=20;dt=h/kappa;L=80.;N=int(round(L/h));x=np.arange(N)*h
rows=[]
for method,label,a,T in CASES:
 d=np.load(P/f'nyquist_profile_{method}_h002.npz');Fcs=CubicSpline(d['eta'],d['F'],extrapolate=False)
 A=a*h*h; X0=20/h+0.173
 js=np.arange(N)
 U=A*((-1.0)**js)*np.nan_to_num(Fcs(js-X0),nan=0.0)
 ca,cl=coeffs(method,h)
 stride=50 # save dt .05
 nsteps=int(round(T/dt)); ts=[]; Xs=[]; As=[]; rels=[]
 def fit(U,Xpred,W=25):
  j0=int(round(Xpred)); jj=np.arange(j0-W,j0+W+1); y=U[jj%N]
  def eva(X):
   s=(-1.0)**jj*np.nan_to_num(Fcs(jj-X),nan=0.0); aa=(y@s)/(s@s);r=y-aa*s
   return r@r,aa,s
  sol=minimize_scalar(lambda X:eva(X)[0],bounds=(Xpred-6,Xpred+6),method='bounded',options={'xatol':1e-10})
  ss,aa,s=eva(sol.x);r=y-aa*s
  return sol.x,aa,np.linalg.norm(r)/(np.linalg.norm(y)+1e-300)
 # use semidisc v predictor for tracking
 v0=float(d['V'])*a
 Xpred=X0
 for n in range(nsteps+1):
  if n%stride==0:
   t=n*dt
   if n>0: Xpred=Xs[-1]+v0*(t-ts[-1])/h
   xx,aa,rr=fit(U,Xpred)
   ts.append(t);Xs.append(xx);As.append(aa);rels.append(rr);Xpred=xx
  if n<nsteps: U=midpoint_step(U,dt,ca,cl,newton=4,tol=2e-13)
 ts=np.array(ts);Xs=np.unwrap(np.array(Xs)*h/L*2*np.pi)*L/(2*np.pi);As=np.array(As);rels=np.array(rels)
 # discard first 0.3
 mask=ts>=0.3
 coef,cov=np.polyfit(ts[mask],Xs[mask],1,cov=True)
 rows.append(dict(method=method,packet=label,input_a=a,fit_a=np.median(abs(As[mask]))/h**2,
                  velocity=coef[0],velocity_stderr=np.sqrt(cov[0,0]),profile_residual_median=np.median(rels[mask]),
                  profile_residual_max=np.max(rels[mask]),T=T))
 print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv(P/'prepared_wave_validation_recomputed.csv',index=False)
