import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from wp3_integrators import SpatialSystem
from wp3_gl6 import gl6_step

def fit_center(U,Fcs,Xpred,W=18):
 N=len(U);j0=int(round(Xpred));js=np.arange(j0-W,j0+W+1);y=U[js%N]
 def ev(X):
  s=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0.0);den=s@s;A=(y@s)/den if den else 0.0;r=y-A*s
  return r@r,A,s
 sol=minimize_scalar(lambda X:ev(X)[0],bounds=(Xpred-2.0,Xpred+2.0),method='bounded',options={'xatol':5e-11})
 ss,A,s=ev(sol.x);r=y-A*s
 return sol.x,A,np.linalg.norm(r)/(np.linalg.norm(y)+1e-300)

def run(kappa,a,h,N=512,travel_cells=35):
 d=np.load(str(Path(__file__).resolve().parent/f'nyquist_profile_h{h:.2f}.npz'));Fcs=CubicSpline(d['eta'],d['F'],extrapolate=False);Vh=float(d['V'])
 sysm=SpatialSystem(N,h);dt=h/kappa;X0=N/4+0.271;j=np.arange(N)
 U=(a*h*h)*((-1.)**j)*np.nan_to_num(Fcs(j-X0),nan=0.0)
 delta0=Vh*a/kappa
 nsteps=max(50,int(np.ceil(travel_cells/max(delta0,1e-8))))
 save_every=max(1,int(round(0.20/max(delta0,1e-8))))
 rows=[];Xpred=X0
 for n in range(nsteps+1):
  if n%save_every==0 or n==nsteps:
   X,A,rel=fit_center(U,Fcs,Xpred)
   X += round((Xpred-X)/N)*N;Xpred=X
   rows.append((n,X,abs(A)/h**2,rel))
  if n<nsteps:
   U=gl6_step(sysm,U,dt,maxit=7,tol=5e-13)
   Xpred += delta0
 df=pd.DataFrame(rows,columns=['step','center','A_over_h2','profile_relL2'])
 q=df[(df.step>=.20*nsteps)&(df.profile_relL2<.08)]
 coef=np.polyfit(q.step,q.center,1);v=kappa*coef[0]
 pred=np.polyval(coef,q.step);rmse=np.sqrt(np.mean((q.center-pred)**2))
 Amed=q.A_over_h2.median();chi=Amed/kappa;v0=Vh*Amed
 return {'method':'gl6','h':h,'kappa':kappa,'input_a':a,'chi':chi,'nsteps':nsteps,'nfit':len(q),
         'A_fit_median':Amed,'A_fit_std':q.A_over_h2.std(ddof=1),'profile_residual_median':q.profile_relL2.median(),
         'profile_residual_max':q.profile_relL2.max(),'center_regression_RMSE_cells':rmse,
         'velocity_fit':v,'semidiscrete_velocity':v0,'relative_velocity_error':(v-v0)/v0}

cases=[]
for h in [.01,.02]:
 for k in [1,2,3,4,5,8,10,20]:cases.append((h,k,.10))
 for a in [.04,.06,.08,.12,.15,.18]:cases.append((h,2,a))
rows=[]
for h,k,a in cases:
 t=time.time();r=run(k,a,h);r['runtime_s']=time.time()-t;rows.append(r)
 print(json.dumps(r),flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'free_wave_gl6_benchmark.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
