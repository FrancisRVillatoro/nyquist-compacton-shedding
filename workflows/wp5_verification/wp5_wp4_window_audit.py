import numpy as np, pandas as pd, sys, math
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).resolve().parents[1]/'wp4_space'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from wp4_spatial_methods import compacton
h=.02
specs=[
 {'method':'ismail','packet':'I','tmin':4.8,'tmax':9.5,'anchor_t':6.0,'anchor_n':94.37429368345374,'v_guess':1.8634682649021972,'Ws':[3,5,7,10]},
 {'method':'ismail','packet':'II','tmin':10.15,'tmax':15.0,'anchor_t':15.0,'anchor_n':526.8123112408773,'v_guess':2.0345280329342916,'Ws':[3,5,7,10]},
 {'method':'pade8','packet':'I','tmin':16.95,'tmax':25.10,'anchor_t':20.0,'anchor_n':847.554867,'v_guess':4.701100306157743,'Ws':[12,16,20,25,30]},
 {'method':'pade8','packet':'II','tmin':17.0,'tmax':26.0,'anchor_t':20.0,'anchor_n':725.189791,'v_guess':4.110906217522077,'Ws':[12,16,20,25,30]},
]
rows=[];sumrows=[]
for s in specs:
 m=s['method']; d=np.load(ROOT/f'natural_{m}_h002_k20.npz'); x=np.asarray(d['x']);tt=np.asarray(d['t']);UU=np.asarray(d['U']);x0=float(d['x0']);L=x[-1]+h
 p=np.load(ROOT/f'nyquist_profile_{m}_h002.npz');Fcs=CubicSpline(p['eta'],p['F'],extrapolate=False)
 inds=np.where((tt>=s['tmin']-1e-12)&(tt<=s['tmax']+1e-12))[0]
 for W in s['Ws']:
  local=[]
  for k in inds:
   tq=float(tt[k]); e=np.asarray(UU[k],float)-compacton(x,x0+tq,L)
   npred=s['anchor_n']+(s['v_guess']-1)/h*(tq-s['anchor_t'])
   edge=x0+tq+2*np.pi;jpred=edge/h+npred;j0=int(np.rint(jpred));js=np.arange(j0-W,j0+W+1);y=e[js%len(x)]
   def eva(X):
    shp=(-1.0)**js*np.nan_to_num(Fcs(js-X),nan=0.0);aa=(y@shp)/(shp@shp);r=y-aa*shp
    return r@r,aa,shp
   sol=minimize_scalar(lambda X:eva(X)[0],bounds=(jpred-(3 if m=='ismail' else 5),jpred+(3 if m=='ismail' else 5)),method='bounded',options={'xatol':1e-10})
   ss,aa,shp=eva(sol.x);r=y-aa*shp
   row={'method':m,'packet':s['packet'],'W':W,'t':tq,'n_fit':sol.x-edge/h,'A_over_h2':abs(aa)/h**2,'relL2':np.linalg.norm(r)/(np.linalg.norm(y)+1e-300)}
   rows.append(row);local.append(row)
  g=pd.DataFrame(local).sort_values('t');coef,cov=np.polyfit(g.t.values,g.n_fit.values*h,1,cov=True)
  sumrows.append({'method':m,'packet':s['packet'],'W':W,'n':len(g),'A_over_h2_median':g.A_over_h2.median(),'A_over_h2_std':g.A_over_h2.std(ddof=1),'profile_relL2_median':g.relL2.median(),'profile_relL2_max':g.relL2.max(),'velocity':1+coef[0],'velocity_stderr':math.sqrt(cov[0,0])})
pd.DataFrame(rows).to_csv(OUT/'wp4_window_audit_snapshot_fits.csv',index=False)
sumdf=pd.DataFrame(sumrows);sumdf.to_csv(OUT/'wp4_window_audit_summary.csv',index=False)
print(sumdf.to_string(index=False))
