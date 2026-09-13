from pathlib import Path
import sys,time
import numpy as np,pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from wp3_integrators import SpatialSystem,compacton
from wp3_gl4_banded import gl4_cell_banded
h=.01;kappa=5;L=20.;N=int(round(L/h));x=np.arange(N)*h;phi=.318530717959;target=6.5
m=int(round((target+2*np.pi)/h-phi));x0=(m+phi)*h-2*np.pi
s=SpatialSystem(N,h);U0=compacton(x,x0,L);U=U0.copy();dt=h/kappa
pr=np.load(str(Path(__file__).resolve().parent/'nyquist_profile_h0.01.npz'));Fcs=CubicSpline(pr['eta'],pr['F'],extrapolate=False)
track=pd.read_csv(ROOT/'natural_gl4_h001_k5.csv').set_index('cell')
def fit(du,jguess,W=20):
 j0=int(round(jguess));js=np.arange(j0-W,j0+W+1);y=du[js%N]
 def obj(X):
  sh=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0);A=(y@sh)/(sh@sh);return np.sum((y-A*sh)**2)
 sol=minimize_scalar(obj,bounds=(jguess-4.5,jguess+4.5),method='bounded',options={'xatol':1e-10})
 X=sol.x;sh=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0);A=(y@sh)/(sh@sh);r=y-A*sh
 return js,X,A,y,r
outs=[]
for n in range(441):
 if n in [400,440]:
  row=track.loc[n];jguess=int(round(x0/h+2*np.pi/h+row.best_fit_center_rel_cells))
  js,X,A,y,r=fit(U-U0,jguess,W=25)
  outs.append(pd.DataFrame({'cell':n,'xi':js-X,'demodulated_normalized':y*((-1.)**js)/A,
                            'Fh':np.nan_to_num(Fcs(js-X),nan=0),'residual_over_A':r*((-1.)**js)/A,
                            'A_over_h2':abs(A)/h**2}))
 if n<440:U=gl4_cell_banded(s,U,dt,kappa,maxit=3,tol=1e-10)
out=pd.concat(outs,ignore_index=True);out.to_csv(ROOT/'gl4_natural_packet_profile_snapshots.csv',index=False)
print(out.groupby('cell').agg(A=('A_over_h2','first'),maxres=('residual_over_A',lambda x:np.max(abs(x)))).to_string())
