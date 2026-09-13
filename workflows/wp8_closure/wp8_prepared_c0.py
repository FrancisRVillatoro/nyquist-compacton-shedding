import sys,numpy as np,pandas as pd,time
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from wp8_controls import step
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'derived'; OUT.mkdir(exist_ok=True)
P=np.load(ROOT/'workflows/wp4_space/nyquist_profile_defrutos_h002.npz')
Fcs=CubicSpline(P['eta'],P['F'],extrapolate=False);Vh=float(P['V'])
h=.02;dt=.001;L=32.;N=int(round(L/h));j=np.arange(N);a=.10;A=a*h*h

def fit(U,guess):
 W=35;js=np.arange(int(round(guess))-W,int(round(guess))+W+1);y=U[js%N]
 def ev(X):
  s=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0.);den=s@s; aa=(y@s)/den;r=y-aa*s;return r@r,aa,s
 sol=minimize_scalar(lambda X:ev(X)[0],bounds=(guess-5,guess+5),method='bounded',options={'xatol':1e-9});ss,aa,s=ev(sol.x)
 return sol.x,aa,np.sqrt(ss)/(np.linalg.norm(y)+1e-300)

def run(c0,T):
 X0=N*.25;U=A*((-1.)**j)*np.nan_to_num(Fcs(j-X0),nan=0.)
 rows=[];guess=X0;stride=10;nsteps=int(round(T/dt));unwrap=X0
 for n in range(nsteps+1):
  if n%stride==0:
   X,aa,rr=fit(U,guess%N)
   # choose periodic image nearest previous unwrap
   Xu=X
   while Xu-unwrap>N/2: Xu-=N
   while Xu-unwrap<-N/2: Xu+=N
   unwrap=Xu;guess=Xu
   rows.append((n*dt,Xu,aa/h**2,rr))
  if n<nsteps:U=step(U,dt,h,c0,'abs')
 df=pd.DataFrame(rows,columns=['time','X_cells_unwrapped','A_over_h2','relL2'])
 # discard initial transient first 20%
 q=df[df.time>=.2*T]
 vel=h*np.polyfit(q.time,q.X_cells_unwrapped,1)[0]
 am=q.A_over_h2.median();chi=am/(h/dt)
 nonlinear=Vh*am*(1-11.659289*chi**2+452.119772*chi**4-2.10497e4*chi**6)
 pred=5*c0+nonlinear
 df.to_csv(OUT/f'wp8_prepared_c0_{c0:.2f}.csv',index=False)
 print(dict(c0=c0,velocity=vel,A_over_h2=am,relL2=float(q.relL2.median()),pred=pred,nonlinear=nonlinear,relerr=(vel-pred)/pred))
 return [c0,vel,am,float(q.relL2.median()),pred,nonlinear,(vel-pred)/pred]
if __name__=='__main__':
 import argparse;ap=argparse.ArgumentParser();ap.add_argument('c0',type=float);ap.add_argument('--T',type=float,default=2.0);a0=ap.parse_args();run(a0.c0,a0.T)
