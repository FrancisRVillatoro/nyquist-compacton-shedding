import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from wp3_integrators import SpatialSystem,compacton
from wp3_gl6 import gl6_cell

def hp(z):return (2*z-np.roll(z,1)-np.roll(z,-1))/4

def fit_at(du,Fcs,jguess,W=20,search=4):
 N=len(du);j0=int(round(jguess));js=np.arange(j0-W,j0+W+1);y=du[js%N]
 def ev(X):
  s=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0);den=s@s;A=(y@s)/den if den else 0;r=y-A*s
  return r@r,A,s
 sol=minimize_scalar(lambda X:ev(X)[0],bounds=(jguess-search,jguess+search),method='bounded',options={'xatol':1e-9})
 ss,A,s=ev(sol.x);r=y-A*s
 return sol.x,A,np.linalg.norm(r)/(np.linalg.norm(y)+1e-300),abs(y@s)/(np.linalg.norm(y)*np.linalg.norm(s)+1e-300)

def choose_x0(h,phi,target):
 m=int(round((target+2*np.pi)/h-phi));return (m+phi)*h-2*np.pi

h=.01;kappa=5;L=20.;target=6.5;phiR=.318530717959;T=6.;save_every=5
N=int(round(L/h));x=np.arange(N)*h;x0=choose_x0(h,phiR,target)
sysm=SpatialSystem(N,h);Uc=compacton(x,x0,L);U=Uc.copy();dt=h/kappa
pr=np.load(str(Path(__file__).resolve().parent/'nyquist_profile_h0.01.npz'));Fcs=CubicSpline(pr['eta'],pr['F'],extrapolate=False);Vh=float(pr['V'])
mR=int(round((x0+2*np.pi)/h))%N
maxahead=int(((L-(x0+2*np.pi))/h)-30);nrel=np.arange(5,maxahead+1);idx=(mR+nrel)%N
nmax=int(round(T/h));rows=[];t0=time.time()
for n in range(nmax+1):
 if n%save_every==0 or n==nmax:
  du=U-Uc;z=hp(du);vals=np.abs(z[idx])/h**2
  peaks,_=find_peaks(vals,distance=3)
  if len(peaks)==0:peaks=np.array([int(np.argmax(vals))])
  order=peaks[np.argsort(vals[peaks])[::-1][:30]];fits=[]
  for pp in order:
   if vals[pp]<.003:continue
   try:
    X,A,rel,corr=fit_at(du,Fcs,mR+nrel[pp],W=20,search=4.5)
    while X-mR>N/2:X-=N
    while X-mR<-N/2:X+=N
    fits.append((rel,-corr,X,A,corr,nrel[pp],vals[pp]))
   except Exception:pass
  best=min(fits) if fits else None
  if best is None: rel=corr=X=A=np.nan;pc=float(nrel[np.argmax(vals)]);pa=float(vals.max())
  else: rel,negcorr,X,A,corr,pc,pa=best
  rows.append({'method':'gl6','h':h,'kappa':kappa,'cell':n,'time':n*h,
   'max_ahead_HP_over_h2':float(vals.max()),'raw_peak_cell':float(nrel[np.argmax(vals)]),
   'candidate_peak_HP_over_h2':float(pa),'candidate_peak_cell':float(pc),
   'best_fit_center_rel_cells':float(X-mR) if np.isfinite(X) else np.nan,
   'best_fit_A_over_h2':float(abs(A)/h**2) if np.isfinite(A) else np.nan,
   'best_fit_relL2':float(rel),'best_fit_corr':float(corr)})
 if n<nmax:U=gl6_cell(sysm,U,dt,kappa)
 if n and n%100==0:print(n,nmax,'elapsed',time.time()-t0,flush=True)
df=pd.DataFrame(rows);df.to_csv(ROOT/'natural_gl6_h001_k5.csv',index=False)
print('max ahead',df.max_ahead_HP_over_h2.max())
# print candidate clean rows
print(df[(df.best_fit_relL2<.02)&(df.best_fit_A_over_h2>.03)&(df.best_fit_center_rel_cells>8)].to_string(index=False))
