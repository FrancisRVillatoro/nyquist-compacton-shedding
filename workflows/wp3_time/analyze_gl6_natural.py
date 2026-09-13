from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).parent
meta=json.loads((ROOT/'natural_gl6_meta.json').read_text());h=meta['h'];L=meta['L'];N=meta['N'];x0=meta['x0'];save=meta['save_every'];kappa=meta['kappa']
S=np.load(ROOT/'natural_gl6_snapshots.npy',mmap_mode='r');x=np.arange(N)*h
pr=np.load(str(Path(__file__).resolve().parent/'nyquist_profile_h0.01.npz'));Fcs=CubicSpline(pr['eta'],pr['F'],extrapolate=False);Vh=float(pr['V'])
def comp(x,xc):
 z=(x-xc+L/2)%L-L/2;u=np.zeros_like(x);m=np.abs(z)<=2*np.pi;u[m]=4/3*np.cos(z[m]/4)**2;return u
def hp(z):return (2*z-np.roll(z,1)-np.roll(z,-1))/4
def fit_at(du,jguess,W=20,search=4.5):
 j0=int(round(jguess));js=np.arange(j0-W,j0+W+1);y=du[js%N]
 def ev(X):
  sh=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0.0);A=(y@sh)/(sh@sh);r=y-A*sh;return r@r,A,sh
 sol=minimize_scalar(lambda X:ev(X)[0],bounds=(jguess-search,jguess+search),method='bounded',options={'xatol':1e-9})
 ss,A,sh=ev(sol.x);r=y-A*sh;return sol.x,A,np.linalg.norm(r)/(np.linalg.norm(y)+1e-300),abs(y@sh)/(np.linalg.norm(y)*np.linalg.norm(sh)+1e-300)
mR0=int(round((x0+2*np.pi)/h))%N
rows=[]
for ii,U in enumerate(S):
 cell=ii*save;t=cell*h;Uc=comp(x,x0);du=np.asarray(U)-Uc;z=hp(du)
 # moving edge index is fixed in comoving stored states? gl6_cell shifts each cell, so compacton remains at x0; yes.
 mR=mR0
 nrel=np.arange(5,int((L-(x0+2*np.pi))/h)-30);idx=(mR+nrel)%N;vals=np.abs(z[idx])/h**2
 pk,_=find_peaks(vals,distance=3)
 if not len(pk):pk=np.array([np.argmax(vals)])
 order=pk[np.argsort(vals[pk])[::-1][:40]];fits=[]
 for p in order:
  if vals[p]<.002:continue
  try:
   X,A,rel,corr=fit_at(du,mR+nrel[p])
   while X-mR>N/2:X-=N
   while X-mR<-N/2:X+=N
   fits.append((rel,-corr,X,A,corr,nrel[p],vals[p]))
  except Exception:pass
 best=min(fits) if fits else None
 if best is None:rel=corr=X=A=np.nan;pc=nrel[np.argmax(vals)];pa=vals.max()
 else:rel,negcorr,X,A,corr,pc,pa=best
 rows.append({'method':'gl6','h':h,'kappa':kappa,'cell':cell,'time':t,'max_ahead_HP_over_h2':float(vals.max()),
  'raw_peak_cell':float(nrel[np.argmax(vals)]),'candidate_peak_HP_over_h2':float(pa),'candidate_peak_cell':float(pc),
  'best_fit_center_rel_cells':float(X-mR) if np.isfinite(X) else np.nan,
  'best_fit_A_over_h2':float(abs(A)/h**2) if np.isfinite(A) else np.nan,'best_fit_relL2':float(rel),'best_fit_corr':float(corr)})
df=pd.DataFrame(rows);df.to_csv(ROOT/'natural_gl6_h001_k5.csv',index=False)
print('clean candidates')
print(df[(df.best_fit_relL2<.02)&(df.best_fit_A_over_h2>.02)&(df.best_fit_center_rel_cells>8)].to_string(index=False))
