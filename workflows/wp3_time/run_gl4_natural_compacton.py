import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from wp3_integrators import *
from wp3_gl4_banded import gl4_cell_banded

def fast_gl4_cell(sysm,U,dt,kappa):
    return gl4_cell_banded(sysm,U,dt,kappa,maxit=3,tol=1e-10)

def hp(z):return (2*z-np.roll(z,1)-np.roll(z,-1))/4

def profile(h):
 d=np.load(Path(__file__).resolve().parent/f'nyquist_profile_h{h:.2f}.npz')
 return CubicSpline(d['eta'],d['F'],extrapolate=False),float(d['V'])

def fit_at(du,Fcs,jguess,W=20,search=4):
 N=len(du);j0=int(round(jguess));js=np.arange(j0-W,j0+W+1);y=du[js%N]
 def ev(X):
  s=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0.0);den=s@s
  A=(y@s)/den if den else 0.0;r=y-A*s
  return r@r,A,s
 sol=minimize_scalar(lambda X:ev(X)[0],bounds=(jguess-search,jguess+search),method='bounded',options={'xatol':1e-9})
 ss,A,s=ev(sol.x);r=y-A*s
 return sol.x,A,np.linalg.norm(r)/(np.linalg.norm(y)+1e-300),abs(y@s)/(np.linalg.norm(y)*np.linalg.norm(s)+1e-300)

def choose_x0(L,h,phi,target):
 m=int(round((target+2*np.pi)/h-phi));return (m+phi)*h-2*np.pi

def run(h,kappa,L,target_center,phiR,T,label,save_every_cells=10):
 N=int(round(L/h));x=np.arange(N)*h;x0=choose_x0(L,h,phiR,target_center)
 sysm=SpatialSystem(N,h);Uc=compacton(x,x0,L);U=Uc.copy();dt=h/kappa
 mR=int(np.rint((x0+2*np.pi)/h))%N
 maxahead=int(((L-(x0+2*np.pi))/h)-30)
 nrel=np.arange(5,maxahead+1);idx=(mR+nrel)%N
 Fcs,Vh=profile(h)
 nmax=int(round(T/h));rows=[]
 for n in range(nmax+1):
  if n%save_every_cells==0 or n==nmax:
   du=U-Uc;z=hp(du);vals=np.abs(z[idx])/h**2
   peaks,_=find_peaks(vals,distance=3)
   if len(peaks)==0:peaks=np.array([int(np.argmax(vals))])
   order=peaks[np.argsort(vals[peaks])[::-1][:20]]
   best=None
   for pp in order:
    if vals[pp]<0.003:continue
    jguess=mR+nrel[pp]
    try:
     X,A,rel,corr=fit_at(du,Fcs,jguess,W=20,search=4.5)
     while X-mR>N/2:X-=N
     while X-mR<-N/2:X+=N
     rec=(rel,-corr,X,A,corr,nrel[pp],vals[pp])
     if best is None or rec<best:best=rec
    except:pass
   if best is None:
    rel=corr=X=A=np.nan;peakcell=float(nrel[np.argmax(vals)]);peakamp=float(vals.max())
   else:
    rel,negcorr,X,A,corr,peakcell,peakamp=best
   rows.append({'case':label,'h':h,'kappa':kappa,'method':'gl4','cell':n,'time':n*h,
                'max_ahead_HP_over_h2':float(vals.max()),'raw_peak_cell':float(nrel[np.argmax(vals)]),
                'candidate_peak_HP_over_h2':float(peakamp),'candidate_peak_cell':float(peakcell),
                'best_fit_center_rel_cells':float(X-mR) if np.isfinite(X) else np.nan,
                'best_fit_A_over_h2':float(abs(A)/h**2) if np.isfinite(A) else np.nan,
                'best_fit_relL2':float(rel),'best_fit_corr':float(corr)})
  if n<nmax:U=fast_gl4_cell(sysm,U,dt,kappa)
  if n and n%(max(1,nmax//5))==0:print(label,n,nmax,flush=True)
 df=pd.DataFrame(rows);df.to_csv(ROOT/f'natural_gl4_{label}.csv',index=False)
 clean=df[(df.best_fit_relL2<0.02)&(df.best_fit_A_over_h2>0.03)&(df.best_fit_center_rel_cells>8)].copy()
 # Find longest consecutive-ish clean sequence in output spacing
 if len(clean)>=4:
  clean=clean.sort_values('cell'); groups=(clean.cell.diff().fillna(save_every_cells)>1.5*save_every_cells).cumsum()
  gid=groups.value_counts().idxmax();cl=clean[groups==gid]
  coef=np.polyfit(cl.cell,cl.best_fit_center_rel_cells,1);v=1+coef[0];amed=cl.best_fit_A_over_h2.median();v0=Vh*amed
  summary={'case':label,'h':h,'kappa':kappa,'first_clean_time':cl.time.min(),'last_clean_time':cl.time.max(),'n_clean':len(cl),
           'A_over_h2_median':amed,'profile_residual_median':cl.best_fit_relL2.median(),'profile_residual_max':cl.best_fit_relL2.max(),
           'velocity_fit':v,'semidiscrete_velocity':v0,'velocity_relative_error':(v-v0)/v0,
           'max_ahead_HP_over_h2':df.max_ahead_HP_over_h2.max(),'x0':x0,'phi_right':((x0+2*np.pi)/h)%1}
 else:
  summary={'case':label,'h':h,'kappa':kappa,'first_clean_time':np.nan,'last_clean_time':np.nan,'n_clean':len(clean),
           'A_over_h2_median':np.nan,'profile_residual_median':np.nan,'profile_residual_max':np.nan,
           'velocity_fit':np.nan,'semidiscrete_velocity':np.nan,'velocity_relative_error':np.nan,
           'max_ahead_HP_over_h2':df.max_ahead_HP_over_h2.max(),'x0':x0,'phi_right':((x0+2*np.pi)/h)%1}
 print(json.dumps(summary,default=float),flush=True)
 return summary

if __name__=='__main__':
 sums=[]
 sums.append(run(.01,5,20.,6.5,0.318530717959,6.,'h001_k5',5))
 pd.DataFrame(sums).to_csv(ROOT/'natural_gl4_summary.csv',index=False)
 sums.append(run(.02,20,28.,7.,0.159265358980,12.,'h002_k20',5))
 pd.DataFrame(sums).to_csv(ROOT/'natural_gl4_summary.csv',index=False)
