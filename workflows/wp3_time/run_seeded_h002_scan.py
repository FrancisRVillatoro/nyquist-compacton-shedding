import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from wp3_integrators import *
from wp3_gl4_banded import gl4_cell_banded

def hp(z):return (2*z-np.roll(z,1)-np.roll(z,-1))/4

def fit_at(du,Fcs,jguess,W=20,search=4):
 N=len(du);j0=int(round(jguess));js=np.arange(j0-W,j0+W+1);y=du[js%N]
 def ev(X):
  s=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0);den=s@s;A=(y@s)/den if den else 0;r=y-A*s
  return r@r,A,s
 sol=minimize_scalar(lambda X:ev(X)[0],bounds=(jguess-search,jguess+search),method='bounded',options={'xatol':1e-9})
 ss,A,s=ev(sol.x);r=y-A*s
 return sol.x,A,np.linalg.norm(r)/(np.linalg.norm(y)+1e-300),abs(y@s)/(np.linalg.norm(y)*np.linalg.norm(s)+1e-300)

def real_mode(v,idx):
 if np.max(np.abs(v.imag))<1e-12:return v.real
 j=idx[np.argmax(np.abs(v[idx]))];return np.real(v*np.exp(-1j*np.angle(v[j])))

def onecell(sysm,U,dt,k,method):
 if method=='midpoint':return sysm.cell_map(U,dt,k,'midpoint')
 return gl4_cell_banded(sysm,U,dt,k,maxit=3,tol=1e-10)

def run(method,k,label,sign,eps=.005,nmax=250):
 h=.02;p=ROOT/f'compacton_{method}_h0.020_k{k}.npz';d=np.load(p);Us=d['U'];x=d['x'];x0=float(d['x0']);N=len(Us);dt=h/k
 sysm=SpatialSystem(N,h);rr=leading_ritz(sysm,Us,h,k,method,x0,nb=8,maxiter=360,tol=1e-8,seed=555+k)
 mR=int(np.rint((x0+2*np.pi)/h))%N;idxR=(mR+np.arange(-35,36))%N
 r=real_mode(rr['leading_vector'],idxR);r/=np.max(np.abs(hp(r)[idxR]))
 U=Us+sign*h*h*eps*r
 pr=np.load(str(Path(__file__).resolve().parent/'nyquist_profile_h0.02.npz'));Fcs=CubicSpline(pr['eta'],pr['F'],extrapolate=False)
 nrel=np.arange(5,N//2-25);idx=(mR+nrel)%N
 rows=[]
 for n in range(nmax+1):
  if n%2==0:
   du=U-Us;z=hp(du);vals=np.abs(z[idx])/h**2;pk,_=find_peaks(vals,distance=3)
   if len(pk)==0:pk=np.array([int(np.argmax(vals))])
   order=pk[np.argsort(vals[pk])[::-1][:30]];fits=[]
   for pp in order:
    if vals[pp]<.005:continue
    try:
     X,A,rel,corr=fit_at(du,Fcs,mR+nrel[pp],20,4.5)
     while X-mR>N/2:X-=N
     while X-mR<-N/2:X+=N
     fits.append((rel,-corr,X-mR,abs(A)/h**2,corr,nrel[pp],vals[pp]))
    except:pass
   fits=sorted(fits)
   for rank,rec in enumerate(fits[:5],1):
    rel,negc,Xr,Amp,corr,pc,pa=rec
    rows.append({'case':label,'sign':sign,'cell':n,'time':n*h,'rank':rank,'rho_F':abs(rr['values'][0]),
      'max_ahead_HP_over_h2':float(vals.max()),'fit_center_rel_cells':Xr,'fit_A_over_h2':Amp,'fit_relL2':rel,'fit_corr':corr,
      'candidate_peak_cell':pc,'candidate_peak_amp':pa,'edge_HP_over_h2':np.max(np.abs(z[idxR]))/h**2})
  if n<nmax:U=onecell(sysm,U,dt,k,method)
  if n and n%50==0:print(label,sign,n,flush=True)
 df=pd.DataFrame(rows);df.to_csv(ROOT/f'seeded_scan_{label}_sign{sign:+d}.csv',index=False)
 return {'case':label,'method':method,'kappa':k,'sign':sign,'rho_F':abs(rr['values'][0]),'max_ahead_HP':df.max_ahead_HP_over_h2.max(),
         'min_fit_residual':df.fit_relL2.min(),'n_fits_below_0p02':int((df.fit_relL2<.02).sum())}

if __name__=='__main__':
 cases=[('midpoint',20,'h002_mid_k20'),('gl4',20,'h002_gl4_k20'),('gl4',80,'h002_ref_k80')]
 ss=[]
 for method,k,label in cases:
  for sign in [-1,1]:
   t=time.time();s=run(method,k,label,sign);s['runtime_s']=time.time()-t;ss.append(s);print(json.dumps(s),flush=True)
   pd.DataFrame(ss).to_csv(ROOT/'seeded_h002_scan_summary.csv',index=False)
