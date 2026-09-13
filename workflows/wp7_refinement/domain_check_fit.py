import sys,numpy as np,pandas as pd
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
R=Path(__file__).resolve().parents[1]/'wp4_space';sys.path.insert(0,str(R))
from wp4_spatial_methods import solve_profile,compacton

def fit_file(p):
 z=np.load(p);x=z['x'];cells=z['cells'];UU=z['U'];x0=float(z['x0']);h=float(z['h']);L=float(z['L']);N=len(x);base=compacton(x,x0,L);edge=(x0+2*np.pi)/h
 sol=solve_profile('defrutos',h,Leta=90,Neta=16384,tol=3e-13,maxit=2500);Fcs=CubicSpline(sol['eta'],sol['F'],extrapolate=False);V=sol['V']
 def fit(e,jpred,W=15,search=4):
  js=np.arange(int(round(jpred))-W,int(round(jpred))+W+1);y=e[js%N]
  def eva(X):
   s=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0);a=(y@s)/(s@s);r=y-a*s;return r@r,a,s
  q=minimize_scalar(lambda X:eva(X)[0],bounds=(jpred-search,jpred+search),method='bounded');ss,a,s=eva(q.x);r=y-a*s
  return q.x,abs(a)/h**2,np.linalg.norm(r)/(np.linalg.norm(y)+1e-300)
 # Find earliest good candidate
 seed=None
 for k,(cell,row) in enumerate(zip(cells,UU)):
  if cell<80:continue
  e=np.asarray(row,float)-base;hp=(2*e-np.roll(e,1)-np.roll(e,-1))/4;rel=(np.arange(N)-edge)%N;mask=(rel>20)&(rel<min(650,.4*N));v=np.abs(hp);v[~mask]=0;pk,_=find_peaks(v,distance=3,prominence=max(1e-13,.003*v.max()))
  for j in pk[np.argsort(v[pk])[::-1][:30]] if len(pk) else []:
   X,a,rr=fit(e,j)
   if rr<.01 and a>.01:
    seed=(k,cell,X,a,rr);break
  if seed:break
 if not seed:return {'file':str(p),'L':L,'status':'none'}
 k0,c0,X,a,rr=seed;rec=[];prev=c0
 for k in range(k0,min(len(cells),k0+60)):
  cell=int(cells[k]);e=np.asarray(UU[k],float)-base;pred=X+(V*a-1)*(cell-prev);X,a,rr=fit(e,pred,15,5);relc=(X-edge)%N
  if rr>.01:break
  rec.append((cell,X,a,rr,relc));prev=cell
 tr=np.array(rec,float);coef=np.polyfit(tr[:,0]*h,tr[:,4]*h,1);vel=1+coef[0]
 return {'file':str(p),'L':L,'status':'clean','first_cell':int(tr[0,0]),'a_median':float(np.median(tr[:,2])),'rr_median':float(np.median(tr[:,3])),'velocity':vel,'n':len(tr)}
rows=[]
rows.append(fit_file(str(Path(__file__).resolve().parent/'comoving_defrutos_M1676_h0.00749783_k5.npz')))
rows.append(fit_file(str(Path(__file__).resolve().parent/'domaincheck/comoving_defrutos_M1676_h0.00749783_k5.npz')))
d=pd.DataFrame(rows);d.to_csv(Path(__file__).resolve().parent/'wp7_domain_check.csv',index=False);print(d.to_string(index=False))
