import numpy as np,pandas as pd,sys,time
from pathlib import Path
from scipy.optimize import minimize_scalar
sys.path.insert(0,str(Path(__file__).resolve().parent))
from wp2_fast_control import cell_total_hyper,highpass
OUT=Path(__file__).resolve().parent

def fit_compacton(U,x,h,L,x0guess):
 def profile(center):
  z=(x-center+L/2)%L-L/2;f=np.zeros_like(x);m=np.abs(z)<=2*np.pi;f[m]=4/3*np.cos(z[m]/4)**2;return f
 def obj(center):
  f=profile(center);a=(U@f)/(f@f);return np.sum((U-a*f)**2)
 sol=minimize_scalar(obj,bounds=(x0guess-.5,x0guess+.5),method='bounded')
 f=profile(sol.x);a=(U@f)/(f@f);r=U-a*f
 return sol.x,a,np.linalg.norm(r)/(np.linalg.norm(U)+1e-300)

def run(basefile,epsvals,tend):
 d=np.load(OUT/basefile);h=float(d['h']);dt=float(d['dt']);k=int(d['kappa']);L=float(d['Ldom']);x=d['x'];x0=float(d['x0']);Uc=d['Uc'];Us=d['Us'];N=len(x)
 rows=[]
 for eps in epsvals:
  U=Uc.copy();nc=int(round(tend/h));t0=time.time()
  for n in range(nc+1):
   if n in sorted(set([0,int(.25*nc),int(.5*nc),int(.75*nc),nc])):
    ctr,a,err=fit_compacton(U,x,h,L,x0)
    rows.append(dict(base=basefile,epsilon=eps,time=n*h,cell=n,mass=h*np.sum(U),l2sq=h*np.sum(U**2),peak=np.max(U),fit_center=ctr,fit_amplitude_scale=a,fit_relL2=err,hp_global=np.max(np.abs(highpass(U)))/h**2))
   if n<nc:U=cell_total_hyper(U,h,dt,k,eps)
  print(basefile,eps,'sec',time.time()-t0,flush=True)
 return pd.DataFrame(rows)

if __name__=='__main__':
 a=run('h001_k5_L30_base.npz',[0,1e-5,.36866,.45],1.0)
 b=run('h002_k20_L16_base.npz',[0,1e-5,.1910,.23],1.0)
 pd.concat([a,b],ignore_index=True).to_csv(OUT/'ordinary_hyperviscosity_distortion.csv',index=False)
