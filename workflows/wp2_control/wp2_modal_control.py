import numpy as np,pandas as pd,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from wp2_fast_control import cell_orbit_control,highpass
OUT=Path(__file__).resolve().parent

def run_rank1(basefile,alphas,ncells,prefix):
 d=np.load(OUT/basefile);h=float(d['h']);dt=float(d['dt']);L=float(d['Ldom']);x0=float(d['x0']);Uc=d['Uc'];Us=d['Us'];mids=d['orbit_mids'];r=d['r'];l=d['l'];lhat=d['lhat'];lam=float(d['lam']);idxR=d['idxR'].astype(int)
 N=len(Uc);mR=int(np.rint((x0+2*np.pi)/h))%N;nmax=int((L/2-2*np.pi)/h)-5;idxA=(mR+np.arange(5,nmax+1))%N
 rows=[]
 for alpha in alphas:
  U=Uc.copy();t0=time.time()
  for n in range(ncells+1):
   du=U-Us;hp=np.abs(highpass(du))/h**2;q=float(lhat@(du/h**2))
   rows.append(dict(case=prefix,control='rank1',alpha=alpha,alpha_ratio=alpha/(1-1/abs(lam)),cell=n,time=n*h,q=q,abs_q=abs(q),hp_edge=np.max(hp[idxR]),hp_ahead_max=np.max(hp[idxA]),hp_ahead_energy=np.sum(hp[idxA]**2)))
   if n<ncells:
    y=cell_orbit_control(U,mids,h,dt,0.0)-Us
    U=Us+y-alpha*r*float(l@y)
  print(prefix,'alpha',alpha,'sec',time.time()-t0,flush=True)
 pd.DataFrame(rows).to_csv(OUT/f'{prefix}_rank1_control.csv',index=False)

def run_rank3(basefile,projfile,alphas,ncells,prefix):
 d=np.load(OUT/basefile);p=np.load(OUT/projfile);h=float(d['h']);dt=float(d['dt']);L=float(d['Ldom']);x0=float(d['x0']);Uc=d['Uc'];Us=d['Us'];mids=d['orbit_mids'];idxR=d['idxR'].astype(int);R=p['R'];Lft=p['L'];Minv=p['Minv'];eigs=p['eigs'];rh=max(abs(eigs))
 N=len(Uc);mR=int(np.rint((x0+2*np.pi)/h))%N;nmax=int((L/2-2*np.pi)/h)-5;idxA=(mR+np.arange(5,nmax+1))%N
 rows=[]
 for alpha in alphas:
  U=Uc.copy();t0=time.time()
  for n in range(ncells+1):
   du=U-Us;hp=np.abs(highpass(du))/h**2;coords=Minv@(Lft.T@du)
   rows.append(dict(case=prefix,control='rank3',alpha=alpha,alpha_ratio=alpha/(1-1/rh),cell=n,time=n*h,unstable_subspace_norm=np.linalg.norm(coords)/h**2,hp_edge=np.max(hp[idxR]),hp_ahead_max=np.max(hp[idxA]),hp_ahead_energy=np.sum(hp[idxA]**2)))
   if n<ncells:
    y=cell_orbit_control(U,mids,h,dt,0.0)-Us
    proj=R@(Minv@(Lft.T@y));U=Us+y-alpha*proj
  print(prefix,'rank3 alpha',alpha,'sec',time.time()-t0,flush=True)
 pd.DataFrame(rows).to_csv(OUT/f'{prefix}_rank3_control.csv',index=False)

def run_projection_initial(basefile,betas,ncells,prefix):
 d=np.load(OUT/basefile);h=float(d['h']);dt=float(d['dt']);L=float(d['Ldom']);x0=float(d['x0']);Uc=d['Uc'];Us=d['Us'];mids=d['orbit_mids'];r=d['r'];l=d['l'];lhat=d['lhat'];idxR=d['idxR'].astype(int)
 N=len(Uc);mR=int(np.rint((x0+2*np.pi)/h))%N;nmax=int((L/2-2*np.pi)/h)-5;idxA=(mR+np.arange(5,nmax+1))%N
 qphys=float(l@(Uc-Us))
 rows=[]
 for beta in betas:
  U=Uc+(beta-1)*qphys*r;t0=time.time()
  for n in range(ncells+1):
   du=U-Us;hp=np.abs(highpass(du))/h**2;q=float(lhat@(du/h**2))
   rows.append(dict(case=prefix,beta=beta,cell=n,time=n*h,q=q,abs_q=abs(q),hp_edge=np.max(hp[idxR]),hp_ahead_max=np.max(hp[idxA]),hp_ahead_energy=np.sum(hp[idxA]**2)))
   if n<ncells:U=cell_orbit_control(U,mids,h,dt,0.0)
  print(prefix,'beta',beta,'sec',time.time()-t0,flush=True)
 pd.DataFrame(rows).to_csv(OUT/f'{prefix}_initial_projection.csv',index=False)

if __name__=='__main__':
 d=np.load(OUT/'h001_k5_L30_base.npz');ac=1-1/abs(float(d['lam']))
 run_rank1('h001_k5_L30_base.npz',[0,.8*ac,ac,1.2*ac,1.0],650,'h001_k5')
 run_projection_initial('h001_k5_L30_base.npz',[0,1,4],650,'h001_k5')
 d=np.load(OUT/'h002_k20_L16_base.npz');ac=1-1/abs(float(d['lam']))
 run_rank1('h002_k20_L16_base.npz',[0,1.2*ac,1.0],250,'h002_k20')
 p=np.load(OUT/'h002_k20_unstable_projector.npz');ac3=1-1/max(abs(p['eigs']))
 run_rank3('h002_k20_L16_base.npz','h002_k20_unstable_projector.npz',[0,1.2*ac3,1.0],250,'h002_k20')
