import numpy as np,pandas as pd,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from wp2_fast_control import cell_orbit_control,highpass
OUT=Path(__file__).resolve().parent

def embed_mode(oldx,oldx0,v,newx,newx0,h):
 Nold=len(oldx);Nnew=len(newx);n=np.arange(Nold);rel=((n-int(round(oldx0/h))+Nold//2)%Nold)-Nold//2
 out=np.zeros(Nnew);j0=int(round(newx0/h));out[(j0+rel)%Nnew]=v
 return out

def run(casebase,modefile,eps,a0,ncells,label):
 d=np.load(OUT/casebase);md=np.load(OUT/modefile)
 h=float(d['h']);dt=float(d['dt']);L=float(d['Ldom']);x0=float(d['x0']);Uc=d['Uc'];Us=d['Us'];mids=d['orbit_mids'];idxR=d['idxR'].astype(int)
 if 'v' in md:v=md['v']
 else:v=md['vr']
 if len(v)!=len(Us):v=embed_mode(md['x'],float(md['x0']),v,d['x'],x0,h)
 # Re-normalize to unit right-edge HP peak on the actual domain.
 v=v/(np.max(np.abs(highpass(v)[idxR]))+1e-300)
 U=Us+h*h*a0*v;N=len(U);mR=int(np.rint((x0+2*np.pi)/h))%N;nmax=max(10,int((L/2-2*np.pi)/h)-5);idxA=(mR+np.arange(5,nmax+1))%N
 rows=[];t0=time.time()
 for n in range(ncells+1):
  du=U-Us;hp=np.abs(highpass(du))/h**2
  rows.append(dict(label=label,epsilon=eps,a0=a0,cell=n,time=n*h,hp_edge=np.max(hp[idxR]),hp_ahead_max=np.max(hp[idxA]),hp_ahead_energy=np.sum(hp[idxA]**2),mismatch_l2=np.linalg.norm(du)/h**2))
  if n<ncells:U=cell_orbit_control(U,mids,h,dt,eps)
 pd.DataFrame(rows).to_csv(OUT/f'seed_{label}.csv',index=False)
 print(label,'sec',time.time()-t0,'maxedge',max(r['hp_edge'] for r in rows),'maxahead',max(r['hp_ahead_max'] for r in rows),flush=True)

if __name__=='__main__':
 run('h001_k5_L30_base.npz','h001_k5_mode_eps_0.3000.npz',.3,.005,1000,'h001_k5_eps030')
 run('h001_k5_L30_base.npz','h001_k5_mode_eps_0.3687.npz',.3687,.005,1000,'h001_k5_epscrit')
 run('h001_k5_L30_base.npz','h001_k5_mode_eps_0.4200.npz',.42,.005,1000,'h001_k5_eps042')
 run('h002_k20_L16_base.npz','h002_k20_mode_eps_0.1600.npz',.16,.005,500,'h002_k20_eps016')
 run('h002_k20_L16_base.npz','h002_k20_mode_eps_0.1900.npz',.19,.005,500,'h002_k20_eps019')
 run('h002_k20_L16_base.npz','h002_k20_mode_eps_0.2200.npz',.22,.005,500,'h002_k20_eps022')
