import sys,time,numpy as np,pandas as pd
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'workflows/wp2_control'))
from wp2_control_helper import highpass,compacton
from wp2_fast_control import cell_total_hyper
ratio=float(sys.argv[1]); nmax=int(sys.argv[2]) if len(sys.argv)>2 else 300; OUT=Path(__file__).resolve().parent/'derived'; OUT.mkdir(exist_ok=True);ec=0.190972;eps=ratio*ec
d=np.load(ROOT/'workflows/wp3_time/compacton_midpoint_h0.020_k20.npz'); h=float(d['h']); kappa=int(d['kappa']); dt=h/kappa; L=float(d['Ldom']); x=d['x']; x0=float(d['x0']); Us=d['U']; Uc=compacton(x,x0,L); N=len(x); mR=int(np.rint((x0+2*np.pi)/h))%N; idxR=(mR+np.arange(-45,46))%N; idxA=(mR+np.arange(5,min(300,N//2-5)))%N
U=Uc.copy();rows=[];mx=0.;onset={.002:None,.005:None,.01:None,.02:None,.05:None};t0=time.time()
for n in range(nmax+1):
 du=U-Us; hp=np.abs(highpass(du))/h**2; a=float(np.max(hp[idxA]));mx=max(mx,a)
 for th in onset:
  if onset[th] is None and a>=th:onset[th]=n
 if n%5==0:rows.append(dict(ratio=ratio,epsilon=eps,cell=n,time=n*h,hp_ahead_max=a,hp_edge=float(np.max(hp[idxR]))))
 if n<nmax: U=cell_total_hyper(U,h,dt,kappa,eps)
sumr=dict(ratio=ratio,epsilon=eps,max_hp_ahead=mx,onset_0p002=onset[.002],onset_0p005=onset[.005],onset_0p01=onset[.01],onset_0p02=onset[.02],onset_0p05=onset[.05],elapsed=time.time()-t0)
pd.DataFrame(rows).to_csv(OUT/f'wp8_hyper_trace_r{ratio:.2f}.csv',index=False); pd.DataFrame([sumr]).to_csv(OUT/f'wp8_hyper_summary_r{ratio:.2f}.csv',index=False)
print(sumr)
