from __future__ import annotations
import sys, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks
OUT=Path(__file__).resolve().parent;ROOT4=Path(__file__).resolve().parents[1]/'wp4_space';sys.path.insert(0,str(ROOT4))
from wp4_spatial_methods import solve_profile,compacton
FILES=sorted(OUT.glob('comoving_defrutos_M*_k5.npz'))

def fit_center(Fcs,e,jpred,W=15,search=3):
 N=len(e);js=np.arange(int(round(jpred))-W,int(round(jpred))+W+1);y=e[js%N]
 def eva(X):
  s=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0.0);den=s@s
  if den<1e-30:return 1e300,0,s
  aa=(y@s)/den;r=y-aa*s;return float(r@r),float(aa),s
 q=minimize_scalar(lambda X:eva(X)[0],bounds=(jpred-search,jpred+search),method='bounded',options={'xatol':1e-10})
 ss,aa,s=eva(q.x);r=y-aa*s
 return float(q.x),float(aa),float(np.linalg.norm(r)/(np.linalg.norm(y)+1e-300)),float(abs(y@s)/(np.linalg.norm(y)*np.linalg.norm(s)+1e-300))

def analyze_file(f):
 z=np.load(f);x=z['x'];cells=z['cells'];UU=z['U'];x0=float(z['x0']);h=float(z['h']);L=float(z['L']);N=len(x);M=int(z['Mwidth']);kappa=int(z['kappa'])
 sol=solve_profile('defrutos',h,Leta=90,Neta=16384,tol=3e-13,maxit=2500);Fcs=CubicSpline(sol['eta'],sol['F'],extrapolate=False);V=float(sol['V'])
 baseline=compacton(x,x0,L); edge_idx=(x0+2*np.pi)/h
 cand=[]
 for kk,(cell,row) in enumerate(zip(cells,UU)):
  if cell<80:continue
  e=np.asarray(row,float)-baseline;hp=(2*e-np.roll(e,1)-np.roll(e,-1))/4
  rel=(np.arange(N)-edge_idx)%N;maxrel=min(650,.40*N);mask=(rel>20)&(rel<maxrel)
  vals=np.abs(hp);vv=vals.copy();vv[~mask]=0
  peaks,_=find_peaks(vv,distance=3,prominence=max(1e-13,0.003*vv.max()))
  if len(peaks): peaks=peaks[np.argsort(vv[peaks])[::-1][:30]]
  for j in peaks:
   X,A,rr,corr=fit_center(Fcs,e,float(j),15,3);a=abs(A)/h**2;relc=(X-edge_idx)%N
   if a<0.002 or relc<20 or relc>maxrel:continue
   cand.append(dict(Mwidth=M,h=h,snapshot=kk,cell=int(cell),t=float(cell*h),center_index=X,relative_center_cells=relc,A=abs(A),A_over_h2=a,profile_relL2=rr,correlation=corr,hp_peak=vv[j]))
 c=pd.DataFrame(cand)
 if c.empty:return z,sol,c,None,pd.DataFrame()
 good=c[(c.profile_relL2<0.02)&(c.correlation>.999)&(c.A_over_h2>.01)].sort_values(['cell','profile_relL2'])
 # Try candidate seeds, choose persistent track, prefer earliest if persistence comparable.
 tracks=[]
 for _,seed in good.head(60).iterrows():
  rec=[];X=seed.center_index;a=seed.A_over_h2;prev=int(seed.cell);fail=0
  for kk in range(int(seed.snapshot),min(len(cells),int(seed.snapshot)+100)):
   cell=int(cells[kk]);e=np.asarray(UU[kk],float)-baseline
   Xpred=X+V*a*(cell-prev) # absolute speed in lab cells per compacton cell-crossing; comoving center advances (v-1), not v
   # correction: cell map has shifted compacton left one cell, so relative packet advance is v-1 per cell.
   Xpred=X+(V*a-1.0)*(cell-prev)
   Xn,An,rr,corr=fit_center(Fcs,e,Xpred,15,5);an=abs(An)/h**2;relc=(Xn-edge_idx)%N
   if rr>.025 or corr<.995 or an<.003 or relc<18 or relc>.42*N:
    fail+=1
    if fail>=2:break
   else:fail=0
   rec.append(dict(Mwidth=M,h=h,snapshot=kk,cell=cell,t=cell*h,center_index=Xn,relative_center_cells=relc,A=abs(An),A_over_h2=an,profile_relL2=rr,correlation=corr))
   X,a,prev=Xn,an,cell
  tr=pd.DataFrame(rec)
  clean=tr[(tr.profile_relL2<.01)&(tr.correlation>.999)] if len(tr) else tr
  if len(clean)>=8:
   tracks.append((len(clean),float(seed.cell),float(clean.profile_relL2.median()),seed,clean))
 if not tracks:return z,sol,c,None,pd.DataFrame()
 # keep tracks with max length within 10%, choose earliest; this identifies first persistent clean packet.
 maxlen=max(t[0] for t in tracks);eligible=[t for t in tracks if t[0]>=max(8,int(.9*maxlen))];chosen=min(eligible,key=lambda q:(q[1],q[2]));tr=chosen[4].reset_index(drop=True)
 if len(tr)>60:tr=tr.iloc[:60].copy()
 return z,sol,c,chosen[3],tr

summ=[];cands=[];trs=[];profrows=[]
for f in FILES:
 z,sol,c,seed,tr=analyze_file(f);h=float(z['h']);M=int(z['Mwidth']);L=float(z['L']);profrows.append(dict(Mwidth=M,h=h,V_h=sol['V'],FWHM_cells=sol['fwhm'],profile_residual=sol['residual']))
 if len(c):cands.append(c)
 if tr.empty:
  # report best candidate for diagnostics
  br=float(c.profile_relL2.min()) if len(c) else np.nan
  summ.append(dict(Mwidth=M,h=h,L=L,status='no_clean_track',V_h=sol['V'],best_candidate_residual=br))
  print('NO',M,h,'best',br)
  continue
 trs.append(tr)
 # velocity in comoving frame: physical relative center = rel_cells*h, slope = v-1; v = 1+slope
 coef,cov=np.polyfit(tr.t.values,tr.relative_center_cells.values*h,1,cov=True);v=1+coef[0];vse=float(np.sqrt(cov[0,0]))
 a=float(tr.A_over_h2.median());A=float(tr.A.median());rr=float(tr.profile_relL2.median());corr=float(tr.correlation.median())
 # representative observed local norms
 mid=tr.iloc[len(tr)//2]; row=np.asarray(z['U'][int(mid.snapshot)],float)-compacton(z['x'],float(z['x0']),L);j0=int(round(mid.center_index));js=np.arange(j0-30,j0+31);yl=row[js%len(row)]
 obsinf=float(np.max(abs(yl)));obs1=float(h*np.sum(abs(yl)));obs2=float(np.sqrt(h*np.sum(yl**2)))
 massabs=float(np.trapezoid(abs(sol['F']),sol['eta']));mass2=float(np.trapezoid(sol['F']**2,sol['eta']))
 fit1=A*h*massabs;fit2=A*np.sqrt(h*mass2)
 summ.append(dict(Mwidth=M,h=h,L=L,status='clean',V_h=sol['V'],FWHM_cells=sol['fwhm'],FWHM_x=sol['fwhm']*h,
  first_clean_cell=int(tr.cell.min()),last_track_cell=int(tr.cell.max()),n_track=len(tr),A_median=A,A_over_h2_median=a,A_over_h2_std=float(tr.A_over_h2.std(ddof=1)),profile_relL2_median=rr,profile_relL2_max=float(tr.profile_relL2.max()),correlation_median=corr,
  velocity=v,velocity_stderr=vse,v_over_Ah2=v/a,relative_velocity_vs_semidiscrete=(v-sol['V']*a)/(sol['V']*a),observed_Linf=obsinf,observed_L1_local=obs1,observed_L2_local=obs2,fitted_Linf=A,fitted_L1=fit1,fitted_L2=fit2,best_candidate_residual=float(c.profile_relL2.min())))
 print('TRACK',M,'h',h,'cell',tr.cell.min(),tr.cell.max(),'a',a,'rr',rr,'v',v)

s=pd.DataFrame(summ).sort_values('h');s.to_csv(OUT/'wp7_refinement_summary.csv',index=False);pd.DataFrame(profrows).sort_values('h').to_csv(OUT/'wp7_profile_family.csv',index=False)
if cands:pd.concat(cands,ignore_index=True).to_csv(OUT/'wp7_candidate_scan.csv',index=False)
if trs:pd.concat(trs,ignore_index=True).to_csv(OUT/'wp7_clean_packet_tracks.csv',index=False)
print('\n',s.to_string(index=False))
g=s[s.status=='clean'];fits=[]
for col,exp in [('A_median',2),('A_over_h2_median',0),('FWHM_x',1),('fitted_L1',3),('fitted_L2',2.5),('observed_Linf',2),('observed_L1_local',3),('observed_L2_local',2.5)]:
 x=np.log(g.h.values);y=np.log(g[col].values);coef,cov=np.polyfit(x,y,1,cov=True);yp=np.polyval(coef,x);r2=1-np.sum((y-yp)**2)/np.sum((y-y.mean())**2) if np.sum((y-y.mean())**2)>0 else 1
 fits.append(dict(quantity=col,observed_exponent=coef[0],stderr=float(np.sqrt(cov[0,0])),expected_exponent=exp,R2=r2,prefactor=np.exp(coef[1])))
f=pd.DataFrame(fits);f.to_csv(OUT/'wp7_scaling_fits.csv',index=False);print('\nFITS\n',f.to_string(index=False))
