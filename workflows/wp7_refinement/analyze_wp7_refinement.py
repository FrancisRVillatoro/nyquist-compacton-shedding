from __future__ import annotations
import sys, math, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

OUT=Path(__file__).resolve().parent
ROOT4=Path(__file__).resolve().parents[1]/'wp4_space'
sys.path.insert(0,str(ROOT4))
from wp4_spatial_methods import solve_profile,compacton

FILES=sorted(OUT.glob('natural_defrutos_M*_k5.npz'))
# Keep newest L~32 versions if duplicates in basename impossible; files overwritten for shared basenames.

def profile_for_h(h):
    sol=solve_profile('defrutos',h,Leta=90,Neta=16384,tol=3e-13,maxit=2500)
    return sol,CubicSpline(sol['eta'],sol['F'],extrapolate=False)

def fit_center(Fcs,e,jpred,W=15,search=3.0):
    N=len(e); j0=int(round(jpred)); js=np.arange(j0-W,j0+W+1); y=e[js%N]
    def eva(X):
        s=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0.0)
        den=s@s
        if den<1e-30:return 1e300,0.0,s
        aa=(y@s)/den; r=y-aa*s
        return float(r@r),float(aa),s
    sol=minimize_scalar(lambda X:eva(X)[0],bounds=(jpred-search,jpred+search),method='bounded',options={'xatol':1e-10})
    ss,aa,s=eva(sol.x); r=y-aa*s
    rel=float(np.linalg.norm(r)/(np.linalg.norm(y)+1e-300))
    corr=float(abs(y@s)/(np.linalg.norm(y)*np.linalg.norm(s)+1e-300))
    return float(sol.x),float(aa),rel,corr

def candidate_scan(z, Fcs, W=15):
    x=np.asarray(z['x']); tt=np.asarray(z['t']); UU=np.asarray(z['U']);x0=float(z['x0']);h=float(z['h']);L=float(z['L']);N=len(x)
    rows=[]
    for kk,(t,row) in enumerate(zip(tt,UU)):
        if t/h<120: continue
        e=np.asarray(row,float)-compacton(x,x0+t,L)
        hp=(2*e-np.roll(e,1)-np.roll(e,-1))/4.0
        edge_idx=(x0+t+2*np.pi)/h
        rel=(np.arange(N)-edge_idx)%N
        # Forward sector, far enough from compacton and well before periodic half box.
        maxcells=min(650.0,0.38*N)
        mask=(rel>22)&(rel<maxcells)
        vals=np.abs(hp); vv=vals.copy();vv[~mask]=0
        prom=max(1e-12,0.004*vv.max())
        peaks,_=find_peaks(vv,distance=3,prominence=prom)
        if len(peaks):peaks=peaks[np.argsort(vv[peaks])[::-1][:24]]
        for j in peaks:
            X,aa,rr,corr=fit_center(Fcs,e,float(j),W=W,search=3.0)
            relc=float((X-edge_idx)%N); a=abs(aa)/h**2
            if a<0.005 or relc<=22 or relc>=maxcells: continue
            rows.append(dict(snapshot=kk,t=float(t),cell_time=float(t/h),center_index=X,relative_center_cells=relc,
                             A=float(abs(aa)),A_over_h2=float(a),profile_relL2=rr,correlation=corr,hp_peak=float(vv[j])))
    return pd.DataFrame(rows)

def choose_track(cand,z,Fcs,Vh,W=15):
    # Candidate pool of convincingly own-family packets.
    if cand.empty:return None,pd.DataFrame()
    good=cand[(cand.profile_relL2<0.01)&(cand.correlation>0.999)&(cand.A_over_h2>0.015)].copy()
    if good.empty:return None,pd.DataFrame()
    # For every candidate at an early clean time, attempt forward/backward tracking and score persistence.
    x=np.asarray(z['x']);tt=np.asarray(z['t']);UU=np.asarray(z['U']);x0=float(z['x0']);h=float(z['h']);L=float(z['L']);N=len(x)
    dt_save=float(np.median(np.diff(tt)))
    seeds=good.sort_values(['cell_time','profile_relL2']).head(30)
    best_score=None; best_track=None; best_seed=None
    for _,seed in seeds.iterrows():
        # Track forward from seed using predicted cell speed V*a and fit around predicted center.
        records=[]; X=float(seed.center_index); a=float(seed.A_over_h2)
        start=int(seed.snapshot)
        failures=0
        for k in range(start,min(len(tt),start+90)):
            t=float(tt[k]);e=np.asarray(UU[k],float)-compacton(x,x0+t,L)
            if k==start: Xpred=X
            else: Xpred=X + Vh*a*(t-prev_t)/h
            Xn,aan,rr,corr=fit_center(Fcs,e,Xpred,W=W,search=5.0)
            an=abs(aan)/h**2
            edge_idx=(x0+t+2*np.pi)/h;relc=float((Xn-edge_idx)%N)
            if rr>0.02 or corr<0.995 or an<0.005 or relc<20 or relc>0.40*N:
                failures+=1
                if failures>=2:break
            else:
                failures=0
            records.append(dict(snapshot=k,t=t,cell_time=t/h,center_index=Xn,relative_center_cells=relc,A=abs(aan),A_over_h2=an,profile_relL2=rr,correlation=corr))
            X,a,prev_t=Xn,an,t
        tr=pd.DataFrame(records)
        clean=tr[(tr.profile_relL2<0.01)&(tr.correlation>0.999)] if len(tr) else tr
        if len(clean)<8:continue
        # score: persistence first, then earlier time, then smaller residual
        score=(len(clean),-float(seed.cell_time),-float(clean.profile_relL2.median()))
        if best_score is None or score>best_score:
            best_score=score;best_track=clean;best_seed=seed
    if best_track is None:return None,pd.DataFrame()
    # Keep a contiguous clean segment up to 60 snapshots; use middle/late part to avoid formation transition.
    tr=best_track.copy().reset_index(drop=True)
    if len(tr)>60: tr=tr.iloc[:60].copy()
    return best_seed,tr

summ=[]; allcand=[];alltracks=[];profiles=[]
for f in FILES:
    z=np.load(f); h=float(z['h']);M=int(z['Mwidth']); kappa=int(z['kappa']); phase=float(z['phase']);L=float(z['L'])
    sol,Fcs=profile_for_h(h);V=float(sol['V']);profiles.append(dict(Mwidth=M,h=h,V_h=V,FWHM_cells=sol['fwhm'],profile_residual=sol['residual']))
    cand=candidate_scan(z,Fcs,W=15);cand['Mwidth']=M;cand['h']=h;allcand.append(cand)
    seed,tr=choose_track(cand,z,Fcs,V,W=15)
    if tr.empty:
        summ.append(dict(Mwidth=M,h=h,kappa=kappa,phase=phase,L=L,status='no_clean_track',V_h=V,FWHM_cells=sol['fwhm']))
        print('NO TRACK',M,h)
        continue
    tr['Mwidth']=M;tr['h']=h;alltracks.append(tr)
    # Linear fit of unwrapped physical center.
    Xphys=np.unwrap(tr.center_index.values*h/L*2*np.pi)*L/(2*np.pi)
    coef,cov=np.polyfit(tr.t.values,Xphys,1,cov=True);v=float(coef[0]);vse=float(np.sqrt(cov[0,0]))
    a_med=float(tr.A_over_h2.median());A_med=float(tr.A.median()); rrmed=float(tr.profile_relL2.median());rrmax=float(tr.profile_relL2.max());corrmed=float(tr.correlation.median())
    # Observed local norms on +/-30 cells at a representative median-time snapshot.
    imid=len(tr)//2; rec=tr.iloc[imid]; k=int(rec.snapshot); x=np.asarray(z['x']); row=np.asarray(z['U'][k],float);x0=float(z['x0']);t=float(rec.t);N=len(x)
    e=row-compacton(x,x0+t,L); j0=int(round(rec.center_index)); js=np.arange(j0-30,j0+31);yloc=e[js%N]
    Linf=float(np.max(np.abs(yloc)));L1=float(h*np.sum(np.abs(yloc)));L2=float(np.sqrt(h*np.sum(yloc**2)))
    # Fitted-template norms, using continuum eta quadrature from profile.
    eta=sol['eta'];F=sol['F']; mass_abs=float(np.trapezoid(np.abs(F),eta));mass2=float(np.trapezoid(F*F,eta))
    fit_Linf=A_med; fit_L1=A_med*h*mass_abs;fit_L2=A_med*np.sqrt(h*mass2)
    summ.append(dict(Mwidth=M,h=h,kappa=kappa,phase=phase,L=L,status='clean',V_h=V,FWHM_cells=sol['fwhm'],FWHM_x=sol['fwhm']*h,
                     first_clean_cell_time=float(tr.cell_time.min()),first_clean_t=float(tr.t.min()),track_snapshots=len(tr),A_median=A_med,A_over_h2_median=a_med,
                     A_over_h2_std=float(tr.A_over_h2.std(ddof=1)),profile_relL2_median=rrmed,profile_relL2_max=rrmax,correlation_median=corrmed,
                     velocity=v,velocity_stderr=vse,v_over_Ah2=v/a_med,semidiscrete_Vh=V,relative_velocity_vs_Va=(v-V*a_med)/(V*a_med),
                     observed_Linf=Linf,observed_L1_local=L1,observed_L2_local=L2,fitted_Linf=fit_Linf,fitted_L1=fit_L1,fitted_L2=fit_L2))
    print('TRACK',M,'h',h,'cells',tr.cell_time.min(),tr.cell_time.max(),'a',a_med,'rr',rrmed,'v',v)

pd.DataFrame(profiles).sort_values('h').to_csv(OUT/'wp7_profile_family.csv',index=False)
if allcand:pd.concat(allcand,ignore_index=True).to_csv(OUT/'wp7_candidate_scan.csv',index=False)
if alltracks:pd.concat(alltracks,ignore_index=True).to_csv(OUT/'wp7_clean_packet_tracks.csv',index=False)
sdf=pd.DataFrame(summ).sort_values('h');sdf.to_csv(OUT/'wp7_refinement_summary.csv',index=False)
print('\nSUMMARY\n',sdf.to_string(index=False))

# Power-law regressions for clean rows.
g=sdf[sdf.status=='clean'].copy()
fitrows=[]
for col,expect in [('A_median',2.0),('FWHM_x',1.0),('fitted_Linf',2.0),('fitted_L1',3.0),('fitted_L2',2.5),('observed_Linf',2.0),('observed_L1_local',3.0),('observed_L2_local',2.5)]:
    x=np.log(g.h.values);y=np.log(g[col].values);coef,cov=np.polyfit(x,y,1,cov=True);pred=np.polyval(coef,x);r2=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
    fitrows.append(dict(quantity=col,observed_exponent=coef[0],stderr=float(np.sqrt(cov[0,0])),expected_exponent=expect,R2=r2,prefactor=np.exp(coef[1])))
# A/h2 const trend: regress vs h and log slope.
col='A_over_h2_median';x=np.log(g.h.values);y=np.log(g[col].values);coef,cov=np.polyfit(x,y,1,cov=True);pred=np.polyval(coef,x);r2=1-np.sum((y-pred)**2)/np.sum((y-y.mean())**2)
fitrows.append(dict(quantity=col,observed_exponent=coef[0],stderr=float(np.sqrt(cov[0,0])),expected_exponent=0.0,R2=r2,prefactor=np.exp(coef[1])))
pd.DataFrame(fitrows).to_csv(OUT/'wp7_scaling_fits.csv',index=False)
print('\nFITS\n',pd.DataFrame(fitrows).to_string(index=False))
