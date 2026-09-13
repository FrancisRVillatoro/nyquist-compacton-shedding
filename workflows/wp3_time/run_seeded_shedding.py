import sys,time,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from wp3_integrators import *

def hp(z): return (2*z-np.roll(z,1)-np.roll(z,-1))/4.0

def load_profile(h):
    p=Path(__file__).resolve().parent/f'nyquist_profile_h{h:.2f}.npz'
    d=np.load(p);return CubicSpline(d['eta'],d['F'],extrapolate=False),float(d['V'])

def phase_real(v,idx):
    if np.max(np.abs(v.imag))<1e-12: return v.real
    j=idx[np.argmax(np.abs(v[idx]))]
    w=v*np.exp(-1j*np.angle(v[j]))
    return w.real

def fit_packet(du,Fcs,jpred,W=20,search=4.0):
    N=len(du);j0=int(np.rint(jpred));js=np.arange(j0-W,j0+W+1);y=du[js%N]
    def eva(X):
        s=(-1.0)**js*np.nan_to_num(Fcs(js-X),nan=0.0)
        den=s@s
        if den<1e-30:return 1e99,0,s
        A=(y@s)/den;r=y-A*s
        return r@r,A,s
    sol=minimize_scalar(lambda X:eva(X)[0],bounds=(jpred-search,jpred+search),method='bounded',options={'xatol':1e-9})
    ss,A,s=eva(sol.x);r=y-A*s
    rel=np.linalg.norm(r)/(np.linalg.norm(y)+1e-300)
    corr=abs(y@s)/(np.linalg.norm(y)*np.linalg.norm(s)+1e-300)
    return sol.x,A,rel,corr

def run(h,kappa,method,label,eps=0.005,L=16.0):
    p=ROOT/f'compacton_{method}_h{h:.3f}_k{kappa}.npz'
    d=np.load(p);x=d['x'];Us=d['U'];x0=float(d['x0']);N=len(Us)
    sysm=SpatialSystem(N,h);dt=h/kappa
    rr=leading_ritz(sysm,Us,h,kappa,method,x0,nb=10,maxiter=500,tol=1e-8,seed=44000+int(h*10000)+kappa)
    r=phase_real(rr['leading_vector'],np.arange(N))
    mR=int(np.rint((x0+2*np.pi)/h))%N
    idxR=(mR+np.arange(-35,36))%N
    hr=np.max(np.abs(hp(r)[idxR]));rhat=r/hr
    U=Us+h*h*eps*rhat
    Fcs,Vh=load_profile(h)
    nmax=120 if h<0.015 else 60
    nrel=np.arange(-80, min(170,N//2-1)+1)
    idx=(mR+nrel)%N
    rows=[];track_started=False;Xpred=None;lastX=None;lastslope=1.0
    for n in range(nmax+1):
        du=U-Us;z=hp(du);zz=z[idx]/h**2;e=zz*zz;es=e.sum()+1e-300
        edgepeak=np.max(np.abs(z[idxR]))/h**2
        aheadmask=nrel>=5
        av=np.abs(zz[aheadmask]);nna=nrel[aheadmask]
        jj=int(np.argmax(av));aheadpeak=float(av[jj]);peakrel=float(nna[jj])
        centroid=float(np.sum(nrel*e)/es);aheadfrac=float(e[nrel>0].sum()/es);corefrac=float(e[np.abs(nrel)<=35].sum()/es)
        fitX=np.nan;fitA=np.nan;fitrel=np.nan;fitcorr=np.nan
        if track_started:
            jguess=Xpred
        else:
            jguess=mR+peakrel
        if aheadpeak>0.01 or track_started:
            try:
                fitX,fitA,fitrel,fitcorr=fit_packet(du,Fcs,jguess,W=20,search=5.0)
                # unwrapped relative coordinate to mR
                while fitX-mR>N/2: fitX-=N
                while fitX-mR<-N/2: fitX+=N
                if (not track_started) and fitrel<0.08 and (fitX-mR)>5 and abs(fitA)/h**2>0.03:
                    track_started=True
                if track_started:
                    if lastX is not None: lastslope=0.7*lastslope+0.3*(fitX-lastX)
                    lastX=fitX;Xpred=fitX+lastslope
            except Exception:
                pass
        rows.append({'case':label,'h':h,'method':method,'kappa':kappa,'cell':n,'time':n*h,
                     'rho_F':abs(rr['values'][0]),'mode_type':'real' if abs(rr['values'][0].imag)<1e-8 else 'complex',
                     'edge_HP_over_h2':edgepeak,'ahead_HP_over_h2':aheadpeak,'peak_rel_cells':peakrel,
                     'centroid_rel_cells':centroid,'ahead_energy_fraction':aheadfrac,'core35_energy_fraction':corefrac,
                     'fit_center_rel_cells':fitX-mR if np.isfinite(fitX) else np.nan,
                     'fit_A_over_h2':abs(fitA)/h**2 if np.isfinite(fitA) else np.nan,
                     'fit_relL2':fitrel,'fit_corr':fitcorr,'track_started':track_started})
        if n<nmax: U=sysm.cell_map(U,dt,kappa,method)
    out=pd.DataFrame(rows)
    out.to_csv(ROOT/f'seeded_shedding_{label}.csv',index=False)
    clean=out[(out.fit_relL2<0.02)&(out.fit_center_rel_cells>8)&(out.fit_A_over_h2>0.03)]
    if len(clean)>=5:
        # use first contiguous clean segment before potential wrap/interactions
        clean=clean.sort_values('cell')
        start=int(clean.cell.iloc[0]);clean=clean[clean.cell<=start+min(40,nmax-start)]
        coef=np.polyfit(clean.cell,clean.fit_center_rel_cells,1)
        vfit=1+coef[0]
        Amed=clean.fit_A_over_h2.median(); vsemi=Vh*Amed
        rel=(vfit-vsemi)/vsemi
    else:vfit=Amed=vsemi=rel=np.nan
    # first clear forward shedding marker
    sh=out[(out.centroid_rel_cells>10)&(out.ahead_energy_fraction>0.6)]
    tsh=float(sh.time.iloc[0]) if len(sh) else np.nan
    summ={'case':label,'h':h,'method':method,'kappa':kappa,'rho_F':abs(rr['values'][0]),
          'lambda_real':rr['values'][0].real,'lambda_imag':rr['values'][0].imag,
          'seed_edge_HP_over_h2':eps,'first_forward_shift_time':tsh,
          'max_edge_HP_over_h2':out.edge_HP_over_h2.max(),'max_ahead_HP_over_h2':out.ahead_HP_over_h2.max(),
          'first_clean_cell':int(clean.cell.min()) if len(clean) else np.nan,
          'n_clean':len(clean),'median_clean_A_over_h2':Amed,'fitted_absolute_velocity':vfit,
          'semidiscrete_velocity_from_A':vsemi,'velocity_relative_error_vs_semidiscrete':rel,
          'median_clean_profile_residual':clean.fit_relL2.median() if len(clean) else np.nan,
          'min_profile_residual':out.fit_relL2.min()}
    return out,summ

if __name__=='__main__':
    cases=[(.01,5,'midpoint','h001_mid_k5'),(.01,5,'gl4','h001_gl4_k5'),(.01,80,'gl4','h001_ref_k80'),
           (.02,20,'midpoint','h002_mid_k20'),(.02,20,'gl4','h002_gl4_k20'),(.02,80,'gl4','h002_ref_k80')]
    sums=[]
    for c in cases:
        t=time.time();_,s=run(*c);s['runtime_s']=time.time()-t;sums.append(s);print(json.dumps(s,default=float),flush=True)
        pd.DataFrame(sums).to_csv(ROOT/'seeded_shedding_summary.csv',index=False)
