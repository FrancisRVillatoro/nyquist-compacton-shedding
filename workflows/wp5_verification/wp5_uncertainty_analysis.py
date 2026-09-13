import numpy as np, pandas as pd, math
from pathlib import Path
from scipy.stats import bootstrap

ROOT=Path(__file__).resolve().parent
ROOT.mkdir(exist_ok=True)
RNG=np.random.default_rng(20260904)
B=5000

# Revised temporal correction coefficients from WP3.
C_MP={2:-11.659289,4:452.119772,6:-21049.697543}
C_GL4={4:-44.079876,6:719.228502,8:8148.209713}

def corr(method,chi):
    C=C_MP if method=='midpoint' else C_GL4
    return 1+sum(c*chi**p for p,c in C.items())

def unwrap_phys(x,L=250.0):
    return np.unwrap(np.asarray(x)/L*2*np.pi)*L/(2*np.pi)

def slope(x,y):
    return np.polyfit(x,y,1)[0]

def percentile_ci(a,alpha=.05):
    return tuple(np.quantile(a,[alpha/2,1-alpha/2]))

rows=[]; bootrows=[]
# ------------------------------------------------------------------
# Archived midpoint packets: combine snapshot resampling and fit-window variants.
# ------------------------------------------------------------------
rob=pd.read_csv(Path(__file__).resolve().parents[2]/'data'/'derived'/'historical'/'first_packet_profile_fit_window_robustness.csv')
for case,h,kappa,tmin,tmax,Vh in [
    ('CFL5',.01,5,6,18,20.66153231806912),
    ('CFL20',.02,20,27,35,20.660521654243198),
]:
    d=rob[(rob.case==case)&(rob.t>=tmin)&(rob.t<=tmax)].copy()
    Ws=sorted(d.W.unique())
    # Point estimate with canonical W=20.
    c=d[d.W==20].sort_values('t')
    xx=unwrap_phys(c.xfit.values)
    vhat=slope(c.t.values,xx)
    ahat=float(np.median(c.A_over_h2))
    epshat=float(np.median(c.relL2))
    vp=Vh*ahat*corr('midpoint',ahat/kappa)
    # Window-only envelope.
    vals=[]
    for W,g in d.groupby('W'):
        g=g.sort_values('t'); xg=unwrap_phys(g.xfit.values)
        a=float(np.median(g.A_over_h2)); v=slope(g.t.values,xg)
        vt=Vh*a*corr('midpoint',a/kappa)
        vals.append((W,a,v,vt,float(np.median(g.relL2))))
    vw=pd.DataFrame(vals,columns=['W','a','v','vth','resid'])
    # Bootstrap: choose one of the available windows each replicate and resample times.
    av=[];vv=[];vt=[];rr=[]
    groups={W:d[d.W==W].sort_values('t').reset_index(drop=True) for W in Ws}
    n=len(c)
    for b in range(B):
        W=int(RNG.choice(Ws)); g=groups[W]
        idx=RNG.integers(0,len(g),len(g))
        # ensure >=2 unique times, otherwise redraw (rare)
        for _ in range(10):
            if np.unique(g.t.values[idx]).size>=2:break
            idx=RNG.integers(0,len(g),len(g))
        gb=g.iloc[idx].sort_values('t')
        x=unwrap_phys(gb.xfit.values)
        v=slope(gb.t.values,x)
        a=float(np.median(gb.A_over_h2))
        av.append(a);vv.append(v);vt.append(Vh*a*corr('midpoint',a/kappa));rr.append(float(np.median(gb.relL2)))
    av=np.array(av);vv=np.array(vv);vt=np.array(vt);rr=np.array(rr)
    rows.append(dict(dataset=f'archived midpoint {case}',method='midpoint',h=h,kappa=kappa,n_snapshots=n,
                     A_over_h2=ahat,A_ci_low=percentile_ci(av)[0],A_ci_high=percentile_ci(av)[1],
                     measured_velocity=vhat,v_ci_low=percentile_ci(vv)[0],v_ci_high=percentile_ci(vv)[1],
                     predicted_velocity=vp,vth_ci_low=percentile_ci(vt)[0],vth_ci_high=percentile_ci(vt)[1],
                     delta_v=vhat-vp,delta_ci_low=percentile_ci(vv-vt)[0],delta_ci_high=percentile_ci(vv-vt)[1],
                     profile_residual=epshat,resid_ci_low=percentile_ci(rr)[0],resid_ci_high=percentile_ci(rr)[1],
                     window_velocity_peak_to_peak=float(vw.v.max()-vw.v.min()),
                     window_A_peak_to_peak=float(vw.a.max()-vw.a.min()),
                     window_prediction_peak_to_peak=float(vw.vth.max()-vw.vth.min())))
    pd.DataFrame({'A':av,'v':vv,'vth':vt,'delta':vv-vt,'resid':rr}).to_csv(ROOT/f'bootstrap_{case}.csv',index=False)
    vw.to_csv(ROOT/f'window_sensitivity_{case}.csv',index=False)

# ------------------------------------------------------------------
# Natural GL4 packets (WP3), using clean ranges in natural_gl4_h001_k5.csv.
# ------------------------------------------------------------------
gl4=pd.read_csv(Path(__file__).resolve().parents[1]/'wp3_time'/'natural_gl4_h001_k5.csv')
# track definitions from WP3 summary
trackdefs=[('GL4 packet I',370,420,20.66153231806912),('GL4 packet II',425,455,20.66153231806912)]
for name,cmin,cmax,Vh in trackdefs:
    g=gl4[(gl4.cell>=cmin)&(gl4.cell<=cmax)&gl4.best_fit_center_rel_cells.notna()&(gl4.best_fit_relL2<0.01)].copy().sort_values('cell')
    t=g.time.values
    # center relative to edge in cells. absolute position speed = 1 + h*d(center_rel)/dt, or fit absolute x = t + h*n_rel.
    y=t + .01*g.best_fit_center_rel_cells.values
    vhat=slope(t,y)
    ahat=float(np.median(g.best_fit_A_over_h2)); epshat=float(np.median(g.best_fit_relL2))
    vp=Vh*ahat*corr('gl4',ahat/5)
    av=[];vv=[];vt=[];rr=[]
    for b in range(B):
        idx=RNG.integers(0,len(g),len(g))
        for _ in range(10):
            if np.unique(t[idx]).size>=2:break
            idx=RNG.integers(0,len(g),len(g))
        gb=g.iloc[idx].sort_values('time'); tb=gb.time.values;yb=tb+.01*gb.best_fit_center_rel_cells.values
        v=slope(tb,yb);a=float(np.median(gb.best_fit_A_over_h2))
        av.append(a);vv.append(v);vt.append(Vh*a*corr('gl4',a/5));rr.append(float(np.median(gb.best_fit_relL2)))
    av=np.array(av);vv=np.array(vv);vt=np.array(vt);rr=np.array(rr)
    slug=name.replace(' ','_')
    rows.append(dict(dataset=name,method='gl4',h=.01,kappa=5,n_snapshots=len(g),
                     A_over_h2=ahat,A_ci_low=percentile_ci(av)[0],A_ci_high=percentile_ci(av)[1],
                     measured_velocity=vhat,v_ci_low=percentile_ci(vv)[0],v_ci_high=percentile_ci(vv)[1],
                     predicted_velocity=vp,vth_ci_low=percentile_ci(vt)[0],vth_ci_high=percentile_ci(vt)[1],
                     delta_v=vhat-vp,delta_ci_low=percentile_ci(vv-vt)[0],delta_ci_high=percentile_ci(vv-vt)[1],
                     profile_residual=epshat,resid_ci_low=percentile_ci(rr)[0],resid_ci_high=percentile_ci(rr)[1],
                     window_velocity_peak_to_peak=np.nan,window_A_peak_to_peak=np.nan,window_prediction_peak_to_peak=np.nan))
    pd.DataFrame({'A':av,'v':vv,'vth':vt,'delta':vv-vt,'resid':rr}).to_csv(ROOT/f'bootstrap_{slug}.csv',index=False)

# ------------------------------------------------------------------
# Natural Ismail and Pade8 packets from WP4. Semidiscrete prediction used here,
# because WP4 validation compares natural and prepared under the same midpoint dynamics.
# ------------------------------------------------------------------
wp4fits=pd.read_csv(Path(__file__).resolve().parents[1]/'wp4_space'/'natural_packet_snapshot_fits.csv')
wp4sum=pd.read_csv(Path(__file__).resolve().parents[1]/'wp4_space'/'natural_vs_prepared_wave_validation.csv')
for (method,packet),g in wp4fits.groupby(['method','packet']):
    g=g.sort_values('t').copy(); s=wp4sum[(wp4sum.method==method)&(wp4sum.packet==packet)].iloc[0]
    t=g.t.values;y=t+.02*g.n_fit.values
    vhat=slope(t,y);ahat=float(np.median(g.A_over_h2));epshat=float(np.median(g.profile_relL2))
    # prepared wave is the direct fully-discrete control prediction.
    vp=float(s.prepared_velocity)
    av=[];vv=[];rr=[]
    for b in range(B):
        idx=RNG.integers(0,len(g),len(g))
        for _ in range(10):
            if np.unique(t[idx]).size>=2:break
            idx=RNG.integers(0,len(g),len(g))
        gb=g.iloc[idx].sort_values('t');tb=gb.t.values;yb=tb+.02*gb.n_fit.values
        av.append(float(np.median(gb.A_over_h2)));vv.append(slope(tb,yb));rr.append(float(np.median(gb.profile_relL2)))
    av=np.array(av);vv=np.array(vv);rr=np.array(rr)
    # prepared velocity uncertainty from reported regression stderr; use Gaussian MC (negligible but explicit)
    pv=RNG.normal(float(s.prepared_velocity),float(s.prepared_velocity_stderr),B)
    slug=f'{method}_{packet}'
    rows.append(dict(dataset=f'{method} packet {packet}',method=method,h=.02,kappa=20,n_snapshots=len(g),
                     A_over_h2=ahat,A_ci_low=percentile_ci(av)[0],A_ci_high=percentile_ci(av)[1],
                     measured_velocity=vhat,v_ci_low=percentile_ci(vv)[0],v_ci_high=percentile_ci(vv)[1],
                     predicted_velocity=vp,vth_ci_low=percentile_ci(pv)[0],vth_ci_high=percentile_ci(pv)[1],
                     delta_v=vhat-vp,delta_ci_low=percentile_ci(vv-pv)[0],delta_ci_high=percentile_ci(vv-pv)[1],
                     profile_residual=epshat,resid_ci_low=percentile_ci(rr)[0],resid_ci_high=percentile_ci(rr)[1],
                     window_velocity_peak_to_peak=np.nan,window_A_peak_to_peak=np.nan,window_prediction_peak_to_peak=np.nan))
    pd.DataFrame({'A':av,'v':vv,'vprep':pv,'delta':vv-pv,'resid':rr}).to_csv(ROOT/f'bootstrap_{slug}.csv',index=False)

out=pd.DataFrame(rows)
out['relative_velocity_difference'] = out.delta_v/out.predicted_velocity
out['relative_velocity_ci_low'] = out.delta_ci_low/out.predicted_velocity
out['relative_velocity_ci_high'] = out.delta_ci_high/out.predicted_velocity
out.to_csv(ROOT/'packet_uncertainty_summary.csv',index=False)
print(out.to_string(index=False))
