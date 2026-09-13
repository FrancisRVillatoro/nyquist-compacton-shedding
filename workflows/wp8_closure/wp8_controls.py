import numpy as np, pandas as pd, time, json, sys
from pathlib import Path
from scipy.linalg import solve_banded
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'derived'; OUT.mkdir(exist_ok=True)
OFFS=np.array([-2,-1,0,1,2],int)
CA=np.array([1,26,66,26,1],float)/120.0
B0=np.array([-1,-10,0,10,1],float)/24.0
C0=np.array([-1,2,0,-2,1],float)/2.0

def apply_op(u,c):
    y=np.zeros_like(u)
    for k,v in zip(OFFS,c):
        if v: y += v*np.roll(u,-k)
    return y

def compacton(x,center,L,c=1.0):
    z=(x-center+L/2)%L-L/2
    u=np.zeros_like(x); m=np.abs(z)<=2*np.pi
    u[m]=4*c/3*np.cos(z[m]/4)**2
    return u

def cyclic_solve(rhs,dph,dt,h,c0):
    cb=B0/h; cl=B0/h+C0/h**3
    N=len(rhs); ab=np.zeros((5,N)); corners=[]
    for k,ak,bk,lk in zip(OFFS,CA,cb,cl):
        vals=(2/dt)*ak-c0*bk+lk*dph
        row=2-k; j0=max(0,k); j1=min(N,N+k)
        if j1>j0: ab[row,j0:j1]=vals[j0:j1]
        omitted=range(0,k) if k>0 else (range(N+k,N) if k<0 else ())
        for j in omitted:
            i=(j-k)%N; corners.append((i,j,vals[j]))
    r=len(corners); B=np.zeros((N,1+r)); B[:,0]=rhs
    for q,(i,j,val) in enumerate(corners): B[i,q+1]=val
    X=solve_banded((2,2),ab,B,check_finite=False)
    x=X[:,0]
    if r:
        Y=X[:,1:]; M=np.eye(r); vx=np.empty(r)
        for p,(i,j,val) in enumerate(corners): vx[p]=x[j]; M[p,:]+=Y[j,:]
        x-=Y@np.linalg.solve(M,vx)
    return x

def step(U,dt,h,c0,kind='abs',maxit=4):
    cb=B0/h; cl=B0/h+C0/h**3; W=U.copy()
    for _ in range(maxit):
        if kind=='abs': ph=np.abs(W)*W; dph=2*np.abs(W)
        elif kind=='square': ph=W*W; dph=2*W
        else: raise ValueError(kind)
        G=(2/dt)*apply_op(W-U,CA)-c0*apply_op(W,cb)+apply_op(ph,cl)
        d=cyclic_solve(-G,dph,dt,h,c0); W+=d
        if np.max(np.abs(d))<2e-13: break
    return 2*W-U

def hp(z): return (2*z-np.roll(z,1)-np.roll(z,-1))/4.0

# DF profile
P=np.load(ROOT/'workflows/wp4_space/nyquist_profile_defrutos_h002.npz')
eta=P['eta']; FF=P['F']; Vh=float(P['V']); Fcs=CubicSpline(eta,FF,extrapolate=False)

def fit_at(du,jguess,W=24,search=5):
    N=len(du); j0=int(round(jguess)); js=np.arange(j0-W,j0+W+1); y=du[js%N]
    def ev(X):
        s=(-1.)**js*np.nan_to_num(Fcs(js-X),nan=0.0); den=s@s
        A=(y@s)/den if den else 0.; r=y-A*s
        return r@r,A,s
    sol=minimize_scalar(lambda X:ev(X)[0],bounds=(jguess-search,jguess+search),method='bounded',options={'xatol':1e-7})
    ss,A,s=ev(sol.x); r=y-A*s
    return sol.x,A,np.linalg.norm(r)/(np.linalg.norm(y)+1e-300),abs(y@s)/(np.linalg.norm(y)*np.linalg.norm(s)+1e-300)

def run(label,c0=0.,kind='abs',kappa=20.,T=15.,L=16.,h=.02,c=1.,save_dt=.02,phiR=.25):
    dt=h/kappa; N=int(round(L/h)); x=np.arange(N)*h
    # choose x0 near 15 with desired right-edge phase
    target=15.; m=int(round((target+2*np.pi)/h-phiR)); x0=(m+phiR)*h-2*np.pi
    U=compacton(x,x0,L,c)
    nsteps=int(round(T/dt)); stride=max(1,int(round(save_dt/dt)))
    rows=[]; snaps=[]; times=[]; tstart=time.time()
    for n in range(nsteps+1):
        if n%stride==0 or n==nsteps:
            t=n*dt; center=x0+(c-c0)*t; uc=compacton(x,center,L,c); du=U-uc; z=hp(du)/h**2
            edge=center+2*np.pi; rel=(x-edge+L/2)%L-L/2
            mask=(rel>0.08)&(rel<max(1.0,L/2-0.5))
            idx=np.where(mask)[0]; vals=np.abs(z[idx]); maxhp=float(vals.max()) if len(vals) else np.nan
            best=None
            if len(vals):
                peaks,_=find_peaks(vals,distance=3)
                if len(peaks)==0: peaks=np.array([int(np.argmax(vals))])
                order=peaks[np.argsort(vals[peaks])[::-1][:20]]
                for pp in order:
                    if vals[pp]<0.003: continue
                    jg=idx[pp]
                    try:
                        X,A,rr,corr=fit_at(du,jg)
                        xpos=(X*h)%L
                        rpos=(xpos-edge+L/2)%L-L/2
                        if not (0.05<rpos<30): continue
                        rec=(rr,-corr,rpos,A,corr)
                        if best is None or rec<best: best=rec
                    except Exception: pass
            if best:
                rr,negc,rpos,A,corr=best
                afit=abs(A)/h**2
            else:
                rr=rpos=A=corr=afit=np.nan
            rows.append(dict(label=label,time=t,step=n,c0=c0,kind=kind,h=h,kappa=kappa,dt=dt,compacton_speed=c-c0,
                             max_ahead_HP_over_h2=maxhp,best_fit_relL2=rr,best_fit_corr=corr,best_fit_A_over_h2=afit,best_fit_center_ahead=rpos))
            snaps.append(U.astype(np.float32)); times.append(t)
        if n<nsteps: U=step(U,dt,h,c0,kind)
        if n and n%(max(1,nsteps//4))==0: print(label,'progress',n,'/',nsteps,flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/f'wp8_{label}_trace.csv',index=False)
    np.savez_compressed(OUT/f'wp8_{label}_snapshots.npz',x=x,t=np.array(times),U=np.stack(snaps),x0=x0,h=h,kappa=kappa,dt=dt,c0=c0,c=c,kind=kind)
    clean=df[(df.best_fit_relL2<.02)&(df.best_fit_corr>.98)&(df.best_fit_A_over_h2>.03)&(df.best_fit_center_ahead>.15)].copy()
    summary=dict(label=label,c0=c0,kind=kind,h=h,kappa=kappa,dt=dt,T=T,x0=x0,phiR=((x0+2*np.pi)/h)%1,
                 max_ahead_HP_over_h2=float(df.max_ahead_HP_over_h2.max()),n_clean=int(len(clean)),first_clean_time=(float(clean.time.min()) if len(clean) else None),
                 A_over_h2_median=(float(clean.best_fit_A_over_h2.median()) if len(clean) else None),profile_residual_median=(float(clean.best_fit_relL2.median()) if len(clean) else None),
                 elapsed=time.time()-tstart,Vh=Vh)
    # velocity of longest contiguous clean segment if enough points
    if len(clean)>=5:
        clean=clean.sort_values('time'); gap=1.6*save_dt; groups=(clean.time.diff().fillna(save_dt)>gap).cumsum(); gid=groups.value_counts().idxmax(); cl=clean[groups==gid]
        if len(cl)>=5:
            # absolute packet position unwrapped = front + rel
            pos=(x0+2*np.pi)+(c-c0)*cl.time.values+cl.best_fit_center_ahead.values
            coef=np.polyfit(cl.time.values,pos,1); summary['packet_velocity']=float(coef[0]);summary['segment_n']=int(len(cl));summary['segment_tmin']=float(cl.time.min());summary['segment_tmax']=float(cl.time.max())
            am=float(cl.best_fit_A_over_h2.median());summary['segment_A_over_h2']=am;summary['pred_velocity_shifted']=float(5*c0+Vh*am*(1-11.659289*(am/kappa)**2+452.119772*(am/kappa)**4-2.10497e4*(am/kappa)**6))
    print('SUMMARY',json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('case')
    a=ap.parse_args()
    cfg={
      'c0_half':dict(c0=.5,kind='abs',kappa=20.,T=25.),
      'c0_equal_c':dict(c0=1.,kind='abs',kappa=20.,T=12.),
      'K22_c0zero':dict(c0=0.,kind='square',kappa=20.,T=12.),
      'kappa20p5':dict(c0=0.,kind='abs',kappa=20.5,T=12.),
      'baseline':dict(c0=0.,kind='abs',kappa=20.,T=12.)}
    sm=run(a.case,**cfg[a.case]);
    import json; OUT/('wp8_'+a.case+'_summary.json').write_text(json.dumps(sm,indent=2))
