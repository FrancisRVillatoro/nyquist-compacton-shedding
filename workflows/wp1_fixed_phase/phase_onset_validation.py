#!/usr/bin/env python3
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def build_ops(N, h):
    r=np.arange(N); rows=[]; cols=[]; av=[]; lv=[]
    ca=np.array([1,26,66,26,1],float)/120.0
    cb=np.array([-1,-10,0,10,1],float)/(24*h)
    cc=np.array([-1,2,0,-2,1],float)/(2*h**3)
    for kk,k in enumerate([-2,-1,0,1,2]):
        c=(r+k)%N; rows.extend(r); cols.extend(c)
        av.extend(np.full(N,ca[kk])); lv.extend(np.full(N,cb[kk]+cc[kk]))
    return sp.csc_matrix((av,(rows,cols)),shape=(N,N)), sp.csc_matrix((lv,(rows,cols)),shape=(N,N))


def compacton(x,x0,Ldom):
    z=(x-x0+Ldom/2)%Ldom-Ldom/2
    u=np.zeros_like(x); m=np.abs(z)<=2*np.pi
    u[m]=4/3*np.cos(z[m]/4)**2
    return u


def midpoint(U,A,L,dt,store=False):
    W=np.asarray(U).copy()
    for _ in range(7):
        G=(2/dt)*(A@(W-U))+L@(np.abs(W)*W)
        J=(2/dt)*A+L@sp.diags(2*np.abs(W),0,format='csc')
        d=spla.spsolve(J,-G,permc_spec='NATURAL')
        W+=d
        if np.max(np.abs(d))<8e-13: break
    U1=2*W-U
    if not store: return U1
    Jp=A/dt+L@sp.diags(np.abs(W),0,format='csc')
    Jm=A/dt-L@sp.diags(np.abs(W),0,format='csc')
    return U1,spla.splu(Jp,permc_spec='NATURAL'),Jm


def cell_map(U,A,L,dt,kappa,record=False):
    V=np.asarray(U).copy(); fac=[]
    for _ in range(int(kappa)):
        if record:
            V,lu,Jm=midpoint(V,A,L,dt,True); fac.append((lu,Jm))
        else: V=midpoint(V,A,L,dt)
    V=np.roll(V,-1)
    return (V,fac) if record else V


def solve_tw(Uc,A,L,dt,kappa):
    def F(U): return cell_map(np.asarray(U),A,L,dt,kappa)-U
    try:
        Us=opt.newton_krylov(F,Uc,f_tol=2e-10,maxiter=10,inner_maxiter=20,rdiff=1e-7)
    except opt.NoConvergence as e:
        Us=np.asarray(e.args[0])
    return Us,float(np.linalg.norm(F(Us),np.inf))


def apply_P(z,fac):
    y=np.asarray(z).copy()
    for lu,Jm in fac: y=lu.solve(Jm@y)
    return np.roll(y,-1)


def apply_PT(z,fac):
    y=np.roll(np.asarray(z),1)
    for lu,Jm in fac[::-1]: y=Jm.T@lu.solve(y,trans='T')
    return y


def highpass(z):
    return (2*z-np.roll(z,1)-np.roll(z,-1))/4


def setup(h,kappa,phi,Ldom=16.0,nit=600):
    dt=h/kappa; N=int(round(Ldom/h)); x=np.arange(N)*h
    A,L=build_ops(N,h)
    nR=int(np.rint((Ldom/2+2*np.pi)/h)); xR=h*(nR+phi); x0=xR-2*np.pi
    Uc=compacton(x,x0,Ldom)
    Us,fp=solve_tw(Uc,A,L,dt,int(kappa))
    PUs,fac=cell_map(Us,A,L,dt,int(kappa),record=True)
    R0=PUs-Us
    qL=(x0-2*np.pi)/h; mL=int(np.rint(qL))%N; mR=nR%N
    win=np.arange(-40,41); idxR=(mR+win)%N; idxL=(mL+win)%N
    rng=np.random.default_rng(20260903)
    r=np.zeros(N); ell=np.zeros(N)
    r[idxR]=rng.normal(size=len(idxR)); r[idxL]+=0.3*rng.normal(size=len(idxL))
    ell[idxR]=rng.normal(size=len(idxR)); ell[idxL]+=0.3*rng.normal(size=len(idxL))
    r/=np.linalg.norm(r); ell/=np.linalg.norm(ell)
    for _ in range(nit):
        y=apply_P(r,fac); r=y/np.linalg.norm(y)
        y=apply_PT(ell,fac); ell=y/np.linalg.norm(y)
    lam=float(r@apply_P(r,fac))
    lamL=float(ell@apply_PT(ell,fac))
    if ell@r<0: ell=-ell
    ell=ell/(ell@r)
    hr=float(np.max(np.abs(highpass(r)[idxR])))
    rhat=r/hr; lhat=hr*ell
    q0=float(lhat@((Uc-Us)/h**2))
    eigres=float(np.linalg.norm(apply_P(r,fac)-lam*r)/np.linalg.norm(r))
    leftres=float(np.linalg.norm(apply_PT(ell,fac)-lam*ell)/np.linalg.norm(ell))
    return dict(h=h,kappa=kappa,phi=phi,dt=dt,N=N,Ldom=Ldom,x=x,x0=x0,A=A,L=L,Uc=Uc,Us=Us,
                R0=R0,fp=fp,fac=fac,lam=lam,lamL=lamL,eigres=eigres,leftres=leftres,
                r=r,ell=ell,rhat=rhat,lhat=lhat,idxR=idxR,idxL=idxL,q0=q0)


def trace(base,nmax,map_kind):
    h=base['h']; Us=base['Us']; U=base['Uc'].copy(); q0=base['q0']; sgn=1 if q0>=0 else -1
    A=base['A'];L=base['L'];dt=base['dt'];kappa=base['kappa'];R0=base['R0'];lhat=base['lhat'];idxR=base['idxR']
    rows=[]
    for n in range(nmax+1):
        du=U-Us
        q=float(lhat@(du/h**2))
        qg=sgn*((-1)**n)*q
        hp=float(np.max(np.abs(highpass(du)[idxR]))/h**2)
        rows.append({'cell':n,'time':n*h,'q':q,'q_gauge':qg,'abs_q':abs(q),'hp_edge':hp})
        if n<nmax:
            U=cell_map(U,A,L,dt,kappa)
            if map_kind=='residual_corrected': U=U-R0
    return pd.DataFrame(rows)


def diagnostics(g,base,qturn):
    mu=abs(base['lam']); q0=abs(base['q0'])
    out={}
    for threshold in [0.005,0.01,0.02,0.04,0.06]:
        z=g[g.q_gauge>=threshold]
        obs=float(z.cell.iloc[0]) if len(z) else np.nan
        pred=math.log(threshold/q0)/math.log(mu) if threshold>q0 else 0.0
        out[f'n_obs_q{threshold:g}']=obs
        out[f'n_pred_q{threshold:g}']=pred
        out[f'error_cells_q{threshold:g}']=obs-pred if np.isfinite(obs) else np.nan
    q=g.q_gauge.values
    cand=[i for i in range(1,len(q)-1) if q[i]>=0.02 and q[i]>=q[i-1] and q[i]>q[i+1]]
    if cand:
        i=cand[0]
        out['turn_cell_obs']=float(g.cell.iloc[i]);out['turn_time_obs']=float(g.time.iloc[i]);out['turn_q_obs']=float(q[i])
    else:
        out['turn_cell_obs']=np.nan;out['turn_time_obs']=np.nan;out['turn_q_obs']=np.nan
    out['turn_cell_pred']=math.log(qturn/q0)/math.log(mu)
    out['turn_time_pred']=h*0 if False else base['h']*out['turn_cell_pred']
    out['turn_cell_error']=out['turn_cell_obs']-out['turn_cell_pred'] if np.isfinite(out['turn_cell_obs']) else np.nan
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',default=str(Path(__file__).resolve().parent))
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cases=[
        ('h0.01_k5_min',0.01,5,0.9375,0.07147,140),
        ('h0.01_k5_max',0.01,5,0.3750,0.07147,120),
        ('h0.02_k20_min',0.02,20,0.9375,0.08920,90),
        ('h0.02_k20_max',0.02,20,0.3750,0.08920,75),
    ]
    summaries=[]
    for name,h,k,phi,qturn,nmax in cases:
        print('setup',name,flush=True);tic=time.time();base=setup(h,k,phi)
        np.savez_compressed(out/f'base_{name}.npz',Us=base['Us'],Uc=base['Uc'],r=base['r'],ell=base['ell'],rhat=base['rhat'],lhat=base['lhat'],R0=base['R0'],x=base['x'],h=h,kappa=k,phi=phi,lambda1=base['lam'],q0=base['q0'])
        for kind in ['true','residual_corrected']:
            g=trace(base,nmax,kind);g['case']=name;g['map_kind']=kind;g.to_csv(out/f'trace_{name}_{kind}.csv',index=False)
            d=diagnostics(g,base,qturn)
            row={'case':name,'h':h,'kappa':k,'phi_right':phi,'seed_class':'min' if 'min' in name else 'max','map_kind':kind,
                 'lambda1':base['lam'],'mu':abs(base['lam']),'q0':base['q0'],'q0_abs':abs(base['q0']),
                 'fixed_point_residual':base['fp'],'constant_residual_norm':float(np.linalg.norm(base['R0'],np.inf)),
                 'right_eigen_residual':base['eigres'],'left_eigen_residual':base['leftres'],'runtime_setup_s':time.time()-tic,
                 **d}
            summaries.append(row)
            print(name,kind,'q0',row['q0_abs'],'turn obs/pred',row['turn_cell_obs'],row['turn_cell_pred'],flush=True)
    s=pd.DataFrame(summaries);s.to_csv(out/'phase_onset_validation_summary.csv',index=False)
    # Pairwise latency differences within each parameter pair and map kind.
    pairs=[]
    for (h,k,kind),g in s.groupby(['h','kappa','map_kind']):
        mn=g[g.seed_class=='min'].iloc[0];mx=g[g.seed_class=='max'].iloc[0]
        pred=math.log(mx.q0_abs/mn.q0_abs)/math.log((mx.mu+mn.mu)/2)
        pairs.append({'h':h,'kappa':k,'map_kind':kind,'phi_min':mn.phi_right,'phi_max':mx.phi_right,
                      'q0_min':mn.q0_abs,'q0_max':mx.q0_abs,'seed_ratio':mx.q0_abs/mn.q0_abs,
                      'delta_turn_cells_observed':mn.turn_cell_obs-mx.turn_cell_obs,
                      'delta_turn_cells_predicted':pred,
                      'delta_turn_time_observed':h*(mn.turn_cell_obs-mx.turn_cell_obs),
                      'delta_turn_time_predicted':h*pred,
                      'latency_difference_error_cells':(mn.turn_cell_obs-mx.turn_cell_obs)-pred})
    pd.DataFrame(pairs).to_csv(out/'phase_onset_latency_pairs.csv',index=False)
    (out/'metadata.json').write_text(json.dumps({'cases':[c[0] for c in cases],'description':'Direct nonlinear onset-latency validation at phases of minimum and maximum unstable-mode seed.'},indent=2))

if __name__=='__main__': main()
