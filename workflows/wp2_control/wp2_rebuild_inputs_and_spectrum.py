#!/usr/bin/env python3
"""Rebuild the deterministic WP2 base orbits, unstable projector, hyperviscous
continuation table, and the mode files consumed by the nonlinear control runs.

This is the portable orchestration layer added in WP6. It uses the numerical
kernels from wp2_control_helper.py and writes all outputs beside this script.
"""
from pathlib import Path
import numpy as np, pandas as pd
from wp2_control_helper import (solve_base, base_orbit, dominant_real_pair,
    hyper_block_ritz, real_unstable_projector, highpass)

OUT=Path(__file__).resolve().parent

def save_base(base, name):
    orbit=base_orbit(base)
    pair=dominant_real_pair(base,nit=700)
    np.savez_compressed(OUT/name,
        h=base.h,kappa=base.kappa,dt=base.dt,Ldom=base.Ldom,x0=base.x0,x=base.x,
        Uc=base.Uc,Us=base.Us,fp_res=base.fp_res,orbit_mids=np.stack(orbit[1]),
        idxR=base.idxR,idxL=base.idxL,r=pair['r'],l=pair['l'],lhat=pair['lhat'],
        lam=pair['lam'])
    return orbit,pair

def continuation(base, orbit, eps_list, case, nb=16, niter=220):
    rows=[];Q=None
    for eps in eps_list:
        Q,rr,_=hyper_block_ritz(base,float(eps),Q=Q,nb=nb,niter=niter,orbit=orbit)
        edge=sorted([z for z in rr if z['edgefrac']>.90],key=lambda z:abs(z['eig']),reverse=True)
        for j,z in enumerate(edge[:4]):
            rows.append(dict(epsilon=eps,rho=abs(z['eig']),real=np.real(z['eig']),imag=np.imag(z['eig']),
                             residual=z['res'],edge_fraction=z['edgefrac'],case=case,
                             branch=('real' if abs(np.imag(z['eig']))<1e-7 else 'complex'),rank=j+1))
    return rows

def save_selected_mode(base,orbit,eps,name,complex_branch=False,nb=18,niter=450):
    _,rr,_=hyper_block_ritz(base,eps,Q=None,nb=nb,niter=niter,orbit=orbit)
    edge=sorted([z for z in rr if z['edgefrac']>.90],key=lambda z:abs(z['eig']),reverse=True)
    if complex_branch:
        cand=[z for z in edge if abs(np.imag(z['eig']))>.02]
        z=cand[0] if cand else edge[0]
        vr=np.real(z['vec']);vi=np.imag(z['vec'])
        scale=max(np.max(np.abs(highpass(vr)[base.idxR])),np.max(np.abs(highpass(vi)[base.idxR])))
        np.savez_compressed(OUT/name,x=base.x,x0=base.x0,vr=vr/scale,vi=vi/scale,eig=z['eig'],rho=abs(z['eig']),res=z['res'])
    else:
        z=edge[0];v=np.real(z['vec']);v/=np.max(np.abs(highpass(v)[base.idxR]))
        np.savez_compressed(OUT/name,x=base.x,x0=base.x0,v=v,eig=z['eig'],rho=abs(z['eig']),res=z['res'])


def main():
    b5=solve_base(.01,5,Ldom=30.0); orb5,_=save_base(b5,'h001_k5_L30_base.npz')
    b20=solve_base(.02,20,Ldom=16.0); orb20,_=save_base(b20,'h002_k20_L16_base.npz')

    # Unstable real-subspace projector used by the rank-3 test.
    pr=real_unstable_projector(b20,rho_tol=1.001,nb=18,niter=700)
    np.savez_compressed(OUT/'h002_k20_unstable_projector.npz',R=pr['R'],L=pr['L'],Minv=pr['Minv'],eigs=pr['eigs'])

    eps5=[0,.02,.05,.10,.15,.20,.25,.30,.34,.35,.36,.365,.3675,.36875,.37,.375,.38,.40,.42,.45]
    eps20=[0,.05,.10,.15,.17,.175,.18,.1825,.185,.1875,.19,.19125,.1925,.195,.20,.21,.22,.23]
    rows=continuation(b5,orb5,eps5,'h=0.01, kappa=5',nb=16,niter=260)
    rows+=continuation(b20,orb20,eps20,'h=0.02, kappa=20',nb=18,niter=300)
    pd.DataFrame(rows).to_csv(OUT/'orbit_preserving_hyperviscosity_spectrum_rebuilt.csv',index=False)

    for eps in [.30,.3687,.42]:
        save_selected_mode(b5,orb5,eps,f'h001_k5_mode_eps_{eps:.4f}.npz',False,16,420)
    for eps in [.16,.19,.22]:
        save_selected_mode(b20,orb20,eps,f'h002_k20_mode_eps_{eps:.4f}.npz',True,18,520)
    print('WP2 deterministic inputs and continuation rebuilt in',OUT)

if __name__=='__main__': main()
