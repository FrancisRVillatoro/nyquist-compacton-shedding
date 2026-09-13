from __future__ import annotations
import numpy as np
from scipy.linalg import solve_banded

OFFS=np.array([-2,-1,0,1,2],int)
CA=np.array([1,26,66,26,1],float)/120.0
D4C=np.array([1,-4,6,-4,1],float)

def coeff_L(h):
    cb=np.array([-1,-10,0,10,1],float)/(24*h)
    cc=np.array([-1,2,0,-2,1],float)/(2*h**3)
    return cb+cc

def apply_op(u,c):
    y=np.zeros_like(u)
    for k,v in zip(OFFS,c):
        if v!=0:y+=v*np.roll(u,-k)
    return y

def cyclic_penta_solve(dphi,rhs,dt,cl,epsilon=0.0):
    N=len(rhs);ab=np.zeros((5,N),float);corners=[]
    for k,ak,lk,dk in zip(OFFS,CA,cl,D4C):
        vals=(2.0/dt)*ak+lk*dphi+epsilon*dk
        bandrow=2-k
        j0=max(0,k);j1=min(N,N+k)
        if j1>j0:ab[bandrow,j0:j1]=vals[j0:j1]
        if k>0:omitted=range(0,k)
        elif k<0:omitted=range(N+k,N)
        else:omitted=()
        for j in omitted:
            i=(j-k)%N;corners.append((i,j,vals[j]))
    r=len(corners);B=np.zeros((N,1+r),float);B[:,0]=rhs
    for q,(i,j,val) in enumerate(corners):B[i,1+q]=val
    X=solve_banded((2,2),ab,B,check_finite=False)
    x=X[:,0]
    if r:
        Y=X[:,1:];M=np.eye(r);vx=np.empty(r)
        for p,(i,j,val) in enumerate(corners):
            vx[p]=x[j];M[p,:]+=Y[j,:]
        x-=Y@np.linalg.solve(M,vx)
    return x

def midpoint_orbit_control(U,Wstar,h,dt,epsilon,cl=None):
    if cl is None:cl=coeff_L(h)
    W=U.copy()
    for _ in range(4):
        G=(2/dt)*apply_op(W-U,CA)+apply_op(np.abs(W)*W,cl)+epsilon*apply_op(W-Wstar,D4C)
        d=cyclic_penta_solve(2*np.abs(W),-G,dt,cl,epsilon)
        W+=d
        if np.max(np.abs(d))<1e-12:break
    return 2*W-U

def cell_orbit_control(U,orbit_mids,h,dt,epsilon):
    cl=coeff_L(h);V=U.copy()
    for Wstar in orbit_mids:V=midpoint_orbit_control(V,Wstar,h,dt,epsilon,cl)
    return np.roll(V,-1)

def midpoint_total_hyper(U,h,dt,epsilon,cl=None):
    if cl is None:cl=coeff_L(h)
    W=U.copy()
    zero=np.zeros_like(U)
    for _ in range(4):
        G=(2/dt)*apply_op(W-U,CA)+apply_op(np.abs(W)*W,cl)+epsilon*apply_op(W,D4C)
        d=cyclic_penta_solve(2*np.abs(W),-G,dt,cl,epsilon)
        W+=d
        if np.max(np.abs(d))<1e-12:break
    return 2*W-U

def cell_total_hyper(U,h,dt,kappa,epsilon):
    cl=coeff_L(h);V=U.copy()
    for _ in range(kappa):V=midpoint_total_hyper(V,h,dt,epsilon,cl)
    return np.roll(V,-1)

def highpass(z):return (2*z-np.roll(z,1)-np.roll(z,-1))/4.0
