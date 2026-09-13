import numpy as np
from scipy.linalg import solve_banded
from wp3_integrators import phi,dphi

OFFS=np.array([-2,-1,0,1,2],int)

def cyclic_penta_solve(sysm,dv,dt,rhs):
    N=sysm.N
    ca=np.array([1,26,66,26,1],float)/120.0
    h=sysm.h
    cl=np.array([-1,2,0,-2,1],float)/(2*h**3)+np.array([-1,-10,0,10,1],float)/(24*h)
    ab=np.zeros((5,N),float); corners=[]
    # J=(2/dt)A+L diag(dv); row j, col q=j+k
    for kk,k in enumerate(OFFS):
        j0=max(0,-k); j1=min(N,N-k)
        if j1>j0:
            j=np.arange(j0,j1);q=j+k
            vals=(2/dt)*ca[kk]+cl[kk]*dv[q]
            ab[2+j[0]-q[0],q]=vals
        if k>0:js=range(N-k,N)
        elif k<0:js=range(0,-k)
        else:js=()
        for j in js:
            q=(j+k)%N
            val=(2/dt)*ca[kk]+cl[kk]*dv[q]
            corners.append((j,q,val))
    r=len(corners);B=np.zeros((N,1+r),float);B[:,0]=rhs
    for p,(i,j,val) in enumerate(corners):B[i,p+1]=val
    X=solve_banded((2,2),ab,B,check_finite=False,overwrite_ab=True)
    x=X[:,0]
    if r:
        Y=X[:,1:];M=np.eye(r);vx=np.empty(r)
        for p,(i,j,val) in enumerate(corners):
            vx[p]=x[j];M[p,:]+=Y[j,:]
        x-=Y@np.linalg.solve(M,vx)
    return x

def midpoint_step_banded(sysm,U,dt,maxit=6,tol=2e-13):
    U=np.asarray(U);W=U.copy()
    for _ in range(maxit):
        G=(2/dt)*(sysm.A@(W-U))+sysm.L@phi(W)
        dd=cyclic_penta_solve(sysm,dphi(W),dt,-G)
        W+=dd
        if np.max(np.abs(dd))<tol:break
    return 2*W-U

def midpoint_cell_banded(sysm,U,dt,kappa,maxit=6,tol=2e-13):
    V=np.asarray(U).copy()
    for _ in range(int(kappa)):V=midpoint_step_banded(sysm,V,dt,maxit=maxit,tol=tol)
    return np.roll(V,-1)
