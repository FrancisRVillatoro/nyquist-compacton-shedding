import numpy as np
from scipy.linalg import solve_banded
from wp4_spatial_methods import METHODS,OFFS,compacton

def coeffs(method,h):
    m=METHODS[method]
    return np.asarray(m['A'],float),np.asarray(m['B0'],float)/h+np.asarray(m['C0'],float)/h**3

def apply_stencil(u,c):
    y=np.zeros_like(u)
    for k,v in zip(OFFS,c):
        if v!=0:y+=v*np.roll(u,-k)
    return y

def cyclic_penta_solve(dphi,rhs,dt,ca,cl):
    N=len(rhs);ab=np.zeros((5,N),float);corners=[]
    for k,ak,lk in zip(OFFS,ca,cl):
        vals=(2/dt)*ak+lk*dphi
        row=2-k
        j0=max(0,k);j1=min(N,N+k)
        if j1>j0:ab[row,j0:j1]=vals[j0:j1]
        omitted=range(0,k) if k>0 else (range(N+k,N) if k<0 else ())
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

def midpoint_step(U,dt,ca,cl,newton=3,tol=2e-13):
    W=U.copy()
    for _ in range(newton):
        G=(2/dt)*apply_stencil(W-U,ca)+apply_stencil(np.abs(W)*W,cl)
        d=cyclic_penta_solve(2*np.abs(W),-G,dt,ca,cl)
        W+=d
        if np.max(np.abs(d))<tol:break
    return 2*W-U

def simulate(method,h,kappa,Ldom,T,x0,save_dt=0.05):
    dt=h/kappa;N=int(round(Ldom/h));x=np.arange(N)*h
    ca,cl=coeffs(method,h);U=compacton(x,x0,Ldom)
    nsteps=int(round(T/dt));stride=max(1,int(round(save_dt/dt)))
    ns=nsteps//stride+1
    snaps=np.empty((ns,N),np.float32);times=np.empty(ns,float)
    snaps[0]=U;times[0]=0.;isave=1
    for n in range(1,nsteps+1):
        U=midpoint_step(U,dt,ca,cl)
        if n%stride==0:
            snaps[isave]=U;times[isave]=n*dt;isave+=1
    return x,times[:isave],snaps[:isave]
