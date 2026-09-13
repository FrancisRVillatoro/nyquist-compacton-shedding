import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.optimize as opt
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any

SQRT3 = np.sqrt(3.0)
GL2_A = np.array([[0.25, 0.25-SQRT3/6.0],
                  [0.25+SQRT3/6.0, 0.25]], dtype=float)
GL2_B = np.array([0.5, 0.5], dtype=float)


def build_ops(N: int, h: float):
    r = np.arange(N)
    rows=[]; cols=[]; av=[]; lv=[]
    ca=np.array([1,26,66,26,1],float)/120.0
    cb=np.array([-1,-10,0,10,1],float)/(24*h)
    cc=np.array([-1,2,0,-2,1],float)/(2*h**3)
    for kk,k in enumerate([-2,-1,0,1,2]):
        c=(r+k)%N
        rows.extend(r); cols.extend(c)
        av.extend(np.full(N,ca[kk]))
        lv.extend(np.full(N,cb[kk]+cc[kk]))
    A=sp.csc_matrix((av,(rows,cols)),shape=(N,N))
    L=sp.csc_matrix((lv,(rows,cols)),shape=(N,N))
    return A,L


def compacton(x, x0, Ldom, c=1.0):
    z=(x-x0+Ldom/2)%Ldom-Ldom/2
    u=np.zeros_like(x)
    m=np.abs(z)<=2*np.pi
    u[m]=4*c/3*np.cos(z[m]/4)**2
    return u


def phi(u):
    return np.abs(u)*u


def dphi(u):
    return 2*np.abs(u)


@dataclass
class MidpointStepData:
    lu_plus: Any
    Jminus: sp.csc_matrix


@dataclass
class GL4StepData:
    lu_stage: Any
    D1: np.ndarray
    D2: np.ndarray


class SpatialSystem:
    def __init__(self, N: int, h: float):
        self.N=N; self.h=h
        self.A,self.L=build_ops(N,h)
        self.A_lu=spla.splu(self.A,permc_spec='NATURAL')

    def midpoint_step(self,U,dt,store=False,newton_tol=2e-13,maxit=8):
        W=np.asarray(U).copy()
        for _ in range(maxit):
            G=(2/dt)*(self.A@(W-U))+self.L@phi(W)
            J=(2/dt)*self.A+self.L@sp.diags(dphi(W),0,format='csc')
            dd=spla.spsolve(J,-G,permc_spec='NATURAL')
            W += dd
            if np.max(np.abs(dd))<newton_tol:
                break
        U1=2*W-U
        if not store:
            return U1
        Jp=self.A/dt+self.L@sp.diags(np.abs(W),0,format='csc')
        Jm=self.A/dt-self.L@sp.diags(np.abs(W),0,format='csc')
        return U1,MidpointStepData(spla.splu(Jp,permc_spec='NATURAL'),Jm)

    def gl4_step(self,U,dt,store=False,newton_tol=2e-13,maxit=8):
        U=np.asarray(U)
        # First-order stage prediction using the semidiscrete derivative.
        f0=-self.A_lu.solve(self.L@phi(U))
        c1=0.5-SQRT3/6.0; c2=0.5+SQRT3/6.0
        Y1=U+c1*dt*f0
        Y2=U+c2*dt*f0
        for _ in range(maxit):
            p1=phi(Y1); p2=phi(Y2)
            R1=self.A@(Y1-U)+dt*(GL2_A[0,0]*(self.L@p1)+GL2_A[0,1]*(self.L@p2))
            R2=self.A@(Y2-U)+dt*(GL2_A[1,0]*(self.L@p1)+GL2_A[1,1]*(self.L@p2))
            D1=sp.diags(dphi(Y1),0,format='csc')
            D2=sp.diags(dphi(Y2),0,format='csc')
            LD1=self.L@D1; LD2=self.L@D2
            J11=self.A+dt*GL2_A[0,0]*LD1
            J12=dt*GL2_A[0,1]*LD2
            J21=dt*GL2_A[1,0]*LD1
            J22=self.A+dt*GL2_A[1,1]*LD2
            J=sp.bmat([[J11,J12],[J21,J22]],format='csc')
            rhs=-np.concatenate([R1,R2])
            dd=spla.spsolve(J,rhs,permc_spec='COLAMD')
            d1=dd[:self.N]; d2=dd[self.N:]
            Y1+=d1; Y2+=d2
            if max(np.max(np.abs(d1)),np.max(np.abs(d2)))<newton_tol:
                break
        # Rebuild at converged stages for tangent storage.
        p1=phi(Y1); p2=phi(Y2)
        increment=-dt*0.5*self.A_lu.solve(self.L@(p1+p2))
        U1=U+increment
        if not store:
            return U1
        d1v=dphi(Y1); d2v=dphi(Y2)
        D1=sp.diags(d1v,0,format='csc'); D2=sp.diags(d2v,0,format='csc')
        LD1=self.L@D1; LD2=self.L@D2
        J=sp.bmat([
            [self.A+dt*GL2_A[0,0]*LD1, dt*GL2_A[0,1]*LD2],
            [dt*GL2_A[1,0]*LD1, self.A+dt*GL2_A[1,1]*LD2]
        ],format='csc')
        return U1,GL4StepData(spla.splu(J,permc_spec='COLAMD'),d1v,d2v)

    def step(self,U,dt,method='midpoint',store=False):
        if method=='midpoint':
            return self.midpoint_step(U,dt,store=store)
        if method=='gl4':
            return self.gl4_step(U,dt,store=store)
        raise ValueError(method)

    def flow(self,U,dt,nsteps,method='midpoint',store=False):
        V=np.asarray(U).copy(); data=[]
        for _ in range(nsteps):
            if store:
                V,d=self.step(V,dt,method=method,store=True); data.append(d)
            else:
                V=self.step(V,dt,method=method,store=False)
        return (V,data) if store else V

    def cell_map(self,U,dt,kappa,method='midpoint',store=False):
        if store:
            V,data=self.flow(U,dt,int(kappa),method=method,store=True)
            return np.roll(V,-1),data
        return np.roll(self.flow(U,dt,int(kappa),method=method),-1)

    def tangent_step(self,Z,dt,data,method):
        Z=np.asarray(Z); vec=(Z.ndim==1)
        if vec: Z=Z[:,None]
        if method=='midpoint':
            rhs=data.Jminus@Z
            if np.iscomplexobj(rhs):
                out=data.lu_plus.solve(rhs.real)+1j*data.lu_plus.solve(rhs.imag)
            else:
                out=data.lu_plus.solve(rhs)
        elif method=='gl4':
            AZ=self.A@Z
            rhs=np.vstack([AZ,AZ])
            if np.iscomplexobj(rhs):
                Y=data.lu_stage.solve(rhs.real)+1j*data.lu_stage.solve(rhs.imag)
            else:
                Y=data.lu_stage.solve(rhs)
            Y1=Y[:self.N]; Y2=Y[self.N:]
            S=data.D1[:,None]*Y1+data.D2[:,None]*Y2
            rhs2=self.L@S
            if np.iscomplexobj(rhs2):
                corr=self.A_lu.solve(rhs2.real)+1j*self.A_lu.solve(rhs2.imag)
            else:
                corr=self.A_lu.solve(rhs2)
            out=Z-dt*0.5*corr
        else: raise ValueError(method)
        return out[:,0] if vec else out

    def tangent_step_adj(self,Z,dt,data,method):
        Z=np.asarray(Z); vec=(Z.ndim==1)
        if vec: Z=Z[:,None]
        if method=='midpoint':
            if np.iscomplexobj(Z):
                X=data.lu_plus.solve(Z.real,trans='T')+1j*data.lu_plus.solve(Z.imag,trans='T')
            else:
                X=data.lu_plus.solve(Z,trans='T')
            out=data.Jminus.T@X
        elif method=='gl4':
            # Kout^T Z = dt/2 L A^{-1} Z, since A^T=A and L^T=-L.
            if np.iscomplexobj(Z):
                AinvZ=self.A_lu.solve(Z.real)+1j*self.A_lu.solve(Z.imag)
            else:
                AinvZ=self.A_lu.solve(Z)
            g=dt*0.5*(self.L@AinvZ)
            rhs=np.vstack([data.D1[:,None]*g,data.D2[:,None]*g])
            if np.iscomplexobj(rhs):
                X=data.lu_stage.solve(rhs.real,trans='T')+1j*data.lu_stage.solve(rhs.imag,trans='T')
            else:
                X=data.lu_stage.solve(rhs,trans='T')
            X1=X[:self.N]; X2=X[self.N:]
            out=Z+self.A@(X1+X2)
        else: raise ValueError(method)
        return out[:,0] if vec else out

    def tangent_cell(self,Z,dt,data,method):
        Y=np.asarray(Z).copy()
        for d in data:
            Y=self.tangent_step(Y,dt,d,method)
        return np.roll(Y,-1,axis=0)

    def tangent_cell_adj(self,Z,dt,data,method):
        Y=np.roll(np.asarray(Z),1,axis=0)
        for d in data[::-1]:
            Y=self.tangent_step_adj(Y,dt,d,method)
        return Y


def solve_traveling_compacton(sys: SpatialSystem, Uguess, h, kappa, method,
                               f_tol=2e-10,maxiter=10,inner_maxiter=20):
    dt=h/kappa
    def F(U):
        return sys.cell_map(np.asarray(U),dt,kappa,method=method)-U
    try:
        Us=opt.newton_krylov(F,Uguess,f_tol=f_tol,maxiter=maxiter,
                             inner_maxiter=inner_maxiter)
    except opt.NoConvergence as e:
        Us=np.array(e.args[0])
    res=float(np.linalg.norm(F(Us),np.inf))
    return Us,res,dt


def leading_ritz(sys: SpatialSystem, Us, h, kappa, method, x0, nb=8,
                 maxiter=500,tol=5e-9,seed=1234):
    dt=h/kappa
    _,data=sys.cell_map(Us,dt,kappa,method=method,store=True)
    N=sys.N
    mR=int(np.rint((x0+2*np.pi)/h))%N
    mL=int(np.rint((x0-2*np.pi)/h))%N
    idxR=(mR+np.arange(-35,36))%N
    idxL=(mL+np.arange(-35,36))%N
    rng=np.random.default_rng(seed)
    Q=rng.normal(size=(N,nb))*1e-5
    Q[idxR,0]+=rng.normal(size=len(idxR)); Q[idxL,1]+=rng.normal(size=len(idxL))
    Q,_=np.linalg.qr(Q)
    old=None; stable=0
    for it in range(1,maxiter+1):
        Z=sys.tangent_cell(Q,dt,data,method)
        Q,_=np.linalg.qr(Z)
        if it%20==0 or it==maxiter:
            PQ=sys.tangent_cell(Q,dt,data,method)
            H=Q.T@PQ
            ew,ev=np.linalg.eig(H)
            order=np.argsort(np.abs(ew))[::-1]; ew=ew[order];ev=ev[:,order]
            v=Q@ev[:,0]
            rr=np.linalg.norm(sys.tangent_cell(v,dt,data,method)-ew[0]*v)/np.linalg.norm(v)
            rho=abs(ew[0])
            if old is not None and abs(rho-old)<1e-10 and rr<tol: stable+=1
            else: stable=0
            old=rho
            if stable>=2: break
    vals=[]; resids=[]
    for j in range(min(4,len(ew))):
        v=Q@ev[:,j]
        Pv=sys.tangent_cell(v,dt,data,method)
        vals.append(ew[j]);resids.append(float(np.linalg.norm(Pv-ew[j]*v)/np.linalg.norm(v)))
    v=Q@ev[:,0]; e=np.abs(v)**2; et=e.sum()+1e-300
    edge=(e[idxR].sum()+e[idxL].sum())/et
    return dict(values=np.array(vals),residuals=np.array(resids),
                right_frac=float(e[idxR].sum()/et),left_frac=float(e[idxL].sum()/et),
                edge_frac=float(edge),iterations=it,data=data,
                leading_vector=v)
