import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from dataclasses import dataclass
from wp3_integrators import phi,dphi

s15=np.sqrt(15.0)
A3=np.array([
 [5/36,2/9-s15/15,5/36-s15/30],
 [5/36+s15/24,2/9,5/36-s15/24],
 [5/36+s15/30,2/9+s15/15,5/36]
],float)
b3=np.array([5/18,4/9,5/18],float)
c3=np.array([.5-s15/10,.5,.5+s15/10],float)

@dataclass
class GL6Data:
    lu: object
    D: list

def gl6_step(sysm,U,dt,store=False,maxit=6,tol=2e-13):
    U=np.asarray(U);N=sysm.N
    f0=-sysm.A_lu.solve(sysm.L@phi(U))
    Y=[U+ci*dt*f0 for ci in c3]
    for _ in range(maxit):
        LP=[sysm.L@phi(y) for y in Y]
        R=[sysm.A@(Y[i]-U)+dt*sum(A3[i,j]*LP[j] for j in range(3)) for i in range(3)]
        Dm=[sp.diags(dphi(y),0,format='csc') for y in Y]
        LD=[sysm.L@dd for dd in Dm]
        blocks=[]
        for i in range(3):
            row=[]
            for j in range(3):
                row.append((sysm.A if i==j else 0)+dt*A3[i,j]*LD[j])
            blocks.append(row)
        J=sp.bmat(blocks,format='csc')
        dd=spla.spsolve(J,-np.concatenate(R),permc_spec='COLAMD')
        mx=0
        for i in range(3):
            di=dd[i*N:(i+1)*N];Y[i]+=di;mx=max(mx,np.max(np.abs(di)))
        if mx<tol:break
    P=sum(b3[j]*phi(Y[j]) for j in range(3))
    U1=U-dt*sysm.A_lu.solve(sysm.L@P)
    if not store:return U1
    dv=[dphi(y) for y in Y];Dm=[sp.diags(v,0,format='csc') for v in dv];LD=[sysm.L@dd for dd in Dm]
    blocks=[]
    for i in range(3):
        row=[]
        for j in range(3):row.append((sysm.A if i==j else 0)+dt*A3[i,j]*LD[j])
        blocks.append(row)
    J=sp.bmat(blocks,format='csc')
    return U1,GL6Data(spla.splu(J,permc_spec='COLAMD'),dv)

def gl6_cell(sysm,U,dt,kappa,store=False):
    V=U.copy();data=[]
    for _ in range(int(kappa)):
        if store:V,d=gl6_step(sysm,V,dt,True);data.append(d)
        else:V=gl6_step(sysm,V,dt)
    V=np.roll(V,-1)
    return (V,data) if store else V

def _lusolve(lu,rhs,trans='N'):
    if np.iscomplexobj(rhs):return lu.solve(rhs.real,trans=trans)+1j*lu.solve(rhs.imag,trans=trans)
    return lu.solve(rhs,trans=trans)

def tangent_step(sysm,Z,dt,data):
    Z=np.asarray(Z);vec=Z.ndim==1
    if vec:Z=Z[:,None]
    AZ=sysm.A@Z;rhs=np.vstack([AZ,AZ,AZ]);Y=_lusolve(data.lu,rhs);N=sysm.N
    S=sum(b3[j]*data.D[j][:,None]*Y[j*N:(j+1)*N] for j in range(3))
    LS=sysm.L@S
    corr=sysm.A_lu.solve(LS.real)+1j*sysm.A_lu.solve(LS.imag) if np.iscomplexobj(LS) else sysm.A_lu.solve(LS)
    out=Z-dt*corr
    return out[:,0] if vec else out

def tangent_step_adj(sysm,Z,dt,data):
    Z=np.asarray(Z);vec=Z.ndim==1
    if vec:Z=Z[:,None]
    AZ=sysm.A_lu.solve(Z.real)+1j*sysm.A_lu.solve(Z.imag) if np.iscomplexobj(Z) else sysm.A_lu.solve(Z)
    g=dt*(sysm.L@AZ) # -dt A^-1 L transpose = +dt L A^-1
    rhs=np.vstack([b3[j]*data.D[j][:,None]*g for j in range(3)])
    X=_lusolve(data.lu,rhs,trans='T');N=sysm.N
    out=Z+sysm.A@sum(X[j*N:(j+1)*N] for j in range(3))
    return out[:,0] if vec else out

def tangent_cell(sysm,Z,dt,data):
    Y=Z.copy()
    for d in data:Y=tangent_step(sysm,Y,dt,d)
    return np.roll(Y,-1,axis=0)

def tangent_cell_adj(sysm,Z,dt,data):
    Y=np.roll(Z,1,axis=0)
    for d in data[::-1]:Y=tangent_step_adj(sysm,Y,dt,d)
    return Y
