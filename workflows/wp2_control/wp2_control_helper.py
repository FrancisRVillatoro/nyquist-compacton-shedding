from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.optimize as opt
from dataclasses import dataclass

@dataclass
class Base:
    h: float
    kappa: int
    dt: float
    Ldom: float
    x0: float
    x: np.ndarray
    A: sp.csc_matrix
    L: sp.csc_matrix
    Uc: np.ndarray
    Us: np.ndarray
    fp_res: float
    fac: list
    ksym: np.ndarray
    idxR: np.ndarray
    idxL: np.ndarray
    idxEdge: np.ndarray
    Pnl: object


def build_ops(N: int, h: float):
    r=np.arange(N); rows=[]; cols=[]; av=[]; lv=[]
    ca=np.array([1,26,66,26,1],float)/120.0
    cb=np.array([-1,-10,0,10,1],float)/(24*h)
    cc=np.array([-1,2,0,-2,1],float)/(2*h**3)
    for kk,k in enumerate([-2,-1,0,1,2]):
        c=(r+k)%N
        rows.extend(r.tolist()); cols.extend(c.tolist())
        av.extend(np.full(N,ca[kk]).tolist())
        lv.extend(np.full(N,cb[kk]+cc[kk]).tolist())
    return (sp.csc_matrix((av,(rows,cols)),shape=(N,N)),
            sp.csc_matrix((lv,(rows,cols)),shape=(N,N)))


def compacton(x: np.ndarray, x0: float, Ldom: float, c: float=1.0):
    z=(x-x0+Ldom/2)%Ldom-Ldom/2
    u=np.zeros_like(x)
    m=np.abs(z)<=2*np.pi
    u[m]=4*c/3*np.cos(z[m]/4)**2
    return u


def midpoint(U,A,L,dt,store=False):
    W=np.asarray(U,dtype=float).copy()
    for _ in range(7):
        G=(2/dt)*(A@(W-U))+L@(np.abs(W)*W)
        J=(2/dt)*A+L@sp.diags(2*np.abs(W),0,format='csc')
        d=spla.spsolve(J,-G,permc_spec='NATURAL')
        W+=d
        if np.max(np.abs(d))<5e-13:
            break
    U1=2*W-U
    if not store:
        return U1
    Jp=A/dt+L@sp.diags(np.abs(W),0,format='csc')
    Jm=A/dt-L@sp.diags(np.abs(W),0,format='csc')
    return U1,spla.splu(Jp,permc_spec='NATURAL'),Jm


def cell_map(U,A,L,dt,kappa,record=False):
    V=np.asarray(U,dtype=float).copy(); fac=[]
    for _ in range(kappa):
        if record:
            V,lu,Jm=midpoint(V,A,L,dt,True); fac.append((lu,Jm))
        else:
            V=midpoint(V,A,L,dt)
    V=np.roll(V,-1)
    return (V,fac) if record else V


def solve_base(h: float,kappa: int,Ldom: float=16.0):
    dt=h/kappa; x0=Ldom/2
    N=int(round(Ldom/h)); x=np.arange(N)*h
    A,L=build_ops(N,h); Uc=compacton(x,x0,Ldom)
    def Pnl(U): return cell_map(U,A,L,dt,kappa,False)
    def F(U): return Pnl(np.asarray(U))-U
    try:
        Us=opt.newton_krylov(F,Uc,f_tol=2e-11,maxiter=12,inner_maxiter=24)
    except opt.NoConvergence as e:
        Us=np.array(e.args[0])
    fp=float(np.linalg.norm(F(Us),np.inf))
    _,fac=cell_map(Us,A,L,dt,kappa,True)
    theta=2*np.pi*np.fft.fftfreq(N)
    asym=(2*np.cos(2*theta)+52*np.cos(theta)+66)/120.0
    d4=16*np.sin(theta/2)**4
    ksym=d4/(120.0*asym)  # Nyquist normalized to one
    mR=int(np.rint((x0+2*np.pi)/h))%N
    mL=int(np.rint((x0-2*np.pi)/h))%N
    idxR=(mR+np.arange(-45,46))%N
    idxL=(mL+np.arange(-45,46))%N
    idxEdge=np.unique(np.r_[idxR,idxL])
    return Base(h,kappa,dt,Ldom,x0,x,A,L,Uc,Us,fp,fac,ksym,idxR,idxL,idxEdge,Pnl)


def apply_P(base: Base,Z):
    Y=np.asarray(Z).copy(); one=Y.ndim==1
    if one: Y=Y[:,None]
    for lu,Jm in base.fac:
        rhs=Jm@Y
        if np.iscomplexobj(rhs):
            Y=lu.solve(rhs.real)+1j*lu.solve(rhs.imag)
        else:
            Y=lu.solve(rhs)
    Y=np.roll(Y,-1,axis=0)
    return Y[:,0] if one else Y


def apply_PT(base: Base,Z):
    Y=np.asarray(Z).copy(); one=Y.ndim==1
    if one:Y=Y[:,None]
    Y=np.roll(Y,1,axis=0)
    for lu,Jm in base.fac[::-1]:
        if np.iscomplexobj(Y):
            q=lu.solve(Y.real,trans='T')+1j*lu.solve(Y.imag,trans='T')
        else:
            q=lu.solve(Y,trans='T')
        Y=Jm.T@q
    return Y[:,0] if one else Y


def hyper_filter(base: Base,Z,delta: float):
    Y=np.asarray(Z); one=Y.ndim==1
    if one:Y=Y[:,None]
    out=np.fft.ifft(np.exp(-delta*base.ksym)[:,None]*np.fft.fft(Y,axis=0),axis=0)
    if not np.iscomplexobj(Z): out=out.real
    return out[:,0] if one else out


def controlled_map_hyper(base: Base,U,delta: float):
    return base.Us + hyper_filter(base,base.Pnl(U)-base.Us,delta)


def highpass(z):
    return (2*z-np.roll(z,1)-np.roll(z,-1))/4.0


def dominant_real_pair(base: Base,nit=700):
    N=len(base.Us);rng=np.random.default_rng(991)
    r=np.zeros(N);l=np.zeros(N)
    r[base.idxR]=rng.normal(size=len(base.idxR));l[base.idxR]=rng.normal(size=len(base.idxR))
    r/=np.linalg.norm(r);l/=np.linalg.norm(l)
    for _ in range(nit):
        y=apply_P(base,r);r=y/np.linalg.norm(y)
        y=apply_PT(base,l);l=y/np.linalg.norm(y)
    lam=float(np.dot(r,apply_P(base,r)))
    rres=float(np.linalg.norm(apply_P(base,r)-lam*r))
    lres=float(np.linalg.norm(apply_PT(base,l)-lam*l))
    ov=float(np.dot(l,r))
    if ov<0:l=-l;ov=-ov
    l=l/ov
    hr=float(np.max(np.abs(highpass(r)[base.idxR])))
    rhat=r/hr; lhat=hr*l
    return dict(lam=lam,r=r,l=l,rhat=rhat,lhat=lhat,hr=hr,rres=rres,lres=lres)


def block_ritz(base: Base,delta: float,Q=None,nb=14,niter=260,seed=1234):
    N=len(base.Us)
    if Q is None:
        rng=np.random.default_rng(seed)
        Q=rng.normal(size=(N,nb))*1e-5
        Q[base.idxR,0]+=rng.normal(size=len(base.idxR))
        Q[base.idxL,1]+=rng.normal(size=len(base.idxL))
        Q,_=np.linalg.qr(Q)
    def Pd(Z):return hyper_filter(base,apply_P(base,Z),delta)
    for _ in range(niter):
        Q,_=np.linalg.qr(Pd(Q))
    PQ=Pd(Q); H=Q.conj().T@PQ
    ew,ev=np.linalg.eig(H); order=np.argsort(np.abs(ew))[::-1]
    ew=ew[order];ev=ev[:,order]
    rows=[]
    for j in range(len(ew)):
        v=Q@ev[:,j]
        res=float(np.linalg.norm(Pd(v)-ew[j]*v)/np.linalg.norm(v))
        ef=float(np.sum(np.abs(v[base.idxEdge])**2)/np.sum(np.abs(v)**2))
        rows.append(dict(eig=ew[j],res=res,edgefrac=ef,vec=v))
    return Q,rows


def modal_control_map(base: Base,pair,U,alpha: float):
    y=base.Pnl(U)-base.Us
    return base.Us + y-alpha*pair['r']*float(np.dot(pair['l'],y))


def trace_control(base: Base,mapfun,ncells: int,U0=None,store_every=1):
    U=base.Uc.copy() if U0 is None else np.asarray(U0).copy()
    pair=dominant_real_pair(base,nit=500)
    mR=int(np.rint((base.x0+2*np.pi)/base.h))%len(U)
    nAhead=np.arange(5,min(401,len(U)//3))
    idxAhead=(mR+nAhead)%len(U)
    rows=[]
    for n in range(ncells+1):
        if n%store_every==0:
            du=U-base.Us
            q=float(pair['lhat']@(du/base.h**2))
            hp=np.abs(highpass(du))/base.h**2
            ea=float(np.sum(hp[idxAhead]**2))
            rows.append(dict(cell=n,time=n*base.h,q=q,abs_q=abs(q),
                             hp_edge=float(np.max(hp[base.idxR])),
                             hp_ahead_max=float(np.max(hp[idxAhead])),
                             hp_ahead_energy=ea,
                             solution_l2=float(np.linalg.norm(U)),
                             mismatch_l2=float(np.linalg.norm(du))))
        if n<ncells:U=mapfun(U)
    return rows,U,pair

def build_D4(N:int):
    r=np.arange(N);rows=[];cols=[];vals=[]
    coeff={-2:1.0,-1:-4.0,0:6.0,1:-4.0,2:1.0}
    for k,v in coeff.items():
        c=(r+k)%N
        rows.extend(r.tolist());cols.extend(c.tolist());vals.extend(np.full(N,v).tolist())
    return sp.csc_matrix((vals,(rows,cols)),shape=(N,N))


def base_orbit(base:Base):
    V=base.Us.copy(); starts=[]; mids=[]; ends=[]
    for _ in range(base.kappa):
        starts.append(V.copy())
        V1,lu,Jm=midpoint(V,base.A,base.L,base.dt,True)
        # Recover midpoint exactly from endpoints.
        W=0.5*(V+V1)
        mids.append(W);ends.append(V1.copy());V=V1
    return starts,mids,ends


def hyper_tangent_factors(base:Base,epsilon:float,orbit=None):
    if orbit is None:orbit=base_orbit(base)
    starts,mids,ends=orbit
    D4=build_D4(len(base.Us));fac=[]
    for W in mids:
        Jp=base.A/base.dt+base.L@sp.diags(np.abs(W),0,format='csc')+0.5*epsilon*D4
        Jm=base.A/base.dt-base.L@sp.diags(np.abs(W),0,format='csc')-0.5*epsilon*D4
        fac.append((spla.splu(Jp,permc_spec='NATURAL'),Jm))
    return fac


def apply_factors_shift(fac,Z):
    Y=np.asarray(Z).copy();one=Y.ndim==1
    if one:Y=Y[:,None]
    for lu,Jm in fac:
        rhs=Jm@Y
        if np.iscomplexobj(rhs):Y=lu.solve(rhs.real)+1j*lu.solve(rhs.imag)
        else:Y=lu.solve(rhs)
    Y=np.roll(Y,-1,axis=0)
    return Y[:,0] if one else Y


def hyper_block_ritz(base:Base,epsilon:float,Q=None,nb=14,niter=200,orbit=None,seed=1234):
    fac=hyper_tangent_factors(base,epsilon,orbit)
    N=len(base.Us)
    if Q is None:
        rng=np.random.default_rng(seed);Q=rng.normal(size=(N,nb))*1e-5
        Q[base.idxR,0]+=rng.normal(size=len(base.idxR));Q[base.idxL,1]+=rng.normal(size=len(base.idxL))
        Q,_=np.linalg.qr(Q)
    def Pz(Z):return apply_factors_shift(fac,Z)
    for _ in range(niter):Q,_=np.linalg.qr(Pz(Q))
    PQ=Pz(Q);H=Q.conj().T@PQ
    ew,ev=np.linalg.eig(H);order=np.argsort(np.abs(ew))[::-1];ew=ew[order];ev=ev[:,order]
    rows=[]
    for j in range(len(ew)):
        v=Q@ev[:,j]
        res=float(np.linalg.norm(Pz(v)-ew[j]*v)/np.linalg.norm(v))
        ef=float(np.sum(np.abs(v[base.idxEdge])**2)/np.sum(np.abs(v)**2))
        rows.append(dict(eig=ew[j],res=res,edgefrac=ef,vec=v))
    return Q,rows,fac


def controlled_hyper_step(U,base:Base,epsilon:float,k:int,orbit,D4=None):
    if D4 is None:D4=build_D4(len(U))
    Wstar=orbit[1][k]
    W=np.asarray(U).copy()
    for _ in range(8):
        G=(2/base.dt)*(base.A@(W-U))+base.L@(np.abs(W)*W)+epsilon*(D4@(W-Wstar))
        J=(2/base.dt)*base.A+base.L@sp.diags(2*np.abs(W),0,format='csc')+epsilon*D4
        d=spla.spsolve(J,-G,permc_spec='NATURAL');W+=d
        if np.max(np.abs(d))<5e-13:break
    return 2*W-U


def controlled_map_orbit_hyper(base:Base,U,epsilon:float,orbit=None,D4=None):
    if orbit is None:orbit=base_orbit(base)
    if D4 is None:D4=build_D4(len(U))
    V=np.asarray(U).copy()
    for k in range(base.kappa):V=controlled_hyper_step(V,base,epsilon,k,orbit,D4)
    return np.roll(V,-1)

def block_ritz_plain(base:Base,adjoint=False,Q=None,nb=12,niter=300,seed=2468):
    N=len(base.Us);op=apply_PT if adjoint else apply_P
    if Q is None:
        rng=np.random.default_rng(seed);Q=rng.normal(size=(N,nb))*1e-5
        Q[base.idxR,0]+=rng.normal(size=len(base.idxR));Q[base.idxL,1]+=rng.normal(size=len(base.idxL));Q,_=np.linalg.qr(Q)
    for _ in range(niter):Q,_=np.linalg.qr(op(base,Q))
    PQ=op(base,Q);H=Q.conj().T@PQ;ew,ev=np.linalg.eig(H);order=np.argsort(np.abs(ew))[::-1];ew=ew[order];ev=ev[:,order]
    rows=[]
    for j in range(len(ew)):
        v=Q@ev[:,j];res=float(np.linalg.norm(op(base,v)-ew[j]*v)/np.linalg.norm(v));ef=float(np.sum(np.abs(v[base.idxEdge])**2)/np.sum(np.abs(v)**2));rows.append(dict(eig=ew[j],vec=v,res=res,edgefrac=ef))
    return Q,rows


def real_unstable_projector(base:Base,rho_tol=1.001,nb=16,niter=600):
    _,rr=block_ritz_plain(base,False,nb=nb,niter=niter,seed=333)
    _,ll=block_ritz_plain(base,True,nb=nb,niter=niter,seed=777)
    rights=[z for z in rr if abs(z['eig'])>rho_tol and z['edgefrac']>.9 and z['res']<.02]
    # unique: one real and one representative from each conjugate pair
    selected=[]
    for z in rights:
        if abs(np.imag(z['eig']))<1e-7:
            if not any(abs(np.imag(w['eig']))<1e-7 and abs(w['eig']-z['eig'])<1e-4 for w in selected):selected.append(z)
        elif np.imag(z['eig'])>0:
            selected.append(z)
    Rcols=[];Lcols=[];eigs=[]
    for z in selected:
        lam=z['eig'];r=z['vec']
        # P^T eigenvalue should be conjugate(lam) for Hermitian left eigenvector.
        target=np.conjugate(lam)
        lz=min(ll,key=lambda w:abs(w['eig']-target))
        l=lz['vec']
        if abs(np.imag(lam))<1e-7:
            Rcols.append(np.real(r));Lcols.append(np.real(l));eigs.append(lam)
        else:
            Rcols.extend([np.real(r),np.imag(r)]);Lcols.extend([np.real(l),np.imag(l)]);eigs.extend([lam,np.conjugate(lam)])
    R=np.column_stack(Rcols);L=np.column_stack(Lcols)
    M=L.T@R
    Minv=np.linalg.inv(M)
    def project(z):return R@(Minv@(L.T@z))
    return dict(R=R,L=L,M=M,Minv=Minv,project=project,eigs=np.array(eigs),right_rows=rr,left_rows=ll,
                idempotence=float(np.linalg.norm((Minv@(L.T@R))-np.eye(R.shape[1]))))
