import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.optimize as opt
from dataclasses import dataclass

SQRT3=np.sqrt(3.0)
GL4_A=np.array([[0.25,0.25-SQRT3/6.0],[0.25+SQRT3/6.0,0.25]],float)
GL4_B=np.array([0.5,0.5],float)
GL4_C=np.array([0.5-SQRT3/6.0,0.5+SQRT3/6.0],float)

METHODS={
 'ismail':{
   'label':'Ismail--Taha',
   'orders':'(2,2)',
   'A':np.array([0,0,1,0,0],float),
   'B0':np.array([0,-0.5,0,0.5,0],float),
   'C0':np.array([-0.5,1,0,-1,0.5],float),
 },
 'defrutos':{
   'label':'de Frutos',
   'orders':'(6,4)',
   'A':np.array([1,26,66,26,1],float)/120.0,
   'B0':np.array([-1,-10,0,10,1],float)/24.0,
   'C0':np.array([-0.5,1,0,-1,0.5],float),
 },
 'pade6':{
   'label':'Pade-6',
   'orders':'(4,6)',
   'A':np.array([1,56,126,56,1],float)/240.0,
   'B0':np.array([-1,-10,0,10,1],float)/24.0,
   'C0':np.array([-0.5,1,0,-1,0.5],float),
 },
 'pade8':{
   'label':'Pade-8',
   'orders':'(8,2)',
   'A':np.array([1,16,36,16,1],float)/70.0,
   'B0':np.array([-5,-32,0,32,5],float)/84.0,
   'C0':np.array([-0.5,1,0,-1,0.5],float),
 },
}
OFFS=np.array([-2,-1,0,1,2],int)

def periodic_stencil(N,coeff,scale=1.0):
    r=np.arange(N); rows=[];cols=[];vals=[]
    for k,c0 in zip(OFFS,coeff):
        if c0==0: continue
        c=(r+k)%N
        rows.extend(r);cols.extend(c);vals.extend(np.full(N,scale*c0))
    return sp.csc_matrix((vals,(rows,cols)),shape=(N,N))

def build_ops(method,N,h):
    m=METHODS[method]
    A=periodic_stencil(N,m['A'])
    B=periodic_stencil(N,m['B0'],1/h)
    C=periodic_stencil(N,m['C0'],1/h**3)
    return A,B+C

def compacton(x,x0,Ldom,c=1.0):
    z=(x-x0+Ldom/2)%Ldom-Ldom/2
    u=np.zeros_like(x)
    q=np.abs(z)<=2*np.pi
    u[q]=4*c/3*np.cos(z[q]/4)**2
    return u

def phi(u): return np.abs(u)*u
def dphi(u): return 2*np.abs(u)

@dataclass
class GL4Data:
    lu: object
    D1: np.ndarray
    D2: np.ndarray

class SpatialSystem:
    def __init__(self,method,N,h):
        self.method=method;self.N=N;self.h=h
        self.A,self.L=build_ops(method,N,h)
        self.A_lu=spla.splu(self.A,permc_spec='NATURAL')

    def gl4_step(self,U,dt,store=False,tol=2e-13,maxit=8):
        U=np.asarray(U)
        f0=-self.A_lu.solve(self.L@phi(U))
        Y1=U+GL4_C[0]*dt*f0
        Y2=U+GL4_C[1]*dt*f0
        for _ in range(maxit):
            p1=phi(Y1);p2=phi(Y2)
            R1=self.A@(Y1-U)+dt*(GL4_A[0,0]*(self.L@p1)+GL4_A[0,1]*(self.L@p2))
            R2=self.A@(Y2-U)+dt*(GL4_A[1,0]*(self.L@p1)+GL4_A[1,1]*(self.L@p2))
            d1v=dphi(Y1);d2v=dphi(Y2)
            LD1=self.L@sp.diags(d1v,0,format='csc')
            LD2=self.L@sp.diags(d2v,0,format='csc')
            J=sp.bmat([[self.A+dt*GL4_A[0,0]*LD1,dt*GL4_A[0,1]*LD2],
                       [dt*GL4_A[1,0]*LD1,self.A+dt*GL4_A[1,1]*LD2]],format='csc')
            dd=spla.spsolve(J,-np.r_[R1,R2],permc_spec='COLAMD')
            d1=dd[:self.N];d2=dd[self.N:]
            Y1+=d1;Y2+=d2
            if max(np.max(np.abs(d1)),np.max(np.abs(d2)))<tol: break
        p1=phi(Y1);p2=phi(Y2)
        U1=U-dt*0.5*self.A_lu.solve(self.L@(p1+p2))
        if not store:return U1
        d1v=dphi(Y1);d2v=dphi(Y2)
        LD1=self.L@sp.diags(d1v,0,format='csc')
        LD2=self.L@sp.diags(d2v,0,format='csc')
        J=sp.bmat([[self.A+dt*GL4_A[0,0]*LD1,dt*GL4_A[0,1]*LD2],
                   [dt*GL4_A[1,0]*LD1,self.A+dt*GL4_A[1,1]*LD2]],format='csc')
        return U1,GL4Data(spla.splu(J,permc_spec='COLAMD'),d1v,d2v)

    def flow(self,U,dt,kappa,store=False):
        V=np.asarray(U).copy();data=[]
        for _ in range(int(kappa)):
            if store:
                V,d=self.gl4_step(V,dt,True);data.append(d)
            else:V=self.gl4_step(V,dt)
        return (V,data) if store else V

    def cell_map(self,U,dt,kappa,store=False):
        if store:
            V,data=self.flow(U,dt,kappa,True)
            return np.roll(V,-1),data
        return np.roll(self.flow(U,dt,kappa),-1)

    def tangent_step(self,Z,dt,data):
        Z=np.asarray(Z);vec=Z.ndim==1
        if vec:Z=Z[:,None]
        AZ=self.A@Z;rhs=np.vstack([AZ,AZ])
        if np.iscomplexobj(rhs):Y=data.lu.solve(rhs.real)+1j*data.lu.solve(rhs.imag)
        else:Y=data.lu.solve(rhs)
        Y1=Y[:self.N];Y2=Y[self.N:]
        S=data.D1[:,None]*Y1+data.D2[:,None]*Y2
        rhs2=self.L@S
        if np.iscomplexobj(rhs2):corr=self.A_lu.solve(rhs2.real)+1j*self.A_lu.solve(rhs2.imag)
        else:corr=self.A_lu.solve(rhs2)
        out=Z-dt*0.5*corr
        return out[:,0] if vec else out

    def tangent_step_adj(self,Z,dt,data):
        Z=np.asarray(Z);vec=Z.ndim==1
        if vec:Z=Z[:,None]
        if np.iscomplexobj(Z):AinvZ=self.A_lu.solve(Z.real)+1j*self.A_lu.solve(Z.imag)
        else:AinvZ=self.A_lu.solve(Z)
        g=dt*0.5*(self.L@AinvZ)
        rhs=np.vstack([data.D1[:,None]*g,data.D2[:,None]*g])
        if np.iscomplexobj(rhs):X=data.lu.solve(rhs.real,trans='T')+1j*data.lu.solve(rhs.imag,trans='T')
        else:X=data.lu.solve(rhs,trans='T')
        out=Z+self.A@(X[:self.N]+X[self.N:])
        return out[:,0] if vec else out

    def tangent_cell(self,Z,dt,data):
        Y=np.asarray(Z).copy()
        for d in data:Y=self.tangent_step(Y,dt,d)
        return np.roll(Y,-1,axis=0)

    def tangent_cell_adj(self,Z,dt,data):
        Y=np.roll(np.asarray(Z),1,axis=0)
        for d in data[::-1]:Y=self.tangent_step_adj(Y,dt,d)
        return Y

def fixed_phase_x0(Ldom,h,phase=0.25):
    m=int(np.rint((Ldom/2+2*np.pi)/h-phase))
    xR=(m+phase)*h
    return xR-2*np.pi

def solve_tw(sys,Uguess,h,kappa,ftol=2e-10,maxiter=12):
    dt=h/kappa
    def F(U):return sys.cell_map(np.asarray(U),dt,kappa)-U
    try:Us=opt.newton_krylov(F,Uguess,f_tol=ftol,maxiter=maxiter,inner_maxiter=25)
    except opt.NoConvergence as e:Us=np.array(e.args[0])
    return Us,float(np.linalg.norm(F(Us),np.inf)),dt

def leading_ritz(sys,Us,h,kappa,x0,nb=10,maxiter=700,tol=5e-9,seed=1234):
    dt=h/kappa
    _,data=sys.cell_map(Us,dt,kappa,True)
    N=sys.N
    mR=int(np.rint((x0+2*np.pi)/h))%N;mL=int(np.rint((x0-2*np.pi)/h))%N
    idxR=(mR+np.arange(-40,41))%N;idxL=(mL+np.arange(-40,41))%N
    rng=np.random.default_rng(seed)
    Q=rng.normal(size=(N,nb))*1e-5
    Q[idxR,0]+=rng.normal(size=len(idxR));Q[idxL,1]+=rng.normal(size=len(idxL))
    Q,_=np.linalg.qr(Q)
    old=None;stable=0
    for it in range(1,maxiter+1):
        Q,_=np.linalg.qr(sys.tangent_cell(Q,dt,data))
        if it%20==0 or it==maxiter:
            PQ=sys.tangent_cell(Q,dt,data);H=Q.T@PQ
            ew,ev=np.linalg.eig(H);o=np.argsort(np.abs(ew))[::-1];ew=ew[o];ev=ev[:,o]
            v=Q@ev[:,0];rr=np.linalg.norm(sys.tangent_cell(v,dt,data)-ew[0]*v)/np.linalg.norm(v)
            rho=abs(ew[0])
            if old is not None and abs(rho-old)<2e-10 and rr<tol:stable+=1
            else:stable=0
            old=rho
            if stable>=2:break
    vals=[];resids=[]
    for j in range(min(6,len(ew))):
        v=Q@ev[:,j];Pv=sys.tangent_cell(v,dt,data)
        vals.append(ew[j]);resids.append(np.linalg.norm(Pv-ew[j]*v)/np.linalg.norm(v))
    v=Q@ev[:,0];e=np.abs(v)**2;et=e.sum()+1e-300
    return {'values':np.array(vals),'residuals':np.array(resids),'vector':v,
            'edge_frac':float((e[idxR].sum()+e[idxL].sum())/et),
            'right_frac':float(e[idxR].sum()/et),'left_frac':float(e[idxL].sum()/et),
            'iterations':it,'data':data,'idxR':idxR,'idxL':idxL}

def M_symbol(method,q,h):
    c=np.cos(q);c2=np.cos(2*q)
    sinc=np.ones_like(q);nz=np.abs(q)>1e-14;sinc[nz]=np.sin(q[nz])/q[nz]
    if method=='ismail':return sinc*(2*(c+1)-h*h)
    if method=='defrutos':return 10*sinc*((h*h+12)*c-5*h*h+12)/(c2-26*c+33)
    if method=='pade6':return 20*sinc*((h*h+12)*c-5*h*h+12)/(c2-56*c+63)
    if method=='pade8':return (5/3)*sinc*(42*(c+1)+h*h*(5*c-16))/(c2-16*c+18)
    raise ValueError(method)

def solve_profile(method,h,Leta=90,Neta=32768,tol=2e-13,maxit=3000):
    eta=np.linspace(-Leta,Leta,Neta,endpoint=False);deta=eta[1]-eta[0]
    q=2*np.pi*np.fft.fftfreq(Neta,d=deta);Ms=M_symbol(method,q,h)
    width={'ismail':.45,'defrutos':.28,'pade6':.20,'pade8':.23}[method]
    F=1/np.cosh(width*eta)**3;F/=F.max()
    alpha={'ismail':.25,'defrutos':.35,'pade6':.35,'pade8':.25}[method]
    for it in range(maxit):
        G=np.fft.ifft(Ms*np.fft.fft(F*F)).real;V=G.max();Fn=G/V
        Fn=np.roll(Fn,Neta//2-np.argmax(Fn));err=np.max(np.abs(Fn-F));F=(1-alpha)*F+alpha*Fn
        if err<tol:break
    G=np.fft.ifft(Ms*np.fft.fft(F*F)).real;V=G.max();F/=F.max()
    res=np.linalg.norm(V*F-G)/np.linalg.norm(V*F)
    ii=np.where(F>=.5)[0];fwhm=eta[ii[-1]]-eta[ii[0]]
    mass=np.trapezoid(F,eta);mass2=np.trapezoid(F*F,eta)
    return {'eta':eta,'F':F,'V':float(V),'residual':float(res),'fwhm':float(fwhm),
            'mass':float(mass),'mass2':float(mass2),'iterations':it+1,'last_error':float(err)}

@dataclass
class MPData:
    lu: object
    Jm: sp.csc_matrix

def midpoint_step_general(sys,U,dt,store=False,tol=2e-13,maxit=10):
    W=np.asarray(U).copy()
    for _ in range(maxit):
        G=(2/dt)*(sys.A@(W-U))+sys.L@phi(W)
        J=(2/dt)*sys.A+sys.L@sp.diags(dphi(W),0,format='csc')
        dd=spla.spsolve(J,-G,permc_spec='NATURAL')
        W+=dd
        if np.max(np.abs(dd))<tol:break
    U1=2*W-U
    if not store:return U1
    Jp=sys.A/dt+sys.L@sp.diags(np.abs(W),0,format='csc')
    Jm=sys.A/dt-sys.L@sp.diags(np.abs(W),0,format='csc')
    return U1,MPData(spla.splu(Jp,permc_spec='NATURAL'),Jm)

def mp_cell_map(sys,U,dt,kappa,store=False):
    V=np.asarray(U).copy();data=[]
    for _ in range(int(kappa)):
        if store:
            V,d=midpoint_step_general(sys,V,dt,True);data.append(d)
        else:V=midpoint_step_general(sys,V,dt)
    V=np.roll(V,-1)
    return (V,data) if store else V

def mp_tangent_cell(sys,Z,data):
    Y=np.asarray(Z).copy()
    for d in data:
        rhs=d.Jm@Y
        if np.iscomplexobj(rhs):Y=d.lu.solve(rhs.real)+1j*d.lu.solve(rhs.imag)
        else:Y=d.lu.solve(rhs)
    return np.roll(Y,-1,axis=0)

def mp_tangent_cell_adj(sys,Z,data):
    Y=np.roll(np.asarray(Z),1,axis=0)
    for d in data[::-1]:
        if np.iscomplexobj(Y):X=d.lu.solve(Y.real,trans='T')+1j*d.lu.solve(Y.imag,trans='T')
        else:X=d.lu.solve(Y,trans='T')
        Y=d.Jm.T@X
    return Y

def solve_tw_mp(sys,Uguess,h,kappa,ftol=2e-10,maxiter=14):
    dt=h/kappa
    def F(U):return mp_cell_map(sys,np.asarray(U),dt,kappa)-U
    try:Us=opt.newton_krylov(F,Uguess,f_tol=ftol,maxiter=maxiter,inner_maxiter=30)
    except opt.NoConvergence as e:Us=np.array(e.args[0])
    return Us,float(np.linalg.norm(F(Us),np.inf)),dt

def leading_ritz_mp(sys,Us,h,kappa,x0,nb=8,maxiter=500,tol=5e-9,seed=1234):
    dt=h/kappa
    _,data=mp_cell_map(sys,Us,dt,kappa,True)
    N=sys.N
    mR=int(np.rint((x0+2*np.pi)/h))%N;mL=int(np.rint((x0-2*np.pi)/h))%N
    idxR=(mR+np.arange(-40,41))%N;idxL=(mL+np.arange(-40,41))%N
    rng=np.random.default_rng(seed)
    Q=rng.normal(size=(N,nb))*1e-5
    Q[idxR,0]+=rng.normal(size=len(idxR));Q[idxL,1]+=rng.normal(size=len(idxL))
    Q,_=np.linalg.qr(Q)
    old=None;stable=0
    for it in range(1,maxiter+1):
        Q,_=np.linalg.qr(mp_tangent_cell(sys,Q,data))
        if it%20==0 or it==maxiter:
            PQ=mp_tangent_cell(sys,Q,data);H=Q.T@PQ
            ew,ev=np.linalg.eig(H);o=np.argsort(np.abs(ew))[::-1];ew=ew[o];ev=ev[:,o]
            v=Q@ev[:,0];rr=np.linalg.norm(mp_tangent_cell(sys,v,data)-ew[0]*v)/np.linalg.norm(v)
            rho=abs(ew[0])
            if old is not None and abs(rho-old)<2e-10 and rr<tol:stable+=1
            else:stable=0
            old=rho
            if stable>=2:break
    vals=[];resids=[]
    for j in range(min(6,len(ew))):
        v=Q@ev[:,j];Pv=mp_tangent_cell(sys,v,data)
        vals.append(ew[j]);resids.append(np.linalg.norm(Pv-ew[j]*v)/np.linalg.norm(v))
    v=Q@ev[:,0];e=np.abs(v)**2;et=e.sum()+1e-300
    return {'values':np.array(vals),'residuals':np.array(resids),'vector':v,
            'edge_frac':float((e[idxR].sum()+e[idxL].sum())/et),
            'right_frac':float(e[idxR].sum()/et),'left_frac':float(e[idxL].sum()/et),
            'iterations':it,'data':data,'idxR':idxR,'idxL':idxL}
