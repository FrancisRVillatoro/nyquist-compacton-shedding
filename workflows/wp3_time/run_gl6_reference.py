from pathlib import Path
import sys,time,json
import numpy as np
import pandas as pd
import scipy.optimize as opt
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from wp3_integrators import SpatialSystem
from wp3_gl6 import gl6_cell,tangent_cell,tangent_cell_adj


def solve_gl6(sysm,Uguess,h,kappa,f_tol=5e-10,maxiter=8):
    dt=h/kappa
    def F(U): return gl6_cell(sysm,np.asarray(U),dt,kappa)-U
    r0=float(np.linalg.norm(F(Uguess),np.inf))
    if r0 <= max(2e-9,5*f_tol):
        return Uguess.copy(),r0,0.0,False
    t=time.time()
    try:
        U=opt.newton_krylov(F,Uguess,f_tol=f_tol,maxiter=maxiter,inner_maxiter=15)
    except opt.NoConvergence as e:
        U=np.asarray(e.args[0])
    return np.asarray(U),float(np.linalg.norm(F(U),np.inf)),time.time()-t,True


def leading_ritz_gl6(sysm,U,h,kappa,x0,nb=6,maxiter=340,tol=1e-8,seed=9901):
    dt=h/kappa
    _,data=gl6_cell(sysm,U,dt,kappa,store=True)
    N=sysm.N
    mR=int(np.rint((x0+2*np.pi)/h))%N
    mL=int(np.rint((x0-2*np.pi)/h))%N
    idxR=(mR+np.arange(-35,36))%N
    idxL=(mL+np.arange(-35,36))%N
    rng=np.random.default_rng(seed)
    Q=rng.normal(size=(N,nb))*1e-5
    Q[idxR,0]+=rng.normal(size=len(idxR));Q[idxL,1]+=rng.normal(size=len(idxL))
    Q,_=np.linalg.qr(Q)
    old=None;stable=0;ew=ev=None
    t=time.time()
    for it in range(1,maxiter+1):
        Q,_=np.linalg.qr(tangent_cell(sysm,Q,dt,data))
        if it%20==0 or it==maxiter:
            PQ=tangent_cell(sysm,Q,dt,data)
            H=Q.T@PQ
            ew,ev=np.linalg.eig(H)
            ix=np.argsort(np.abs(ew))[::-1];ew=ew[ix];ev=ev[:,ix]
            v=Q@ev[:,0]
            rr=np.linalg.norm(tangent_cell(sysm,v,dt,data)-ew[0]*v)/np.linalg.norm(v)
            rho=abs(ew[0])
            if old is not None and abs(rho-old)<2e-10 and rr<tol: stable+=1
            else: stable=0
            old=rho
            if stable>=2: break
    vals=[];res=[]
    for j in range(min(4,len(ew))):
        v=Q@ev[:,j];Pv=tangent_cell(sysm,v,dt,data)
        vals.append(ew[j]);res.append(np.linalg.norm(Pv-ew[j]*v)/np.linalg.norm(v))
    v=Q@ev[:,0];ee=np.abs(v)**2;tot=ee.sum()+1e-300
    return dict(values=np.array(vals),residuals=np.array(res),iterations=it,
                spectrum_runtime_s=time.time()-t,
                right_frac=float(ee[idxR].sum()/tot),left_frac=float(ee[idxL].sum()/tot))


def adjoint_check(sysm,U,h,kappa):
    dt=h/kappa;_,data=gl6_cell(sysm,U,dt,kappa,store=True)
    rng=np.random.default_rng(17);u=rng.normal(size=sysm.N);v=rng.normal(size=sysm.N)
    Pu=tangent_cell(sysm,u,dt,data);PTv=tangent_cell_adj(sysm,v,dt,data)
    return abs(np.vdot(Pu,v)-np.vdot(u,PTv))/(abs(np.vdot(Pu,v))+abs(np.vdot(u,PTv))+1e-300)

rows=[]
for h in [0.01,0.02]:
    p=ROOT/f'compacton_gl4_h{h:.3f}_k80.npz'
    d=np.load(p);x=d['x'];x0=float(d['x0']);Uref=d['U'];N=len(Uref)
    sysm=SpatialSystem(N,h)
    for kappa in [5,10,20]:
        print('case',h,kappa,flush=True)
        U,res,solve_s,solved=solve_gl6(sysm,Uref,h,kappa)
        print(' fixed res',res,'solved',solved,'time',solve_s,flush=True)
        sp=leading_ritz_gl6(sysm,U,h,kappa,x0,seed=991+int(h*10000)+kappa)
        ac=adjoint_check(sysm,U,h,kappa)
        vals=sp['values'];r=sp['residuals']
        row={'h':h,'method':'gl6','kappa':kappa,'dt':h/kappa,'N':N,'x0':x0,
             'fixed_point_residual':res,'fixed_point_was_solved':solved,'solve_runtime_s':solve_s,
             'spectrum_runtime_s':sp['spectrum_runtime_s'],'ritz_iterations':sp['iterations'],
             'adjoint_relative_error':ac,'right_edge_energy_fraction':sp['right_frac'],
             'left_edge_energy_fraction':sp['left_frac'],'edge_energy_fraction':sp['right_frac']+sp['left_frac']}
        for j,z in enumerate(vals,1):
            row[f'lambda{j}_real']=z.real;row[f'lambda{j}_imag']=z.imag;row[f'lambda{j}_abs']=abs(z);row[f'lambda{j}_residual']=r[j-1]
        row['rho_F']=abs(vals[0]);row['mode_type']='real' if abs(vals[0].imag)<1e-8 else 'complex'
        rows.append(row)
        np.savez(ROOT/f'compacton_gl6_h{h:.3f}_k{kappa}.npz',x=x,x0=x0,U=U,h=h,kappa=kappa,res=res)
        pd.DataFrame(rows).to_csv(ROOT/'gl6_reference_spectrum.csv',index=False)
        print(json.dumps({k:(float(v) if isinstance(v,(np.floating,float,int)) else v) for k,v in row.items() if k in ['h','kappa','rho_F','lambda1_real','lambda1_imag','lambda1_residual','fixed_point_residual','adjoint_relative_error']}),flush=True)
print(pd.DataFrame(rows).to_string(index=False))
