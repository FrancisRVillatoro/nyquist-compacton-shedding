import sys,time,json
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT))
from wp3_integrators import *

def fixed_phase_x0(Ldom,h,phi=0.25):
    m=int(np.rint((Ldom/2+2*np.pi)/h-phi))
    return (m+phi)*h-2*np.pi

def state_path(method,h,k): return ROOT/f'compacton_{method}_h{h:.3f}_k{k}.npz'

def load_guess(method,h,k,x,x0,Ldom):
    p=state_path(method,h,k)
    if p.exists(): return np.load(p)['U']
    other='midpoint' if method=='gl4' else 'gl4'
    po=state_path(other,h,k)
    if po.exists(): return np.load(po)['U']
    # nearest saved same method/other
    candidates=[]
    for meth in [method,other]:
        for q in ROOT.glob(f'compacton_{meth}_h{h:.3f}_k*.npz'):
            try: kk=int(q.stem.split('_k')[-1]); candidates.append((abs(kk-k),q))
            except: pass
    if candidates:
        return np.load(sorted(candidates,key=lambda z:z[0])[0][1])['U']
    return compacton(x,x0,Ldom)

def main(h,kappa,method,Ldom=16.0):
    x0=fixed_phase_x0(Ldom,h,.25);N=int(round(Ldom/h));x=np.arange(N)*h
    sysm=SpatialSystem(N,h); guess=load_guess(method,h,kappa,x,x0,Ldom)
    dt=h/kappa
    t0=time.time()
    guess_res=float(np.linalg.norm(sysm.cell_map(guess,dt,kappa,method=method)-guess,np.inf))
    if guess_res < 5e-9:
        Us=guess.copy(); res=guess_res
    else:
        Us,res,dt=solve_traveling_compacton(sysm,guess,h,kappa,method,
                                            f_tol=1e-9,maxiter=5,inner_maxiter=12)
    tsolve=time.time()-t0
    np.savez_compressed(state_path(method,h,kappa),x=x,U=Us,x0=x0,h=h,kappa=kappa,Ldom=Ldom,residual=res)
    t1=time.time()
    rr=leading_ritz(sysm,Us,h,kappa,method,x0,nb=10,maxiter=650,tol=5e-9,
                    seed=20260904+int(1e4*h)+kappa+(0 if method=='midpoint' else 10000))
    tspec=time.time()-t1
    vals=rr['values'];rres=rr['residuals']
    row={'h':h,'Ldom':Ldom,'x0':x0,'phi_right':((x0+2*np.pi)/h)%1,
         'method':method,'kappa':kappa,'dt':dt,'N':N,'fixed_point_residual':res,
         'solve_runtime_s':tsolve,'spectrum_runtime_s':tspec,'ritz_iterations':rr['iterations'],
         'edge_energy_fraction':rr['edge_frac'],'right_edge_energy_fraction':rr['right_frac'],
         'left_edge_energy_fraction':rr['left_frac']}
    for j in range(4):
        row[f'lambda{j+1}_real']=float(vals[j].real);row[f'lambda{j+1}_imag']=float(vals[j].imag)
        row[f'lambda{j+1}_abs']=float(abs(vals[j]));row[f'lambda{j+1}_residual']=float(rres[j])
    row['rho_F']=row['lambda1_abs'];row['mode_type']='real' if abs(row['lambda1_imag'])<1e-8 else 'complex'
    outf=ROOT/'time_integrator_spectrum_cases.csv'
    if outf.exists():
        df=pd.read_csv(outf);df=df[~((df.h==h)&(df.kappa==kappa)&(df.method==method))]
        df=pd.concat([df,pd.DataFrame([row])],ignore_index=True)
    else:df=pd.DataFrame([row])
    df.sort_values(['h','method','kappa']).to_csv(outf,index=False)
    print(json.dumps(row),flush=True)

if __name__=='__main__':
    main(float(sys.argv[1]),int(sys.argv[2]),sys.argv[3],float(sys.argv[4]) if len(sys.argv)>4 else 16.0)
