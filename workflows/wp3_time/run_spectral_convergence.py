import sys,time,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).parent))
from wp3_integrators import SpatialSystem,compacton,solve_traveling_compacton,leading_ritz

OUT=Path(__file__).parent

def fixed_phase_x0(Ldom,h,phi=0.25):
    m=int(np.rint((Ldom/2+2*np.pi)/h-phi))
    return (m+phi)*h-2*np.pi

def run_h(h,Ldom=16.0,kappas=(5,10,20,40,80,160)):
    x0=fixed_phase_x0(Ldom,h,0.25)
    N=int(round(Ldom/h)); x=np.arange(N)*h
    sysm=SpatialSystem(N,h)
    Uanalytic=compacton(x,x0,Ldom)
    states={}
    rows=[]
    # midpoint first; continuation in kappa
    Uprev=Uanalytic
    for method in ['midpoint','gl4']:
        if method=='gl4':
            Uprev=None
        for kappa in kappas:
            keym=('midpoint',kappa)
            if method=='gl4':
                guess=states[keym].copy()
            else:
                guess=Uprev
            t0=time.time()
            Us,res,dt=solve_traveling_compacton(sysm,guess,h,kappa,method,
                                                f_tol=1e-10,maxiter=10,inner_maxiter=25)
            tsolve=time.time()-t0
            Uprev=Us; states[(method,kappa)]=Us.copy()
            np.savez_compressed(OUT/f'compacton_{method}_h{h:.3f}_k{kappa}.npz',
                                x=x,U=Us,x0=x0,h=h,kappa=kappa,Ldom=Ldom,residual=res)
            t1=time.time()
            rr=leading_ritz(sysm,Us,h,kappa,method,x0,nb=8,maxiter=420,tol=1e-8,
                            seed=20260904+int(1000*h)+kappa)
            tspec=time.time()-t1
            vals=rr['values']; rres=rr['residuals']
            row={'h':h,'Ldom':Ldom,'x0':x0,'phi_right':((x0+2*np.pi)/h)%1,
                 'method':method,'kappa':kappa,'dt':dt,'N':N,
                 'fixed_point_residual':res,'solve_runtime_s':tsolve,
                 'spectrum_runtime_s':tspec,'ritz_iterations':rr['iterations'],
                 'edge_energy_fraction':rr['edge_frac'],
                 'right_edge_energy_fraction':rr['right_frac'],
                 'left_edge_energy_fraction':rr['left_frac']}
            for j in range(4):
                if j<len(vals):
                    row[f'lambda{j+1}_real']=float(vals[j].real)
                    row[f'lambda{j+1}_imag']=float(vals[j].imag)
                    row[f'lambda{j+1}_abs']=float(abs(vals[j]))
                    row[f'lambda{j+1}_residual']=float(rres[j])
            row['rho_F']=row['lambda1_abs']
            row['mode_type']='real' if abs(row['lambda1_imag'])<1e-8 else 'complex'
            rows.append(row)
            pd.DataFrame(rows).to_csv(OUT/f'time_integrator_spectrum_h{h:.3f}.csv',index=False)
            print(json.dumps({'method':method,'h':h,'kappa':kappa,'rho':row['rho_F'],
                              'lambda':[row['lambda1_real'],row['lambda1_imag']],
                              'fp_res':res,'eig_res':row['lambda1_residual'],
                              'solve_s':tsolve,'spec_s':tspec}),flush=True)
    return pd.DataFrame(rows)

if __name__=='__main__':
    h=float(sys.argv[1]) if len(sys.argv)>1 else 0.02
    run_h(h)
