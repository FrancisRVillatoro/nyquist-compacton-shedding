#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
from fixed_phase_map import analyze_case

p=argparse.ArgumentParser()
p.add_argument('--h',type=float,required=True)
p.add_argument('--kappa',type=int,required=True)
p.add_argument('--phi',type=float,default=0.25)
p.add_argument('--Ldom',type=float,default=16.0)
p.add_argument('--outdir',default=str(Path(__file__).resolve().parent/'cases'))
p.add_argument('--maxiter',type=int,default=600)
a=p.parse_args()
out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
row,_=analyze_case(a.h,a.kappa,a.phi,a.Ldom,Uguess=None,maxiter=a.maxiter)
row['excess_percent']=100*(row['rho_F']-1)
row['log_growth_per_cell']=__import__('numpy').log(row['rho_F'])
f=out/f"h{a.h:g}_k{a.kappa:g}_phi{a.phi:g}.csv"
pd.DataFrame([row]).to_csv(f,index=False)
print(pd.DataFrame([row])[['h','kappa','phi_right','rho_F','lambda1_real','lambda1_imag','mode_type','fixed_point_residual','lambda1_residual','runtime_s']].to_string(index=False),flush=True)
print(f,flush=True)
