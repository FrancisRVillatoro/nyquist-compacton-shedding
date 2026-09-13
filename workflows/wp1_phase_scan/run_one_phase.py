import sys,json,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from phase_floquet_scan import analyze_phase
p=argparse.ArgumentParser(); p.add_argument('--h',type=float,required=True); p.add_argument('--kappa',type=float,required=True); p.add_argument('--phi',type=float,required=True); p.add_argument('--out',required=True); p.add_argument('--maxiter',type=int,default=500); a=p.parse_args()
row,_=analyze_phase(a.h,a.kappa,a.phi,16.0,Uguess=None,nb=10,maxiter=a.maxiter)
row['case']=f'h{a.h:g}_k{a.kappa:g}'
Path(a.out).write_text(json.dumps(row,indent=2))
print(json.dumps({k:row[k] for k in ['case','phi_right','rho_F','lambda_real','lambda_imag','q0_abs','mode_type','fixed_point_residual','right_eigen_residual','left_eigen_residual','runtime_s']}))
