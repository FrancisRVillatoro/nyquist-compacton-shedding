import sys,time,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT))
from run_free_wave_benchmark import run
cases=[]
for method in ['midpoint','gl4']:
 for k in [1,2,3,5,10,20,40]:cases.append((method,k,.10))
 for a in [.04,.06,.08,.12,.15,.18]:cases.append((method,5,a))
rows=[]
for method,k,a in cases:
 t=time.time();r=run(method,k,a,h=.02,N=512,travel_cells=35);r['runtime_s']=time.time()-t;rows.append(r);print(json.dumps(r),flush=True)
 pd.DataFrame(rows).to_csv(ROOT/'free_wave_time_integrator_benchmark_h002.csv',index=False)
