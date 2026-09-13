import pandas as pd,numpy as np
from pathlib import Path
OUT=Path(__file__).resolve().parent
rows=[]
def fit(t,y):return np.polyfit(t,y,1)[0]
# archived midpoint W20
rob=pd.read_csv(Path(__file__).resolve().parents[2]/'data'/'derived'/'historical'/'first_packet_profile_fit_window_robustness.csv')
for case,h,tmin,tmax in [('CFL5',.01,6,18),('CFL20',.02,27,35)]:
 g=rob[(rob.case==case)&(rob.W==20)&(rob.t>=tmin)&(rob.t<=tmax)].sort_values('t').reset_index(drop=True)
 for left in range(0,min(3,len(g)-2)):
  for right in range(0,min(3,len(g)-left-2)):
   q=g.iloc[left:len(g)-right if right else None]
   x=np.unwrap(q.xfit.values/250*2*np.pi)*250/(2*np.pi)
   rows.append({'dataset':f'archived {case}','trim_left':left,'trim_right':right,'n':len(q),'velocity':fit(q.t.values,x),'A_over_h2':q.A_over_h2.median(),'residual':q.relL2.median()})
# GL4 natural packets
ng=pd.read_csv(Path(__file__).resolve().parents[1]/'wp3_time'/'natural_gl4_h001_k5.csv')
for name,cmin,cmax in [('GL4 I',370,420),('GL4 II',425,455)]:
 g=ng[(ng.cell>=cmin)&(ng.cell<=cmax)&(ng.best_fit_relL2<.01)].sort_values('time').reset_index(drop=True)
 for left in range(0,min(2,len(g)-2)):
  for right in range(0,min(2,len(g)-left-2)):
   q=g.iloc[left:len(g)-right if right else None]; y=q.time.values+.01*q.best_fit_center_rel_cells.values
   rows.append({'dataset':name,'trim_left':left,'trim_right':right,'n':len(q),'velocity':fit(q.time.values,y),'A_over_h2':q.best_fit_A_over_h2.median(),'residual':q.best_fit_relL2.median()})
# WP4 natural packets, trimming 0,5%,10% worth of samples from both ends separately
w=pd.read_csv(Path(__file__).resolve().parents[1]/'wp4_space'/'natural_packet_snapshot_fits.csv')
for (m,p),g in w.groupby(['method','packet']):
 g=g.sort_values('t').reset_index(drop=True); trims=sorted(set([0,max(1,int(.05*len(g))),max(1,int(.10*len(g))) ]))
 for left in trims:
  for right in trims:
   if left+right>=len(g)-2:continue
   q=g.iloc[left:len(g)-right if right else None];y=q.t.values+.02*q.n_fit.values
   rows.append({'dataset':f'{m} {p}','trim_left':left,'trim_right':right,'n':len(q),'velocity':fit(q.t.values,y),'A_over_h2':q.A_over_h2.median(),'residual':q.profile_relL2.median()})
df=pd.DataFrame(rows);df.to_csv(OUT/'fit_range_sensitivity.csv',index=False)
s=df.groupby('dataset').agg(v_min=('velocity','min'),v_max=('velocity','max'),v_ptp=('velocity',lambda x:x.max()-x.min()),A_min=('A_over_h2','min'),A_max=('A_over_h2','max'),A_ptp=('A_over_h2',lambda x:x.max()-x.min()),resid_min=('residual','min'),resid_max=('residual','max')).reset_index();s.to_csv(OUT/'fit_range_sensitivity_summary.csv',index=False);print(s.to_string(index=False))
