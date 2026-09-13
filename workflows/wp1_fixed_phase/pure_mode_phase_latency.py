#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OUT=Path(__file__).resolve().parent

def build_ops(N,h):
 r=np.arange(N);rows=[];cols=[];av=[];lv=[]
 ca=np.array([1,26,66,26,1],float)/120
 cb=np.array([-1,-10,0,10,1],float)/(24*h)
 cc=np.array([-1,2,0,-2,1],float)/(2*h**3)
 for kk,k in enumerate([-2,-1,0,1,2]):
  c=(r+k)%N;rows.extend(r);cols.extend(c);av.extend(np.full(N,ca[kk]));lv.extend(np.full(N,cb[kk]+cc[kk]))
 return sp.csc_matrix((av,(rows,cols)),shape=(N,N)),sp.csc_matrix((lv,(rows,cols)),shape=(N,N))

def midpoint(U,A,L,dt):
 W=U.copy()
 for _ in range(7):
  G=(2/dt)*(A@(W-U))+L@(np.abs(W)*W)
  J=(2/dt)*A+L@sp.diags(2*np.abs(W),0,format='csc')
  d=spla.spsolve(J,-G,permc_spec='NATURAL');W+=d
  if np.max(np.abs(d))<8e-13:break
 return 2*W-U

def cell(U,A,L,dt,kappa):
 V=U.copy()
 for _ in range(int(kappa)):V=midpoint(V,A,L,dt)
 return np.roll(V,-1)

def hp(z):return (2*z-np.roll(z,1)-np.roll(z,-1))/4

cases=[
 ('h0.01_k5_min',.01,5,150),('h0.01_k5_max',.01,5,120),
 ('h0.02_k20_min',.02,20,90),('h0.02_k20_max',.02,20,75)]
rows=[];summ=[]
for name,h,kappa,nmax in cases:
 b=np.load(OUT/f'base_{name}.npz')
 Us=b['Us'];rhat=b['rhat'];lhat=b['lhat'];R0=b['R0'];q0=float(b['q0']);lam=float(b['lambda1']);N=len(Us);dt=h/kappa
 A,L=build_ops(N,h)
 U=Us+h*h*q0*rhat
 sgn=1 if q0>=0 else -1
 tr=[]
 for n in range(nmax+1):
  du=U-Us;q=float(lhat@(du/h**2));qg=sgn*((-1)**n)*q
  tr.append({'case':name,'cell':n,'time':n*h,'q':q,'q_gauge':qg,'hp_edge':float(np.max(np.abs(hp(du)))/h**2)})
  if n<nmax:U=cell(U,A,L,dt,kappa)-R0
 g=pd.DataFrame(tr);g.to_csv(OUT/f'pure_mode_trace_{name}.csv',index=False);rows.extend(tr)
 q=g.q_gauge.values;c=[i for i in range(1,len(q)-1) if q[i]>=.02 and q[i]>=q[i-1] and q[i]>q[i+1]]
 i=c[0] if c else int(np.argmax(q))
 qturn=.07147 if h==.01 else .08920
 pred=np.log(qturn/abs(q0))/np.log(abs(lam))
 summ.append({'case':name,'h':h,'kappa':kappa,'phi_right':float(b['phi']),'seed_class':'min' if 'min' in name else 'max',
              'q0':q0,'q0_abs':abs(q0),'mu':abs(lam),'turn_cell_obs':int(g.cell.iloc[i]),'turn_time_obs':float(g.time.iloc[i]),
              'turn_q_obs':float(q[i]),'turn_cell_pred_using_reference_qturn':pred,'turn_cell_error':int(g.cell.iloc[i])-pred})
pd.DataFrame(rows).to_csv(OUT/'pure_mode_phase_latency_traces.csv',index=False)
s=pd.DataFrame(summ);s.to_csv(OUT/'pure_mode_phase_latency_summary.csv',index=False)
pairs=[]
for (h,k),g in s.groupby(['h','kappa']):
 mn=g[g.seed_class=='min'].iloc[0];mx=g[g.seed_class=='max'].iloc[0]
 pred=np.log(mx.q0_abs/mn.q0_abs)/np.log((mx.mu+mn.mu)/2)
 pairs.append({'h':h,'kappa':k,'delta_turn_cells_observed':mn.turn_cell_obs-mx.turn_cell_obs,'delta_turn_cells_predicted':pred,
               'delta_time_observed':h*(mn.turn_cell_obs-mx.turn_cell_obs),'delta_time_predicted':h*pred,
               'error_cells':(mn.turn_cell_obs-mx.turn_cell_obs)-pred})
pd.DataFrame(pairs).to_csv(OUT/'pure_mode_phase_latency_pairs.csv',index=False)
print(s.to_string(index=False));print(pd.DataFrame(pairs).to_string(index=False))
