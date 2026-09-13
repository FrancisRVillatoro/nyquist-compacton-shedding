#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; D=HERE/'derived'; OUT=HERE/'figures'; OUT.mkdir(exist_ok=True)
S=pd.read_csv(D/'wp8_hyperviscosity_spectrum_strict.csv')
# tolerate either field name from staged/final table
rcol='epsilon_over_eps_c' if 'epsilon_over_eps_c' in S else 'epsilon_over_eps_c_strict'
ons=np.array([44.,145.,287.,615.]); rhos=S.iloc[:4].rho.to_numpy(float); x=1/np.log(rhos); fit=np.polyfit(x,ons,1)
plt.rcParams.update({'font.size':12,'axes.labelsize':12,'xtick.labelsize':10.5,'ytick.labelsize':10.5,'legend.fontsize':10})
fig,ax=plt.subplots(1,2,figsize=(10.6,4.2));ax[0].plot(S[rcol],S.rho,'o-');ax[0].axhline(1,lw=1);ax[0].axvline(1,lw=1,ls='--');ax[0].set(xlabel=r'$\varepsilon/\varepsilon_c$',ylabel=r'$\rho_F$');ax[0].grid(alpha=.2)
xx=np.linspace(.95*x.min(),1.03*x.max(),300);ax[1].plot(x,ons,'o',label='nonlinear onset');ax[1].plot(xx,np.polyval(fit,xx),'--',label=r'fit, $R^2=0.99815$');ax[1].set(xlabel=r'$1/\log\rho_F$',ylabel='onset cell crossing');ax[1].grid(alpha=.2);ax[1].legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'control_threshold_onset.pdf',bbox_inches='tight');plt.close(fig)
# Noninteger-kappa early histories
b=pd.read_csv(D/'wp8_baseline_trace.csv'); k=pd.read_csv(D/'wp8_kappa20p5_trace.csv');fig,ax=plt.subplots(figsize=(6.4,4.2));ax.plot(b.time,b.max_ahead_HP_over_h2,label=r'$\kappa=20$');ax.plot(k.time,k.max_ahead_HP_over_h2,label=r'$\kappa=20.5$');ax.set(xlabel='time',ylabel=r'max. forward high-pass / $h^2$');ax.legend(frameon=False);fig.tight_layout();fig.savefig(OUT/'noninteger_kappa_control.pdf',bbox_inches='tight');plt.close(fig)
# c0 natural maxima + prepared packet speed
rows=[]
for c0 in (0.,.05,.10,.20,.50):
    q=pd.read_csv(D/f'wp8_prepared_c0_{c0:.2f}.csv');sel=q.iloc[max(1,int(.1*len(q))):max(2,int(.9*len(q)))];v=.02*np.polyfit(sel.time,sel.X_cells_unwrapped,1)[0];rows.append((c0,v))
rr=np.array(rows); small=rr[rr[:,0]<=.2]; cf=np.polyfit(small[:,0],small[:,1],1)
base=float(b.max_ahead_HP_over_h2.max()); ch=pd.read_csv(D/'wp8_c0_half_trace.csv'); ch2=pd.read_csv(D/'wp8_c0_half_trace_25_50.csv'); ce=pd.read_csv(D/'wp8_c0_equal_c_trace.csv'); vals=[base,max(ch.max_ahead_HP_over_h2.max(),ch2.max_ahead_HP_over_h2.max()),ce.max_ahead_HP_over_h2.max()]
fig,ax=plt.subplots(1,2,figsize=(10.6,4.2));ax[0].bar([r'$c_0=0$',r'$c_0=c/2$',r'$c_0=c$'],vals);ax[0].set_ylabel(r'max. forward high-pass / $h^2$');ax[1].plot(rr[:,0],rr[:,1],'o');xx=np.linspace(0,.2,100);ax[1].plot(xx,np.polyval(cf,xx),'--');ax[1].set(xlabel=r'$c_0$',ylabel='prepared-packet speed');fig.tight_layout();fig.savefig(OUT/'c0_generation_and_drift.pdf',bbox_inches='tight');plt.close(fig)
