import numpy as np
from scipy.linalg import solve_banded
from wp3_integrators import GL2_A,SQRT3,phi,dphi

OFFS=np.array([-2,-1,0,1,2],int)

def cyclic_block_penta_solve(sysm,d1,d2,dt,rhs1,rhs2):
    N=sysm.N; n=2*N; lower=upper=5
    ca=np.array([1,26,66,26,1],float)/120.0
    h=sysm.h
    cl=np.array([-1,2,0,-2,1],float)/(2*h**3)+np.array([-1,-10,0,10,1],float)/(24*h)
    ds=[d1,d2]
    ab=np.zeros((lower+upper+1,n),float)
    corners=[]
    jall=np.arange(N)
    for i in range(2):
      for s in range(2):
        for kk,k in enumerate(OFFS):
          # row node j, col node q=j+k
          j0=max(0,-k); j1=min(N,N-k)
          if j1>j0:
            j=np.arange(j0,j1);q=j+k
            rows=2*j+i;cols=2*q+s
            vals=(ca[kk] if i==s else 0.0)+dt*GL2_A[i,s]*cl[kk]*ds[s][q]
            ab[upper+rows[0]-cols[0],cols]=vals
          # wrapped rows
          if k>0:
            js=range(N-k,N)
          elif k<0:
            js=range(0,-k)
          else:
            js=()
          for j in js:
            q=(j+k)%N;row=2*j+i;col=2*q+s
            val=(ca[kk] if i==s else 0.0)+dt*GL2_A[i,s]*cl[kk]*ds[s][q]
            corners.append((row,col,val))
    r=len(corners)
    B=np.zeros((n,1+r),float)
    B[0::2,0]=rhs1;B[1::2,0]=rhs2
    for q,(row,col,val) in enumerate(corners):B[row,q+1]=val
    X=solve_banded((lower,upper),ab,B,check_finite=False,overwrite_ab=True,overwrite_b=False)
    x=X[:,0]
    if r:
      Y=X[:,1:]
      M=np.eye(r);vx=np.empty(r)
      for p,(row,col,val) in enumerate(corners):
        vx[p]=x[col];M[p,:]+=Y[col,:]
      x-=Y@np.linalg.solve(M,vx)
    return x[0::2],x[1::2]

def gl4_step_banded(sysm,U,dt,maxit=3,tol=1e-10):
    U=np.asarray(U)
    f0=-sysm.A_lu.solve(sysm.L@phi(U))
    c1=.5-SQRT3/6;c2=.5+SQRT3/6
    Y1=U+c1*dt*f0;Y2=U+c2*dt*f0
    for _ in range(maxit):
      p1=phi(Y1);p2=phi(Y2)
      Lp1=sysm.L@p1;Lp2=sysm.L@p2
      R1=sysm.A@(Y1-U)+dt*(GL2_A[0,0]*Lp1+GL2_A[0,1]*Lp2)
      R2=sysm.A@(Y2-U)+dt*(GL2_A[1,0]*Lp1+GL2_A[1,1]*Lp2)
      dd1,dd2=cyclic_block_penta_solve(sysm,dphi(Y1),dphi(Y2),dt,-R1,-R2)
      Y1+=dd1;Y2+=dd2
      if max(np.max(np.abs(dd1)),np.max(np.abs(dd2)))<tol:break
    return U-dt*.5*sysm.A_lu.solve(sysm.L@(phi(Y1)+phi(Y2)))

def gl4_cell_banded(sysm,U,dt,kappa,maxit=3,tol=1e-10):
    V=U
    for _ in range(int(kappa)):V=gl4_step_banded(sysm,V,dt,maxit=maxit,tol=tol)
    return np.roll(V,-1)
