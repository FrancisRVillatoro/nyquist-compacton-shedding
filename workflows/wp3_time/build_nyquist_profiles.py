import numpy as np
from pathlib import Path
OUT=Path(__file__).resolve().parent

def solve_Fh(h,Leta=60,Neta=8192):
    eta=np.linspace(-Leta,Leta,Neta,endpoint=False)
    de=eta[1]-eta[0]
    q=2*np.pi*np.fft.fftfreq(Neta,d=de)
    den=np.cos(2*q)-26*np.cos(q)+33
    sinc=np.ones_like(q);nz=np.abs(q)>1e-14;sinc[nz]=np.sin(q[nz])/q[nz]
    Mh=10*sinc*((h*h+12)*np.cos(q)-5*h*h+12)/den
    F=1/np.cosh(0.28*eta)**3;F/=F.max()
    for it in range(1500):
        G=np.fft.ifft(Mh*np.fft.fft(F*F)).real
        V=G.max();Fn=G/V;Fn=np.roll(Fn,Neta//2-np.argmax(Fn))
        err=np.max(np.abs(Fn-F));F=.65*F+.35*Fn
        if err<2e-13:break
    G=np.fft.ifft(Mh*np.fft.fft(F*F)).real;V=G.max();F/=F.max()
    return eta,F,V,np.trapezoid(F,eta),it+1,err
for h in [0.01,0.02]:
    eta,F,V,M,it,err=solve_Fh(h)
    np.savez(OUT/f'nyquist_profile_h{h:.2f}.npz',eta=eta,F=F,V=V,M=M,h=h)
    print(h,V,M,it,err)
