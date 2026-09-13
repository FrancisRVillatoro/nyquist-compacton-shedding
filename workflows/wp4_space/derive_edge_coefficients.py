import sympy as sp
from wp4_spatial_methods import METHODS,OFFS
phi=sp.symbols('phi', real=True)
C0=[sp.Rational(-1,2),1,0,-1,sp.Rational(1,2)]
for method,m in METHODS.items():
    A=[sp.Rational(str(float(x))).limit_denominator(100000) for x in m['A']]
    D={}
    for n in range(-3,4):
        z=0
        for k,a,c in zip(OFFS,A,C0):
            if n+k<=0:
                y=phi-(n+k)
                z += a*y/sp.Integer(6)+c*y**4/sp.Integer(144)
        z=sp.factor(z)
        if z!=0:D[n]=z
    DN=sp.factor(sum(sp.Integer(-1)**n*z for n,z in D.items()))
    print(method)
    for n,z in D.items():print(f'  D_{n} = {z}')
    print('  D_N =',DN)
