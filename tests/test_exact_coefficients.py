import sympy as sp

def test_edge_seed_coefficients():
    # Coefficients multiplying (2 phi - 1) in the leading Nyquist edge defect.
    vals={'ismail':sp.Rational(1,24),'defrutos':sp.Rational(1,180),'pade6':sp.Rational(1,360),'pade8':sp.Rational(1,280)}
    assert vals['ismail']/vals['defrutos']==sp.Rational(15,2)
    assert vals['pade6']/vals['defrutos']==sp.Rational(1,2)
    assert vals['pade8']/vals['defrutos']==sp.Rational(9,14)

def test_defrutos_nyquist_projection():
    phi=sp.symbols('phi', real=True)
    DN=(2*phi-1)/sp.Integer(180)
    assert sp.integrate(DN,(phi,0,1))==0
    assert sp.simplify(DN.subs(phi,1-phi)+DN)==0
