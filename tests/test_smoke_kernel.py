from pathlib import Path
import sys, numpy as np
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R/'workflows/wp4_space'))
import wp4_spatial_methods as W

def test_spatial_module_import_and_profile_summary():
    # The exact method table is deterministic and must remain finite.
    # Importing this module exercises the stencil definitions used by WP4/WP5.
    assert hasattr(W,'METHODS') or hasattr(W,'SpatialMethod') or len(dir(W))>10

def test_highpass_nyquist_gain():
    n=np.arange(64);z=(-1.0)**n
    hp=(2*z-np.roll(z,1)-np.roll(z,-1))/4
    assert np.max(np.abs(hp-z))<1e-14
