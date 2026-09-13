#!/usr/bin/env python3
from pathlib import Path
import argparse, subprocess, sys, hashlib, json, compileall
ROOT=Path(__file__).resolve().parent

def verify_hashes():
    mf=ROOT/'metadata/FILES.sha256'
    bad=[];n=0
    for line in mf.read_text().splitlines():
        if not line.strip():continue
        expected,rel=line.split('  ',1);p=ROOT/rel;n+=1
        if not p.exists():bad.append((rel,'missing'));continue
        h=hashlib.sha256(p.read_bytes()).hexdigest()
        if h!=expected:bad.append((rel,'hash'))
    if bad: raise SystemExit(f'Integrity failures: {bad[:10]}')
    print(f'Integrity OK: {n} tracked files')

def check(skip_hashes=False):
    if not skip_hashes: verify_hashes()
    ok=compileall.compile_dir(str(ROOT/'workflows'),quiet=1)
    if not ok: raise SystemExit('Python compile check failed')
    subprocess.run([sys.executable,'-m','pytest','-q',str(ROOT/'tests/test_headline_results.py'),str(ROOT/'tests/test_exact_coefficients.py')],check=True,cwd=ROOT)
    print('Headline and exact-algebra checks OK')

def figures():
    subprocess.run([sys.executable,str(ROOT/'make_figures.py')],check=True,cwd=ROOT)
    subprocess.run([sys.executable,str(ROOT/'workflows/wp7_refinement/make_wp7_figures.py')],check=True,cwd=ROOT/'workflows/wp7_refinement')
    subprocess.run([sys.executable,str(ROOT/'workflows/wp8_closure/make_wp8_figures.py')],check=True,cwd=ROOT/'workflows/wp8_closure')
def smoke(): subprocess.run([sys.executable,'-m','pytest','-q'],check=True,cwd=ROOT)
def full(wp,execute=False):
    cmd=[sys.executable,str(ROOT/'scripts/full_reproduction.py'),'--wp',wp]
    if execute: cmd.append('--execute')
    subprocess.run(cmd,check=True,cwd=ROOT)

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    c=sub.add_parser('check');c.add_argument('--skip-hashes',action='store_true')
    sub.add_parser('figures');sub.add_parser('smoke')
    f=sub.add_parser('full');f.add_argument('--wp',default='all',choices=['1','2','3','4','5','7','8','all']);f.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.cmd=='check':check(a.skip_hashes)
    elif a.cmd=='figures':figures()
    elif a.cmd=='smoke':smoke()
    else:full(a.wp,a.execute)
if __name__=='__main__':main()
