#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, sys
ROOT=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(ROOT/'metadata/RAW_DATA_MANIFEST.csv')))
missing=[]
for r in rows:
    p=ROOT/'data/raw'/r['filename']
    if not p.exists(): missing.append(r['filename']); continue
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    if h!=r['sha256']: raise SystemExit(f'Hash mismatch: {p.name}')
if missing:
    print('Missing raw files:',*missing,sep='\n  ')
    print('Download from the companion archive/Zenodo record after the final DOI is inserted in DATA_AVAILABILITY.md.')
    sys.exit(2)
print('All raw historical files present and verified.')
