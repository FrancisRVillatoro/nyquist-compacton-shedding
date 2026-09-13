from pathlib import Path
import csv, hashlib
R=Path(__file__).resolve().parents[1]

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def test_raw_manifest_has_two_historical_files():
    rows=list(csv.DictReader(open(R/'metadata/RAW_DATA_MANIFEST.csv')))
    assert len(rows)==2
    assert all(len(r['sha256'])==64 for r in rows)

def test_tracked_manifest_if_present():
    p=R/'metadata/FILES.sha256'
    if not p.exists(): return
    for line in p.read_text().splitlines():
        if not line.strip():continue
        h,rel=line.split('  ',1)
        target=R/rel
        assert target.exists(), rel
        assert sha256(target)==h, rel
