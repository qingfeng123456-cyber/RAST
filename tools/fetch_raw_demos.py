"""Optional pinned mirror retrieval. Listing is the default; --download retrieves 25.82 GB."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cs2.io import download


def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        while chunk:=stream.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'data/raw')
    args=p.parse_args()
    with (ROOT/'data/provenance/source_manifest.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    print(f"{len(rows)} pinned DEMOs; {sum(int(r['raw_bytes']) for r in rows):,} bytes. Training uses the supplied processed data and needs no raw downloads.")
    if not args.download:
        for row in rows:print(row['match_id'],row['event'],row['mirror_download_url'])
    else:
        root=args.output.resolve()
        for row in rows:
            path=(root/row['file_name']).resolve()
            if not path.is_relative_to(root):raise ValueError('Unsafe manifest path')
            path=download(row['mirror_download_url'],path,int(row['raw_bytes']))
            actual=sha256(path)
            if actual!=row['raw_sha256']:raise RuntimeError('Raw digest mismatch: '+str(path))
            print('Verified',row['file_name'])
