#!/usr/bin/env python3
"""Bundle an allowlisted local run report. Does not upload files or checkpoints.

Review output before sharing: an allowlisted filename can still contain private
content. Symlinks and path traversal are rejected. No home-directory sweeping.
"""
from __future__ import annotations
import argparse,hashlib,json,zipfile
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-dir',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--max-total-mb',type=int,default=100)
    a=ap.parse_args();root=a.run_dir.resolve();out=a.output.resolve()
    if not root.is_dir():raise SystemExit('Run directory does not exist')
    if out.exists():raise SystemExit('Refusing to overwrite an existing report bundle')
    if a.max_total_mb<=0:raise SystemExit('Positive size budget required')
    if not (root/'REPORT_BACK.md').is_file():raise SystemExit('Required REPORT_BACK.md is absent')
    allowed_top={'REPORT_BACK.md','REPO_ASSESSMENT.md','run_manifest.json','run_result.json','environment.json',
                 'config.json','metrics.json','metrics.jsonl','metrics.csv','changes.diff','TEST_REPORT.md','CLAIM_UPDATE.md'}
    allowed_dirs={'logs','metrics','reports','decisions','figures'}
    suffixes={'.md','.json','.jsonl','.csv','.txt','.log','.diff','.png','.svg','.pdf'}
    chosen=[];total=0
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise SystemExit(f'Symlink rejected: {p.relative_to(root)}')
        if not p.is_file() or p.resolve()==out:continue
        rel=p.relative_to(root)
        if p.suffix.lower() not in suffixes:continue
        if (len(rel.parts)==1 and rel.name in allowed_top) or (len(rel.parts)>1 and rel.parts[0] in allowed_dirs):
            if any(part.startswith('.') for part in rel.parts):continue
            total+=p.stat().st_size;chosen.append((p,rel))
    if total>a.max_total_mb*1000000:raise SystemExit('Report size exceeds budget; trim logs/figures explicitly')
    out.parent.mkdir(parents=True,exist_ok=True)
    manifest=[]
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p,rel in chosen:
            data=p.read_bytes();z.writestr(str(rel),data)
            manifest.append({'path':str(rel),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        z.writestr('REPORT_BUNDLE_MANIFEST.json',json.dumps({'files':manifest,'warning':'Review content before sharing. No automatic upload.'},indent=2))
    print(json.dumps({'created':str(out),'files':len(chosen),'uncompressed_bytes':total,'uploaded':False}))
if __name__=='__main__':main()
