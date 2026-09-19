#!/usr/bin/env python3
"""Read-only environment probe; no installation, device mutation, or network."""
from __future__ import annotations
import argparse,datetime,json,platform,sys,importlib.metadata
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    report={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'python':sys.version,'platform':platform.platform(),'libraries':{},'devices':[]}
    for name in ['numpy','scipy','torch','jsonschema']:
        try: report['libraries'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: report['libraries'][name]=None
    try:
        import torch
        report['cuda_available']=torch.cuda.is_available()
        report['torch_cuda_build']=torch.version.cuda
        report['mps_available']=bool(hasattr(torch.backends,'mps') and torch.backends.mps.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p=torch.cuda.get_device_properties(i)
                report['devices'].append({'kind':'cuda','index':i,'name':p.name,
                    'total_memory_bytes':p.total_memory,'capability':[p.major,p.minor]})
        else:report['devices'].append({'kind':'cpu'})
    except Exception as exc:
        report['device_probe_error']=f'{type(exc).__name__}: {exc}'
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
