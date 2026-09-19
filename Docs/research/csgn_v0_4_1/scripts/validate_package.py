#!/usr/bin/env python3
"""Check released files, configs, templates, relative links, and local evidence.

This validates package consistency, not scientific truth or complete model code.
"""
from __future__ import annotations
import argparse,hashlib,json,math,re,sys
from pathlib import Path
import jsonschema
ROOT=Path(__file__).resolve().parents[1]

def load(p):
    def reject(x):raise ValueError(f'Nonfinite JSON value: {x}')
    return json.loads(p.read_text(),parse_constant=reject)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--no-hash-check',action='store_true',help='Build-time validation only')
    ap.add_argument('--output',type=Path);a=ap.parse_args();errors=[];checks=0
    def check(ok,msg):
        nonlocal checks
        checks+=1
        if not ok: errors.append(msg)
    schemas={p.stem:load(p) for p in (ROOT/'schemas').glob('*.json')}
    for name,s in schemas.items():
        try:jsonschema.Draft202012Validator.check_schema(s);check(True,name)
        except Exception as e:check(False,f'{name}: {e}')
    for name in ['cpu_smoke','learned_pilot','rule_transfer_pilot']:
        c=load(ROOT/'configs'/f'{name}.json')
        errs=list(jsonschema.Draft202012Validator(schemas['experiment_config.schema']).iter_errors(c))
        check(not errs,f'config {name}: '+'; '.join(e.message for e in errs))
        g,l,w=c['graph'],c['learning'],c['workspace']
        check(g['columns']%g['regions']==0,f'{name}: nonuniform region division')
        check(g['protected_slots']<=g['out_slots']<g['columns'],f'{name}: slot capacity')
        check(g['active_senders']<=g['columns'] and g['updated_node_cap']<=g['columns'],f'{name}: active cap')
        check(g['real_routes_per_sender']<=g['out_slots'],f'{name}: route count')
        q=1-w['kappa']+w['kappa']*(w['Ah_spectral_norm_max']+w['Br_times_Lm_max'])
        check(0<w['kappa']<=1 and q<=w['required_q_max']+1e-12 and q<1,f'{name}: contraction configuration')
        check(c['resources']['allow_paid_compute'] is False,f'{name}: paid compute disallowed')
        check(c['task']['query_feedback'] is False,f'{name}: query feedback leakage')
    cp=load(ROOT/'configs/confirmation_plan.json')
    check(cp['launchable'] is False and cp['primary_metric'] is None,'Confirmation plan must await preregistration')
    for name in ['run_manifest','run_result']:
        value=load(ROOT/'templates'/f'{name}.json')
        errs=list(jsonschema.Draft202012Validator(schemas[name+'.schema']).iter_errors(value))
        check(not errs,'template '+name+': '+'; '.join(e.message for e in errs))
    for p in list(ROOT.glob('*.md'))+list((ROOT/'handoff').glob('*.md')):
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
            if target.startswith(('http:','https:','mailto:','#')):continue
            target=target.split('#',1)[0]
            check((p.parent/target).exists(),f'Broken link {p.relative_to(ROOT)} -> {target}')
    paper=(ROOT/'paper/csgn_paper_v0_4_1.md').read_text()
    tags=re.findall(r'\\tag\{(\d+)\}',paper)
    check(tags==[str(i) for i in range(1,37)],'Equation tags must remain unique 1–36 in order')
    check('four-endpoint' in paper.lower() and '19,438' in paper,'Paper release facts absent')
    rr=load(ROOT/'verification/results_v0_4_1.json')
    check(rr['group_count']==len(rr['groups'])==24,'Reference group count')
    check(rr['case_count']==sum(g['cases'] for g in rr['groups'])==19438,'Reference case count')
    check(all(g['status']=='pass' for g in rr['groups']),'Reference result status')
    for name,sha in rr['source_sha256'].items():
        check(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,'Reference result/source mismatch '+name)
    if not a.no_hash_check:
        manifest=load(ROOT/'MANIFEST.sha256.json')
        for ent in manifest['files']:
            rel=Path(ent['path'])
            check(not rel.is_absolute() and '..' not in rel.parts,'Unsafe manifest path')
            p=ROOT/rel
            check(p.is_file(),f'Missing manifest file {rel}')
            if p.is_file():
                check(p.stat().st_size==ent['bytes'],f'Size mismatch {rel}')
                check(hashlib.sha256(p.read_bytes()).hexdigest()==ent['sha256'],f'Hash mismatch {rel}')
    report={'status':'pass' if not errors else 'fail','checks':checks,'errors':errors,
            'hash_check':not a.no_hash_check,'scope':'package integrity and schema consistency, not model validation'}
    if a.output:a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if errors:sys.exit(1)
if __name__=='__main__':main()
