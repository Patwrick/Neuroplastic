"""Bounded M2 graph and delta lifecycle; fixed theta, no learned-advantage claim."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import sys
import time
import traceback
import torch

from ..baselines.delta import DeltaModel
from ..consolidation import evaluate, predict_example, consolidate
from ..diagnostics import reachability, address_drift
from ..feedback import PendingFeedback, graph_write, graph_decay
from ..graph import GraphModel
from ..memory import ReplayStore
from ..tasks.t1 import T1Lifetime
from .evidence import ROOT, RELEASE, capture_code, sha256, utcnow, write_json, validate_artifact, validate_m2_config, peak_process_memory


def tensor_metrics(values):
    return {k: v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else v for k,v in values.items()}


def summarize(raw, model_name, phase, task_seed):
    result = []
    for context in sorted({m['context'] for m in raw}):
        rows = [m for m in raw if m['context']==context]
        for name, field, unit in [('squared_error','squared_error','SSE'),('accuracy','correct','fraction')]:
            result.append(dict(name=name,value=sum(m[field] for m in rows)/len(rows),unit=unit,
                               task_id='T1',condition=f'{model_name}/{phase}/context_{context}',
                               seed=task_seed,status='measured',notes=f'{len(rows)} correlated queries; fixed theta'))
    return result


def run_lifetime(model, model_name, config, task, task_seed, evaluation_seed, deadline, progress_path):
    state = model.initial_state()
    pending = PendingFeedback(config['resources']['max_pending_records'])
    replay = ReplayStore(config['resources']['max_replay_records'])
    metrics, phases, writes, diagnostics, sleep_reports = [], [], [], [], []
    event = 0
    inference_seconds = write_seconds = diagnostic_seconds = 0.
    snapshot_peak = 0
    operation_totals = {}
    last_snapshot = None
    diagnostic_rng = torch.Generator().manual_seed(evaluation_seed+30000)

    def persist():
        write_json(progress_path,dict(status='in_progress',model=model_name,task_seed=task_seed,
                                      phases=phases,writes=writes,diagnostics=diagnostics,sleep=sleep_reports))

    def budget():
        if time.perf_counter() >= deadline:
            raise TimeoutError('180-second per-job smoke cap reached')

    def queries(phase, contexts, cold=False, episodes=1):
        nonlocal inference_seconds
        raw = []
        # Each episode resets once for cold; no arbitrary per-query resets.
        for episode in range(episodes):
            budget()
            examples = task.queries(contexts,config['task']['queries_per_context'],evaluation_seed+episode)
            rows, _, _, seconds = evaluate(model,state,examples,cold=cold,deadline=deadline)
            raw.extend(rows)
            inference_seconds += seconds
        phases.append(dict(phase=phase,cold=cold,writes_enabled=False,answer_retrieval=False,
                           query_episodes=episodes,rows=raw))
        metrics.extend(summarize(raw,model_name,phase,task_seed))
        persist()

    for context_index, examples in enumerate(task.contexts):
        for support_pass in range(config['task']['support_passes']):
            for example in examples:
                budget()
                start = time.perf_counter()
                _, state, snapshot = predict_example(model,state,example,f'decision-{event}',event,event_type='support')
                inference_seconds += time.perf_counter()-start
                event += 1
                snapshot_peak = max(snapshot_peak,snapshot.storage_bytes())
                if hasattr(snapshot,'routing'):
                    for key,value in snapshot.routing.metrics[0].items():
                        if isinstance(value,(int,float)):
                            if key in {'queue_service_max_event_latency','queue_occupancy','output_path_internal_round_latency'}:
                                operation_totals[key] = max(operation_totals.get(key,0),value)
                            else:
                                operation_totals[key] = operation_totals.get(key,0)+value
                    # Enforce actual decoder/design identity before any write.
                    torch.testing.assert_close(snapshot.reconstruct(),snapshot.prediction,atol=1e-6,rtol=1e-5)
                if not pending.add(snapshot):
                    continue
                ready = pending.take(snapshot.decision_id,event,state.version,
                                     model.topology.generation if model_name=='graph' else None)
                if ready is None:
                    continue
                before_decay = state
                start = time.perf_counter()
                if model_name=='graph':
                    state = graph_decay(state,config)
                    before_write = state.clone(detach=True)
                    state,result = graph_write(state,ready,example.target,config,apply_decay=False)
                    write_values = tensor_metrics(vars(result))
                else:
                    before_write = state.clone(detach=True)
                    state,result = model.write(state,ready,example.target,
                                                beta=config['learning']['beta'],eps=config['learning']['rate_epsilon'],
                                                feedback_time=event)
                    write_values = tensor_metrics(result)
                write_values['passive_decay_l1'] = float((before_write.Pa-before_decay.Pa).abs().sum()+(before_write.Pu-before_decay.Pu).abs().sum())
                write_seconds += time.perf_counter()-start
                writes.append(dict(decision_id=snapshot.decision_id,context=context_index,metrics=write_values,
                                   routing=snapshot.routing.metrics[0] if hasattr(snapshot,'routing') else None))
                pending.counts['applied'] += 1
                replay.append(example)
                if (model_name=='graph' and len(diagnostics)<config['evaluation']['max_oracle_calls']
                        and torch.rand((),generator=diagnostic_rng).item()<config['evaluation']['diagnostic_snapshot_rate']):
                    start = time.perf_counter()
                    probe = task.queries(list(range(context_index+1)),config['task']['queries_per_context'],evaluation_seed)
                    no_write,_,_,_ = evaluate(model,before_write,probe,deadline=deadline)
                    written,_,_,_ = evaluate(model,state,probe,deadline=deadline)
                    # Neither evaluation branch is returned to the live learner.
                    _,_,read = predict_example(model,state.clone(detach=True),example,'drift',event)
                    utility = sum(x['squared_error']-y['squared_error'] for x,y in zip(no_write,written))/len(probe)
                    diagnostics.append(dict(decision_id=snapshot.decision_id,write_utility=utility,
                                             reachability=reachability(snapshot,before_write,example.target,config),
                                             address_drift=address_drift(snapshot,read)))
                    diagnostic_seconds += time.perf_counter()-start
                last_snapshot = snapshot
                persist()
        queries(f'warm_after_context_{context_index}',list(range(context_index+1)))
        if context_index==0:
            # Query-only distractor, independent descriptor and cue, no target to model.
            distractor = T1Lifetime(config,task_seed+700001).contexts[0][0]
            start = time.perf_counter()
            _,state,_ = predict_example(model,state,distractor,f'distractor-{event}',event,event_type='distractor')
            inference_seconds += time.perf_counter()-start
            event += 1
    queries('cold_before_sleep',[0,1],cold=True,episodes=config['evaluation']['per_task_episodes'])
    validation = []
    for episode in range(config['acceptance']['development_episodes']):
        validation.append(task.queries([0,1],config['task']['queries_per_context'],evaluation_seed+10000+episode))
    budget()
    state,sleep_report = consolidate(model,state,replay,validation,config,pending=pending,deadline=deadline)
    sleep_report.update(transaction_id=1,event_cutoff=event,
                        pending_policy='clear unresolved records only on accepted reset-boundary publication')
    sleep_reports.append(sleep_report)
    queries('cold_after_sleep',[0,1],cold=True,episodes=config['evaluation']['per_task_episodes'])
    budget()
    storage = model.storage_bytes() if hasattr(model,'storage_bytes') else dict(theta_bytes=0,key_dimension=model.key_dim,
                                                                              layout='dense_context_tensor_product_matrix')
    return dict(model=model_name,task_seed=task_seed,evaluation_seed=evaluation_seed,task_split_sha256=task.split_hash(),
                metrics=metrics,phases=phases,writes=writes,diagnostics=diagnostics,sleep=sleep_reports,
                resources=dict(inference_seconds=inference_seconds,write_seconds=write_seconds,
                               diagnostic_seconds=diagnostic_seconds,state_tensor_bytes=state.storage_bytes(),
                               model_storage=storage,snapshot_peak_bytes=snapshot_peak,
                               pending_peak_bytes=pending.peak_bytes,pending_counts=pending.counts,
                               replay_peak_bytes=replay.peak_bytes,replay_evictions=replay.evicted,
                               replay_records=len(replay.records),
                               support_routing_totals=operation_totals,
                               pending_byte_cap=pending.byte_capacity,replay_byte_cap=replay.byte_capacity),
                operator_norms=model.operator_metrics() if model_name=='graph' else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/m2_smoke.json')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_artifact(config,'experiment_config.schema.json')
    validate_m2_config(config)
    args.output.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    deadline = start+config['resources']['max_job_seconds']
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    identity = capture_code(args.output)
    run_id = args.output.name
    manifest = json.loads((RELEASE/'templates/run_manifest.json').read_text())
    manifest.update(identity,run_id=run_id,status='in_progress',milestone='M2',
                    config_sha256=sha256(args.config),train_seed=config['seed'],task_seed=config['seed']+100,
                    evaluation_seed=config['seed']+200,started_at_utc=utcnow(),
                    environment=dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
                                     device='cpu',precision='float32',threads=1,cuda_available=torch.cuda.is_available()),
                    resource_caps=config['resources'],commands=[],
                    deviations=['M2 uses fixed random theta; no shared-encoder training or advantage claim.',
                                'Tensor-product delta key is engineered, not shared with graph encoder; costs not matched.',
                                'WM has a bounded empty/no-write profile; episodic retrieval is disabled.',
                                'Post-cold supersession, T2 sweeps and M3-M6 comparisons are not run.',
                                'Replay and pending byte caps are 32768 and 16000000; reject-new pending, FIFO replay.',
                                'Warm queries are cloned probes; live support/distractor state continues without probe state.'],
                    notes=['All 47 release files are a working subset; original completeness is not claimed.',
                           'Query episodes share tables; they are not independent trained seeds.'])
    write_json(args.output/'config.json',config)
    write_json(args.output/'run_manifest.json',manifest)
    all_runs = []
    status, exit_code, error = 'completed',0,None
    try:
        graph = GraphModel(config,seed=config['seed'])
        for parameter in graph.parameters():
            parameter.requires_grad_(False)
        delta = DeltaModel(config,seed=config['seed'])
        for lifetime in range(config['training']['outer_episodes']):
            task_seed,evaluation_seed = config['seed']+100+lifetime,config['seed']+200+lifetime
            task = T1Lifetime(config,task_seed)
            for model_name,model in (('graph',graph),('delta',delta)):
                if time.perf_counter()>=deadline:
                    raise TimeoutError('smoke cap reached before next model')
                run = run_lifetime(model,model_name,config,task,task_seed,evaluation_seed,deadline,
                                   args.output/f'{model_name}_lifetime_{lifetime}.partial.json')
                all_runs.append(run)
                write_json(args.output/f'{model_name}_lifetime_{lifetime}.json',run)
                print(json.dumps(dict(model=model_name,lifetime=lifetime,
                                      cold=[m for m in run['metrics'] if '/cold_after_sleep/' in m['condition']],
                                      sleep_accepted=run['sleep'][0]['accepted'])),flush=True)
        if time.perf_counter()>=deadline:
            raise TimeoutError('smoke cap reached')
        output_bytes = sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())
        if output_bytes>config['resources']['max_output_bytes']:
            raise RuntimeError('output byte cap exceeded')
    except Exception as exc:
        status = 'aborted' if isinstance(exc,TimeoutError) else 'failed'
        exit_code = 124 if status=='aborted' else 1
        error = traceback.format_exc()
        (args.output/'failure.txt').write_text(error)
        print(error,file=sys.stderr)
    elapsed = time.perf_counter()-start
    result = dict(schema_version='0.4.1',run_id=run_id,status=status,evidence_class='implementation_smoke',
                  test_summary=dict(passed=0,failed=0,skipped=0),
                  metrics=[m for run in all_runs for m in run['metrics']],
                  resource_use=dict(wall_seconds=elapsed,completed_model_lifetimes=len(all_runs),
                                    process_memory=peak_process_memory(),
                                    stop_reason='completed' if error is None else status,
                                    per_lifetime=[run['resources'] for run in all_runs]),
                  artifacts=sorted(set([p.name for p in args.output.iterdir()]+['run_result.json'])),
                  limitations=['Fixed-theta implementation smoke; not architecture validation.',
                               'No independent trained seeds, uncertainty estimates, matched cost frontier, or M3 training.',
                               'test_summary=0 denotes no pytest invocation inside this experiment; separate test logs report tests.',
                               'Activation/Python-object peak allocation is not fully decomposed; dense storage is explicit.'],
                  claim_assessment=dict(H1='untested',H2='untested',H3='untested'))
    manifest.update(status=status,ended_at_utc=utcnow(),
                    task_split_sha256=hashlib.sha256(''.join(run['task_split_sha256'] for run in all_runs).encode()).hexdigest(),
                    commands=[dict(command=' '.join([sys.executable,'-B','-m','neuroplastic_v041.experiments.smoke',*sys.argv[1:]]),
                                   exit_code=exit_code,duration_seconds=elapsed,log_path=f'../{run_id}.log')],
                    artifact_paths=sorted(set([p.name for p in args.output.iterdir()]+['run_result.json'])))
    validate_artifact(manifest,'run_manifest.schema.json')
    validate_artifact(result,'run_result.schema.json')
    write_json(args.output/'run_manifest.json',manifest)
    write_json(args.output/'run_result.json',result)
    print(json.dumps(dict(status=status,wall_seconds=elapsed,output=str(args.output))),flush=True)
    raise SystemExit(exit_code)


if __name__=='__main__':
    main()
