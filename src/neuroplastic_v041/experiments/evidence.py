"""Local identity, exact dirty patch, and schema-checked run artifacts."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import zipfile
import jsonschema


ROOT = Path(__file__).resolve().parents[3]
RELEASE = ROOT / 'Docs/research/csgn_v0_4_1'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT)


def capture_code(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_relative_to(ROOT.resolve()):
        relative_output = str(directory.relative_to(ROOT.resolve()))
        ignored = subprocess.run(['git', 'check-ignore', '--quiet', '--', relative_output], cwd=ROOT)
        if ignored.returncode != 0:
            raise ValueError('Evidence output inside the repository must be git-ignored')
    def tracked_and_untracked():
        return sorted(set(filter(None, git('ls-files','--cached','--others','--exclude-standard','-z').decode().split('\0'))))
    files = tracked_and_untracked()
    before = {relative: sha256(ROOT/relative) for relative in files if (ROOT/relative).is_file()}
    patch = git('diff','HEAD','--binary')
    untracked = git('ls-files','--others','--exclude-standard','-z').decode().split('\0')
    for relative in filter(None,untracked):
        result = subprocess.run(['git','diff','--no-index','--binary','--','/dev/null',relative],
                                cwd=ROOT,capture_output=True)
        if result.returncode not in (0,1):
            raise RuntimeError(result.stderr.decode(errors='replace'))
        patch += result.stdout
    (directory/'dirty.patch').write_bytes(patch)
    source_files = [p for p in files if p.startswith(('src/','tests/','scripts/','configs/')) or
                    p in {'pyproject.toml','README.md','AGENTS.md'}]
    inventory = {}
    with zipfile.ZipFile(directory/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for relative in source_files:
            path = ROOT/relative
            if path.is_file():
                contents = path.read_bytes()
                inventory[relative] = dict(sha256=hashlib.sha256(contents).hexdigest(),bytes=len(contents))
                archive.writestr(relative,contents)
    after_files = tracked_and_untracked()
    after = {relative: sha256(ROOT/relative) for relative in after_files if (ROOT/relative).is_file()}
    if files != after_files or before != after:
        raise RuntimeError('Code changed while capturing provenance; retry after edits stop')
    write_json(directory/'source_manifest.json',inventory)
    return dict(code_commit=git('rev-parse','HEAD').decode().strip(),
                git_dirty=bool(git('status','--porcelain').strip()),
                dirty_diff_sha256=hashlib.sha256(patch).hexdigest(),
                package_manifest_sha256=sha256(RELEASE/'MANIFEST.sha256.json'),
                paper_sha256=sha256(RELEASE/'paper/csgn_paper_v0_4_1.md'))


def validate_artifact(value, schema):
    jsonschema.validate(value,json.loads((RELEASE/'schemas'/schema).read_text()))


def process_peak_working_set_bytes():
    """Windows process high-water mark; unavailable platforms return None."""
    if os.name != 'nt':
        return None
    try:
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [
                (name,ctypes.c_size_t) for name in (
                    'PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage',
                    'PagefileUsage','PeakPagefileUsage')]
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        psapi = ctypes.WinDLL('psapi',use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
            return None
        return int(counters.PeakWorkingSetSize)
    except (AttributeError,OSError):
        return None


def peak_process_memory():
    return dict(peak_working_set_bytes=process_peak_working_set_bytes(),
                scope='Process-lifetime peak working set including interpreter/native libraries; not tensor-only or exclusive run memory.')


def validate_m2_config(config):
    """Reject unsupported profiles instead of silently ignoring their settings."""
    def finite(value):
        if isinstance(value,dict):
            for item in value.values():
                finite(item)
        elif isinstance(value,list):
            for item in value:
                finite(item)
        elif isinstance(value,float) and not math.isfinite(value):
            raise ValueError('M2 config numbers must be finite')
    finite(config)
    validate_artifact(config,'experiment_config.schema.json')
    def require(condition,message):
        if not condition:
            raise ValueError(message)
    def cap(value,maximum,name,minimum=1):
        require(minimum <= value <= maximum,f'M2 {name} must be in [{minimum}, {maximum}]')
    require(config['stage']=='M2' and config['evidence_class']=='implementation_smoke', 'M2 implementation_smoke only')
    require(config['model']=='graph_sleep' and config['task_id']=='T1','M2 runner compares graph_sleep and delta on T1 only')
    task,training,sleep,resources = (config[name] for name in ('task','training','sleep','resources'))
    require(tuple(task[name] for name in ('cue_dim','context_dim','classes','target_dim'))==(16,8,8,8),'fixed T1 public dimensions required')
    require(task['contexts_per_lifetime']==2,'M2 requires two old/new contexts')
    require(task['feedback_delays']==[0] and not task['query_feedback'] and not task['episode_answer_retrieval'], 'M2 uses immediate feedback, query writes off and retrieval off')
    require(0 < task['target_scale'] <= 1,'bounded positive T1 target scale required')
    for field,maximum in (('cues_per_context',4),('support_passes',1),('queries_per_context',8)):
        cap(task[field],maximum,field)
    require(not training['train_shared_theta'] and training['batch_lifetimes']==1,'fixed theta and one independent lifetime per call required')
    cap(training['outer_episodes'],2,'outer_episodes')
    cap(training['max_outer_steps'],2,'max_outer_steps',0)
    require(resources['device']=='cpu' and resources['precision']=='float32','M2 requires CPU float32')
    require(not resources['allow_paid_compute'] and not resources['allow_external_downloads'],'M2 jobs do not use paid compute or downloads')
    for field,maximum in (('max_job_seconds',180),('max_output_bytes',100000000),('max_pending_records',32),('max_replay_records',128),('max_hot_edge_identities',512)):
        cap(resources[field],maximum,field)
    require(sleep['enabled'] and sleep['mode']=='synchronous_reset_boundary' and sleep['shared_theta_frozen'] and sleep['fast_disabled_during_fit'], 'synchronous slow-only frozen-theta sleep required')
    require(sleep['optimizer']=='Adam' and sleep['candidate_quantization']=='none','M2 uses Adam sleep without quantization')
    cap(sleep['max_steps'],4,'sleep.max_steps')
    cap(sleep['replay_batch_size'],8,'sleep.replay_batch_size')
    require(sleep['lr']>0,'positive sleep rate required')
    evaluation,acceptance = config['evaluation'],config['acceptance']
    require(all(evaluation[name] for name in ('cold_disable_residuals','cold_disable_writes','cold_reset_workspace_wm','cold_disable_answer_retrieval')),
            'M2 cold protocol requires all fast, write and retrieval channels disabled')
    cap(evaluation['per_task_episodes'],8,'evaluation.per_task_episodes')
    cap(evaluation['max_oracle_calls'],8,'max_oracle_calls',0)
    require(0<=evaluation['diagnostic_snapshot_rate']<=1,'diagnostic rate must be in [0,1]')
    require(acceptance['level']=='empirical_screen' and acceptance['formal_delta_lifetime'] is None,'M2 has no formal acceptance guarantee')
    cap(acceptance['development_episodes'],8,'development_episodes')
    require(acceptance['max_old_loss_increase']>=0 and acceptance['min_new_improvement']>=0,'nonnegative acceptance thresholds required')
    graph = config['graph']
    for field,maximum,minimum in (('columns',64,16),('regions',8,1),('out_slots',8,4),('state_dim',32,1),('message_dim',16,8),('heads',2,1),('descriptor_dim',8,1),('active_senders',16,1),('real_routes_per_sender',4,1),('incoming_capacity',32,1),('inner_rounds',4,1)):
        cap(graph[field],maximum,'graph.'+field,minimum)
    require(graph['columns']%graph['regions']==0 and graph['out_slots']<graph['columns'],'equal regions and unique nonself slots required')
    require(graph['protected_slots']==4 and graph['output_receivers']==[0,1,2,3],'four protected slots and fixed output decoder required')
    require(graph['support_policy']=='protected_output_slot_plus_topk' and graph['address_mode']=='state_dependent','unsupported routing variant')
    cap(graph['active_regions'],graph['regions'],'active_regions')
    cap(graph['updated_node_cap'],graph['columns'],'updated_node_cap',4)
    require(graph['real_routes_per_sender']<=graph['out_slots'],'route count exceeds reserved slots')
    require(graph['columns']*graph['out_slots']<=resources['max_hot_edge_identities'],'dense residual allocation exceeds configured hot identity cap')
    learning=config['learning']
    require(learning['feasibility']=='four_endpoint' and learning['gradient_mode']=='continuous_full_path_conditional_support' and learning['projection']=='exact_intersection_box' and learning['hot_state_layout']=='dense_correctness_reference','unsupported M2 write profile')
    require(learning['admissible_decay']==1,'M2 shared retention comparison requires admissible decay 1')
    require(learning['per_edge_allowance']>=0 and learning['total_write_budget']>=0,'nonnegative write allowances required')
    require({'rewiring','INT4','offloading','asynchronous_sleep','online_theta_updates','learned_internal_teacher','Minecraft'}<=set(config['disabled_extensions']),'M2 extensions must remain disabled')
