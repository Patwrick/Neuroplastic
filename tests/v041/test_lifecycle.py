"""M2 causal boundaries, full-graph sleep, and immutable cold controls."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch

from neuroplastic_v041.graph import GraphModel
from neuroplastic_v041.baselines.delta import DeltaModel
from neuroplastic_v041.tasks.t1 import T1Lifetime, validate_observation
from neuroplastic_v041.feedback import PendingFeedback, graph_write
from neuroplastic_v041.memory import ReplayStore
from neuroplastic_v041.consolidation import evaluate, predict_example, consolidate
from neuroplastic_v041.diagnostics import reachability


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def config():
    torch.set_num_threads(1)
    return json.loads((ROOT/'Docs/research/csgn_v0_4_1/configs/cpu_smoke.json').read_text())


@pytest.fixture
def fixture(config):
    model = GraphModel(config,seed=7)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    task = T1Lifetime(config,100)
    example = task.contexts[0][0]
    prediction,state,snapshot = predict_example(model,model.initial_state(),example,'one',0,event_type='support')
    return model,task,example,prediction,state,snapshot


@pytest.mark.parametrize('forbidden',['target','seed','info','hidden_map','future_reward'])
def test_target_leakage_red_guard(fixture,forbidden):
    observation = fixture[2].observation('one')
    observation[forbidden] = fixture[2].target
    with pytest.raises(ValueError,match='allowlist'):
        validate_observation(observation)


def test_task_reproducibility_and_observation_does_not_alias_target(config):
    a,b = T1Lifetime(config,100),T1Lifetime(config,100)
    assert a.split_hash()==b.split_hash()
    assert a.split_hash()!=T1Lifetime(config,101).split_hash()
    example = a.contexts[0][0]
    observation = example.observation('id')
    assert set(observation)=={'cue','context','event_type','decision_id'}
    observation['cue'].zero_()
    assert torch.count_nonzero(example.cue)>0


def test_pending_causal_reorder_duplicate_generation_and_expiry(fixture):
    model,_,_,_,state,snapshot = fixture
    pending = PendingFeedback(capacity=2,max_delay=4)
    delayed = replace(snapshot,feedback_available_time=3)
    assert pending.add(delayed)
    with pytest.raises(ValueError,match='duplicate'):
        pending.add(delayed)
    with pytest.raises(ValueError,match='before available'):
        pending.take('one',1,state.version,model.topology.generation)
    second = replace(snapshot,decision_id='two',feedback_available_time=1)
    assert pending.add(second)
    assert pending.take('two',2,state.version,model.topology.generation).decision_id=='two'
    assert pending.take('one',3,state.version,model.topology.generation).decision_id=='one'
    with pytest.raises(ValueError,match='consumed'):
        pending.take('one',3,state.version)
    pending.add(snapshot)
    assert pending.take('one',1,state.version,model.topology.generation+1) is None
    assert pending.counts['version_rejected']==1
    pending.add(snapshot)
    assert pending.take('one',5,state.version,model.topology.generation) is None
    assert pending.counts['expired']==1


def test_pending_overflow_byte_budget_version_and_commit(fixture):
    model,_,_,_,state,snapshot = fixture
    pending = PendingFeedback(capacity=1,byte_capacity=snapshot.storage_bytes())
    assert pending.add(snapshot)
    assert not pending.add(replace(snapshot,decision_id='two'))
    pending.reconcile_commit()
    assert not pending.records and pending.bytes==0 and pending.counts['invalidated_at_sleep']==1
    pending.add(snapshot)
    assert pending.take('one',1,replace(state.version,feature=1),model.topology.generation) is None


def test_delay_uses_captured_features_and_two_tier_post_decay(config,fixture):
    model,task,example,_,state,snapshot = fixture
    state = replace(state,Pu=torch.full_like(state.Pu,.001))
    _,state,_ = predict_example(model,state,task.contexts[1][0],'later',1)
    captured = snapshot.A.clone()
    written,metrics = graph_write(state,snapshot,example.target,config,elapsed=4)
    torch.testing.assert_close(snapshot.A,captured,rtol=0,atol=0)
    assert metrics.loss_after <= metrics.loss_before+1e-6
    assert metrics.l1_write <= config['learning']['total_write_budget']+1e-6
    assert torch.equal(written.Pa[~snapshot.writable_mask],state.Pa[~snapshot.writable_mask])
    assert (written.Pu<state.Pu).all()
    diagnostic = reachability(snapshot,state,example.target,config)
    assert diagnostic['numerical_bound_check']
    assert diagnostic['loss_lower']<=diagnostic['loss_upper']+1e-8


@pytest.mark.parametrize('model_name',['graph','delta'])
def test_sleep_shadow_rejection_and_entire_cold_rollout(config,model_name):
    model = GraphModel(config,seed=7) if model_name=='graph' else DeltaModel(config)
    if model_name=='graph':
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    task = T1Lifetime(config,100)
    state = model.initial_state()
    replay = ReplayStore()
    for i,example in enumerate(task.contexts[0]+task.contexts[1]):
        _,state,snapshot = predict_example(model,state,example,str(i),i,event_type='support')
        if model_name=='graph':
            state,_ = graph_write(state,snapshot,example.target,config)
        else:
            state,_ = model.write(state,snapshot,example.target)
        replay.append(example)
    saved = state.clone(detach=True)
    examples = task.queries([0,1],4,500)
    pending = PendingFeedback()
    pending.add(snapshot)
    rejected,report = consolidate(model,state,replay,examples,config,pending=pending,force_reject=True)
    assert not report['accepted']
    assert report['optimizer_steps']==4 and report['replay_examples']==32
    assert len(pending.records)==1  # rejection preserves unresolved live records
    for field in ('S','Pa','Pu','h'):
        assert torch.equal(getattr(state,field),getattr(saved,field))
        assert torch.equal(getattr(rejected,field),getattr(saved,field))
    result,report = consolidate(model,state,replay,examples,config,pending=pending)
    assert isinstance(report['accepted'],bool)
    if report['accepted']:
        assert not pending.records
        assert torch.count_nonzero(result.Pa)==torch.count_nonzero(result.Pu)==0
    # Poisoned transient/fast values cannot become an answer channel in cold mode.
    poisoned = replace(result,Pa=torch.full_like(result.Pa,.2),Pu=torch.full_like(result.Pu,.1),
                       h=torch.full_like(result.h,.5))
    first,end,_,_ = evaluate(model,poisoned,examples,cold=True)
    second,_,_,_ = evaluate(model,result,examples,cold=True)
    assert first==second
    assert torch.equal(end.S,result.S)
    assert torch.count_nonzero(end.Pa)==torch.count_nonzero(end.Pu)==0
    with pytest.raises(RuntimeError,match='disabled'):
        replay.retrieve_answer(examples[0])


def test_replay_record_and_byte_caps(config):
    task = T1Lifetime(config,100)
    replay = ReplayStore(capacity=2,byte_capacity=320)
    for item in task.contexts[0]:
        replay.append(item)
    assert len(replay.records)==2 and replay.bytes<=320 and replay.evicted==2
    assert replay.snapshot()[0].cue.data_ptr()!=replay.records[0].cue.data_ptr()
