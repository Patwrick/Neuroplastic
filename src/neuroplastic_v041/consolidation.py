"""Reset-boundary slow fitting and whole-candidate empirical acceptance."""
from dataclasses import replace
import time
import torch
from .tasks.t1 import validate_observation


def cold_state(state):
    state = state.clone(detach=True).reset_transient()
    return replace(state, Pa=torch.zeros_like(state.Pa), Pu=torch.zeros_like(state.Pu))


def predict_example(model, state, example, identifier, time_index, *, cold=False, event_type='query'):
    observation = example.observation(identifier, event_type)
    validate_observation(observation)
    # Only the two lawful numeric observation channels enter the encoder.
    return model.predict(state, observation['cue'], observation['context'],
                         observation['decision_id'], time_index, time_index, cold=cold)


def evaluate(model, state, examples, *, cold=False, deadline=None):
    """No feedback API or replay store is reachable from this function."""
    evaluation_state = cold_state(state) if cold else state.clone(detach=True)
    original_s = state.S.detach().clone()
    metrics, snapshots = [], []
    start = time.perf_counter()
    queues = getattr(state,'demand_queues',())
    last_time = (getattr(state,'metadata',None) or {}).get('last_prediction_time',-1)
    start_time = 0 if cold else max(last_time,max((t for q in queues for _,t in q),default=-1))+1
    with torch.no_grad():
        for index, example in enumerate(examples):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError('smoke wall-clock cap reached during evaluation')
            prediction, evaluation_state, snap = predict_example(model, evaluation_state, example,
                                                                  f'eval-{index}', start_time+index, cold=cold)
            metrics.append(dict(context=example.context_index, item=example.item_index,
                                squared_error=float((prediction-example.target).square().sum()),
                                correct=int(prediction.argmax(-1).item()==example.target.argmax(-1).item())))
            snapshots.append(snap)
            if cold and (torch.count_nonzero(evaluation_state.Pa) or torch.count_nonzero(evaluation_state.Pu)):
                raise AssertionError('cold evaluation reconstructed fast state')
    if not torch.equal(state.S, original_s) or not torch.equal(evaluation_state.S, original_s):
        raise AssertionError('evaluation changed slow storage')
    return metrics, evaluation_state, snapshots, time.perf_counter()-start


def context_loss(metrics, context):
    selected = [m['squared_error'] for m in metrics if m['context']==context]
    return sum(selected)/len(selected) if selected else None


def consolidate(model, state, replay, validation_examples, config, *, pending=None,
                force_reject=False, deadline=None):
    """Fit S through the full graph; never optimize a frozen snapshot as sleep."""
    sleep = config['sleep']
    start = time.perf_counter()
    source = state.clone(detach=True)
    cutoff_records = replay.snapshot()
    if not cutoff_records:
        raise ValueError('sleep requires admissible replay')
    train_s = source.S.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([train_s], lr=sleep['lr'])
    losses, replay_examples, replay_bytes = [], 0, 0
    for step in range(sleep['max_steps']):
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError('smoke wall-clock cap reached during sleep')
        fit = replace(cold_state(source), S=train_s)
        batch = cutoff_records[:sleep['replay_batch_size']]
        optimizer.zero_grad(set_to_none=True)
        loss = train_s.sum()*0
        for index, example in enumerate(batch):
            if deadline is not None and time.perf_counter() >= deadline:
                raise TimeoutError('smoke wall-clock cap reached during replay')
            prediction, fit, _ = predict_example(model, fit, example, f'sleep-{step}-{index}', index, cold=True)
            loss = loss + (prediction-example.target).square().sum()/len(batch)
            replay_examples += 1
            replay_bytes += sum(t.numel()*t.element_size() for t in (example.cue, example.context, example.target))
        if not torch.isfinite(loss):
            raise ValueError('nonfinite sleep loss')
        # Only S is an optimizer parameter; theta is frozen by the smoke runner.
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_s.clamp_(-config['learning']['wmax'], config['learning']['wmax'])
        losses.append(float(loss.detach()))
    fit_seconds = time.perf_counter()-start
    candidate = replace(cold_state(source), S=train_s.detach().clone())
    # A flat list denotes one episode. Explicit episode lists reset separately.
    episodes = validation_examples if validation_examples and isinstance(validation_examples[0],(list,tuple)) else [validation_examples]
    before,after = [],[]
    validation_before_seconds = validation_after_seconds = 0.
    for episode in episodes:
        rows,_,_,seconds = evaluate(model,source,episode,cold=True,deadline=deadline)
        before.extend(rows)
        validation_before_seconds += seconds
        rows,_,_,seconds = evaluate(model,candidate,episode,cold=True,deadline=deadline)
        after.extend(rows)
        validation_after_seconds += seconds
    old_before, old_after = context_loss(before, 0), context_loss(after, 0)
    new_before, new_after = context_loss(before, 1), context_loss(after, 1)
    finite = bool(torch.isfinite(candidate.S).all()) and bool((candidate.S.abs()<=config['learning']['wmax']).all())
    finite = finite and not bool(torch.count_nonzero(candidate.Pa) or torch.count_nonzero(candidate.Pu))
    old_ok = old_before is None or old_after-old_before <= config['acceptance']['max_old_loss_increase']
    new_ok = new_before is None or new_before-new_after >= config['acceptance']['min_new_improvement']
    accepted = finite and old_ok and new_ok and not force_reject and state.version==source.version
    if accepted:
        published = candidate.clone(detach=True)
        if pending is not None:
            pending.reconcile_commit()
    else:
        published = state.clone(detach=True)
    # Never mutate the live object, including on rejection.
    for field in ('S', 'Pa', 'Pu', 'h'):
        if not torch.equal(getattr(state, field), getattr(source, field)):
            raise AssertionError('sleep shadow mutated live state')
    optimizer_bytes = sum(t.numel()*t.element_size() for values in optimizer.state.values()
                          for t in values.values() if isinstance(t, torch.Tensor))
    report = dict(accepted=accepted, acceptance_level='empirical_screen',
                  loss_history=losses, optimizer_steps=len(losses), replay_examples=replay_examples,
                  replay_tensor_bytes_read=replay_bytes, optimizer_tensor_bytes=optimizer_bytes,
                  gradient_bytes=train_s.numel()*train_s.element_size(),
                  candidate_bytes=candidate.storage_bytes(),
                  old_before=old_before, old_after=old_after, new_before=new_before, new_after=new_after,
                  fit_seconds=fit_seconds, validation_seconds=validation_before_seconds+validation_after_seconds,
                  validation_examples=len(before)+len(after),validation_episodes=2*len(episodes),
                  total_seconds=time.perf_counter()-start,
                  conditional_guarantee='none; one fixed candidate, development queries only')
    return published, report
