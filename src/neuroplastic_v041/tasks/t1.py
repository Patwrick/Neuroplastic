"""Causal, locally generated T1 tables. Only observation() crosses into a model."""
from dataclasses import dataclass
import hashlib
import torch


ALLOWED_OBSERVATION_KEYS = frozenset({'cue', 'context', 'event_type', 'decision_id'})


def validate_observation(observation):
    if set(observation) != ALLOWED_OBSERVATION_KEYS:
        raise ValueError('observation leakage allowlist violation')
    if observation['event_type'] not in {'support', 'query', 'distractor'}:
        raise ValueError('invalid event type')
    for key in ('cue', 'context'):
        value = observation[key]
        if value.ndim != 2 or not torch.isfinite(value).all() or (value.abs() > 1).any():
            raise ValueError('invalid bounded observable')


@dataclass(frozen=True)
class ScoredExample:
    cue: torch.Tensor
    context: torch.Tensor
    target: torch.Tensor
    context_index: int
    item_index: int

    def observation(self, decision_id, event_type='query'):
        result = dict(cue=self.cue.clone(), context=self.context.clone(),
                      event_type=event_type, decision_id=str(decision_id))
        validate_observation(result)
        return result

    def clone(self):
        return ScoredExample(self.cue.clone(), self.context.clone(), self.target.clone(),
                             self.context_index, self.item_index)


class T1Lifetime:
    """Environment/scorer owns targets; model receives independently drawn descriptors."""
    def __init__(self, config, task_seed, dtype=torch.float32):
        cfg = config['task']
        # Independent streams prevent descriptors depending on label sampling order.
        features = torch.Generator().manual_seed(task_seed)
        labels = torch.Generator().manual_seed(task_seed + 1000003)
        self.contexts = []
        for c in range(cfg['contexts_per_lifetime']):
            context = torch.randn(1, cfg['context_dim'], generator=features, dtype=dtype)
            context = context / context.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cues = torch.randn(cfg['cues_per_context'], cfg['cue_dim'], generator=features, dtype=dtype)
            cues = cues / cues.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            table = torch.randint(cfg['classes'], (cfg['cues_per_context'],), generator=labels)
            codes = torch.eye(cfg['classes'], dtype=dtype)[table] * cfg['target_scale']
            self.contexts.append([ScoredExample(cues[i:i+1].clone(), context.clone(),
                                               codes[i:i+1].clone(), c, i)
                                  for i in range(cfg['cues_per_context'])])

    def queries(self, context_indices, count_per_context, evaluation_seed):
        rng = torch.Generator().manual_seed(evaluation_seed)
        result = []
        for context in context_indices:
            indices = torch.randint(len(self.contexts[context]), (count_per_context,), generator=rng)
            result.extend(self.contexts[context][int(i)].clone() for i in indices)
        return result

    def split_hash(self):
        digest = hashlib.sha256()
        for context in self.contexts:
            for example in context:
                for value in (example.cue, example.context, example.target):
                    digest.update(value.numpy().tobytes())
        return digest.hexdigest()
