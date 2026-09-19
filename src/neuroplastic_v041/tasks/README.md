# Causal T1

The environment/scorer owns random class tables and target codes. Its observation
allowlist exposes only bounded cue/context, event type and decision ID. Feature
and target RNG streams are separate; the model never receives generator seeds.
