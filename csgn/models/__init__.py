"""Model definitions for CSGN baselines."""

from csgn.models.plastic_mlp import PlasticLinear, PlasticMLP
from csgn.models.tiny_bc import TinyBC

__all__ = ["TinyBC", "PlasticLinear", "PlasticMLP"]
