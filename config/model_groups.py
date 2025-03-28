"""Main configuration file for model-specific attention head groups."""

from .gpt2_groups import GPT2_GROUPS
from .pythia_groups import PYTHIA_GROUPS

MODEL_SPECIFIC_GROUPS = {
    "gpt2-small": GPT2_GROUPS,
    "pythia-2.8b": PYTHIA_GROUPS
} 