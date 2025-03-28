"""
Model specifications and architecture details.
These are code-related configurations that don't need to be user-configurable.
"""

from typing import Dict, Any

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "gpt2-small": {
        "num_layers": 12,
        "num_heads": 12,
        "hidden_size": 768,
        "vocab_size": 50257,
        "max_position_embeddings": 1024,
    },
    "pythia-2.8b": {
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 2560,
        "vocab_size": 50304,
        "max_position_embeddings": 2048,
    }
} 