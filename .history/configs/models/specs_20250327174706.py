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

# Predefined attention head groups for each model
MODEL_HEAD_GROUPS = {
    "gpt2-small": [
        {
            "name": "Position Embedding Heads",
            "description": "Heads that primarily attend to positional information",
            "vertices": [
                (0, 0), (0, 1),  # Layer 0
                (1, 2), (1, 3),  # Layer 1
                (2, 4), (2, 5),  # Layer 2
            ]
        },
        {
            "name": "Copy Heads",
            "description": "Heads that copy information from previous tokens",
            "vertices": [
                (3, 6), (3, 7),  # Layer 3
                (4, 8), (4, 9),  # Layer 4
            ]
        }
    ],
    "pythia-2.8b": [
        {
            "name": "Position Embedding Heads",
            "description": "Heads that primarily attend to positional information",
            "vertices": [
                (0, 0), (0, 1),  # Layer 0
                (1, 2), (1, 3),  # Layer 1
            ]
        }
    ]
} 