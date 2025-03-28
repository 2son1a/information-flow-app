"""
Configuration for model-specific attention head groups.
Each model can have predefined groups of attention heads that work together.
"""

MODEL_SPECIFIC_GROUPS = {
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