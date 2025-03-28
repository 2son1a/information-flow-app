"""
Default configurations for different models.
Includes default text, model parameters, and other settings.
"""

MODEL_DEFAULTS = {
    "gpt2-small": {
        "default_text": "The quick brown fox jumps over the lazy dog.",
        "num_layers": 12,
        "num_heads": 12,
        "hidden_size": 768,
        "max_length": 1024,
        "description": "GPT-2 Small model with 117M parameters"
    },
    "pythia-2.8b": {
        "default_text": "In the beginning, there was light.",
        "num_layers": 32,
        "num_heads": 32,
        "hidden_size": 2560,
        "max_length": 2048,
        "description": "Pythia 2.8B model"
    }
} 