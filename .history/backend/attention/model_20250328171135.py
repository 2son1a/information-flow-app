import torch
from transformer_lens import HookedTransformer
from transformer_lens.head_detector import detect_head 
from typing import Dict, List, Tuple, Any
import numpy as np

# Available models
AVAILABLE_MODELS = {
    "gpt2-small": "gpt2-small",
    "pythia-2.8b": "pythia-2.8b"
}

class AttentionPatternExtractor:
    def __init__(self, model_name: str = "gpt2-small"):
        """Initialize the model for attention pattern extraction.
        
        Args:
            model_name: Name of the pretrained model to use
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available models: {', '.join(AVAILABLE_MODELS.keys())}")
            
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading model {model_name} on {self.device}...")
        self.model = HookedTransformer.from_pretrained(
            AVAILABLE_MODELS[model_name],
            device=self.device,
            dtype=torch.float32
        )
        self.model.eval()
        
        # Cache model configuration
        self.n_layers = self.model.cfg.n_layers
        self.n_heads = self.model.cfg.n_heads
        
        print(f"Model loaded: {model_name} with {self.n_layers} layers and {self.n_heads} heads")
        
    def get_attention_patterns(self, text: str) -> Dict[str, Any]:
        """Extract attention patterns from input text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing:
                - tokens: List of tokens
                - layerAttentions: List of attention patterns per layer
        """
        # Tokenize input text
        tokens_list = self.model.to_str_tokens(text)
        
        # Store attention patterns
        patterns = []
        
        def save_pattern(activation, hook):
            # Remove batch dimension and store
            pattern = activation.detach().squeeze(0).cpu().numpy()
            patterns.append(pattern[:,1:,1:])
        
        # Run model with hooks to capture attention patterns
        pattern_filter = lambda name: "hook_pattern" in name
        self.model.run_with_hooks(
            text,
            return_type=None,
            fwd_hooks=[(pattern_filter, save_pattern)]
        )
        
        
        return {
            "tokens": tokens_list, 
            "layerAttentions": layer_attentions,
            "headTypes": head_types
        }

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process text and return data in the format expected by the frontend.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary containing attention patterns and metadata
        """
        # Get attention patterns and tokens
        result = self.get_attention_patterns(text)
        tokens = result["tokens"]
        layer_attentions = result["layerAttentions"]
        head_types = result["headTypes"]
        
        # Transform data into the format expected by the frontend
        attention_patterns = []
        
        # For each layer
        for layer_data in layer_attentions:
            layer = layer_data["layer"]
            # For each head
            for head in range(self.n_heads):
                head_pattern = layer_data["heads"][head]  # This is now a 2D matrix
                # For each token pair
                for src_idx in range(len(tokens)):
                    for dest_idx in range(len(tokens)):
                        # Get attention weight directly from the pattern matrix
                        weight = float(head_pattern[dest_idx, src_idx])  # Note: matrix is [dest, src]
                        attention_patterns.append({
                            "sourceLayer": layer,
                            "sourceToken": src_idx,
                            "destLayer": layer + 1,  # Next layer
                            "destToken": dest_idx,
                            "weight": weight,
                            "head": head,
                            "headType": head_types.get((layer, head), "unknown")
                        })
        
        return {
            "numLayers": self.n_layers + 1,  # +1 because we show source and destination layers
            "numTokens": len(tokens),
            "numHeads": self.n_heads,
            "tokens": tokens[1:],
            "attentionPatterns": attention_patterns,
            "model_name": self.model_name,
            "model_info": {
                "name": self.model_name,
                "layers": self.n_layers,
                "heads": self.n_heads,
            }
        }

# Example usage:
if __name__ == "__main__":
    extractor = AttentionPatternExtractor()
    text = "The quick brown fox jumped over the lazy dog"
    patterns = extractor.process_text(text)
    print(f"Number of tokens: {patterns['numTokens']}")
    print(f"Number of attention patterns: {len(patterns['attentionPatterns'])}")
    print(f"Tokens: {patterns['tokens']}")
    print("\nSample attention patterns for first head:")
    # Print first few patterns for head 0 to verify weights
    head_0_patterns = [p for p in patterns["attentionPatterns"] if p["head"] == 0][:5]
    for p in head_0_patterns:
        print(f"Source token: {patterns['tokens'][p['sourceToken']]} -> Dest token: {patterns['tokens'][p['destToken']]}, Weight: {p['weight']:.4f}")
