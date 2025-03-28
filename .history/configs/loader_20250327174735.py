"""
Configuration loader that combines YAML user settings with Python code configs.
"""

import os
from typing import Dict, Any
import yaml
from .models.specs import MODEL_SPECS, MODEL_HEAD_GROUPS
from .visualization.styles import COLOR_PALETTE, CSS_STYLES, LAYOUT_CONSTANTS

class ConfigLoader:
    def __init__(self, config_path: str = "configs/user_settings.yaml"):
        self.config_path = config_path
        self.user_settings = self._load_yaml()
        self.env_vars = self._load_env()
        
    def _load_yaml(self) -> Dict[str, Any]:
        """Load user settings from YAML file."""
        try:
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: {self.config_path} not found. Using default settings.")
            return {}
            
    def _load_env(self) -> Dict[str, str]:
        """Load environment variables."""
        return {
            "api_key": os.getenv("API_KEY"),
            "api_base_url": os.getenv("API_BASE_URL", self.user_settings.get("api", {}).get("base_url", "http://localhost:8000"))
        }
        
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get complete model configuration combining specs and user settings."""
        specs = MODEL_SPECS.get(model_name, {})
        user_settings = self.user_settings.get("models", {}).get(model_name, {})
        return {**specs, **user_settings}
        
    def get_model_head_groups(self, model_name: str) -> list:
        """Get predefined head groups for a model."""
        return MODEL_HEAD_GROUPS.get(model_name, [])
        
    def get_visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration combining styles and user settings."""
        user_viz = self.user_settings.get("visualization", {})
        return {
            "colors": COLOR_PALETTE,
            "css": CSS_STYLES,
            "layout": LAYOUT_CONSTANTS,
            "graph": user_viz.get("graph", {}),
            "theme": user_viz.get("theme", {})
        }
        
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration combining defaults and user settings."""
        user_api = self.user_settings.get("api", {})
        return {
            "base_url": self.env_vars["api_base_url"],
            "timeout": user_api.get("timeout", 30),
            "health_check_timeout": user_api.get("health_check_timeout", 10)
        }

# Create a global config loader instance
config = ConfigLoader() 