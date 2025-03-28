"""
Main configuration package.
Exposes a global config loader that combines YAML user settings with Python code configs.
"""

from .loader import config

__all__ = ['config'] 