"""
Visualization styles and layouts.
These are code-related configurations that don't need to be user-configurable.
"""

# Color palette for attention head groups
COLOR_PALETTE = [
    '#3B82F6',  # blue-500
    '#10B981',  # emerald-500
    '#F59E0B',  # amber-500
    '#EF4444',  # red-500
    '#8B5CF6',  # violet-500
    '#EC4899',  # pink-500
    '#14B8A6',  # teal-500
    '#F97316',  # orange-500
    '#6366F1',  # indigo-500
    '#D946EF',  # fuchsia-500
]

# CSS styles for the application
CSS_STYLES = """
    .main {
        padding: 0rem 1rem;
    }
    .stTitle {
        font-size: 2rem !important;
        font-weight: 500 !important;
    }
    .stSelectbox > div > div {
        background-color: white;
        border: 1px solid #e5e7eb;
    }
    .head-button {
        display: inline-block;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.8rem;
        color: white;
    }
    .group-container {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }
    .group-description {
        color: #6b7280;
        font-size: 0.875rem;
        margin: 0.5rem 0;
    }
    .graph-container {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
"""

# Layout constants
LAYOUT_CONSTANTS = {
    "column_ratio": [1, 3],  # For main content columns
    "control_column_ratio": [1, 1],  # For control section columns
    "text_area_height": 100,
    "slider_step": 0.01,
    "default_threshold": 0.4,
} 