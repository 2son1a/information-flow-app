"""
UI layout configurations and styling.
"""

# Streamlit page configuration
PAGE_CONFIG = {
    "layout": "wide",
    "page_title": "Information Flow Visualization",
    "initial_sidebar_state": "expanded",
}

# CSS styles
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