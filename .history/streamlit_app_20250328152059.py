import streamlit as st
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import requests
from dataclasses import dataclass
from datetime import datetime
import logging
import io
import os
import plotly.graph_objects as go
from config.model_groups import MODEL_SPECIFIC_GROUPS
from config.model_defaults import MODEL_DEFAULTS

# Set up logging with a StringIO buffer to capture logs
log_buffer = io.StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=log_buffer
)
logger = logging.getLogger(__name__)

# Constants
COLOR_PALETTE = [
    "#4ECDC4",  # Turquoise
    "#FFD166",  # Warm yellow
    "#7EDC11",  # Bright lime
    "#FF1493",  # Deep pink
    "#1A9CE0",  # Azure Blue
    "#FF8C42",  # Bright orange
    "#06D6A0",  # Bright turquoise
    "#FFBB33",  # Amber
    "#4B0082",  # Indigo
    "#A7E541",  # Lime green
    "#FF5C5C",  # Coral Red
    "#66D7EE",  # Sky Blue
    "#FFE066",  # Yellow Gold
    "#233FD2",  # Royal Blue
    "#74E39A",  # Mint Green
    "#FF3377",  # Hot Pink
    "#5BC0EB",  # Light blue
    "#FFA07A",  # Light salmon orange
    "#118AB2",  # Blue
    "#C1FF72",  # Lime Green
    "#D90368",  # Magenta
    "#00AA5B",  # Emerald Green
    "#FF6B35",  # Deep orange
    "#F8E16C",  # Light yellow
    "#9F0162",  # Deep Magenta
    "#1EAE98",  # Teal green
    "#FF3366",  # Coral pink
    "#731DD8",  # Electric Purple
]

def get_random_color() -> str:
    """Get a random color from the palette."""
    return COLOR_PALETTE[np.random.randint(0, len(COLOR_PALETTE))]

@dataclass
class AttentionPattern:
    sourceLayer: int
    sourceToken: int
    destLayer: int
    destToken: int
    weight: float
    head: int
    headType: Optional[str] = None

@dataclass
class HeadPair:
    layer: int
    head: int
    color: Optional[str] = None  # Add color field

@dataclass
class HeadGroup:
    id: int
    name: str
    heads: List[HeadPair]
    description: Optional[str] = None
    color: Optional[str] = None  # Add color field

def get_head_color(layer: int, head: int, head_groups: List[HeadGroup]) -> str:
    """Get the color for a head based on its group membership."""
    # Check which group this head belongs to
    for group in head_groups:
        if any(h.layer == layer and h.head == head for h in group.heads):
            # Use group's custom color if set, otherwise use default from palette
            return group.color or COLOR_PALETTE[group.id % len(COLOR_PALETTE)]
    
    # Default blue for heads not in any group
    return '#3B82F6'

class APIService:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def check_backend_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            logger.warning(f"Could not connect to backend at {self.base_url}. Make sure the backend server is running.")
            return False
        except requests.exceptions.Timeout:
            logger.warning("Backend request timed out. The server might be overloaded or not responding.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking backend health: {str(e)}")
            return False

    def process_text(self, text: str, model: str) -> Dict:
        response = requests.post(
            f"{self.base_url}/process_text",
            json={"text": text, "model": model},
            timeout=30
        )
        if response.status_code != 200:
            raise Exception(f"API error: {response.text}")
        return response.json()

def get_default_text_for_model(model: str) -> str:
    """Get the default text for a given model from configuration."""
    return MODEL_DEFAULTS.get(model, {}).get("default_text", "")

def create_attention_graph(
    data: Dict,
    threshold: float,
    selected_heads: List[HeadPair],
    head_groups: List[HeadGroup]
) -> None:
    """Create graph visualization using Plotly."""
    logger.info(f"Creating graph with data: numLayers={data['numLayers']}, numTokens={data['numTokens']}, numPatterns={len(data['attentionPatterns'])}")
    logger.info(f"Selected heads: {[(h.layer, h.head) for h in selected_heads]}")
    logger.info(f"Head groups: {[(g.name, len(g.heads)) for g in head_groups]}")
    
    # Filter attention patterns based on threshold and selected heads
    visible_heads = selected_heads + [head for group in head_groups for head in group.heads]
    filtered_patterns = [
        pattern for pattern in data['attentionPatterns']
        if pattern['weight'] >= threshold and any(
            h.layer == pattern['sourceLayer'] and h.head == pattern['head'] 
            for h in visible_heads
        )
    ]

    # Create the base figure with grid points
    fig = go.Figure()

    # Add grid points for each layer and token
    for layer in range(data['numLayers']):
        for token in range(data['numTokens']):
            fig.add_trace(go.Scatter(
                x=[token],
                y=[layer],
                mode='markers',
                marker=dict(
                    size=8,
                    color='lightgrey',
                ),
                hoverinfo='none',
                showlegend=False
            ))

    # Create a mapping of (layer, head) to group name
    head_to_group = {}
    for group in head_groups:
        for head in group.heads:
            head_to_group[(head.layer, head.head)] = group.name

    # First add all group patterns
    for group in head_groups:
        group_patterns = [
            pattern for pattern in filtered_patterns
            if (pattern['sourceLayer'], pattern['head']) in head_to_group
            and head_to_group[(pattern['sourceLayer'], pattern['head'])] == group.name
        ]
        
        if group_patterns:
            # Add a dummy trace for the group legend entry
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(
                    color=group.color or COLOR_PALETTE[group.id % len(COLOR_PALETTE)],
                    width=3
                ),
                name=group.name,
                showlegend=True
            ))
            
            # Add actual patterns for this group
            for pattern in group_patterns:
                x0, y0 = pattern['sourceToken'], pattern['sourceLayer']
                x1, y1 = pattern['destToken'], pattern['destLayer']
                
                control_x = (x0 + x1) / 2
                control_y = max(y0, y1) + 0.5
                
                t = np.linspace(0, 1, 20)
                x = (1-t)**2 * x0 + 2*(1-t)*t * control_x + t**2 * x1
                y = (1-t)**2 * y0 + 2*(1-t)*t * control_y + t**2 * y1
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    line=dict(
                        color=group.color or COLOR_PALETTE[group.id % len(COLOR_PALETTE)],
                        width=3
                    ),
                    hovertemplate=f'Layer: {pattern["sourceLayer"]}<br>Head: {pattern["head"]}<br>Weight: {pattern["weight"]:.3f}',
                    name=group.name,
                    showlegend=False
                ))

    # Then add individual head patterns, treating each head as its own group
    for head in selected_heads:
        head_patterns = [
            pattern for pattern in filtered_patterns
            if pattern['sourceLayer'] == head.layer and pattern['head'] == head.head
        ]
        
        if head_patterns:
            # Add a dummy trace for the head legend entry
            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(
                    color=head.color or '#3B82F6',
                    width=3
                ),
                name=f"Head {head.layer},{head.head}",
                showlegend=True
            ))
            
            # Add actual patterns for this head
            for pattern in head_patterns:
                x0, y0 = pattern['sourceToken'], pattern['sourceLayer']
                x1, y1 = pattern['destToken'], pattern['destLayer']
                
                control_x = (x0 + x1) / 2
                control_y = max(y0, y1) + 0.5
                
                t = np.linspace(0, 1, 20)
                x = (1-t)**2 * x0 + 2*(1-t)*t * control_x + t**2 * x1
                y = (1-t)**2 * y0 + 2*(1-t)*t * control_y + t**2 * y1
                
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    line=dict(
                        color=head.color or '#3B82F6',
                        width=3
                    ),
                    hovertemplate=f'Layer: {pattern["sourceLayer"]}<br>Head: {pattern["head"]}<br>Weight: {pattern["weight"]:.3f}',
                    name=f"Head {head.layer},{head.head}",
                    showlegend=False
                ))

    # Update layout
    tokens = data.get('tokens', [f'T{i}' for i in range(data['numTokens'])])
    fig.update_layout(
        plot_bgcolor='white',
        width=1000,
        height=700,
        xaxis=dict(
            title='Token',
            ticktext=tokens,
            tickvals=list(range(len(tokens))),
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title='Layer',
            showgrid=False,
            zeroline=False,
            range=[0, data['numLayers']]
        ),
        margin=dict(l=50, r=50, t=30, b=50),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

def convert_predefined_groups_to_head_groups(model: str) -> List[HeadGroup]:
    """Convert predefined groups from MODEL_SPECIFIC_GROUPS to HeadGroup objects."""
    groups = MODEL_SPECIFIC_GROUPS.get(model, [])
    return [
        HeadGroup(
            id=idx,
            name=group["name"],
            description=group.get("description"),
            heads=[HeadPair(layer=layer, head=head) for layer, head in group["vertices"]],
            color=None  # Initialize with no custom color
        )
        for idx, group in enumerate(groups)
    ]

def load_sample_data(model: str) -> Dict:
    """Load sample attention data from JSON file."""
    try:
        file_path = os.path.join("public", "data", f"sample-attention-{model}.json")
        if not os.path.exists(file_path):
            logger.error(f"Sample data file not found: {file_path}")
            st.error(f"Sample data file not found for model {model}. Please ensure the backend is running or contact support.")
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing sample data JSON: {str(e)}")
        st.error("Error parsing sample data. The file might be corrupted.")
        return None
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        st.error("An unexpected error occurred while loading sample data.")
        return None

def main():
    st.set_page_config(layout="wide", page_title="Information Flow Visualization")

    # Initialize session state
    if 'api_service' not in st.session_state:
        st.session_state.api_service = APIService()
    if 'backend_available' not in st.session_state:
        st.session_state.backend_available = st.session_state.api_service.check_backend_health()
    if 'current_model' not in st.session_state:
        st.session_state.current_model = "gpt2-small"
    if 'threshold' not in st.session_state:
        st.session_state.threshold = 0.4
    if 'selected_heads' not in st.session_state:
        st.session_state.selected_heads = []
    if 'head_groups' not in st.session_state:
        st.session_state.head_groups = convert_predefined_groups_to_head_groups(st.session_state.current_model)
    if 'input_text' not in st.session_state:
        st.session_state.input_text = get_default_text_for_model(st.session_state.current_model)
    if 'attention_data' not in st.session_state:
        st.session_state.attention_data = None
    if 'loading' not in st.session_state:
        st.session_state.loading = False

    # Custom CSS
    st.markdown("""
        <style>
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
        .controls-container {
            max-height: 400px;
            overflow-y: auto;
            padding-right: 8px;
        }
        .controls-container::-webkit-scrollbar {
            width: 6px;
        }
        .controls-container::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 3px;
        }
        .controls-container::-webkit-scrollbar-thumb {
            background: #888;
            border-radius: 3px;
        }
        .controls-container::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Information Flow Visualization")

    # Model selection
    col1, _ = st.columns([1, 3])
    with col1:
        selected_model = st.selectbox(
            "Model",
            ["gpt2-small", "pythia-2.8b"],
            index=0 if st.session_state.current_model == "gpt2-small" else 1
        )
        
        # Handle model change
        if selected_model != st.session_state.current_model:
            st.session_state.current_model = selected_model
            st.session_state.head_groups = convert_predefined_groups_to_head_groups(selected_model)
            st.session_state.input_text = get_default_text_for_model(selected_model)
            st.session_state.attention_data = None  # Reset attention data when switching models
            st.experimental_rerun()

    # Backend status and sample data loading
    if not st.session_state.backend_available:
        st.warning("Backend is not available. Using sample data.")
        if st.session_state.attention_data is None:
            st.session_state.attention_data = load_sample_data(st.session_state.current_model)
            if st.session_state.attention_data is None:
                st.error("Failed to load sample data. Please try again later.")
                return

    # Controls section
    with st.expander("Controls", expanded=True):
        st.markdown('<div class="controls-container">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Head Groups")
            
            # New group creation
            new_group = st.text_input("New group name", placeholder="Create a new group to organize attention heads")
            if st.button("Create", type="primary"):
                if new_group:
                    new_group_obj = HeadGroup(
                        id=len(st.session_state.head_groups),
                        name=new_group,
                        heads=[],
                        color=None
                    )
                    st.session_state.head_groups.append(new_group_obj)
                    st.experimental_rerun()
                else:
                    st.error("Group name cannot be empty")

            # Display existing groups with color pickers
            for group in st.session_state.head_groups:
                group_container = st.container()
                with group_container:
                    # Create a single row with two columns
                    group_col1, group_col2 = st.columns([6, 1])
                    
                    # Left column with group content
                    with group_col1:
                        st.markdown(f"""
                            <div class="group-container">
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <div style="font-weight: 500;">{group.name}</div>
                                </div>
                                <div class="group-description">{group.description or ''}</div>
                                <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                                    {''.join([f'<span class="head-button" style="background-color: {group.color or COLOR_PALETTE[group.id % len(COLOR_PALETTE)]}">{head.layer},{head.head}</span>' for head in group.heads])}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Right column with color picker button
                    with group_col2:
                        if st.button("🎨", key=f"color_{group.id}", help="Change group color"):
                            group.color = get_random_color()
                            st.experimental_rerun()

        with col2:
            st.subheader("Individual Heads")            
            # Initialize the input state if not exists
            if 'head_input_value' not in st.session_state:
                st.session_state.head_input_value = ""
                
            head_input = st.text_input(
                "Add head",
                placeholder="layer,head or layer,: or :,head or :,:",
                help="Valid: Layer (0-32), Head (0-31). Press Enter to add.",
                key="head_input",
                value=st.session_state.head_input_value
            )
            
            def add_head(head_input: str) -> None:
                if not head_input.strip():
                    return
                    
                try:
                    layer_str, head_str = head_input.split(',')
                    layer_str = layer_str.strip()
                    head_str = head_str.strip()
                    
                    if layer_str == ':' and head_str == ':':
                        for l in range(st.session_state.attention_data['numLayers']):
                            for h in range(st.session_state.attention_data['numHeads']):
                                st.session_state.selected_heads.append(HeadPair(layer=l, head=h, color=COLOR_PALETTE[len(st.session_state.selected_heads) % len(COLOR_PALETTE)]))
                    elif layer_str == ':':
                        head = int(head_str)
                        if not 0 <= head < st.session_state.attention_data['numHeads']:
                            st.error(f"Invalid head number. Must be between 0 and {st.session_state.attention_data['numHeads']-1}")
                            return
                        for l in range(st.session_state.attention_data['numLayers']):
                            st.session_state.selected_heads.append(HeadPair(layer=l, head=head, color=COLOR_PALETTE[len(st.session_state.selected_heads) % len(COLOR_PALETTE)]))
                    elif head_str == ':':
                        layer = int(layer_str)
                        if not 0 <= layer < st.session_state.attention_data['numLayers']:
                            st.error(f"Invalid layer number. Must be between 0 and {st.session_state.attention_data['numLayers']-1}")
                            return
                        for h in range(st.session_state.attention_data['numHeads']):
                            st.session_state.selected_heads.append(HeadPair(layer=layer, head=h, color=COLOR_PALETTE[len(st.session_state.selected_heads) % len(COLOR_PALETTE)]))
                    else:
                        layer = int(layer_str)
                        head = int(head_str)
                        if not 0 <= layer < st.session_state.attention_data['numLayers']:
                            st.error(f"Invalid layer number. Must be between 0 and {st.session_state.attention_data['numLayers']-1}")
                            return
                        if not 0 <= head < st.session_state.attention_data['numHeads']:
                            st.error(f"Invalid head number. Must be between 0 and {st.session_state.attention_data['numHeads']-1}")
                            return
                        st.session_state.selected_heads.append(HeadPair(layer=layer, head=head, color=COLOR_PALETTE[len(st.session_state.selected_heads) % len(COLOR_PALETTE)]))
                    
                    # Clear the input field after successful addition
                    st.session_state.head_input_value = ""
                    st.experimental_rerun()
                except ValueError as e:
                    st.error(f"Invalid input format. Please use numbers or ':' for wildcards (e.g., '1,2' or '1,:')")
                except Exception as e:
                    st.error(f"Error adding head: {str(e)}")

            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Add", type="primary"):
                    add_head(head_input)
            
            # Handle Enter key press
            if head_input and head_input.endswith('\n'):
                add_head(head_input)

            # Display selected heads
            if st.session_state.selected_heads:
                for head in st.session_state.selected_heads:
                    col1, col2 = st.columns([6, 1])
                    with col1:
                        st.markdown(f"""
                            <div style="display: flex; align-items: center; gap: 4px;">
                                <span class="head-button" style="background-color: {head.color or '#3B82F6'}">
                                    {head.layer},{head.head}
                                </span>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        if st.button("×", key=f"remove_{head.layer}_{head.head}", help="Remove head"):
                            st.session_state.selected_heads = [h for h in st.session_state.selected_heads 
                                                             if not (h.layer == head.layer and h.head == head.head)]
                            st.experimental_rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close controls container

    # Threshold control
    st.markdown("<div style='margin: 1rem 0;'>", unsafe_allow_html=True)
    st.session_state.threshold = st.slider(
        "Edge Weight",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.threshold,
        step=0.01
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Text input and processing
    if st.session_state.backend_available:
        st.text_area(
            "Input Text",
            value=st.session_state.input_text,
            height=100,
            key="input_text"
        )
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("Process Text", type="primary"):
                st.session_state.loading = True
                try:
                    st.session_state.attention_data = st.session_state.api_service.process_text(
                        st.session_state.input_text,
                        st.session_state.current_model
                    )
                except Exception as e:
                    st.error(f"Error processing text: {str(e)}")
                finally:
                    st.session_state.loading = False
                    st.experimental_rerun()
    else:
        st.text_area(
            "Input Text",
            value=st.session_state.input_text,
            height=100,
            key="input_text",
            disabled=True,
            help="Text input is disabled when using sample data. Please start the backend server to enable text processing."
        )

    # Visualization
    if st.session_state.attention_data:
        st.markdown('<div class="graph-container">', unsafe_allow_html=True)
        
        try:
            logger.info("Starting graph creation")
            # Log the attention data structure
            logger.info("Attention data structure: " + json.dumps({
                'numLayers': st.session_state.attention_data['numLayers'],
                'numTokens': st.session_state.attention_data['numTokens'],
                'numHeads': st.session_state.attention_data['numHeads'],
                'numPatterns': len(st.session_state.attention_data['attentionPatterns']),
                'hasTokens': 'tokens' in st.session_state.attention_data
            }, indent=2))
            
            # Create and display the graph
            create_attention_graph(
                st.session_state.attention_data,
                st.session_state.threshold,
                st.session_state.selected_heads,
                st.session_state.head_groups
            )
        except Exception as e:
            logger.error(f"Error creating graph: {str(e)}", exc_info=True)
            st.error(f"Error creating graph: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        logger.info("No attention data available yet")
        st.info("Enter text and click 'Process Text' to visualize attention patterns.")

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 