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
from scipy import interpolate

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
    def __init__(self, base_url: str = None):
        # If no base_url is provided, try to get it from environment variable or use local IP
        if base_url is None:
            import os
            # Try to get URL from environment variable first
            env_url = os.getenv('BACKEND_URL')
            if env_url:
                self.base_url = env_url
            else:
                import subprocess
                try:
                    # Get IP address using ifconfig (macOS compatible)
                    result = subprocess.run(['ifconfig'], capture_output=True, text=True)
                    if result.returncode == 0:
                        # Parse the output to find the first non-localhost IP
                        for line in result.stdout.split('\n'):
                            if 'inet ' in line and '127.0.0.1' not in line:
                                local_ip = line.strip().split(' ')[1]
                                self.base_url = f"http://{local_ip}:8000"
                                break
                        else:
                            # Fallback to localhost if no IP found
                            self.base_url = "http://localhost:8000"
                    else:
                        # Fallback to localhost if ifconfig fails
                        self.base_url = "http://localhost:8000"
                except Exception:
                    # Fallback to localhost if any error occurs
                    self.base_url = "http://localhost:8000"
        else:
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
            f"{self.base_url}/process",
            json={"text": text, "model_name": model},
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
    """Create graph visualization using D3.js."""
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

    # Prepare data for D3 visualization
    viz_data = {
        'numLayers': data['numLayers'],
        'numTokens': data['numTokens'],
        'numHeads': data['numHeads'],
        'tokens': data.get('tokens', [f'T{i}' for i in range(data['numTokens'])]),
        'attentionPatterns': filtered_patterns,
        'headGroups': [
            {
                'id': group.id,
                'name': group.name,
                'description': group.description,
                'color': group.color,
                'heads': [{'layer': h.layer, 'head': h.head} for h in group.heads]
            }
            for group in head_groups
        ],
        'selectedHeads': [{'layer': h.layer, 'head': h.head} for h in selected_heads]
    }

    # Create HTML with embedded D3.js visualization
    html = f"""
    <div id="visualization-container" style="width: 100%; height: 800px;"></div>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        .attention-line {{
            stroke-opacity: 0.6;
            stroke-width: 4;
            cursor: pointer;
        }}
        .attention-line:hover {{
            stroke-opacity: 0.9;
            stroke-width: 6;
        }}
        .grid-point {{
            fill: #e5e7eb;
            cursor: pointer;
        }}
        .grid-point:hover {{
            fill: #d1d5db;
            r: 8;
        }}
        .hover-target {{
            fill: transparent;
            cursor: pointer;
        }}
        #graph-tooltip {{
            display: none;
            position: absolute;
            background: white;
            padding: 5px;
            border: 1px solid #ccc;
            border-radius: 4px;
            font-size: 12px;
            pointer-events: none;
            z-index: 1000;
        }}
    </style>
    <script>
        // Color palette for individual heads
        const colorPalette = [
            "#38B2AC", "#9F7AEA", "#F6AD55", "#68D391", "#F687B3",
            "#4FD1C5", "#B794F4", "#7F9CF5", "#C6F6D5", "#FBD38D",
            "#76E4F7", "#E9D8FD", "#90CDF4", "#FEB2B2", "#81E6D9",
            "#D6BCFA", "#FBB6CE", "#B2F5EA", "#667EEA", "#ED64A6"
        ];

        // Graph dimensions
        const graphDimensions = {{
            width: 1000,
            height: 700,
            padding: {{ top: 40, right: 180, bottom: 60, left: 60 }}
        }};

        // Data from Streamlit
        const data = {json.dumps(viz_data)};
        let headGroups = data.headGroups || [];
        let selectedHeads = data.selectedHeads || [];

        // Function to get visible heads
        function getVisibleHeads() {{
            return [...selectedHeads, ...headGroups.flatMap(g => g.heads)];
        }}

        // Function to get head group
        function getHeadGroup(layer, head) {{
            for (const group of headGroups) {{
                if (group.heads.some(h => h.layer === layer && h.head === head)) {{
                    return group.id;
                }}
            }}
            return -1;
        }}

        // Function to get group color
        function getGroupColor(groupId) {{
            const group = headGroups.find(g => g.id === groupId);
            return group?.color || colorPalette[groupId % colorPalette.length];
        }}

        // Main drawing function
        function drawGraph() {{
            const svg = d3.select("#visualization-container")
                .append("svg")
                .attr("width", graphDimensions.width)
                .attr("height", graphDimensions.height);

            const width = graphDimensions.width;
            const height = graphDimensions.height;
            const padding = graphDimensions.padding;
            const legendWidth = padding.right;
            const graphWidth = width - padding.left - padding.right;
            const graphHeight = height - padding.top - padding.bottom;
            const tokenWidth = graphWidth / data.numTokens;
            const layerHeight = graphHeight / (data.numLayers - 1);

            // Create nodes
            const nodes = [];
            for (let l = 0; l < data.numLayers; l++) {{
                for (let t = 0; t < data.numTokens; t++) {{
                    nodes.push({{
                        id: `${{l}}-${{t}}`,
                        layer: l,
                        token: t,
                        x: padding.left + t * tokenWidth + tokenWidth / 2,
                        y: height - (padding.bottom + l * layerHeight)
                    }});
                }}
            }}

            // Create color scales
            const individualHeadColorScale = d3.scaleOrdinal(colorPalette)
                .domain(Array.from({{ length: data.numHeads }}, (_, i) => i.toString()));

            // Filter edges
            const visibleHeadPairs = getVisibleHeads();
            const links = data.attentionPatterns
                .filter(edge => {{
                    const isVisible = visibleHeadPairs.some(h =>
                        h.layer === edge.sourceLayer && h.head === edge.head
                    );
                    return edge.weight >= {threshold} && isVisible;
                }})
                .map(edge => ({{
                    source: `${{edge.sourceLayer}}-${{edge.sourceToken}}`,
                    target: `${{edge.destLayer}}-${{edge.destToken}}`,
                    weight: edge.weight,
                    head: edge.head,
                    groupId: getHeadGroup(edge.sourceLayer, edge.head) ?? -1
                }}));

            // Draw layers and tokens labels
            const g = svg.append("g");

            // Layer labels
            for (let l = 0; l < data.numLayers; l++) {{
                g.append("text")
                    .attr("x", padding.left / 2 + 25)
                    .attr("y", height - (padding.bottom + l * layerHeight))
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "middle")
                    .text(l.toString());
            }}

            // Y-axis label
            g.append("text")
                .attr("x", padding.left / 2)
                .attr("y", height / 2)
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle")
                .attr("font-size", "14px")
                .attr("font-weight", "medium")
                .text("Layer");

            // Token labels
            for (let t = 0; t < data.numTokens; t++) {{
                g.append("text")
                    .attr("x", padding.left + t * tokenWidth + tokenWidth / 2)
                    .attr("y", height - padding.bottom / 2)
                    .attr("text-anchor", "middle")
                    .attr("dominant-baseline", "middle")
                    .text(data.tokens?.[t] || `T${{t}}`);
            }}

            // X-axis label
            g.append("text")
                .attr("x", width / 2)
                .attr("y", height - padding.bottom / 4)
                .attr("text-anchor", "middle")
                .attr("dominant-baseline", "middle")
                .attr("font-size", "14px")
                .attr("font-weight", "medium")
                .text("Token");

            // Draw edges with curved paths
            const linkElements = g.selectAll("path")
                .data(links)
                .enter()
                .append("path")
                .attr("d", d => {{
                    const source = nodes.find(n => n.id === d.source);
                    const target = nodes.find(n => n.id === d.target);
                    const dx = target.x - source.x;
                    const controlPoint1x = source.x + dx * 0.5;
                    const controlPoint1y = source.y;
                    const controlPoint2x = target.x - dx * 0.5;
                    const controlPoint2y = target.y;
                    return `M ${{source.x}} ${{source.y}} C ${{controlPoint1x}} ${{controlPoint1y}}, ${{controlPoint2x}} ${{controlPoint2y}}, ${{target.x}} ${{target.y}}`;
                }})
                .attr("fill", "none")
                .attr("stroke", d => {{
                    if (d.groupId >= 0) {{
                        return getGroupColor(d.groupId);
                    }}
                    return individualHeadColorScale(d.head.toString());
                }})
                .attr("stroke-width", 4)
                .attr("opacity", 0.6)
                .attr("class", "attention-line")
                .on("mouseover", function(event, d) {{
                    d3.select(this)
                        .attr("opacity", 1)
                        .attr("stroke-width", 6);
                    const tooltip = d3.select("#graph-tooltip");
                    const group = headGroups.find(g => g.id === d.groupId);
                    tooltip.style("display", "block")
                        .html(`Head: Layer ${{d.source.split("-")[0]}}, Head ${{d.head}}<br>
                               Weight: ${{d.weight.toFixed(4)}}${{group ?
                               `<br>Group: ${{group.name}}${{group.description ?
                               `<br><span style="font-style: italic; font-size: 11px;">${{group.description}}</span>` : ''}}` :
                               '<br>Individual Head'}}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mouseout", function() {{
                    d3.select(this)
                        .attr("opacity", 0.6)
                        .attr("stroke-width", 4);
                    d3.select("#graph-tooltip").style("display", "none");
                }});

            // Draw nodes
            const nodeElements = g.selectAll("circle")
                .data(nodes)
                .enter()
                .append("circle")
                .attr("class", "grid-point")
                .attr("cx", d => d.x)
                .attr("cy", d => d.y)
                .attr("r", 6)
                .attr("fill", "#e5e7eb")
                .on("mouseover", function(event, d) {{
                    d3.select(this)
                        .attr("r", 8)
                        .attr("fill", "#d1d5db");
                    const tooltip = d3.select("#graph-tooltip");
                    tooltip.style("display", "block")
                        .html(`Layer ${{d.layer}}, Token ${{d.token}}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                }})
                .on("mouseout", function() {{
                    d3.select(this)
                        .attr("r", 6)
                        .attr("fill", "#e5e7eb");
                    d3.select("#graph-tooltip").style("display", "none");
                }});

            // Add invisible hover targets for easier interaction
            g.selectAll("circle.hover-target")
                .data(nodes)
                .enter()
                .append("circle")
                .attr("class", "hover-target")
                .attr("cx", d => d.x)
                .attr("cy", d => d.y)
                .attr("r", 12)
                .attr("fill", "transparent")
                .on("mouseover", function(event, d) {{
                    const tooltip = d3.select("#graph-tooltip");
                    tooltip.style("display", "block")
                        .html(`Layer ${{d.layer}}, Token ${{d.token}}`)
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY - 10) + "px");
                    d3.select(this.parentNode)
                        .select(`circle:not(.hover-target)[data-node-id="${{d.id}}"]`)
                        .attr("r", 8)
                        .attr("fill", "#d1d5db");
                }})
                .on("mouseout", function(event, d) {{
                    d3.select("#graph-tooltip").style("display", "none");
                    d3.select(this.parentNode)
                        .select(`circle:not(.hover-target)[data-node-id="${{d.id}}"]`)
                        .attr("r", 6)
                        .attr("fill", "#e5e7eb");
                }});

            // Draw legend
            const legend = g.append("g")
                .attr("transform", `translate(${{width - legendWidth + 20}}, ${{padding.top}})`);

            // Add legend title
            legend.append("text")
                .attr("x", 0)
                .attr("y", 0)
                .attr("font-size", "14px")
                .attr("font-weight", "bold")
                .text("Legend");

            // Add group colors to legend
            headGroups.forEach((group, i) => {{
                const y = 30 + i * 25;
                legend.append("rect")
                    .attr("x", 0)
                    .attr("y", y)
                    .attr("width", 15)
                    .attr("height", 15)
                    .attr("fill", getGroupColor(group.id));
                const groupText = legend.append("text")
                    .attr("x", 25)
                    .attr("y", y + 12)
                    .attr("font-size", "12px")
                    .text(group.name);
                if (group.description) {{
                    groupText
                        .on("mouseenter", function(event) {{
                            const tooltip = d3.select("#graph-tooltip");
                            tooltip.style("display", "block")
                                .html(`<strong>${{group.name}}</strong><br>${{group.description}}`)
                                .style("left", (event.pageX + 10) + "px")
                                .style("top", (event.pageY - 10) + "px");
                        }})
                        .on("mouseleave", function() {{
                            d3.select("#graph-tooltip").style("display", "none");
                        }});
                }}
            }});

            // Add separator
            const separatorY = 30 + headGroups.length * 25 + 10;
            legend.append("line")
                .attr("x1", 0)
                .attr("x2", legendWidth - padding.left)
                .attr("y1", separatorY)
                .attr("y2", separatorY)
                .attr("stroke", "#e5e7eb")
                .attr("stroke-width", 2);

            // Add individual heads section
            legend.append("text")
                .attr("x", 0)
                .attr("y", separatorY + 25)
                .attr("font-size", "12px")
                .attr("font-weight", "bold")
                .text("Individual Heads");

            // Add individual head colors to legend
            const visibleIndividualHeads = selectedHeads.filter(h =>
                !headGroups.some(g => g.heads.some(gh => gh.layer === h.layer && gh.head === h.head))
            );

            visibleIndividualHeads.forEach((head, i) => {{
                const y = separatorY + 40 + i * 25;
                legend.append("rect")
                    .attr("x", 0)
                    .attr("y", y)
                    .attr("width", 15)
                    .attr("height", 15)
                    .attr("fill", individualHeadColorScale(head.head.toString()));
                legend.append("text")
                    .attr("x", 25)
                    .attr("y", y + 12)
                    .attr("font-size", "12px")
                    .text(`Layer ${{head.layer}}, Head ${{head.head}}`);
            }});
        }}

        // Add tooltip div
        const tooltip = document.createElement("div");
        tooltip.id = "graph-tooltip";
        document.body.appendChild(tooltip);

        // Draw the graph
        drawGraph();
    </script>
    """
    
    # Display the visualization
    st.components.v1.html(html, height=800)

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
    if 'color_change_group' not in st.session_state:
        st.session_state.color_change_group = None
    if 'curve_type' not in st.session_state:
        st.session_state.curve_type = "cubic"

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
        /* Style for the color picker button */
        div[data-testid="stHorizontalBlock"] > div:last-child button[data-testid="baseButton-primary"] {
            background: transparent;
            border: none;
            padding: 0;
            margin: 0;
            font-size: 1.2rem;
            cursor: pointer;
            color: #666;
            transition: color 0.2s;
        }
        div[data-testid="stHorizontalBlock"] > div:last-child button[data-testid="baseButton-primary"]:hover {
            color: #000;
        }
        /* Style for primary buttons (Add and Create) */
        button[data-testid="baseButton-primary"]:not(div[data-testid="stHorizontalBlock"] > div:last-child button) {
            background: #FF4B4B !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            margin: 0 !important;
            font-size: 0.875rem !important;
            cursor: pointer !important;
            color: white !important;
            border-radius: 0.25rem !important;
            transition: background-color 0.2s !important;
        }
        button[data-testid="baseButton-primary"]:not(div[data-testid="stHorizontalBlock"] > div:last-child button):hover {
            background: #FF3333 !important;
        }
        /* Additional style for the Add button */
        div[data-testid="stHorizontalBlock"] > div:first-child button[data-testid="baseButton-primary"] {
            background: #FF4B4B !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            margin: 0 !important;
            font-size: 0.875rem !important;
            cursor: pointer !important;
            color: white !important;
            border-radius: 0.25rem !important;
            transition: background-color 0.2s !important;
        }
        div[data-testid="stHorizontalBlock"] > div:first-child button[data-testid="baseButton-primary"]:hover {
            background: #FF3333 !important;
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
                        color=get_random_color()
                    )
                    st.session_state.head_groups.append(new_group_obj)
                else:
                    st.error("Group name cannot be empty")

            # Display existing groups with color pickers
            for group in st.session_state.head_groups:
                # Create a container for the group
                group_container = st.container()
                with group_container:
                    # Create a single row with two columns
                    group_col1, group_col2 = st.columns([6, 1])
                    
                    # Left column with group content
                    with group_col1:
                        st.markdown(f"""
                            <div class="group-container">
                                <div style="display: flex; align-items: center; justify-content: space-between; gap: 8px;">
                                    <div style="font-weight: 500;">{group.name}</div>
                                </div>
                                <div class="group-description">{group.description or ''}</div>
                                <div style="display: flex; flex-wrap: wrap; gap: 4px;">
                                    {''.join([f'<span class="head-button" style="background-color: {group.color or COLOR_PALETTE[group.id % len(COLOR_PALETTE)]}">{head.layer},{head.head}</span>' for head in group.heads])}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Right column with color picker and remove buttons
                    with group_col2:
                        if st.button("🎨", key=f"color_{group.id}", help="Change group color"):
                            group.color = get_random_color()
                        if st.button("×", key=f"remove_group_{group.id}", help="Remove group"):
                            st.session_state.head_groups = [g for g in st.session_state.head_groups if g.id != group.id]

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
                                st.session_state.selected_heads.append(HeadPair(layer=l, head=h, color="#e5e7eb"))
                    elif layer_str == ':':
                        head = int(head_str)
                        if not 0 <= head < st.session_state.attention_data['numHeads']:
                            st.error(f"Invalid head number. Must be between 0 and {st.session_state.attention_data['numHeads']-1}")
                            return
                        for l in range(st.session_state.attention_data['numLayers']):
                            st.session_state.selected_heads.append(HeadPair(layer=l, head=head, color="#e5e7eb"))
                    elif head_str == ':':
                        layer = int(layer_str)
                        if not 0 <= layer < st.session_state.attention_data['numLayers']:
                            st.error(f"Invalid layer number. Must be between 0 and {st.session_state.attention_data['numLayers']-1}")
                            return
                        for h in range(st.session_state.attention_data['numHeads']):
                            st.session_state.selected_heads.append(HeadPair(layer=layer, head=h, color="#e5e7eb"))
                    else:
                        layer = int(layer_str)
                        head = int(head_str)
                        if not 0 <= layer < st.session_state.attention_data['numLayers']:
                            st.error(f"Invalid layer number. Must be between 0 and {st.session_state.attention_data['numLayers']-1}")
                            return
                        if not 0 <= head < st.session_state.attention_data['numHeads']:
                            st.error(f"Invalid head number. Must be between 0 and {st.session_state.attention_data['numHeads']-1}")
                            return
                        st.session_state.selected_heads.append(HeadPair(layer=layer, head=head, color="#e5e7eb"))
                    
                    # Clear the input field after successful addition
                    st.session_state.head_input_value = ""
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
                    col1, col2, col3 = st.columns([5, 1, 1])
                    with col1:
                        st.markdown(f"""
                            <div style="display: flex; align-items: center; gap: 4px;">
                                <span class="head-button" style="background-color: {head.color or '#3B82F6'}">
                                    {head.layer},{head.head}
                                </span>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        if st.button("🎨", key=f"color_head_{head.layer}_{head.head}", help="Change head color"):
                            head.color = get_random_color()

                    with col3:
                        if st.button("×", key=f"remove_{head.layer}_{head.head}", help="Remove head"):
                            st.session_state.selected_heads = [h for h in st.session_state.selected_heads 
                                                             if not (h.layer == head.layer and h.head == head.head)]
        
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
    
    # Add curve type selector
    curve_types = {
        "Cubic Bezier": "cubic",
        "Quadratic Bezier": "quadratic",
        "Linear": "linear",
        "Spline": "spline"
    }
    
    # Create reverse mapping from internal values to display names
    curve_type_to_display = {v: k for k, v in curve_types.items()}
    
    # Get the current display name using the reverse mapping
    current_display = curve_type_to_display.get(st.session_state.curve_type, "Cubic Bezier")
    
    selected_display = st.selectbox(
        "Curve Type",
        options=list(curve_types.keys()),
        index=list(curve_types.keys()).index(current_display)
    )
    st.session_state.curve_type = curve_types[selected_display]
    st.markdown("</div>", unsafe_allow_html=True)

    # Text input and processing
    if st.session_state.backend_available:
        st.text_input(
            "Input Text",
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
    else:
        st.text_input(
            "Input Text",
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