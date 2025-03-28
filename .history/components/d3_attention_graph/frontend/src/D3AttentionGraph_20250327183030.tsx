import {
  StreamlitComponentBase,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib";
import React from "react";
import * as d3 from "d3";

interface Node {
  layer: number;
  token: number;
  x: number;
  y: number;
  label: string;
}

interface AttentionPattern {
  sourceLayer: number;
  sourceToken: number;
  destLayer: number;
  destToken: number;
  weight: number;
  head: number;
}

interface GraphData {
  numLayers: number;
  numTokens: number;
  tokens?: string[];
  attentionPatterns: AttentionPattern[];
}

interface State {
  data: GraphData | null;
  width: number;
  height: number;
}

interface ComponentArgs {
  data: GraphData;
  width: number;
  height: number;
}

interface Props {
  args?: ComponentArgs;
  data?: GraphData;
  width?: number;
  height?: number;
}

class D3AttentionGraph extends React.Component<Props, State> {
  private svgRef: React.RefObject<SVGSVGElement | null>;

  constructor(props: Props) {
    super(props);
    this.svgRef = React.createRef<SVGSVGElement | null>();
  }

  state: State = {
    data: null,
    width: 1000,
    height: 700,
  };

  componentDidMount() {
    // Get the component's dimensions from either Streamlit args or direct props
    const { width, height } = this.props.args || this.props;
    this.setState({ width, height });
  }

  componentDidUpdate() {
    const { data, width, height } = this.props.args || this.props;
    this.setState({ data, width, height }, () => {
      this.renderGraph();
    });
  }

  renderGraph() {
    if (!this.state.data || !this.svgRef.current) return;

    const { attentionPatterns, numLayers, numTokens, tokens } = this.state.data;
    const { width, height } = this.state;

    // Clear previous SVG content
    d3.select(this.svgRef.current).selectAll("*").remove();

    const svg = d3.select(this.svgRef.current);
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Create scales
    const xScale = d3.scaleLinear()
      .domain([0, numTokens - 1])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, numLayers - 1])
      .range([0, innerHeight]);

    // Create the main group
    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Create nodes
    const nodes: Node[] = [];
    for (let l = 0; l < numLayers; l++) {
      for (let t = 0; t < numTokens; t++) {
        nodes.push({
          layer: l,
          token: t,
          x: xScale(t),
          y: yScale(l),
          label: tokens ? tokens[t] : `T${t}`,
        });
      }
    }

    // Draw nodes
    g.selectAll("circle")
      .data(nodes)
      .enter()
      .append("circle")
      .attr("cx", d => d.x)
      .attr("cy", d => d.y)
      .attr("r", 5)
      .attr("fill", "#e5e7eb");

    // Add labels
    g.selectAll("text")
      .data(nodes)
      .enter()
      .append("text")
      .attr("x", d => d.x)
      .attr("y", d => d.y - 10)
      .attr("text-anchor", "middle")
      .attr("font-size", "12px")
      .text(d => `L${d.layer}-${d.label}`);

    // Draw edges
    attentionPatterns.forEach((pattern: AttentionPattern) => {
      const source = nodes.find(n => n.layer === pattern.sourceLayer && n.token === pattern.sourceToken);
      const target = nodes.find(n => n.layer === pattern.destLayer && n.token === pattern.destToken);

      if (source && target) {
        g.append("line")
          .attr("x1", source.x)
          .attr("y1", source.y)
          .attr("x2", target.x)
          .attr("y2", target.y)
          .attr("stroke", "#3B82F6")
          .attr("stroke-width", pattern.weight * 2)
          .attr("opacity", pattern.weight);
      }
    });
  }

  render() {
    return (
      <div style={{ width: "100%", height: "100%" }}>
        <svg
          ref={this.svgRef}
          width={this.state.width}
          height={this.state.height}
          style={{ background: "white" }}
        />
      </div>
    );
  }
}

export default withStreamlitConnection(D3AttentionGraph); 