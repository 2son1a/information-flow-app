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

interface Props {
  args: {
    data: GraphData;
    width: number;
    height: number;
  };
}

const D3AttentionGraph: React.FC<Props> = ({ args }) => {
  const svgRef = React.useRef<SVGSVGElement | null>(null);
  const { data, width, height } = args;

  React.useEffect(() => {
    if (!data || !svgRef.current) return;

    const { attentionPatterns, numLayers, numTokens, tokens } = data;

    // Clear previous SVG content
    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3.select(svgRef.current);
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
  }, [data, width, height]);

  return (
    <div style={{ width: "100%", height: "100%" }}>
      <svg
        ref={svgRef}
        width={width}
        height={height}
        style={{ background: "white" }}
      />
    </div>
  );
};

export default D3AttentionGraph; 