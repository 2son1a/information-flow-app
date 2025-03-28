import React, { useEffect, useState } from 'react';
import D3AttentionGraph from './D3AttentionGraph';
import './App.css';

interface GraphData {
  numLayers: number;
  numTokens: number;
  tokens?: string[];
  attentionPatterns: Array<{
    sourceLayer: number;
    sourceToken: number;
    destLayer: number;
    destToken: number;
    weight: number;
    head: number;
  }>;
}

function App() {
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load the sample data
    fetch('/data/sample-attention-gpt2-small.json')
      .then(response => response.json())
      .then(jsonData => {
        setData(jsonData);
        setLoading(false);
      })
      .catch(error => {
        console.error('Error loading data:', error);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return <div>Loading attention graph data...</div>;
  }

  if (!data) {
    return <div>Error loading data</div>;
  }

  return (
    <div style={{ width: '100vw', height: '100vh', padding: '20px' }}>
      <h1>Attention Graph Visualization</h1>
      <D3AttentionGraph
        args={{
          data: data,
          width: window.innerWidth - 40,
          height: window.innerHeight - 100
        }}
      />
    </div>
  );
}

export default App;
