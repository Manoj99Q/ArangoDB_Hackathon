import gradio as gr
import base64

# 1. Define the complete HTML document with D3.js code for the graph.
html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>D3.js Force-Directed Graph</title>
  <script src="https://d3js.org/d3.v6.min.js"></script>
  <style>
    body { font: 14px sans-serif; }
    .link { stroke: #999; stroke-opacity: 0.6; }
    .node { stroke: #fff; stroke-width: 1.5px; }
  </style>
</head>
<body>
  <h1>Interactive Force-Directed Graph</h1>
  <svg width="600" height="400"></svg>
  <script>
    // Select the SVG element and set dimensions.
    const svg = d3.select("svg");
    const width = +svg.attr("width");
    const height = +svg.attr("height");

    // Define sample nodes and links.
    const nodes = [
      {id: "A"}, {id: "B"}, {id: "C"}, {id: "D"}, {id: "E"}
    ];
    const links = [
      {source: "A", target: "B"},
      {source: "A", target: "C"},
      {source: "B", target: "D"},
      {source: "C", target: "D"},
      {source: "D", target: "E"}
    ];

    // Create a simulation with forces.
    const simulation = d3.forceSimulation(nodes)
                         .force("link", d3.forceLink(links).id(d => d.id).distance(100))
                         .force("charge", d3.forceManyBody().strength(-300))
                         .force("center", d3.forceCenter(width / 2, height / 2));

    // Create and style the links.
    const link = svg.append("g")
                    .attr("class", "links")
                    .selectAll("line")
                    .data(links)
                    .enter().append("line")
                    .attr("class", "link")
                    .attr("stroke-width", 2);

    // Create and style the nodes.
    const node = svg.append("g")
                    .attr("class", "nodes")
                    .selectAll("circle")
                    .data(nodes)
                    .enter().append("circle")
                    .attr("class", "node")
                    .attr("r", 10)
                    .attr("fill", "steelblue")
                    .call(d3.drag()
                        .on("start", dragstarted)
                        .on("drag", dragged)
                        .on("end", dragended));

    // Add a tooltip to each node.
    node.append("title")
        .text(d => d.id);

    // Update positions on each tick of the simulation.
    simulation.on("tick", () => {
      link.attr("x1", d => d.source.x)
          .attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x)
          .attr("y2", d => d.target.y);

      node.attr("cx", d => d.x)
          .attr("cy", d => d.y);
    });

    // Define drag event functions.
    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }
    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }
    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  </script>
</body>
</html>
"""

# 2. Encode the HTML into a Base64 data URL.
encoded_html = base64.b64encode(html_code.encode("utf-8")).decode("utf-8")
iframe_html = f'<iframe src="data:text/html;base64,{encoded_html}" width="620" height="420" frameborder="0"></iframe>'

# 3. Create the Gradio interface that returns the iframe HTML.
iface = gr.Interface(fn=lambda: iframe_html, inputs=[], outputs=gr.HTML())

# 4. Launch the app.
iface.launch()
