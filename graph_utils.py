def create_d3_visualization(nodes, links):
    """
    Creates an HTML string containing a D3.js force-directed graph visualization
    
    Args:
        nodes: List of dictionaries with 'id' and 'name' keys
        links: List of dictionaries with 'source' and 'target' keys
    """
    print("Python: create_d3_visualization function called") # Python-side logging
    
    import json
    nodes_json = json.dumps(nodes)
    links_json = json.dumps(links)
    
    # Generate a unique ID for this graph instance
    import uuid
    container_id = f"graph-{uuid.uuid4()}"
    viz_id = f"viz-{uuid.uuid4()}"
    
    html_template = f"""
    <div id="{container_id}" style="width: 100%; height: 400px; border: 1px solid #ccc;">
        <div id="{viz_id}" style="width: 100%; height: 100%;"></div>
    </div>

    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script>
        (function() {{
            // Create the visualization immediately
            const width = 600;
            const height = 400;
            
            // Wait for D3 to load
            function initViz() {{
                if (typeof d3 === 'undefined') {{
                    setTimeout(initViz, 100);
                    return;
                }}
                
                // Data
                const data = {{
                    nodes: {nodes_json},
                    links: {links_json}
                }};
                
                // Create SVG
                const svg = d3.select('#{viz_id}')
                    .append('svg')
                    .attr('width', '100%')
                    .attr('height', '100%')
                    .attr('viewBox', [0, 0, width, height]);
                
                // Create force simulation
                const simulation = d3.forceSimulation(data.nodes)
                    .force('link', d3.forceLink(data.links)
                        .id(d => d.id)
                        .distance(100))
                    .force('charge', d3.forceManyBody().strength(-200))
                    .force('center', d3.forceCenter(width / 2, height / 2));
                
                // Add links
                const links = svg.append('g')
                    .selectAll('line')
                    .data(data.links)
                    .join('line')
                    .style('stroke', '#999')
                    .style('stroke-opacity', 0.6)
                    .style('stroke-width', 2);
                
                // Add nodes
                const nodes = svg.append('g')
                    .selectAll('circle')
                    .data(data.nodes)
                    .join('circle')
                    .attr('r', 8)
                    .style('fill', '#69b3a2');
                
                // Add labels
                const labels = svg.append('g')
                    .selectAll('text')
                    .data(data.nodes)
                    .join('text')
                    .text(d => d.name)
                    .attr('font-size', '12px')
                    .attr('dx', 12)
                    .attr('dy', 4);
                
                // Add drag behavior
                nodes.call(d3.drag()
                    .on('start', dragstarted)
                    .on('drag', dragged)
                    .on('end', dragended));
                
                function dragstarted(event) {{
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }}
                
                function dragged(event) {{
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }}
                
                function dragended(event) {{
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }}
                
                // Update positions on each tick
                simulation.on('tick', () => {{
                    links
                        .attr('x1', d => d.source.x)
                        .attr('y1', d => d.source.y)
                        .attr('x2', d => d.target.x)
                        .attr('y2', d => d.target.y);
                    
                    nodes
                        .attr('cx', d => d.x)
                        .attr('cy', d => d.y);
                    
                    labels
                        .attr('x', d => d.x)
                        .attr('y', d => d.y);
                }});
            }}

            // Start initialization
            initViz();
        }})();
    </script>
    """
    
    return html_template 