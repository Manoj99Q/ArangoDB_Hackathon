import base64
import gradio as gr
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file


def generate_dynamic_graph_html(query):
    """Dynamically generates a complete HTML document for an interactive D3.js visualization based on the query using an LLM prompt."""
    sample_html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\">
  <title>D3.js Force-Directed Graph</title>
  <script src=\"https://d3js.org/d3.v6.min.js\"></script>
  <style>
    body { font: 14px sans-serif; }
    .link { stroke: #999; stroke-opacity: 0.6; }
    .node { stroke: #fff; stroke-width: 1.5px; }
  </style>
</head>
<body>
  <h1>Interactive Force-Directed Graph</h1>
  <svg width=\"600\" height=\"400\"></svg>
  <script>
    // Select the SVG element and set dimensions.
    const svg = d3.select(\"svg\");
    const width = +svg.attr(\"width\");
    const height = +svg.attr(\"height\");

    // Define sample nodes and links.
    const nodes = [
      {id: \"A\"}, {id: \"B\"}, {id: \"C\"}, {id: \"D\"}, {id: \"E\"}
    ];
    const links = [
      {source: \"A\", target: \"B\"},
      {source: \"A\", target: \"C\"},
      {source: \"B\", target: \"D\"},
      {source: \"C\", target: \"D\"},
      {source: \"D\", target: \"E\"}
    ];

    // Create a simulation with forces.
    const simulation = d3.forceSimulation(nodes)
                         .force(\"link\", d3.forceLink(links).id(d => d.id).distance(100))
                         .force(\"charge\", d3.forceManyBody().strength(-300))
                         .force(\"center\", d3.forceCenter(width / 2, height / 2));

    // Create and style the links.
    const link = svg.append(\"g\")
                    .attr(\"class\", \"links\")
                    .selectAll(\"line\")
                    .data(links)
                    .enter().append(\"line\")
                    .attr(\"class\", \"link\")
                    .attr(\"stroke-width\", 2);

    // Create and style the nodes.
    const node = svg.append(\"g\")
                    .attr(\"class\", \"nodes\")
                    .selectAll(\"circle\")
                    .data(nodes)
                    .enter().append(\"circle\")
                    .attr(\"class\", \"node\")
                    .attr(\"r\", 10)
                    .attr(\"fill\", \"steelblue\")
                    .call(d3.drag()
                        .on(\"start\", dragstarted)
                        .on(\"drag\", dragged)
                        .on(\"end\", dragended));

    // Add a tooltip to each node.
    node.append(\"title\")
        .text(d => d.id);

    // Update positions on each tick of the simulation.
    simulation.on(\"tick\", () => {
      link.attr(\"x1\", d => d.source.x)
          .attr(\"y1\", d => d.source.y)
          .attr(\"x2\", d => d.target.x)
          .attr(\"y2\", d => d.target.y);

      node.attr(\"cx\", d => d.x)
          .attr(\"cy\", d => d.y);
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
</html>"""

    prompt = (
        "You are an expert web developer specializing in interactive D3.js visualizations. "
        "Generate a complete, self-contained HTML document that includes <!DOCTYPE html>, <html>, <head>, and <body> tags. "
        "Integrate D3.js from 'https://d3js.org/d3.v6.min.js' to create an interactive visualization based on the following specification: "
        + query + " "
        "For reference, here is an example of a Force-Directed Graph: "
        + sample_html + " "
        "Only output the complete HTML code without any explanation or markdown formatting."
    )
    try:
        # Get response from ChatOpenAI
        response = llm.invoke(prompt)
        # Extract HTML content from the response
        html_code = response.content
        # Remove markdown formatting if present
        if html_code.startswith("```html"):
            html_code = html_code[7:-3]  # Remove ```html and ``` markers
        elif html_code.startswith("```"):
            html_code = html_code[3:-3]  # Remove ``` markers
            
        # Print the generated HTML code for debugging
        print("\nGenerated HTML Code:")
        print("=" * 50)
        print(html_code)
        print("=" * 50)
        
        encoded_html = base64.b64encode(html_code.encode('utf-8')).decode('utf-8')
        data_url = f"data:text/html;base64,{encoded_html}"
        
        # Following the pattern from main.py
        iframe_html = f'<iframe src="{data_url}" width="620" height="420" frameborder="0"></iframe>'
        return iframe_html
    except Exception as e:
        print(f"Error generating visualization: {str(e)}")
        return "<p>Error generating visualization. Please try again.</p>"


# Create a LangChain Tool for generating graph HTML content based on the user query.
dynamic_graph_tool = Tool(
    name="DynamicGraphGenerator",
    func=generate_dynamic_graph_html,
    description="Dynamically generates a complete HTML document for an interactive D3.js visualization based on user specification using an LLM prompt."
)

# Initialize ChatOpenAI with appropriate parameters
llm = ChatOpenAI(
    model_name="gpt-4o",  # or "gpt-4" if you have access
    temperature=0,
    max_tokens=2000,
    presence_penalty=0,
    frequency_penalty=0
)

def agent_wrapper(query):
    """Takes user query and returns the generated iframe HTML directly."""
    try:
        # Direct call to generate visualization without using agent
        iframe_html = generate_dynamic_graph_html(query)
        return gr.HTML(value=iframe_html)
    except Exception as e:
        print(f"Error in agent_wrapper: {str(e)}")
        return gr.HTML(value="<p>Error processing your request. Please try again.</p>")

# Create and launch the Gradio interface
iface = gr.Interface(
    fn=agent_wrapper,
    inputs=gr.Textbox(
        label="Enter your visualization request",
        placeholder="e.g., Create a force-directed graph with 10 nodes"
    ),
    outputs=gr.HTML(),
    title="D3.js Visualization Generator",
    description="Describe the visualization you want to create (e.g., 'Create a force-directed graph with 10 nodes' or 'Make a bar chart showing monthly sales data')"
)

# Launch the interface
iface.launch() 
