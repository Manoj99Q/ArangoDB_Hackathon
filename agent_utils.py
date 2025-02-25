from typing import List, TypedDict,Annotated,Any
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain_core.tools import BaseTool,tool
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.types import Command
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
import networkx as nx
import re
import base64
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
from langgraph.prebuilt import InjectedState
from pprint import pprint
import gradio as gr
import json
# Load environment variables
load_dotenv()

def add_data(old_data: list[dict[str, Any]], new_data: list[dict[str, Any]] | dict[str, Any]) -> list[dict[str, Any]]:
    """Reducer to append new data to existing data list"""
    if old_data is None:
        old_data = []
    if isinstance(new_data, dict):
        return [*old_data, new_data]
    return [*old_data, *new_data]

# Add State definition at the top with other imports
class GraphState(TypedDict):
    """Represents the state of our graph workflow."""
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str  # To track the current query being processed
    data: Annotated[list[dict[str, Any]], add_data]
    RAG_reply: str
    iframe_html: any
    # graph_schema: dict[str, Any]
    

class GraphAgent:
    def __init__(self, arango_graph=None, networkx_graph=None):
        """
        Initialize the Graph Agent with optional graph instances.
        
        Args:
            arango_graph: ArangoDB graph instance
            networkx_graph: NetworkX graph instance
        """
        self.arango_graph = arango_graph
        self.networkx_graph = networkx_graph
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
        self.claude_llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0,
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
            max_tokens=8192
        )

        
        # Create separate tool sets
        self.RAG_tools = self._create_RAG_tools()
        self.visualization_tools = self._create_visualization_tools()
        
        # Build the graph
        self.agent = self._create_workflow()

        display(Image(self.agent.get_graph(xray=True).draw_mermaid_png()))


    def text_to_aql_to_text(self, query: str,tool_call_id: str, var_name: str):
        """This tool is available to invoke the
        ArangoGraphQAChain object, which enables you to
        translate a Natural Language Query into AQL, execute
        the query, and translate the result back into Natural Language.
        """

        

        chain = ArangoGraphQAChain.from_llm(
            llm=self.llm,
            graph=self.arango_graph,
            verbose=True,
            allow_dangerous_requests=True,
            return_aql_query=True,
            return_aql_result=True,
            aql_examples = """
            #Find the game that has been played by the most players
            WITH Games, plays
            FOR game IN Games 
            LET playerCount = (
                FOR play IN plays 
                FILTER play._to == game._id 
                COLLECT WITH COUNT INTO length 
                RETURN length
            )[0] 
            SORT playerCount DESC 
            LIMIT 1 
            RETURN {game, playerCount: playerCount}
            """

        )
        
        result = chain.invoke(query+ " \n  Important: if returning a node always return the node object with all the properties")

        print("print AQL tool result")
        print(result)
        # return str(result["result"])

        return Command(
            update={
                "data": {var_name: result["aql_result"]},
                "messages": [ToolMessage(str(result["result"]), tool_call_id=tool_call_id)]
            }
        )

    def text_to_nx_algorithm_to_text(self, query: str,tool_call_id: str,name: str):
        """This tool executes NetworkX algorithms based on natural language queries."""
        if self.networkx_graph is None:
            return "Error: NetworkX graph is not initialized"
            
        print("Netwrokx Query received:")
        print(query)
        print("-"*10)
        print("1) Generating NetworkX code")
        text_to_nx = self.llm.invoke(f"""
        I have a NetworkX Graph called `G_adb`. It is a undirected weighted graph. It has the following schema: {self.arango_graph.schema}

        I have the following graph analysis query: {query}.

        Generate the Python Code required to answer the query using the `G_adb` object.
        Be very precise on the NetworkX algorithm you select to answer this query.
        Think step by step.

        Only assume that networkx is installed, and other base python dependencies.
        Always set the last variable as `FINAL_RESULT`, which represents the answer to the original query.
        Only provide python code that I can directly execute via `exec()`. Do not provide any instructions.
        Make sure that `FINAL_RESULT` stores all the information for example a list of nodes, edges, etc.
        Make sure that `FINAL_RESULT` contains not just the ID but the actual node/edge object with all the properties.

        Example:
        Question:Perform PageRank analysis on the SteamGraph to identify the most influential game based on user-game interactions
        # Assuming G_adb is already defined and is a NetworkX Graph object
        import networkx as nx

        # Perform PageRank analysis
        pagerank_scores = nx.pagerank(G_adb, weight='weight')

        # Find the most influential game based on PageRank scores
        # Since the graph is undirected, we need to filter out the game nodes
        game_nodes = [node for node, data in G_adb.nodes(data=True) if data.get('type') == 'Games']

        # Get the game with the highest PageRank score
        most_influential_game = max(game_nodes, key=lambda node: pagerank_scores.get(node, 0))

        # Retrieve the node data for the most influential game
        FINAL_RESULT = G_adb.nodes[most_influential_game]
        Your code:
        """).content

        text_to_nx_cleaned = re.sub(r"^```python\n|```$", "", text_to_nx, flags=re.MULTILINE).strip()
        
        print('-'*10)
        print(text_to_nx_cleaned)
        print('-'*10)

        print("\n2) Executing NetworkX code")
        try:
            context = {"G_adb": self.networkx_graph, "nx": nx}
            exec(text_to_nx_cleaned, context)
            FINAL_RESULT = context["FINAL_RESULT"]

        except Exception as e:
            print(f"EXEC ERROR: {e}")
            return f"EXEC ERROR: {e}"

        print('-'*10)
        print(f"FINAL_RESULT: {FINAL_RESULT}")
        print('-'*10)

        print("3) Formulating final answer")
        response = self.llm.invoke(f"""
        I have a NetworkX Graph called `G_adb`. It has the following schema: {self.arango_graph.schema}
        I have the following graph analysis query: {query}.
        I have executed the following python code to help me answer my query:
        ---
        {text_to_nx_cleaned}
        ---
        The `FINAL_RESULT` variable is set to: {FINAL_RESULT}.
        Based on my original Query and FINAL_RESULT, generate a short and concise response.
        """).content

        # return response
    
        return Command(
            update={
                "data": {name: FINAL_RESULT},
                "messages": [ToolMessage(response,tool_call_id=tool_call_id)]
            }
        )

   
 
    def get_node_by_attribute(self, collection: Annotated[str, "The collection name (e.g., 'Games' or 'Users')"], 
                            attribute: Annotated[str, "The attribute name to search by (e.g., '_id', 'GameName', etc.)"],
                            value: Annotated[str, "The value to search for"],
                            tool_call_id: str):
        """Retrieve a node from ArangoDB by collection name and any attribute."""
        try:
            # Get the collection from ArangoDB
            collection = self.arango_graph.db.collection(collection)
            
            # If searching by _id and it doesn't include collection prefix, add it
            if attribute == '_id' and not ":" in value:
                value = f"{collection.name}:{value}"
            
            # Build and execute AQL query to find document by attribute
            aql = f"""
                FOR doc IN {collection.name}
                    FILTER doc.{attribute} == @value
                    LIMIT 1
                    RETURN doc
            """
            cursor = self.arango_graph.db.aql.execute(aql, bind_vars={'value': value})
            
            # Get the first (and should be only) document
            document = next(cursor, None)
            
            if document is None:
                return Command(
                    update={
                        "messages": [ToolMessage(f"No node found in collection {collection.name} where {attribute} = {value}", tool_call_id=tool_call_id)]
                    }
                )
            
            return Command(
                update={
                    "data": {str(document['_id']): document},
                    "messages": [ToolMessage(f"Retrieved node from collection {collection.name} where {attribute} = {value}. Node data: {document}", tool_call_id=tool_call_id)]
                }
            )
        except Exception as e:
            return Command(
                update={
                    "messages": [ToolMessage(f"Error retrieving node: {str(e)}", tool_call_id=tool_call_id)]
                }
            )

    def _create_RAG_tools(self):
        """Tools for data processing stage"""
        @tool
        def AQL_QueryWrapper(tool_call_id: Annotated[str, InjectedToolCallId], query: Annotated[str, "Natural Language Query"],name: Annotated[str, "Name of the variable to store the result; dont use generic names"]):
            """This tool is available to invoke the
            ArangoGraphQAChain object, which enables you to
            translate a Natural Language Query into AQL, execute
            the query and get the result.
            """
            try:
                return self.text_to_aql_to_text(query,tool_call_id=tool_call_id, var_name=name)
            except Exception as e:
                return f"Error: {e}"
            
        @tool
        def NX_QueryWrapper(tool_call_id: Annotated[str, InjectedToolCallId], query: Annotated[str, "Natural Language Query"],name: Annotated[str, "Name of the variable to store the result; dont use generic names"]):
            """Analyze graph structure and patterns using NetworkX algorithms.
                Best for:
                - Finding shortest paths
                - Calculating centrality
                - Detecting communities
                - Complex network analysis"""
            try:
                return self.text_to_nx_algorithm_to_text(query,tool_call_id=tool_call_id,name=name)
            except Exception as e:
                return f"Error: {e}"

        @tool
        def get_node(tool_call_id: Annotated[str, InjectedToolCallId], 
                    collection: Annotated[str, "The collection name (e.g., 'Games' or 'Users')"],
                    attribute: Annotated[str, "The attribute name to search by (e.g., '_id', 'GameName', etc.)"],
                    value: Annotated[str, "The value to search for"]):
            """Retrieve a specific node from the graph by its collection name and any attribute.
            The collection should be either 'Games' or 'Users'.
            The attribute can be any field in the document (e.g., '_id', 'GameName', etc.).
            If searching by _id, the value can be provided with or without the collection prefix."""
            return self.get_node_by_attribute(collection, attribute, value, tool_call_id)

        return [
            AQL_QueryWrapper,
            NX_QueryWrapper,
            get_node
        ]

    def _create_visualization_tools(self):
        """Tools for visualization stage"""

        @tool
        def graph_vis_Wrapper(tool_call_id: Annotated[str, InjectedToolCallId], query: Annotated[str, "The detailed instructions on which visulation to generate"],state: Annotated[dict, InjectedState]):
            """Generate visual representations of data based on the instructions provided
            This tool has access to the data at hand and the current query. So no need to pass it again.
            """
            return self.generate_dynamic_graph_html(query,tool_call_id=tool_call_id,state=state)
        
        return [
            graph_vis_Wrapper
            # Add other visualization tools as needed
        ]

    def should_continue_after_RAG(self,state: GraphState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "RAG Tools"
        else:
            return "Visualizer"
        
    def should_continue_after_Vis(self, state: GraphState):
        messages = state["messages"]
        last_message = messages[-1]

        if last_message.tool_calls:
            return "Vis Tools"
        return END

    def _create_workflow(self):
        # Define the graph structure
        builder = StateGraph(GraphState)
        
        # Add nodes
        builder.add_node("RAG", self.RAG)
        builder.add_node("RAG Tools", ToolNode(self.RAG_tools))
        builder.add_node("Visualizer", self.Visualizer)
        builder.add_node("Vis Tools", ToolNode(self.visualization_tools))
        builder.add_edge("RAG Tools","RAG")
        builder.add_edge("Vis Tools","Visualizer")
        # Set up edges
        builder.set_entry_point("RAG")
        
        # Update conditional edges to handle message objects properly
        builder.add_conditional_edges(
            "RAG",
            self.should_continue_after_RAG,
            ["RAG Tools", "Visualizer"]

        )
        
        builder.add_conditional_edges(
            "Visualizer",
            self.should_continue_after_Vis  ,
            ["Vis Tools", END]
        )

        return builder.compile()

    def _has_tool_calls(self, state: GraphState) -> bool:
        """Check if the last message has tool calls"""
        if not state.get("messages"):
            return False
        last_msg = state["messages"][-1]
        return hasattr(last_msg, 'tool_calls') and bool(last_msg.tool_calls)

    def create_data_preview(self, data, max_items=2, current_depth=0, max_depth=5):
        """Create a compact preview of data structures for prompts"""
        if current_depth > max_depth:
            return "..."
        
        # Handle NetworkX node data (tuple format)
        if isinstance(data, tuple) and len(data) == 2:
            node_id, attributes = data
            return {
                "node_id": node_id,
                "attributes": self.create_data_preview(attributes, max_items, current_depth+1, max_depth)
            }
            
        # Handle NetworkX NodeAttrDict
        if hasattr(data, 'items'):
            return {str(k): self.create_data_preview(v, max_items, current_depth+1, max_depth) 
                    for k, v in data.items()}

        if isinstance(data, list):
            if not data:
                return []
            sample = data[:min(max_items, len(data))]
            preview = [self.create_data_preview(item, max_items, current_depth+1, max_depth) for item in sample]
            if len(data) > max_items:
                preview.append(f"... ({len(data) - max_items} more items)")
            return preview
            
        elif isinstance(data, dict):
            return {str(k): self.create_data_preview(v, max_items, current_depth+1, max_depth) 
                    for k, v in data.items()}
                    
        # Handle other non-serializable types
        try:
            json.dumps(data)
            return data
        except TypeError:
            return str(data)

    def RAG(self, state: GraphState):
        """RAG Step to generate a plan for the query and execute it to get the results"""

        print("\nProcessing Agent State:")
        pprint(state, indent=2, width=80)

        # Create a preview for the data in the prompt
        data_preview = [self.create_data_preview(item) for item in state.get("data", [])]
        
        plan_prompt = """SYSTEM: You are a Graph Analysis Planner. Follow these steps:
                1. Analyze the user's query for batch processing opportunities
                2. Create optimized steps using available tools
                3. Prefer single comprehensive queries over multiple small ones
                4. Use aggregation and grouping where possible
                5. Combine results for final answer

                Rules:
                → Look for ways to handle multiple items in one tool call
                → Use summary statistics (sums, counts) in queries
                → Retrieve related data in single queries when possible
                → Handle top N results within the same query

               
                Graph Schema:
                {schema}
                Current Data at hand:
                {data_preview}
                Current Query: {user_query}

                YOUR PLAN:
                1."""

        # Log the prompt for verification
        filled_prompt = plan_prompt.format(
            schema=self.arango_graph.schema,
            data_preview=json.dumps(data_preview, indent=2),
            user_query=state["user_query"]
        )
        print("\n==== RAG PROMPT ====")
        print(filled_prompt)
        print("==== END RAG PROMPT ====\n")

        prompt = ChatPromptTemplate.from_messages([
            ("system", plan_prompt),
            ("placeholder", "{messages}"),
        ])

        chain = (
            prompt 
            | self.llm.bind_tools(
                self.RAG_tools,
                parallel_tool_calls=False,
            )
        )
        
        # 1) Invoke the chain on the user query
        new_ai_message = chain.invoke({
            **state,
            "schema": self.arango_graph.schema,
            "data_preview": json.dumps(data_preview, indent=2),
            "user_query": state["user_query"]
        })

        # 2) Return updated messages by *appending* new_ai_message to the state
        return {
            "messages": state["messages"] + [new_ai_message]
        }
    
    
    
    def Visualizer(self, state: GraphState):
        """Agent for visualization phase"""
        print("Visualizer State:")
        pprint(state, indent=2, width=80)
        
        # Create a preview for the data in the prompt
        data_preview = [self.create_data_preview(item) for item in state.get("data", [])]
        data_preview_json = json.dumps(data_preview, indent=2)
        
        system_prompt = """You are a visualization expert. Create visual representations of the data. 
                         RULES:
                        - Generate Only One graph for the whole
                        - Come up with different visualizations based on the data and the query
                        - Example: a bar chart for number of hours played by each user or number users a game has been played by 
                         -Use graph type charts when you have edge type and node type  data"""
        
        # Don't embed JSON directly in the prompt to avoid template variable confusion
        user_prompt = f"Visualize for: {state['user_query']}"
        
        # Log the prompt for verification
        print("\n==== VISUALIZER PROMPT ====")
        print("SYSTEM: " + system_prompt)
        print("\nUSER: " + user_prompt)
        print("DATA: " + data_preview_json)
        print("==== END VISUALIZER PROMPT ====\n")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", f"{user_prompt}\n\nAnd the data at hand is: {{data_json}}"),
            ("placeholder", "{messages}")
        ])
        
        chain = prompt | self.llm.bind_tools(self.visualization_tools, parallel_tool_calls=False)
        
        # Pass data_json as a separate variable to avoid template parsing issues
        new_ai_message = chain.invoke({
            **state,
            "data_json": data_preview_json  # Pass as pre-formatted JSON string
        })

        return {
            "messages": state["messages"] + [new_ai_message]
        }

    def generate_dynamic_graph_html(self, query, tool_call_id: str, state: GraphState):
        """Dynamically generates code for an interactive D3.js visualization.
        Needs clear instruction on what type of plot to generate"""
        
        # Serialize the data to JSON for embedding in the HTML
        import json
        try:
            json_data = json.dumps(state["data"])
        except TypeError as e:
            print(f"Warning: JSON serialization issue: {e}")
            print("Attempting to create serializable version of the data...")
            # Create a serializable version if needed
            safe_data = self.create_serializable_data(state["data"])
            json_data = json.dumps(safe_data)
        
        # Create a preview of the data for the prompt using the class method
        data_preview = [self.create_data_preview(item) for item in state["data"]]
        
        # HTML header template with D3.js import, minimal styling, and embedded data
        html_header = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>D3.js Visualization</title>
  <script src="https://d3js.org/d3.v6.min.js"></script>
  <style>
    body {{ font: 14px sans-serif; margin: 20px; }}
    svg {{ border: 1px solid #ccc; }}
    /* The visualization-specific CSS will be provided by the model */
  </style>
  <script>
    // Data from the application state
    const data = {json_data};
  </script>
</head>
<body>
  <h2 id="visualization-title">Data Visualization</h2>
  <div id="visualization-container">
    <svg width="600" height="400"></svg>
  </div>
"""

        # HTML footer template
        html_footer = """
</body>
</html>"""

        # Example D3.js code to show the model what we expect
        example_d3_code = """
  <script>
    // The data is already available as a global 'data' variable
    // You can access it directly with the 'data' variable
    console.log("Data available:", data);
    
    // Select the SVG element and set dimensions
    const svg = d3.select("svg");
    const width = +svg.attr("width");
    const height = +svg.attr("height");

    // Add necessary CSS styles for this specific visualization
    const style = document.createElement('style');
    style.textContent = `
      .link { 
        stroke: #999; 
        stroke-opacity: 0.6; 
      }
      .node { 
        stroke: #fff; 
        stroke-width: 1.5px; 
      }
    `;
    document.head.appendChild(style);

    // Example visualization code for a force-directed graph
    // You would adapt this to your specific data and requirements
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

    // Create a simulation with forces
    const simulation = d3.forceSimulation(nodes)
                       .force("link", d3.forceLink(links).id(d => d.id).distance(100))
                       .force("charge", d3.forceManyBody().strength(-300))
                       .force("center", d3.forceCenter(width / 2, height / 2));

    // Create and style the links
    const link = svg.append("g")
                  .attr("class", "links")
                  .selectAll("line")
                  .data(links)
                  .enter().append("line")
                  .attr("class", "link")
                  .attr("stroke-width", 2);

    // Create and style the nodes
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

    // Add a tooltip to each node
    node.append("title")
        .text(d => d.id);

    // Update positions on each tick of the simulation
    simulation.on("tick", () => {
      link.attr("x1", d => d.source.x)
          .attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x)
          .attr("y2", d => d.target.y);

      node.attr("cx", d => d.x)
          .attr("cy", d => d.y);
    });

    // Define drag event functions
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
"""

        # Create a more focused prompt that asks only for the D3.js code
        visualization_prompt = (
            "You are an expert D3.js developer specializing in interactive visualizations. "
            "I already have the HTML boilerplate code with D3.js imported and an SVG element set up. "
            "Generate ONLY the JavaScript code within <script> tags to create an interactive visualization based on the following specification: "
            f"{query}\n\n"
            "IMPORTANT: The data is already available as a global JavaScript variable named 'data'. "
            "Here's a preview of the structure (actual data may contain more elements): \n" + 
            json.dumps(data_preview, indent=2) + "\n\n"
            "The HTML structure is already set up with:\n"
            "- D3.js v6 imported\n"
            "- Basic body styling and an SVG element with width 600px and height 400px\n"
            "- The data already loaded and available as a global 'data' variable\n"
            "- NO visualization-specific CSS - YOU MUST include any necessary CSS for your visualization\n\n"
            "IMPORTANT: You must include any visualization-specific CSS within your script by either:\n"
            "1. Adding a <style> element to the document head, or\n"
            "2. Setting inline styles on the SVG elements\n\n"
            "For reference, here is an example of D3.js code with CSS handling for a Force-Directed Graph: "
            f"{example_d3_code}\n\n"
            "Output ONLY the <script> element with your visualization code. Do not include DOCTYPE, HTML, head, or body tags."
        )
        
        # Log the prompt for verification
        print("\n==== VISUALIZATION GENERATION PROMPT ====")
        print(visualization_prompt)
        print("==== END VISUALIZATION GENERATION PROMPT ====\n")
        
        try:
            print("Generating visualization code...")
            
            # Get response from Claude
            response = self.claude_llm.invoke(visualization_prompt)
            visualization_code = response.content
            
            # Extract just the script content if it's wrapped in markdown or other tags
            if "```" in visualization_code:
                # Extract content from code blocks
                import re
                script_blocks = re.findall(r'```(?:html|javascript)?(.*?)```', visualization_code, re.DOTALL)
                if script_blocks:
                    visualization_code = '\n'.join(script_blocks)
            
            # Ensure the code has script tags
            if "<script>" not in visualization_code:
                visualization_code = f"<script>\n{visualization_code}\n</script>"
            
            # Combine the template parts with the generated visualization code
            complete_html = html_header + visualization_code + html_footer
            
            # Print the generated code for debugging
            print("\nGenerated Visualization Code:")
            print("=" * 50)
            print(complete_html)
            print("=" * 50)

            # Encode the HTML and create iframe
            encoded_html = base64.b64encode(complete_html.encode('utf-8')).decode('utf-8')
            data_url = f"data:text/html;base64,{encoded_html}"
            iframe_html = f'<iframe src="{data_url}" width="620" height="420" frameborder="0"></iframe>'
            
            return Command(
                update={
                    "iframe_html": gr.HTML(value=iframe_html),
                    "messages": [ToolMessage("Visualization Done", tool_call_id=tool_call_id)]
                }
            )
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            return Command(
                update={
                    "iframe_html": gr.HTML(value="<p>Error generating visualization. Please try again.</p>"),
                    "messages": [ToolMessage(f"Error generating visualization: {str(e)}", tool_call_id=tool_call_id)]
                }
            )

    def create_serializable_data(self, data):
        """Create a fully JSON-serializable version of the data"""
        if isinstance(data, list):
            return [self.create_serializable_data(item) for item in data]
        elif isinstance(data, dict):
            return {str(k): self.create_serializable_data(v) for k, v in data.items()}
        else:
            # Handle non-serializable data
            try:
                # Test if it's directly serializable
                json.dumps(data)
                return data
            except (TypeError, OverflowError):
                # Convert to string if not serializable
                return str(data)

    def query_graph(self, query: str):
        """Execute a graph query using the appropriate tool."""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            # "vis_messages": [HumanMessage(content=query)],
            "user_query": query,
            "data": [],  # Initialize empty data list
            "iframe_html": "",  # Initialize empty iframe
        }
        
        # Use invoke() instead of stream() to get final state directly
        final_state = self.agent.invoke(initial_state)
        
        print("Final Answer:")
        pprint(final_state, indent=2, width=80)
        
        # Return structured response
        return {
            "html_code": final_state["iframe_html"],
            "reply": final_state["messages"][-1].content
        }

 
