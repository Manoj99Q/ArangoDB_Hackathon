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

   
 
    def get_node_by_id(self, collection: Annotated[str, "The collection name (e.g., 'Games' or 'Users')"], node_id: Annotated[str, "The ID of the node to retrieve"],tool_call_id:str):
        """Retrieve a node from ArangoDB by collection name and ID."""
        try:
            # Get the collection from ArangoDB
            collection = self.arango_graph.db.collection(collection)
            
            # Get the document by ID
            # If ID doesn't include collection prefix, add it
            if not ":" in node_id:
                node_id = f"{collection.name}:{node_id}"
            
            document = collection.get(node_id)
            
            if document is None:
                return Command(
                    update={
                        "messages": [ToolMessage(f"Node {node_id} not found in collection {collection.name}", tool_call_id=tool_call_id)]
                    }
                )
            
            return Command(
                update={
                    "data": {node_id: document},
                    "messages": [ToolMessage(f"Retrieved node {node_id} from collection {collection.name}. Node data: {document}", tool_call_id=tool_call_id)]
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
        def AQL_QueryWrapper(tool_call_id: Annotated[str, InjectedToolCallId], query: Annotated[str, "Natural Language Query"],name: Annotated[str, "Name of the variable to store the result"]):
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
        def NX_QueryWrapper(tool_call_id: Annotated[str, InjectedToolCallId], query: Annotated[str, "Natural Language Query"],name: Annotated[str, "Name of the variable to store the result"]):
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
                    node_id: Annotated[str, "The ID of the node to retrieve"]):
            """Retrieve a specific node from the graph by its collection name and ID.
            The collection should be either 'Games' or 'Users'.
            The node_id can be provided with or without the collection prefix."""
            return self.get_node_by_id(collection, node_id, tool_call_id)

        return [
            AQL_QueryWrapper,
            NX_QueryWrapper,
            get_node
        ]

    def _create_visualization_tools(self):
        """Tools for visualization stage"""

        @tool
        def graph_vis_Wrapper(tool_call_id: Annotated[str, InjectedToolCallId], query: str,state: Annotated[dict, InjectedState]):
            """Generate visual representations of graph data"""
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

    def RAG(self, state: GraphState):
        """RAG Step to generate a plan for the query and execute it to get the results"""

        print("\nProcessing Agent State:")
        pprint(state, indent=2, width=80)

        
        plan_prompt = """SYSTEM: You are a Graph Analysis Planner. Follow these steps:
                1. Analyze the user's query
                2. Create a step-by-step plan using available tools
                3. Execute tools sequentially using previous results
                4. Always break the query into smaller parts and use the tools accordingly
                5. Combine results for final answer

                Rules:
                → Create a plan before tool usage
                → Use one tool per step
                → Reference previous results where needed

                Graph Schema:
                {schema}
                Current Data at hand (This data will be updated as you use the tools):
                {data}
                Current Query: {user_query}

                YOUR PLAN:
                1."""

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
        new_ai_message = chain.invoke({**state,**{"schema":self.arango_graph.schema}})

        # 2) Return updated messages by *appending* new_ai_message to the state
        return {
            "messages": state["messages"] + [new_ai_message]
        }
    
    
    
    def Visualizer(self, state: GraphState):
        """Agent for visualization phase"""
        print("Visualizer State:")
        pprint(state, indent=2, width=80)
        

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a visualization expert. Create visual representations of the data. 
                         RULES:
                        - Generate Only One graph for the whole
                        - Come up with different visualizations based on the data and the query
                        - Example: a bar chart for number of hours played by each user or number users a game has been played by 
                         -Use graph type charts when you have edge type and node type  data"""),
            ("user", "Visualize for: {user_query}\n\n And the data at hand is: {data}\n\n"),
            ("placeholder", "{messages}")
        ])
        
        chain = prompt | self.llm.bind_tools(self.visualization_tools,parallel_tool_calls=False,)
        new_ai_message = chain.invoke(state)

        return {
            "messages": state["messages"] + [new_ai_message]
        }

    def generate_dynamic_graph_html(self,query,tool_call_id: str,state: GraphState):
        """Dynamically generates style and script tags for an interactive D3.js visualization.
        Needs clear instruction on what type of plot to generate"""
        sample_html = """<style>
                        body { font: 14px sans-serif; }
                        .link { stroke: #999; stroke-opacity: 0.6; }
                        .node { stroke: #fff; stroke-width: 1.5px; }
                    </style>
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
  </script>"""
        

        prompt = (
            "You are an expert web developer specializing in interactive D3.js visualizations. "
            "Generate only the <style> and <script> tags needed for a D3.js visualization based on the following specification: "
            + query + " "
            "For reference, here is an example showing both SVG and div-based visualizations: "
            + sample_html + " "
            "IMPORTANT: Your script must define an initViz() function that will be called after the DOM is ready. "
            "All D3.js initialization and manipulation should happen inside this function. "
            "The #visualization div will be empty when your code runs - your code needs to create any necessary elements. "
            "Only output the <style> and <script> tags without any other HTML elements or markdown formatting."
            "Here is the data at hand:" + str(state["data"])
        )
        try:
            print("Generating style and script code for visualization")
            # Get response from ChatOpenAI
            response = self.claude_llm.invoke(prompt)
            # Extract content from the response
            code = response.content
            # Remove markdown formatting if present
            if code.startswith("```html"):
                code = code[7:-3]  # Remove ```html and ``` markers
            elif code.startswith("```"):
                code = code[3:-3]  # Remove ``` markers
                
            # Print the generated code for debugging
            print("\nGenerated Style and Script Code:")
            print("=" * 50)

            print("=" * 50)
            
            # Wrap the style and script tags in a basic HTML structure
            html_code = f"""<!DOCTYPE html>
            <html lang="en">
            <head>
            <meta charset="UTF-8">
            <title>D3.js Visualization</title>
            <script src="https://d3js.org/d3.v6.min.js"></script>
            {code}
            </head>
            <body>
            <div id="visualization">
            </div>
            <script>
                // Wait for DOM to be ready
                document.addEventListener('DOMContentLoaded', function() {{
                    // Initialize D3 visualization after DOM is ready
                    if (typeof initViz === 'function') {{
                        initViz();
                    }}
                }});
            </script>
            </body>
            </html>"""
             # Print the generated code for debugging
            print("\nGenerated Style and Script Code:")
            print("=" * 50)
            print(html_code)
            print("=" * 50)

            encoded_html = base64.b64encode(html_code.encode('utf-8')).decode('utf-8')
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
            return "<p>Error generating visualization. Please try again.</p>"

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

 
