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
from ArangoGraphDirectChain import ArangoGraphDirectChain
# Load environment variables
load_dotenv()

def add_data(old_data: dict[str, Any], new_data: dict[str, Any]) -> dict[str, Any]:
    """Reducer to merge new data into existing data dictionary"""
    if old_data is None:
        old_data = {}
    return {**old_data, **new_data}

# Add State definition at the top with other imports
class GraphState(TypedDict):
    """Represents the state of our graph workflow."""
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str  # To track the current query being processed
    data: Annotated[dict[str, Any], add_data]
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


    def text_to_aql_to_text(self, query: str, tool_call_id: str, var_name: str, state):
        """This tool is available to invoke the
        ArangoGraphQAChain object, which enables you to
        translate a Natural Language Query into AQL, execute
        the query, and translate the result back into Natural Language.
        
        The function can now use injected data from the state variables in the query execution.
        """
        
        chain = ArangoGraphDirectChain.from_llm(
            llm=self.llm,
            graph=self.arango_graph,
            verbose=True,
            allow_dangerous_requests=True,
            return_aql_query=True,
            return_aql_result=True,
            return_direct = True,
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
        print("AQL tool query")
        print(query)


  
        result = chain.invoke(query + " \n  Important: if returning a node always return the node object with all the properties and dont perform any llm processing on the result.I only need the direct aql result")
        reply = "Query executed successfully and updated the state with the result"
 
        
        print("print AQL tool result")
        print(result)
        
        return Command(
            update={
                "data": {var_name: result["aql_result"]},
                "messages": [ToolMessage(reply, tool_call_id=tool_call_id)]
            }
        )

    def text_to_nx_algorithm(self, query: str, tool_call_id: str, name: str, state=None):
        """This tool executes NetworkX algorithms based on natural language queries."""
        if self.networkx_graph is None:
            return "Error: NetworkX graph is not initialized"
        
        # Get graph size info
        num_nodes = len(self.networkx_graph.nodes())
        num_edges = len(self.networkx_graph.edges())
        
        print("NetworkX Query received:")
        print(query)
        print("-"*10)
        print("1) Generating NetworkX code")
        
        # Create data preview from state if available
        data_preview = {}
        if state and "data" in state:
            data_preview = self.create_data_preview(state["data"])
            print("State data available:")
            print(json.dumps(data_preview, indent=2))
        
        # Create parts of the prompt separately to handle the braces properly
        example_code = '''
            ### EXAMPLE 1: Using existing state data ###
            import networkx as nx
            
            # Example of accessing state data directly
            # Assume user_communities variable contains community data
            communities_data = user_communities
            
            # Process the communities data
            results = []
            for community_idx, community_data in enumerate(communities_data["community_data"]):
                # Process each community
                community_size = community_data["community_size"]
                
                # Calculate something useful about this community
                result = {
                    "community_id": community_idx,
                    "community_size": community_size,
                    "some_metric": community_size * 2  # Example calculation
                }
                results.append(result)
                
            FINAL_RESULT = {
                'community_analysis': results,
                'source': 'Used existing community data from state'
            }
            
            ### EXAMPLE 2: Computing from scratch ###
            import networkx as nx
            
            # Use fast Louvain method for community detection
            communities = nx.community.louvain_communities(G_adb)
            
            # Find the largest community
            largest_community = max(communities, key=len)
            
            # Get relevant data
            largest_community_data = [G_adb.nodes[node] for node in largest_community]
            
            # For user similarity (using a subset of users for performance)
            user_nodes = [node for node in G_adb.nodes() if node.startswith('Users/')]
            sample_users = user_nodes[:100]  # Limit to 100 users for performance
            
            # Map users to their games
            user_to_games = {}
            for user in sample_users:
                # Get neighbors (connected nodes) of this user
                neighbors = list(G_adb.neighbors(user))
                # Filter to only keep game nodes
                games = [n for n in neighbors if n.startswith('Games/')]
                user_to_games[user] = set(games)
            
            # Calculate Jaccard similarity between users
            user_similarities = []
            user_ids = list(user_to_games.keys())
            for i in range(len(user_ids)):
                for j in range(i+1, len(user_ids)):
                    user1 = user_ids[i]
                    user2 = user_ids[j]
                    # Calculate Jaccard similarity
                    games1 = user_to_games[user1]
                    games2 = user_to_games[user2]
                    intersection = len(games1.intersection(games2))
                    union = len(games1.union(games2))
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > 0.5:  # Only keep significant similarities
                        user_similarities.append({
                            'user1': user1,
                            'user2': user2,
                            'similarity': similarity,
                            'common_games_count': intersection
                        })
            
            FINAL_RESULT = {
                'num_communities': len(communities),
                'largest_community_size': len(largest_community),
                'largest_community_sample': largest_community_data[:10],
                'similar_users': sorted(user_similarities, key=lambda x: x['similarity'], reverse=True)[:20]
            }
            '''
        
        # Build the prompt with proper escaping of schema
        schema_str = str(self.arango_graph.schema).replace("{", "{{").replace("}", "}}")
        
        # Include data preview in the prompt
        data_section = ""
        if data_preview:
            data_section = f"""
            
            AVAILABLE DATA VARIABLES:
            I have the following data available in state that you can use directly:
            {json.dumps(data_preview, indent=2)}
            
            IMPORTANT INSTRUCTIONS FOR USING STATE DATA:
            1. DO NOT copy and paste the preview data into your code. The preview is truncated and contains placeholders like "... (N more items)".
            2. Instead, use the variable names directly to access the COMPLETE data.
            3. For example, if 'user_communities' is in the state, simply use the 'user_communities' variable in your code to access the full data.
            4. Check if the variable exists before using it with: if 'variable_name' in locals()
            5. When working with large data structures, use loops and direct access rather than hardcoding values.
            """
        
        prompt = f"""
        I have a NetworkX Graph called `G_adb`. It is an undirected weighted graph with {num_nodes} nodes and {num_edges} edges. It has the following schema: {schema_str}

        I have the following graph analysis query: {query}.{data_section}

        Generate the Python Code required to answer the query using the `G_adb` object.
        
        IMPORTANT PERFORMANCE CONSIDERATIONS:
        - This is a LARGE graph with over 14,000 nodes - algorithm selection is critical
        - For community detection, use ONLY fast algorithms like Louvain (nx.community.louvain_communities) or Label Propagation (nx.community.label_propagation_communities)
        - NEVER use Girvan-Newman algorithm as it's too slow for this graph
        - For centrality, prefer approximation methods where available
        - Limit any expensive computations to relevant subgraphs when possible
        - Consider node/edge sampling for visualization or complex algorithms
        - If there are relevant variables already in the state data, USE THEM DIRECTLY instead of recomputing

        
        Be very precise on the NetworkX algorithm you select to answer this query.
        Think step by step about both correctness AND performance.

        Only assume that networkx is installed, and other base python dependencies. Do not assume anything else or import anything else.
        Always set the last variable as `FINAL_RESULT`, which represents the answer to the original query.
        Only provide python code that I can directly execute via `exec()`. Do not provide any instructions.
        Make sure that `FINAL_RESULT` stores all the information for example a list of nodes, edges, etc.
        Make sure that `FINAL_RESULT` contains not just the ID but the actual node/edge object with all the properties.
        Always try to return relevant data to the query that can help in generating a good visualization.

        Example of GOOD code for using state data and community analysis:
        ```python
            {example_code}
        ```
        
        Your code:
        """
        
        try:
            text_to_nx = self.llm.invoke(prompt).content
        except Exception as e:
            print(f"Error in LLM invocation: {e}")
            return Command(
                update={
                    "messages": [ToolMessage(f"Error generating NetworkX code: {str(e)}", tool_call_id=tool_call_id)]
                }
            )

        text_to_nx_cleaned = re.sub(r"^```python\n|```$", "", text_to_nx, flags=re.MULTILINE).strip()
        
        print('-'*10)
        print(text_to_nx_cleaned)
        print('-'*10)

        print("\n2) Executing NetworkX code")
        try:
            # Create execution context with both the graph and any state data variables
            context = {"G_adb": self.networkx_graph, "nx": nx}
            
            # Add state data variables to context if available
            if state and "data" in state:
                print("Adding state data variables to execution context:")
                for var_name, var_value in state["data"].items():
                    print(f"  - {var_name}")
                    context[var_name] = var_value
            
            exec(text_to_nx_cleaned, context)
            FINAL_RESULT = context["FINAL_RESULT"]

        except Exception as e:
            print(f"EXEC ERROR: {e}")
            return Command(
                update={
                    "messages": [ToolMessage(f"Error executing NetworkX code: {str(e)}", tool_call_id=tool_call_id)]
                }
            )

        print('-'*10)
        print(f"FINAL_RESULT: {FINAL_RESULT}")
        print('-'*10)

        return Command(
            update={
                "data": {name: FINAL_RESULT},
                "messages": [ToolMessage("NetworkX query executed successfully and updated the state with the result",tool_call_id=tool_call_id)]
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
        def AQL_QueryWrapper(tool_call_id: Annotated[str, InjectedToolCallId], 
                            query: Annotated[str, "Natural Language Query"],
                            name: Annotated[str, "Name of the variable to store the result; dont use generic names"],
                            state: Annotated[dict, InjectedState] = None):
            """This tool is available to invoke the
            ArangoGraphQAChain object, which enables you to
            translate a Natural Language Query into AQL, execute
            the query and get the result.
            
            Only pass natural language detailed instructions on what to do.
            """
            try:
                return self.text_to_aql_to_text(query, tool_call_id=tool_call_id, var_name=name, state=state)
            except Exception as e:
                return f"Error: {e}"
            
        @tool
        def NX_QueryWrapper(tool_call_id: Annotated[str, InjectedToolCallId], 
                           query: Annotated[str, "Natural Language Query"],
                           name: Annotated[str, "Name of the variable to store the result; dont use generic names"],
                           state: Annotated[dict, InjectedState] = None):
            """Analyze graph structure and patterns using NetworkX algorithms.
                Best for:
                - Finding shortest paths
                - Calculating centrality
                - Detecting communities
                - Complex network analysis"""
            try:
                return self.text_to_nx_algorithm(query, tool_call_id=tool_call_id, name=name, state=state)
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
            return "RAG_Summarizer"
        
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
        builder.add_node("RAG_Summarizer", self.RAG_Summarizer)  # Add the new intermediate node
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
            ["RAG Tools", "RAG_Summarizer"]  # Changed Visualizer to RAG_Summarizer
        )
        
        # Add a direct edge from RAG_Summarizer to Visualizer
        builder.add_edge("RAG_Summarizer", "Visualizer")
        
        builder.add_conditional_edges(
            "Visualizer",
            self.should_continue_after_Vis,
            ["Vis Tools", END]
        )

        return builder.compile()

    def _has_tool_calls(self, state: GraphState) -> bool:
        """Check if the last message has tool calls"""
        if not state.get("messages"):
            return False
        last_msg = state["messages"][-1]
        return hasattr(last_msg, 'tool_calls') and bool(last_msg.tool_calls)

    def create_data_preview(self, data, max_items=2, current_depth=0, max_depth=7):
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
        pprint(state["messages"], indent=2, width=80)

        # Create a preview for the data in the prompt
        data_preview = self.create_data_preview(state.get("data", {}))
        
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
                → Also dont bother about visulization in the plan. There is another agent for that after this step. But always try to return the data that can help in generating a good visualization.

               
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
    
    def RAG_Summarizer(self, state: GraphState):
        """Intermediate node that extracts the last message from RAG and updates the RAG_reply state variable."""
        
        print("\nRAG_Summarizer State:")
        pprint(state["messages"], indent=2, width=80)
        
        # Extract the last message from the RAG node
        messages = state["messages"]
        last_message = messages[-1]
        
        # Update the RAG_reply state variable with the content of the last message
        rag_reply = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        print("\nUpdating RAG_reply with:")
        print(rag_reply)
        
        # Return updated state with the RAG_reply field updated
        return {
            "RAG_reply": rag_reply
        }
    
    def Visualizer(self, state: GraphState):
        """Agent for visualization phase"""
        print("Visualizer State:")
        pprint(state["messages"], indent=2, width=80)
        
        # Create a preview for the data in the prompt
        data_preview = self.create_data_preview(state.get("data", {}))
        data_preview_json = json.dumps(data_preview, indent=2)
        
        system_prompt = """You are a visualization expert. Create visual representations of the data. 
                         RULES:
                        - You can make only one tool call for visualization
                        - Come up with different visualizations based on the data and the query
                        - Provide clear instructions to the tool on what to generate by considering the data at hand and the query
                        - Example: a bar chart for number of hours played by each user or number users a game has been played by 
                        - Example: Use graph type charts whenever you can based on the data at hand like a games and its users and the node size should be based on some metric"""
        
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
        data_preview = self.create_data_preview(state["data"])
        
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
                <svg width="1200" height="800"></svg>
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
        transition: r 0.2s, fill 0.2s;
      }
      .node:hover {
        fill: #ff7700;
        cursor: pointer;
      }
      .node-label {
        font-size: 12px;
        font-weight: bold;
        text-anchor: middle;
        pointer-events: none;
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

    // Create a simulation with forces - adjusted for better handling of many nodes
    // Using more balanced forces similar to the reference code
    const simulation = d3.forceSimulation(nodes)
                       .force("link", d3.forceLink(links).id(d => d.id).distance(30)) // Reduced distance
                       .force("charge", d3.forceManyBody().strength(-30)) // Much weaker repulsion
                       .force("x", d3.forceX(width / 2).strength(0.05)) // Gentle force toward center x
                       .force("y", d3.forceY(height / 2).strength(0.05)); // Gentle force toward center y

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
                      .on("end", dragended))
                  .on("mouseover", function(event, d) {
                    d3.select(this)
                      .attr("r", 15)
                      .attr("fill", "#ff7700");
                  })
                  .on("mouseout", function(event, d) {
                    d3.select(this)
                      .attr("r", 10)
                      .attr("fill", "steelblue");
                  });

    // Add a tooltip to each node
    node.append("title")
        .text(d => d.id);
        
    // Add text labels for nodes
    const labels = svg.append("g")
                  .attr("class", "labels")
                  .selectAll("text")
                  .data(nodes)
                  .enter().append("text")
                  .attr("class", "node-label")
                  .attr("dy", -15)  // Position above the node
                  .text(d => d.id);

    // Update positions on each tick of the simulation
    simulation.on("tick", () => {
      link.attr("x1", d => d.source.x)
          .attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x)
          .attr("y2", d => d.target.y);

      node.attr("cx", d => d.x)
          .attr("cy", d => d.y);
          
      labels.attr("x", d => d.x)
            .attr("y", d => d.y);
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
            "- The data already loaded and available as a global 'data' variable\n"
            "- NO visualization-specific CSS - YOU MUST include any necessary CSS for your visualization\n\n"
            "IMPORTANT: You must include any visualization-specific CSS within your script by either:\n"
            "1. Adding a <style> element to the document head, or\n"
            "2. Setting inline styles on the SVG elements\n\n"
            "For reference, here is an example of D3.js code with CSS handling for a Force-Directed Graph: "
            f"{example_d3_code}\n\n"
            "Output ONLY the <script> element with your visualization code. Do not include DOCTYPE, HTML, head, or body tags."
            "And if the task is simple you can amp it up with effects and animations only if you are confident without errors or else just keep it simple"
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
            "RAG_reply": "",
            "user_query": query,
            "data": {},  # Initialize empty data dictionary
            "iframe_html": "",  # Initialize empty iframe
        }
        
        # Use invoke() instead of stream() to get final state directly
        final_state = self.agent.invoke(initial_state)
        
        # Save the final state to a JSON file for logging
        self.save_state_to_json(final_state, query)
        
        # Return structured response
        return {
            "html_code": final_state["iframe_html"],
            "reply": final_state["messages"][-1].content,
            "rag_reply": final_state["RAG_reply"]  # Include the RAG_reply in the response
        }
        
    def save_state_to_json(self, state, query):
        """Save the state to a JSON file for logging purposes"""
        import os
        import json
        import datetime
        import hashlib
        
        # Create a logs directory if it doesn't exist
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        # Create a timestamp for the filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a short hash of the query for the filename
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        
        # Create the filename
        filename = f"{logs_dir}/state_{timestamp}_{query_hash}.json"
        
        # Convert state to a serializable format
        serializable_state = self.prepare_state_for_serialization(state)
        
        # Save the state to a JSON file
        with open(filename, "w") as f:
            json.dump(serializable_state, f, indent=2)
            
        print(f"State saved to {filename}")
        
    def prepare_state_for_serialization(self, state):
        """Prepare the state for serialization by converting non-serializable objects"""
        import copy
        
        # Make a deep copy to avoid modifying the original state
        state_copy = copy.deepcopy(state)
        
        # Convert messages to a serializable format
        if "messages" in state_copy:
            state_copy["messages"] = [self.message_to_dict(msg) for msg in state_copy["messages"]]
            
        # Convert data to a serializable format using existing method
        if "data" in state_copy:
            state_copy["data"] = self.create_serializable_data(state_copy["data"])
            
        # Remove any potentially problematic fields
        if "iframe_html" in state_copy and isinstance(state_copy["iframe_html"], object) and not isinstance(state_copy["iframe_html"], (str, int, float, bool, list, dict, type(None))):
            state_copy["iframe_html"] = str(state_copy["iframe_html"])
            
        return state_copy
        
    def message_to_dict(self, message):
        """Convert a message object to a dictionary"""
        result = {
            "type": message.__class__.__name__,
            "content": message.content
        }
        
        # Add additional fields if they exist
        if hasattr(message, "name") and message.name:
            result["name"] = message.name
            
        if hasattr(message, "tool_call_id") and message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
            
        if hasattr(message, "additional_kwargs") and message.additional_kwargs:
            # Make sure additional_kwargs is serializable
            result["additional_kwargs"] = self.create_serializable_data(message.additional_kwargs)
            
        return result

 
