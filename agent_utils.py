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
from utils import create_data_preview
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
    Has_Visualization: str
    

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


    def text_to_aql(self, query: str, tool_call_id: str, var_name: str, state):
        """This tool is available to invoke the
        ArangoGraphQAChain object, which enables you to
        translate a Natural Language Query into AQL, execute
        the query, and translate the result back into Natural Language.
        
        The function can now use injected data from the state variables in the query execution.
        """
        
        chain = ArangoGraphDirectChain.from_llm(
            llm=self.claude_llm,
            graph=self.arango_graph,
            verbose=True,
            allow_dangerous_requests=True,
            return_aql_query=True,
            return_aql_result=True,
            return_direct = True,
            data_state = state["data"],
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

            # Analyze a user's gaming patterns
            WITH Users, Games, plays

            // Get user's plays
            LET user_plays = (
                FOR play IN plays
                FILTER play._from == @target_user_id
                RETURN play
            )

            // Calculate total gaming stats
            LET total_games = LENGTH(user_plays)
            LET total_hours = SUM(user_plays[*].weight)

            // Categorize games by playtime without COLLECT
            LET short_games = (
                FOR play IN user_plays
                FILTER play.weight <= @categories.short
                RETURN play
            )

            LET medium_games = (
                FOR play IN user_plays
                FILTER play.weight > @categories.short AND play.weight <= @categories.medium
                RETURN play
            )

            LET long_games = (
                FOR play IN user_plays
                FILTER play.weight > @categories.medium AND play.weight <= @categories.long
                RETURN play
            )

            LET hardcore_games = (
                FOR play IN user_plays
                FILTER play.weight > @categories.long
                RETURN play
            )

            // Get top games
            LET top_games = (
                FOR play IN user_plays
                LET game = DOCUMENT(play._to)
                SORT play.weight DESC
                LIMIT @top_games_limit
                RETURN {
                    name: game.GameName,
                    hours: play.weight
                }
            )

            RETURN {
                user_id: @target_user_id,
                total_stats: {
                    games_played: total_games,
                    total_hours: total_hours,
                    avg_hours_per_game: total_hours / total_games
                },
                categories: [
                    {
                        category: "short",
                        game_count: LENGTH(short_games),
                        total_hours: SUM(short_games[*].weight),
                        percentage: (LENGTH(short_games) / total_games) * 100
                    },
                    {
                        category: "medium",
                        game_count: LENGTH(medium_games),
                        total_hours: SUM(medium_games[*].weight),
                        percentage: (LENGTH(medium_games) / total_games) * 100
                    },
                    {
                        category: "long",
                        game_count: LENGTH(long_games),
                        total_hours: SUM(long_games[*].weight),
                        percentage: (LENGTH(long_games) / total_games) * 100
                    },
                    {
                        category: "hardcore",
                        game_count: LENGTH(hardcore_games),
                        total_hours: SUM(hardcore_games[*].weight),
                        percentage: (LENGTH(hardcore_games) / total_games) * 100
                    }
                ],
                top_games: top_games
            }
    """

        )
        print("AQL tool query")
        print(query)


  
        result = chain.invoke(query + " \n  Important: if returning a node always return the node object with all the properties and dont perform any llm processing on the result.I only need the direct aql result")
        reply = "Query executed successfully and updated the state with the result"
 
        

        
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
            data_preview = create_data_preview(state["data"])
            # print("State data available:")
            # print(json.dumps(data_preview, indent=2))
        
        
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
            
           ### EXAMPLE 2: An community detection and analysis example ###
            import networkx as nx
            from collections import Counter

            # Use Label Propagation for community detection
            communities = nx.community.label_propagation_communities(G_adb)
            communities = list(communities)

            # Get game names for all game nodes
            game_names = {}
            games_query = "FOR g IN Games RETURN {id: g._id, name: g.GameName}"
            for game in G_adb.query(games_query):
                game_names[game['id']] = game.get('name', 'Unknown Game')

            # Analyze all communities
            community_stats = []
            for idx, community in enumerate(communities):
                # Split nodes into Users and Games
                user_nodes = [node for node in community if node.startswith('Users/')]
                
                # For games, get both ID and name
                game_nodes = []
                for node in community:
                    if node.startswith('Games/'):
                        game_nodes.append({
                            'id': node,
                            'name': game_names.get(node, 'Unknown Game')
                        })
                
                # Count node types (Users, Games, etc.)
                node_types = Counter(node.split('/')[0] for node in community)
                
                community_stats.append({
                    'community_id': idx,
                    'users': user_nodes, # ids of user nodes
                    'games': game_nodes, # ids and names of game nodes
                    'size': len(community),
                    'number_of_users': len(user_nodes),
                    'number_of_games': len(game_nodes),
                    'composition': dict(node_types)
                })

            print("Commity Stats Done")
            # Create a mapping of users to their communities
            user_community_map = {}
            for comm_idx, stats in enumerate(community_stats):
                for user_id in stats['users']:
                    user_community_map[user_id] = comm_idx

            # Make a single AQL query to get all play relationships
            plays_query = """
            FOR p IN plays
                RETURN {game: p._to, user: p._from}
            """
            all_plays = list(G_adb.query(plays_query))

            print("AQL query done")
            # Create a mapping of games to their users
            game_to_users = {}
            for play in all_plays:
                game_id = play['game']
                user_id = play['user']
                
                if game_id not in game_to_users:
                    game_to_users[game_id] = []
                
                game_to_users[game_id].append(user_id)

            # Identify bridge games - games that connect users from different communities
            bridge_games = []
            for game_id, users in game_to_users.items():
                if game_id not in game_names:
                    continue
                    
                game_name = game_names[game_id]
                
                # Get the communities of these users
                user_communities = set()
                for user_id in users:
                    if user_id in user_community_map:
                        user_communities.add(user_community_map[user_id])
                
                # If the game connects multiple communities, it's a bridge
                if len(user_communities) > 1:
                    bridge_games.append({
                        'game_id': game_id,
                        'game_name': game_name,
                        'connected_communities': list(user_communities),
                        'bridge_strength': len(user_communities),
                        'player_count': len(users)
                    })
            # Sort bridge games by bridge strength (number of communities connected)
            bridge_games.sort(key=lambda x: x['bridge_strength'], reverse=True)

            FINAL_RESULT = {
                'num_communities': len(communities),
                'all_communities': sorted(community_stats, key=lambda x: x['size'], reverse=True),
                "bridge_games": bridge_games
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
            print("Executing NetworkX code after adding state data variables to execution context:")
            
            try:
                exec(text_to_nx_cleaned, context)
                FINAL_RESULT = context["FINAL_RESULT"]
            except Exception as e:
                import traceback
                print("Full traceback:")
                traceback.print_exc()
                print(f"\nError occurred in NetworkX code: {str(e)}")
                raise  # Re-raise to be caught by outer try-except
            
            # Make sure the result is JSON serializable using the specialized NetworkX serializer
            FINAL_RESULT = self.create_networkx_serializable_data(FINAL_RESULT)
            print("Converted FINAL_RESULT to serializable format")

        except Exception as e:
            print(f"EXEC ERROR: {str(e)}")
            return Command(
                update={
                    "messages": [ToolMessage(f"Error executing NetworkX code: {str(e)}", tool_call_id=tool_call_id)]
                }
            )

        # print('-'*10)
        # print(f"FINAL_RESULT: {FINAL_RESULT}")
        # print('-'*10)

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
            translate a Natural Language question into AQL, execute
            the query and get the result.
            This tool has access to the data at hand. So no need to pass it again.
            Only pass natural language detailed instructions on what to do. Do not pass the query directly.
            """
            try:
                return self.text_to_aql(query, tool_call_id=tool_call_id, var_name=name, state=state)
            except Exception as e:
                return f"Error: {e}"
            
        @tool
        def NX_QueryWrapper(tool_call_id: Annotated[str, InjectedToolCallId], 
                           query: Annotated[str, "Natural Language Query"],
                           name: Annotated[str, "Name of the variable to store the result; dont use generic names"],
                           state: Annotated[dict, InjectedState] = None):
            """Analyze graph structure and patterns using NetworkX algorithms.
                Best for:
                - PageRank Algorithm
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
        
        @tool
        def format_limited_data_reply(tool_call_id: Annotated[str, InjectedToolCallId], 
                                     state: Annotated[dict, InjectedState]):
            """Format a text-based reply for limited data sets (less than ~10 values).
            This tool should be used when the data is too small to warrant a full visualization.
            It formats the data into a clear, readable text response based on the user's query.
            """
            return self.format_small_data_reply(tool_call_id=tool_call_id, state=state)
        
        return [
            graph_vis_Wrapper,
            format_limited_data_reply
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

    def RAG(self, state: GraphState):
        """RAG Step to generate a plan for the query and execute it to get the results"""

        # return{
        #     "messages": AIMessage(content="Done"),
        #     "RAG_reply": "Done"
        # }

        print("\nProcessing Agent State:")
        # pprint(state["messages"], indent=2, width=80)

        # Create a preview for the data in the prompt
        data_preview = create_data_preview(state.get("data", {}))

        print("\nData Preview:")
        pprint(data_preview, indent=2, width=80)
        
        plan_prompt = """SYSTEM: You are a Graph Analysis Agent. Follow these steps:
                ANALYSIS APPROACH:
                1. You can enhance the query to get more related data which can help in answering the user query better
                2. First examine the query complexity and data dependencies
                3. ALWAYS check existing state data before making new tool calls
                4. Choose the optimal strategy:
                   → For simple, direct queries: Use single comprehensive tool calls
                   → For complex analysis: Break down into logical steps
                   → For uncertain data: Start with exploratory queries, then refine
                5. Combine results and formulate a final answer for the user query like a natural language answer without mentioning the data variables but only the values if needed
                6. if the data is not what you want or it is empty; you are allowed to make new tool calls to replace the data for that particular variable name

                Rules:
                → FIRST check if needed data already exists in Current Data at hand
                → ALL tools have access to the Current Data at hand - no need to re-fetch existing da
                → Retrieve related data in single queries when possible
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
        # print("\n==== RAG PROMPT ====")
        # print(filled_prompt)
        # print("==== END RAG PROMPT ====\n")

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
        # pprint(state["messages"], indent=2, width=80)
        
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

        # test code


        # return{
        #     "messages": AIMessage(content="Done"),
        #     "Has_Visualization": "true"
        # }
        
        # Create a preview for the data in the prompt
        data_preview = create_data_preview(state.get("data", {}))
        data_preview_json = json.dumps(data_preview, indent=2)
        
        system_prompt = """You are a visualization expert. Create visual representations of the data or format text replies for appropriate scenarios. 
                         RULES:
                        - You can make only one tool call per response
                        - Choose the right tool based on the data complexity:
                          
                          1. Use format_limited_data_reply ONLY for extremely simple data sets:
                             - Single values or statistics (e.g., "42 users")
                             - Simple lists with under 10 items (e.g., "Top 5 games")
                             - Basic counts with no relationships (e.g., "User X played 3 games")
                          
                          2. Use graph_vis_Wrapper for more complex data with:
                             - Relationships between entities (e.g., games and their players)
                             - Hierarchical or nested structures
                             - Any data with more than 10 values or items
                             - Data that benefits from visual patterns
                        
                        - If the query answer is a single word or the data is just a simple value like a number or string, don't generate any visualization. Just return "No Visualization Needed".
                        
                        - For graph_vis_Wrapper, provide clear instructions on what visualization type to generate. Prefer networkx visualization whenever possible unless the user specifies otherwise or if the query query is not related to graph visualization.
                        
                        - IMPORTANT: NEVER use format_limited_data_reply for relationship data like games and their players - this always requires visualization"""
        
        # Don't embed JSON directly in the prompt to avoid template variable confusion
        user_prompt = f"Visualize or format the data for: {state['user_query']}"
        
        # Log the prompt for verification
        # print("\n==== VISUALIZER PROMPT ====")
        # print("SYSTEM: " + system_prompt)
        # print("\nUSER: " + user_prompt)
        # print("DATA: " + data_preview_json)
        # print("==== END VISUALIZER PROMPT ====\n")
        
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
            print("Successfully serialized the data")
        except TypeError as e:
            print(f"Warning: JSON serialization issue: {e}")
            print("Attempting to create serializable version of the data...")
            # Create a serializable version if needed
            safe_data = self.create_serializable_data(state["data"])
            json_data = json.dumps(safe_data)
        
        # Create a preview of the data for the prompt using the imported function
        data_preview = create_data_preview(state["data"])
        
        # Part 1: HTML head and body opening
        html_head = """
                    <html lang="en">
                    <head>
                    <meta charset="UTF-8">
                    <script src="https://d3js.org/d3.v6.min.js"></script>
                    <style>
                        body { 
                            font: 14px sans-serif; 
                            margin: 0; 
                            padding: 0;
                            width: 100vw;
                            height: 100vh;
                            background-color: white;
                            overflow: hidden;
                        }
                        
                        #visualization-container {
                            width: 100vw;
                            height: 100vh;
                            position: relative;
                            overflow: hidden;
                        }  
                        
                        svg {
                            position: absolute;
                            top: 0;
                            left: 0;
                            width: 100%;
                            height: 100%;
                            display: block;
                        }
                    </style>
                    </head>
                    <body>
                    <div id="visualization-container">
                        <svg></svg>
                    </div>
        """

        # Part 2: Data injection with f-string
        data_script = f"""
                    <script>
                        // Data from the application state
                        const data = {json_data};
        """

        # Part 3: Main JavaScript code
        visualization_script = """
                        // Function to get dimensions with margins
                         function getDimensions() {
                            const container = document.getElementById('visualization-container');
                            const fullWidth = container.clientWidth;
                            const fullHeight = container.clientHeight;
                            
                            const margin = {top: 20, right: 20, bottom: 20, left: 20};
                            const width = fullWidth - margin.left - margin.right;
                            const height = fullHeight - margin.top - margin.bottom;
                            
                            return { width, height, margin, fullWidth, fullHeight };
                        }
                        // Get SVG dimensions
                        const svg = d3.select("svg")
                            .attr("preserveAspectRatio", "xMidYMid meet")
                            .attr("viewBox", function() {
                                const dims = getDimensions();
                                return `0 0 ${dims.fullWidth} ${dims.fullHeight}`;
                            });
                        
                        // Get initial dimensions for setup
                        const initialDimensions = getDimensions();
                        
                        // Create main group
                        const g = svg.append("g")
                            .attr("transform", `translate(${initialDimensions.margin.left},${initialDimensions.margin.top})`);

                        // Update zoom behavior to use dynamic dimensions
                        const zoom = d3.zoom()
                            .scaleExtent([0.1, 4])
                            .on("zoom", (event) => {
                                g.attr("transform", event.transform);
                            });

                        svg.call(zoom);
                        
                        // IMPORTANT: DO NOT use width or height as global variables!
                        // ALWAYS call getDimensions() to get the current dimensions when needed.
                        // Example: const { width, height } = getDimensions();
                    </script>
        """

        # Combine all parts
        html_header = html_head + data_script + visualization_script

        # HTML footer template
        html_footer = """
        </body>
        </html>"""

        # Example D3.js code to show the model what we expect
        example_d3_code = """
                <script>
                // The data is already available as a global 'data' variable
                console.log("Data available:", data);
                
                //using the existing getDimensions
                 {width, height, margin, fullWidth, fullHeight } = getDimensions();

                // Add necessary styles
            const styleSheet = document.createElement("style");
            styleSheet.textContent = `
                .node { cursor: pointer; }
                .link { stroke: #999; stroke-opacity: 0.6; }
                .node-label { font-size: 12px; font-family: sans-serif; pointer-events: none; }
            `;
            document.head.appendChild(styleSheet);

            // Process data into nodes and links
            const nodes = [];
            const links = [];

            data.top_users_and_games[0].forEach(userObj => {
            // Add user node
            nodes.push({
                id: userObj.user._key,
                type: "user",
                size: userObj.total_games,
                name: `User ${userObj.user._key}`
            });
            
            // Add game nodes and links
            userObj.games.forEach(game => {
                // Check if game node already exists
                let gameNode = nodes.find(n => n.id === game._key);
                if (!gameNode) {
                nodes.push({
                    id: game._key,
                    type: "game",
                    name: game.GameName,
                    size: 20
                });
                }
                
                links.push({
                source: userObj.user._key,
                target: game._key
                });
            });
            });

            // Update force simulation to use dynamic dimensions
            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(width * 0.1)) // Make distance relative to width
                .force("charge", d3.forceManyBody().strength(-width * 0.3)) // Make strength relative to width
                .force("center", d3.forceCenter(width / 2, height / 2));

            // Draw links
            const link = g.selectAll(".link")
                .data(links)
                .join("line")
                .attr("class", "link");

            // Draw nodes
            const node = g.selectAll(".node")
                .data(nodes)
                .join("circle")
                .attr("class", "node")
                .attr("r", d => Math.sqrt(d.size) * 2)
                .attr("fill", d => d.type === "user" ? "#ff7f0e" : "#1f77b4")
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));

            // Add labels
            const label = g.selectAll(".node-label")
                .data(nodes)
                .join("text")
                .attr("class", "node-label")
                .text(d => d.name)
                .attr("dy", d => d.type === "user" ? -15 : 25);




            // Update positions on tick
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            });

            // Drag functions
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
            "You are an expert D3.js developer specializing in creating all types of data visualizations. "
            "I have already set up a complete D3.js visualization environment with the following components:\n\n"
            
            "1. Pre-configured SVG and Container:\n"
            "- A responsive SVG element within #visualization-container\n"
            "- Margins and dimensions handled via getDimensions() function\n"
            "- Zoom behavior already implemented and bound to the SVG\n"
            "- A main group 'g' created and transformed with margins\n\n"
            
            "2. Core D3.js Setup:\n"
            "- D3.js v6 imported and ready to use\n"
            "- Base SVG structure with proper viewBox and preserveAspectRatio\n"
            "- Responsive container with automatic resizing\n\n"
            
            "3. Available Variables and Functions:\n"
            "- data: Global variable containing the dataset\n"
            "- getDimensions(): Returns {width, height, margin, fullWidth, fullHeight}\n"
            "- g: Main SVG group for adding visualization elements\n"
            "- svg: Main SVG selection with zoom behavior\n\n"
            
            "HERE IS THE EXACT CODE THAT'S ALREADY SET UP:\n"
            "```javascript\n" + 
            visualization_script.strip() + 
            "\n```\n\n"
            
            f"Create an interactive visualization based on this user query: {query}\n\n"
            
            "CRITICAL REQUIREMENTS:\n"
            "1. ALWAYS call getDimensions() at the beginning of EVERY function that uses width/height\n"
            "2. NEVER use global width/height variables - they don't exist in the template- always use getDimensions() to get width or heigth and others\n"
            "3. Define simulation BEFORE any functions that use it (like drag functions)\n"
            "4. Never define data varible; always use the existing 'data' variable and access its different fields\n"
            "5. You can use new data like names, labels directly depending on the query to generate better visualizations\n"
            "6. Use the existing 'g' group for adding visualization elements\n"
            "7. Add visualization-specific styles via JavaScript\n"
            "8. Add labels or on hover info always depends on the type of visualization\n"
            "9. ALWAYS handle empty, null, or undefined values in the data:\n"
            "   - Check if objects exist before accessing their properties\n"
            "   - Provide default values using || or ternary operators\n"
            "   - Filter out null/undefined entries from arrays\n"
            "   - Use optional chaining (?.) when appropriate\n"
            "   - Add fallbacks for all data-dependent calculations\n"
            "10. Never sample or limit the data; use all the data available"
            "11. IMPORTANT: Create UNBOUNDED visualizations - in any force simulation or node positioning logic, DO NOT clamp or constrain positions to the visible area. Let objects move freely without boundaries\n"

            """Your visualization code MUST follow this exact sequence:
            1. Process data and declare all data-derived variables
            2. Define all scales and utility functions
            3. Then initialze varibales using the scale and utility functions required for the next steps
            4. ONLY THEN create any simulation or force layout
            5. Define event handlers (drag, click, etc.)
            6. Create and append visual elements
            7. Add update functions (simulation tick, etc.)"""

            "data variable Preview:\n" + 
            json.dumps(data_preview, indent=2) + "\n\n"
            
            "Output Requirements:\n"
            "1. Provide ONLY the visualization-specific code within <script> tags\n"
            "2. Include visualization-specific CSS via JavaScript\n"
            "3. Implement proper scales and axes as needed\n"
            "4. Add appropriate interactions and animations\n"
            
            "DO NOT include any HTML structure, just the <script> content for the visualization."
        )
        
        # Log the prompt for verification
        # print("\n==== VISUALIZATION GENERATION PROMPT ====")
        # print(visualization_prompt)
        # print("==== END VISUALIZATION GENERATION PROMPT ====\n")
        
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
            # print("=" * 50)
            # print(complete_html)
            # print("=" * 50)

            # Save the generated HTML to a file
            with open("generated.html", "w") as file:
                file.write(complete_html)
            
            return Command(
                update={
                    "messages": [ToolMessage("Visualization Done", tool_call_id=tool_call_id)],
                    "Has_Visualization": "true"
                }
            )
        except Exception as e:
            print(f"Error generating visualization: {str(e)}")
            return Command(
                update={
                    "messages": [ToolMessage(f"Error generating visualization: {str(e)}", tool_call_id=tool_call_id)],
                    "Has_Visualization": "false"
                }
            )

    def create_serializable_data(self, data):
        """Create a fully JSON-serializable version of the data"""
        if isinstance(data, list):
            return [self.create_serializable_data(item) for item in data]
        elif isinstance(data, dict):
            return {str(k): self.create_serializable_data(v) for k, v in data.items()}
        elif hasattr(data, '__dict__'):  # Handle custom objects
            # Convert custom objects to dictionaries
            return self.create_serializable_data(data.__dict__)
        else:
            # Handle non-serializable data
            try:
                # Test if it's directly serializable
                json.dumps(data)
                return data
            except (TypeError, OverflowError):
                # For dictionaries, recursively process them
                if str(data).startswith('{') and str(data).endswith('}'):
                    # This might be a string representation of a dict
                    try:
                        # Use ast.literal_eval for safely evaluating string representations of Python literals
                        import ast
                        dict_data = ast.literal_eval(str(data))
                        if isinstance(dict_data, dict):
                            return self.create_serializable_data(dict_data)
                    except (SyntaxError, ValueError):
                        pass
                # Convert to string if nothing else works
                return str(data)
                
    def create_networkx_serializable_data(self, data):
        """Create a JSON-serializable version of NetworkX data, extracting only essential node properties.
        This function is optimized for NetworkX node objects and avoids serializing internal metadata."""
        if isinstance(data, list):
            return [self.create_networkx_serializable_data(item) for item in data]
        elif isinstance(data, dict):
            # Explicitly exclude problematic keys that contain connection objects or internal metadata
            excluded_keys = ['db', 'graph', 'node_id', 'parent_keys', 'node_attr_dict_factory', '_conn', '_executor', '_sessions', '_http', '_auth', '_host_resolver']
            
            # Return a new dict with only the keys we want to keep
            return {str(k): self.create_networkx_serializable_data(v) 
                   for k, v in data.items() 
                   if k not in excluded_keys}
        elif hasattr(data, '__dict__'):
            # For objects with __dict__, convert to dict and process
            obj_dict = vars(data)  # This gets the __dict__ attribute
            
            # Exclude the same problematic keys
            excluded_keys = ['db', 'graph', 'node_id', 'parent_keys', 'node_attr_dict_factory', '_conn', '_executor', '_sessions', '_http', '_auth', '_host_resolver']
            
            # Remove excluded keys and process the rest
            return {str(k): self.create_networkx_serializable_data(v) 
                   for k, v in obj_dict.items() 
                   if k not in excluded_keys}
        else:
            # Handle non-serializable data
            try:
                # Test if it's directly serializable
                json.dumps(data)
                return data
            except (TypeError, OverflowError):
                # Convert to string for non-serializable data
                return str(data)

    def query_graph(self, query: str):
        """Execute a graph query using the appropriate tool."""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "RAG_reply": "",
            "user_query": query,
            "data": {},  # Initialize empty data dictionary
            "Has_Visualization": "false"
        }
        
        # Use invoke() instead of stream() to get final state directly
        final_state = self.agent.invoke(initial_state)
        
        # Save the final state to a JSON file for logging
        self.save_state_to_json(final_state, query)
        
        # Return structured response
        return {
            "reply": final_state["messages"][-1].content,
            "rag_reply": final_state["RAG_reply"],  # Include the RAG_reply in the response
            "Has_Visualization": final_state["Has_Visualization"]
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

    def format_small_data_reply(self, tool_call_id: str, state: GraphState):
        """Format a simple text response for very basic data sets
        
        This method handles only very basic data - a few items with simple values.
        It should NOT be used for complex structured data like games with their players,
        which should use visualization instead.
        
        Examples of appropriate data:
        - Simple counts: "5 users played Game X"
        - Basic lists: "Top 3 games: Game A, Game B, Game C"
        - Single statistics: "Average playtime is 45 hours"
        
        The output is purely text-based with no HTML or visualization code.
        """
        # Get the data and user query from the state
        data = state.get("data", {})
        user_query = state.get("user_query", "")
        
        # Create prompt for formatting the reply - directly use the data without any processing
        format_prompt = (
            "Based on the following data and user query, format a clear, plain text response. "
            "This should ONLY be used for extremely simple data sets with few items and single values. "
            "Your task is to create a readable, direct text response.\n\n"
            
            f"User Query: {user_query}\n\n"
            
            "Data (use directly):\n" + 
            str(data) + "\n\n"
            
            "IMPORTANT REQUIREMENTS:\n"
            "1. DO NOT generate any HTML, visualization code, or markdown formatting\n"
            "2. DO NOT attempt to format complex data - if the data contains nested structures like games with players, "
            "   respond with 'This data is too complex for text formatting and requires visualization'\n"
            "3. Provide ONLY a plain text response with simple formatting like bullet points or numbers\n"
            "4. Use standard text characters only - no special HTML entities or symbols\n"
            "5. If the data is a single value or very simple count, present it in a straightforward sentence\n"
            "6. For lists of items (up to ~10), use simple numbered or bulleted lists using standard characters\n"
            
            
            "Example appropriate responses:\n"
            "- The top 3 games are: 1) Fortnite, 2) Minecraft, 3) Call of Duty\n"
            "- Total users: 42\n"
            "- User 123 has played 5 games in the past week\n"

            "Current Response: " +str(state.get("RAG_reply", "")) +  "\n\n" 
            
            "Your response will be displayed directly to the user without any further processing."
        )
        
        try:
            print("Generating plain text reply for simple data...")
            print(f"Data being used directly: {str(data)[:200]}..." if len(str(data)) > 200 else f"Data being used directly: {str(data)}")
            
            # Get response from Claude
            response = self.claude_llm.invoke(format_prompt)
            formatted_reply = response.content
            
            # Check if the response indicates the data is too complex
            if "too complex" in formatted_reply.lower():
                return Command(
                    update={
                        "messages": [ToolMessage("Data is too complex for text formatting, please use visualization instead.", tool_call_id=tool_call_id)],
                        "Has_Visualization": "false"
                    }
                )
            
            # Update the RAG_reply in the state
            return Command(
                update={
                    "messages": [ToolMessage("Text reply formatted successfully.", tool_call_id=tool_call_id)],
                    "RAG_reply": formatted_reply,
                    "Has_Visualization": "false"  # No visualization needed for this type of reply
                }
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return Command(
                update={
                    "messages": [ToolMessage(f"Error formatting reply: {str(e)}", tool_call_id=tool_call_id)],
                    "Has_Visualization": "false"
                }
            )

 
