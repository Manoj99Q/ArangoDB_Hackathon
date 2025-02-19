from typing import List, TypedDict,Annotated
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda
import networkx as nx
import re
from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from IPython.display import Image, display
from langgraph.prebuilt import tools_condition
# Load environment variables
load_dotenv()

# Add State definition at the top with other imports
class GraphState(TypedDict):
    """Represents the state of our graph workflow."""
    messages: Annotated[list[AnyMessage], add_messages]
    user_query: str  # To track the current query being processed
    # data: Annotated[dict, add_messages]  # Changed from tool_merger to add_messages
    

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
        # self.llm = ChatAnthropic(
        #     model="claude-3-5-sonnet-20241022",
        #     temperature=0,
        #     anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
        # )
        self.tools = self._create_tools()
        
        # Create separate tool sets
        self.processing_tools = self._create_processing_tools()
        self.visualization_tools = self._create_visualization_tools()
        
        # Build the graph
        self.agent = self._create_workflow()

        display(Image(self.agent.get_graph(xray=True).draw_mermaid_png()))

    def _create_agent(self):
        """Create the agent - easy to modify if we switch to custom LangGraph"""
        return create_react_agent(self.llm, self.tools)

    def text_to_aql_to_text(self, query: str):
        """Process natural language to AQL query and format results"""
        try:
            aql_query = self.llm.invoke(f"""
            Convert this natural language query to AQL:
            '{query}'
            
            The graph has these collections: {self.arango_graph.schema}
            Return ONLY the AQL query without explanations.
            """).content.strip()

            print("Query Received for AQL translation:")
            print(query)
            print("-"*10)
            print("Generated AQL:\n", aql_query)
            
            # Execute query
            result = self.arango_graph.query(aql_query)
            
            # Enhance results with game names
            enhanced_results = []
            for item in result:
                if 'game_id' in item:
                    game_doc = self.arango_graph.get_document(item['game_id'])
                    item['game_name'] = game_doc.get('GameName', item['game_id'])
                enhanced_results.append(item)
            
            # Format numbers and create response
            formatted = []
            for i, item in enumerate(enhanced_results):
                hours = f"{item['total_hours']:,.1f}".rstrip('.0')
                name = item.get('game_name', item.get('game_id', 'Unknown'))
                formatted.append(f"{i+1}. {name}: {hours} hours")
            
            final_response = "Results:\n" + "\n".join(formatted)
            return final_response

        except Exception as e:
            print(f"Query Error: {e}")
            return f"Error processing query: {str(e)}"

    def text_to_nx_algorithm_to_text(self, query: str):
        """This tool executes NetworkX algorithms based on natural language queries."""
        if self.networkx_graph is None:
            return "Error: NetworkX graph is not initialized"
            
        print("Netwrokx Query received:")
        print(query)
        print("-"*10)
        print("1) Generating NetworkX code")
        text_to_nx = self.llm.invoke(f"""
        I have a NetworkX Graph called `G_adb`. It has the following schema: {self.arango_graph.schema}

        I have the following graph analysis query: {query}.

        Generate the Python Code required to answer the query using the `G_adb` object.
        Be very precise on the NetworkX algorithm you select to answer this query.
        Think step by step.

        Only assume that networkx is installed, and other base python dependencies.
        Always set the last variable as `FINAL_RESULT`, which represents the answer to the original query.
        Only provide python code that I can directly execute via `exec()`. Do not provide any instructions.
        Make sure that `FINAL_RESULT` stores a short & concise answer.

        Your code:
        """).content

        text_to_nx_cleaned = re.sub(r"^```python\n|```$", "", text_to_nx, flags=re.MULTILINE).strip()
        
        print('-'*10)
        print(text_to_nx_cleaned)
        print('-'*10)

        print("\n2) Executing NetworkX code")
        global_vars = {"G_adb": self.networkx_graph, "nx": nx}
        local_vars = {}

        try:
            exec(text_to_nx_cleaned, global_vars, local_vars)
            text_to_nx_final = text_to_nx
            FINAL_RESULT = local_vars["FINAL_RESULT"]
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
        {text_to_nx_final}
        ---
        The `FINAL_RESULT` variable is set to: {FINAL_RESULT}.
        Based on my original Query and FINAL_RESULT, generate a short and concise response.
        """).content

        return response

    def _create_tools(self) -> List[BaseTool]:
        """Create the list of tools available to the agent."""
        return [
            Tool(
                name="AQL_Query",
                func=self.text_to_aql_to_text,
                description="""Use natural language to query the graph database.

                Examples of good queries:
                - "How many users are there?"
                - "What are the top 5 most played games?"
                - "Find users who played Dota 2"

                Do not write AQL queries - this tool translates your question into database queries.

                Best for:
                - Counting entities
                - Finding relationships
                - Data retrieval
                - Aggregating information"""
            ),
            Tool(
                name="NetworkX_Analysis",
                func=self.text_to_nx_algorithm_to_text,
                description="""Analyze graph structure and patterns using NetworkX algorithms.

                Best for:
                - Finding shortest paths
                - Calculating centrality
                - Detecting communities
                - Complex network analysis

                Use natural language to describe what you want to analyze."""
            )
        ]

    def _create_processing_tools(self):
        """Tools for data processing stage"""
        return [
            Tool.from_function(
                func=self.text_to_aql_to_text,
                name="query_processor",
                description="Process natural language queries into database queries"
            ),
            Tool.from_function(
                func=self.text_to_nx_algorithm_to_text,
                name="network_analyzer",
                description="""Analyze graph structure and patterns using NetworkX algorithms.
                Best for:
                - Finding shortest paths
                - Calculating centrality
                - Detecting communities
                - Complex network analysis"""
            )
        ]

    def _create_visualization_tools(self):
        """Tools for visualization stage"""
        return [
            Tool.from_function(
                func=self.generate_graph_visualization,
                name="graph_visualizer",
                description="Generate visual representations of graph data"
            ),
            # Add other visualization tools as needed
        ]

    def _create_workflow(self):
        # Define the graph structure
        builder = StateGraph(GraphState)
        
        # Add nodes
        builder.add_node("processing_agent", self.processing_agent)
        builder.add_node("tools", ToolNode(self.processing_tools))
        # builder.add_node("visualization_agent", self.visualization_agent)
        # builder.add_node("visualization_tools", ToolNode(self.visualization_tools))
        builder.add_edge("tools","processing_agent")
        # builder.add_edge("visualization_tools","visualization_agent")
        # Set up edges
        builder.set_entry_point("processing_agent")
        
        # Update conditional edges to handle message objects properly
        builder.add_conditional_edges(
            "processing_agent",
            tools_condition
        )
        
        # builder.add_conditional_edges(
        #     "visualization_agent",
        #     lambda state: "visualization_tools" if self._has_tool_calls(state) else END,
        #     {"visualization_tools": "visualization_tools", END: END}
        # )

        return builder.compile()

    def _has_tool_calls(self, state: GraphState) -> bool:
        """Check if the last message has tool calls"""
        if not state.get("messages"):
            return False
        last_msg = state["messages"][-1]
        return hasattr(last_msg, 'tool_calls') and bool(last_msg.tool_calls)

    def processing_agent(self, state: GraphState):
        """Agent for data processing phase"""
        # Remove these print statements
        # print("State at processing_agent call:")
        # messages = state.get("messages", [])
        # print("Messages:")
        # ... etc ...
        
        plan_prompt = """SYSTEM: You are a Graph Analysis Planner. Follow these steps:
                1. Analyze the user's query
                2. Create a step-by-step plan using available tools
                3. Execute tools sequentially using previous results
                4. Combine results for final answer

                Rules:
                → Create a plan before tool usage
                → Use one tool per step
                → Reference previous results where needed

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
                self.processing_tools,
                parallel_tool_calls=False,
            )
        )
        
         # 1) Invoke the chain on the user query
        new_ai_message = chain.invoke(state)

        # 2) Return updated messages by *appending* new_ai_message to the state
        return {
            "messages": state["messages"] + [new_ai_message]
        }
    def visualization_agent(self, state: GraphState):
        """Agent for visualization phase"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a visualization expert. Create visual representations of the data."),
            ("user", "Visualize results for: {user_query}\n\nData: {data}")
        ])
        chain = prompt | self.llm.bind_tools(self.visualization_tools)
        return {"messages": [chain.invoke(state)]}

    def generate_graph_visualization(self, query: str):
        """Tool to generate graph visualizations"""
        # Implementation for networkx/Arango visualization
        G = self.networkx_graph or nx.DiGraph()
        # Add visualization logic here
        return "Graph visualization generated"

    def query_graph(self, query: str):
        """Execute a graph query using the appropriate tool."""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "user_query": query
        }
        
        # Use invoke() instead of stream() to get final state directly
        final_state = self.agent.invoke(initial_state)
        
        # Only print the final message content
        print("Final Answer:")
        print(final_state["messages"][-1].content)
        
        return final_state["messages"][-1].content

 
    # def handle_tool_error(state) -> dict:
    #     error = state.get("error")
    #     tool_calls = state["messages"][-1].tool_calls
    #     return {
    #         "messages": [
    #             ToolMessage(
    #                 content=f"Error: {repr(error)}\nPlease fix your mistakes.",
    #                 tool_call_id=tc["id"],
    #             )
    #             for tc in tool_calls
    #         ]
    #     }

    # def create_tool_node_with_fallback(self, tools: list) -> dict:
    #     return ToolNode(tools).with_fallbacks(
    #         [RunnableLambda(self.handle_tool_error)], 
    #         exception_key="error"
    #     )
