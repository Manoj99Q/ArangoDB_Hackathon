from typing import List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.tools import Tool
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_community.chains.graph_qa.arangodb import ArangoGraphQAChain
import networkx as nx
import re
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

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
        
        # Keep agent creation modular for easy replacement
        self.agent = self._create_agent()

    def _create_agent(self):
        """Create the agent - easy to modify if we switch to custom LangGraph"""
        return create_react_agent(self.llm, self.tools)

    def text_to_aql_to_text(self, query: str):
        """This tool is available to invoke the
            ArangoGraphQAChain object, which enables you to
            translate a Natural Language Query into AQL, execute
            the query, and translate the result back into Natural Language.
        """
        if self.arango_graph is None:
            return "Error: ArangoDB graph is not initialized"
        
        print("Query Received for AQL translation:")
        print(query)
        print("-"*10)

        # Add optimization guidelines in the system message
        system_message = """You are an expert in ArangoDB and AQL query generation.
        When generating AQL queries:
        1. Always use the most efficient method for the task
        2. Avoid unnecessary FOR loops when aggregation functions exist
        3. Use built-in AQL functions (LENGTH, COUNT, SUM, etc.) when possible
        4. Consider memory usage and query performance
        5. Keep queries simple and readable
        """

        chain = ArangoGraphQAChain.from_llm(
            llm=self.llm,
            graph=self.arango_graph,
            verbose=True,
            allow_dangerous_requests=True,
            system_message=system_message,
            return_aql_result=True
        )
        
        result = chain.invoke(query)
        print("Arrango Chain Result:")
        print(result)
        return str(result["result"])

    def text_to_nx_algorithm_to_text(self, query: str):
        """This tool executes NetworkX algorithms based on natural language queries."""
        if self.networkx_graph is None:
            return "Error: NetworkX graph is not initialized"
            
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

    def query_graph(self, query: str):
        """Execute a graph query using the appropriate tool."""
        # Use the pre-initialized agent
        final_state = self.agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        for i in final_state["messages"]:
            print(i)
        return final_state["messages"][-1].content
