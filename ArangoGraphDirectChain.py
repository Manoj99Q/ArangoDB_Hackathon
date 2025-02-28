"""Direct AQL execution without natural language processing."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from pydantic import Field

from langchain_community.graphs.arangodb_graph import ArangoGraph
from utils import create_data_preview

# Custom prompt templates for AQL generation and fixing
CUSTOM_AQL_GENERATION_TEMPLATE = """Task: Generate an ArangoDB Query Language (AQL) query from a User Input.

You are an ArangoDB Query Language (AQL) expert responsible for translating a `User Input` into an ArangoDB Query Language (AQL) query.

You are given an `ArangoDB Schema`. It is a JSON Object containing:
1. `Graph Schema`: Lists all Graphs within the ArangoDB Database Instance, along with their Edge Relationships.
2. `Collection Schema`: Lists all Collections within the ArangoDB Database Instance, along with their document/edge properties and a document/edge example.

You are also given a `Data State` containing key-value pairs. Each key represents a variable name, and each value represents the variable's value.
These variables should be used in your AQL query if they are needed to answer the user's query.
IMPORTANT: When using these variables in your AQL query, you MUST prefix them with '@'. For example, if the data state has a variable 'limit' with value 10, you should use it in the query as: 'LIMIT @limit'

You may also be given a set of `AQL Query Examples` to help you create the `AQL Query`. If provided, the `AQL Query Examples` should be used as a reference, similar to how `ArangoDB Schema` should be used.

Things you should do:
- Think step by step.
- Review the `Data State` and use any relevant variables in your query with '@' prefix.
- Rely on `ArangoDB Schema`,`Data State` and `AQL Query Examples` (if provided) to generate the query.
- Begin the `AQL Query` by the `WITH` AQL keyword to specify all of the ArangoDB Collections required.
- Return the `AQL Query` wrapped in 3 backticks (```).
- Use only the provided relationship types and properties in the `ArangoDB Schema` and variables in the `Data State`.
- Only answer to requests related to generating an AQL Query.
- If a request is unrelated to generating AQL Query, say that you cannot help the user.

Things you should not do:
- Do not use any properties/relationships that can't be inferred from the `ArangoDB Schema` or the `AQL Query Examples`. 
- Do not include any text except the generated AQL Query.
- Do not provide explanations or apologies in your responses.
- Do not generate an AQL Query that removes or deletes any data.
- Do not forget to prefix bind variables with '@' when using them in the query.

Under no circumstance should you generate an AQL Query that deletes any data whatsoever.

ArangoDB Schema:
{adb_schema}

Data State:
{data_state}

AQL Query Examples (Optional):
{aql_examples}

User Input:
{user_input}

AQL Query: 
"""

CUSTOM_AQL_GENERATION_PROMPT = PromptTemplate(
    input_variables=["adb_schema", "data_state", "aql_examples", "user_input"],
    template=CUSTOM_AQL_GENERATION_TEMPLATE,
)

CUSTOM_AQL_FIX_TEMPLATE = """Task: Address the ArangoDB Query Language (AQL) error message of an ArangoDB Query Language query.

You are an ArangoDB Query Language (AQL) expert responsible for correcting the provided `AQL Query` based on the provided `AQL Error`. 

The `AQL Error` explains why the `AQL Query` could not be executed in the database.
The `AQL Error` may also contain the position of the error relative to the total number of lines of the `AQL Query`.
For example, 'error X at position 2:5' denotes that the error X occurs on line 2, column 5 of the `AQL Query`.  

You are also given the `ArangoDB Schema`. It is a JSON Object containing:
1. `Graph Schema`: Lists all Graphs within the ArangoDB Database Instance, along with their Edge Relationships.
2. `Collection Schema`: Lists all Collections within the ArangoDB Database Instance, along with their document/edge properties and a document/edge example.

You are also given a `Data State` containing key-value pairs. Each key represents a variable name, and each value represents the variable's value.
These variables should be used in your corrected AQL query if they are needed to answer the original query.
IMPORTANT: When using these variables in your AQL query, you MUST prefix them with '@'. For example, if the data state has a variable 'limit' with value 10, you should use it in the query as: 'LIMIT @limit'

Pay special attention to the error. If it mentions "bind parameter 'X' was not declared", it usually means one of two things:
1. You're trying to use a variable that doesn't exist in the Data State
2. You're not prefixing the variable with '@' in the query

You will output the `Corrected AQL Query` wrapped in 3 backticks (```). Do not include any text except the Corrected AQL Query.

Remember to think step by step.

ArangoDB Schema:
{adb_schema}

Data State:
{data_state}

AQL Query:
{aql_query}

AQL Error:
{aql_error}

Corrected AQL Query:
"""

CUSTOM_AQL_FIX_PROMPT = PromptTemplate(
    input_variables=[
        "adb_schema",
        "data_state",
        "aql_query",
        "aql_error",
    ],
    template=CUSTOM_AQL_FIX_TEMPLATE,
)

class ArangoGraphDirectChain(Chain):
    """Chain for querying a graph by generating AQL statements without natural language processing.
    
    This class executes AQL statements directly and returns the raw results without 
    natural language interpretation, avoiding token limit issues with large result sets.
    """
    
    graph: ArangoGraph = Field(exclude=True)
    aql_generation_chain: LLMChain
    aql_fix_chain: LLMChain
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    # Specifies the maximum number of AQL Query Results to return
    top_k: int = 10

    # Specifies the set of AQL Query Examples that promote few-shot-learning
    aql_examples: str = ""
    
    # Specifies the data state with key-value pairs for variable access
    data_state: Dict[str, Any] = Field(default_factory=dict)

    # Specify whether to return the AQL Query in the output dictionary
    return_aql_query: bool = False

    # Specify whether to return the AQL JSON Result in the output dictionary
    return_aql_result: bool = False

    # Specify the maximum amount of AQL Generation attempts that should be made
    max_aql_generation_attempts: int = 3
    
    allow_dangerous_requests: bool = False
    """Forced user opt-in to acknowledge that the chain can make dangerous requests.

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chain."""
        super().__init__(**kwargs)
        if self.allow_dangerous_requests is not True:
            raise ValueError(
                "In order to use this chain, you must acknowledge that it can make "
                "dangerous requests by setting `allow_dangerous_requests` to `True`."
                "You must narrowly scope the permissions of the database connection "
                "to only include necessary permissions. Failure to do so may result "
                "in data corruption or loss or reading sensitive data if such data is "
                "present in the database."
                "Only use this chain if you understand the risks and have taken the "
                "necessary precautions. "
                "See https://python.langchain.com/docs/security for more information."
            )
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        return "graph_aql_direct_chain"

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        *,
        aql_generation_prompt: BasePromptTemplate = CUSTOM_AQL_GENERATION_PROMPT,
        aql_fix_prompt: BasePromptTemplate = CUSTOM_AQL_FIX_PROMPT,
        **kwargs: Any,
    ) -> ArangoGraphDirectChain:
        """Initialize from LLM."""
        aql_generation_chain = LLMChain(llm=llm, prompt=aql_generation_prompt)
        aql_fix_chain = LLMChain(llm=llm, prompt=aql_fix_prompt)

        return cls(
            aql_generation_chain=aql_generation_chain,
            aql_fix_chain=aql_fix_chain,
            **kwargs,
        )

    def _extract_bind_vars(self, aql_query: str) -> Dict[str, Any]:
        """Extract only the bind variables that are used in the query."""
        # Find all @variable_name patterns in the query
        bind_var_pattern = r'@([a-zA-Z_][a-zA-Z0-9_]*)'
        used_vars = set(re.findall(bind_var_pattern, aql_query))
        
        # Create a new dict with only the bind variables that are used in the query
        return {k: v for k, v in self.data_state.items() if k in used_vars}

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """
        Generate an AQL statement from user input and use it to retrieve a response
        from an ArangoDB Database instance, but skip the natural language processing step.
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        callbacks = _run_manager.get_child()
        user_input = inputs[self.input_key]

        # Create a preview of data_state for the LLM to avoid token limit issues
        data_state_preview = create_data_preview(self.data_state)

        #########################
        # Print Complete Prompt #
        prompt_values = {
            "adb_schema": self.graph.schema,
            "data_state": data_state_preview,  # Use the preview instead of full data_state
            "aql_examples": self.aql_examples,
            "user_input": user_input,
        }
        # Format the prompt with all values
        complete_prompt = self.aql_generation_chain.prompt.format(**prompt_values)
        print("Complete AQL Generation Prompt with all values:")
        print("-" * 80)
        print(complete_prompt)
        print("-" * 80)
        #########################

        #########################
        # Generate AQL Query #
        aql_generation_output = self.aql_generation_chain.run(
            prompt_values,
            callbacks=callbacks,
        )
        #########################

        aql_query = ""
        aql_error = ""
        aql_result = None
        aql_generation_attempt = 1

        while (
            aql_result is None
            and aql_generation_attempt < self.max_aql_generation_attempts + 1
        ):
            #####################
            # Extract AQL Query #
            pattern = r"```(?i:aql)?(.*?)```"
            matches = re.findall(pattern, aql_generation_output, re.DOTALL)
            if not matches:
                _run_manager.on_text(
                    "Invalid Response: ", end="\n", verbose=self.verbose
                )
                _run_manager.on_text(
                    aql_generation_output, color="red", end="\n", verbose=self.verbose
                )
                raise ValueError(f"Response is Invalid: {aql_generation_output}")

            aql_query = matches[0]
            #####################

            _run_manager.on_text(
                f"AQL Query ({aql_generation_attempt}):", verbose=self.verbose
            )
            _run_manager.on_text(
                aql_query, color="green", end="\n", verbose=self.verbose
            )

            #####################
            # Execute AQL Query #
            from arango import AQLQueryExecuteError

            try:
                # Filter bind_vars to only include variables referenced in the query
                used_bind_vars = self._extract_bind_vars(aql_query)
                aql_result = self.graph.query(aql_query, self.top_k, bind_vars=used_bind_vars)
            except AQLQueryExecuteError as e:
                aql_error = e.error_message

                _run_manager.on_text(
                    "AQL Query Execution Error: ", end="\n", verbose=self.verbose
                )
                _run_manager.on_text(
                    aql_error, color="yellow", end="\n\n", verbose=self.verbose
                )

                ########################
                # Retry AQL Generation #
                aql_generation_output = self.aql_fix_chain.run(
                    {
                        "adb_schema": self.graph.schema,
                        "data_state": data_state_preview,  # Use the preview instead of full data_state
                        "aql_query": aql_query,
                        "aql_error": aql_error,
                    },
                    callbacks=callbacks,
                )
                ########################

            #####################

            aql_generation_attempt += 1

        if aql_result is None:
            m = f"""
                Maximum amount of AQL Query Generation attempts reached.
                Unable to execute the AQL Query due to the following error:
                {aql_error}
            """
            raise ValueError(m)

        _run_manager.on_text("AQL Result:", end="\n", verbose=self.verbose)
        _run_manager.on_text(
            str(aql_result), color="green", end="\n", verbose=self.verbose
        )

        # Skip the QA chain step - don't interpret results with natural language
        
        # Prepare result dictionary
        result = {self.output_key: "Query executed successfully"}  # Simple success message

        if self.return_aql_query:
            result["aql_query"] = aql_query

        if self.return_aql_result:
            result["aql_result"] = aql_result

        return result