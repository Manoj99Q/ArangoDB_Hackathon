"""Direct AQL execution without natural language processing."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from pydantic import Field

from langchain_community.chains.graph_qa.prompts import (
    AQL_FIX_PROMPT,
    AQL_GENERATION_PROMPT,
)
from langchain_community.graphs.arangodb_graph import ArangoGraph

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
        aql_generation_prompt: BasePromptTemplate = AQL_GENERATION_PROMPT,
        aql_fix_prompt: BasePromptTemplate = AQL_FIX_PROMPT,
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

        #########################
        # Generate AQL Query #
        aql_generation_output = self.aql_generation_chain.run(
            {
                "adb_schema": self.graph.schema,
                "aql_examples": self.aql_examples,
                "user_input": user_input,
            },
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
                aql_result = self.graph.query(aql_query, self.top_k)
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