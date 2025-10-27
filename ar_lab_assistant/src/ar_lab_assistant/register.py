"""
AR Lab Assistant Workflow Registration.

This module registers the main AR Lab Assistant workflow and imports all functions and tools.
"""

import logging
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.api_server import ChatRequestOrMessage, ChatRequest, ChatResponse, Usage
from nat.data_models.component_ref import LLMRef, FunctionRef
from nat.data_models.function import FunctionBaseConfig
from nat.utils.type_converter import GlobalTypeConverter

# Import all functions and tools to trigger their registration
from ar_lab_assistant.functions import (  # noqa: F401
    entry_node_function,
    router_node_function,
    qa_node_function,
    vpg_node_function,
    reprompt_node_function,
    log_session_node_function,
    end_session_node_function,
)
from ar_lab_assistant.tools import rag_question_answering_function  # noqa: F401

logger = logging.getLogger(__name__)


# Define state here in register.py
class ARLabState(TypedDict):
    """State for the AR Lab Assistant workflow."""
    messages: Annotated[list[BaseMessage], "The conversation messages"]
    current_node: str  # Track which node we're in
    session_data: dict  # Store session information
    user_response: str  # Current user response


class ARLabWorkflowConfig(AgentBaseConfig, name="ar_lab_assistant"):
    """AR Lab Assistant workflow configuration."""
    description: str = Field(
        default="AR Lab Assistant Workflow",
        description="The description of this workflow's use."
    )
    tool_names: list[FunctionRef] = Field(
        default_factory=list,
        description="The list of tools to provide to the AR Lab Assistant."
    )
    verbose: bool = Field(default=True, description="Enable verbose logging")
    max_history: int = Field(
        default=15,
        description="Maximum number of messages to keep in the conversation history."
    )
    
    # Function node references
    entry_node_name: str = Field(default="entry_node", description="Entry node function name")
    router_a_node_name: str = Field(default="router_a_node", description="Router A node function name")
    qa_node_a_name: str = Field(default="qa_node_a", description="Q&A node A function name")
    vpg_node_name: str = Field(default="vpg_node", description="VPG node function name")
    reprompt_node_a_name: str = Field(default="reprompt_node_a", description="Reprompt node A function name")
    router_b_node_name: str = Field(default="router_b_node", description="Router B node function name")
    qa_node_b_name: str = Field(default="qa_node_b", description="Q&A node B function name")
    log_session_node_name: str = Field(default="log_session_node", description="Log session node function name")
    end_session_node_name: str = Field(default="end_session_node", description="End session node function name")
    reprompt_node_b_name: str = Field(default="reprompt_node_b", description="Reprompt node B function name")


@register_function(config_type=ARLabWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def ar_lab_workflow(config: ARLabWorkflowConfig, builder: Builder):
    """AR Lab Assistant workflow using LangGraph."""
    
    # Get all function references
    entry_node_fn = await builder.get_function(config.entry_node_name)
    router_a_node_fn = await builder.get_function(config.router_a_node_name)
    qa_node_a_fn = await builder.get_function(config.qa_node_a_name)
    vpg_node_fn = await builder.get_function(config.vpg_node_name)
    reprompt_node_a_fn = await builder.get_function(config.reprompt_node_a_name)
    router_b_node_fn = await builder.get_function(config.router_b_node_name)
    qa_node_b_fn = await builder.get_function(config.qa_node_b_name)
    log_session_node_fn = await builder.get_function(config.log_session_node_name)
    end_session_node_fn = await builder.get_function(config.end_session_node_name)
    reprompt_node_b_fn = await builder.get_function(config.reprompt_node_b_name)
    
    # Define thin wrapper nodes that call the actual functions
    async def entry_node(state: ARLabState) -> ARLabState:
        return await entry_node_fn.ainvoke(state)
    
    async def router_a_node(state: ARLabState) -> ARLabState:
        return await router_a_node_fn.ainvoke(state)
    
    async def qa_node_a(state: ARLabState) -> ARLabState:
        return await qa_node_a_fn.ainvoke(state)
    
    async def vpg_node(state: ARLabState) -> ARLabState:
        return await vpg_node_fn.ainvoke(state)
    
    async def reprompt_node_a(state: ARLabState) -> ARLabState:
        return await reprompt_node_a_fn.ainvoke(state)
    
    async def router_b_node(state: ARLabState) -> ARLabState:
        return await router_b_node_fn.ainvoke(state)
    
    async def qa_node_b(state: ARLabState) -> ARLabState:
        return await qa_node_b_fn.ainvoke(state)
    
    async def log_session_node(state: ARLabState) -> ARLabState:
        return await log_session_node_fn.ainvoke(state)
    
    async def end_session_node(state: ARLabState) -> ARLabState:
        return await end_session_node_fn.ainvoke(state)
    
    async def reprompt_node_b(state: ARLabState) -> ARLabState:
        return await reprompt_node_b_fn.ainvoke(state)
    
    # Routing functions
    def route_from_a(state: ARLabState) -> Literal["qa", "vpg", "reprompt", "end", "router_b"]:
        """Route from Router A based on user response."""
        current_node = state.get("current_node", "")
        if "router_a->qa" in current_node:
            return "qa"
        elif "router_a->vpg" in current_node:
            return "vpg"
        elif "router_a->end" in current_node:
            return "end"
        elif "router_a->router_b" in current_node:
            return "router_b"
        else:
            return "reprompt"
    
    def route_from_b(state: ARLabState) -> Literal["qa", "log", "reprompt", "end"]:
        """Route from Router B based on user response."""
        current_node = state.get("current_node", "")
        if "router_b->qa" in current_node:
            return "qa"
        elif "router_b->log" in current_node:
            return "log"
        elif "router_b->end" in current_node:
            return "end"
        else:
            return "reprompt"
    
    # Build the graph
    workflow = StateGraph(ARLabState)
    
    # Add nodes
    workflow.add_node("entry", entry_node)
    workflow.add_node("router_a", router_a_node)
    workflow.add_node("qa_a", qa_node_a)
    workflow.add_node("vpg", vpg_node)
    workflow.add_node("reprompt_a", reprompt_node_a)
    workflow.add_node("router_b", router_b_node)
    workflow.add_node("qa_b", qa_node_b)
    workflow.add_node("log_session", log_session_node)
    workflow.add_node("end_session", end_session_node)
    workflow.add_node("reprompt_b", reprompt_node_b)
    
    # Add edges
    workflow.set_entry_point("entry")
    workflow.add_edge("entry", "router_a")
    
    # Router A edges
    workflow.add_conditional_edges(
        "router_a",
        route_from_a,
        {
            "qa": "qa_a",
            "vpg": "vpg",
            "reprompt": "reprompt_a",
            "end": "end_session",
            "router_b": "router_b"
        }
    )
    
    # After Q&A A and Re-prompt A, loop back to Router A for follow-up
    workflow.add_edge("qa_a", "router_a")
    workflow.add_edge("reprompt_a", "router_a")
    
    # After VPG, go to Router B
    workflow.add_edge("vpg", "router_b")
    
    # Router B edges
    workflow.add_conditional_edges(
        "router_b",
        route_from_b,
        {
            "qa": "qa_b",
            "log": "log_session",
            "reprompt": "reprompt_b",
            "end": "end_session"
        }
    )
    
    # After Q&A B and Re-prompt B, loop back to Router B for follow-up
    workflow.add_edge("qa_b", "router_b")
    workflow.add_edge("reprompt_b", "router_b")
    
    # Log session and end session end the workflow
    workflow.add_edge("log_session", END)
    workflow.add_edge("end_session", END)
    
    # Compile the graph
    graph = workflow.compile()
    
    async def _run_workflow(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        """
        Main workflow entry function for the AR Lab Assistant.
        
        This function starts the LangGraph workflow ONCE and lets it run to completion.
        The workflow will pause internally when nodes call user_interaction_manager.prompt_user_input().
        User responses come back through the WebSocket via NAT's HITL infrastructure.
        
        Args:
            chat_request_or_message (ChatRequestOrMessage): The input message (typically empty to start)
            
        Returns:
            ChatResponse | str: The response from the workflow
        """
        try:
            # Start with minimal empty state
            # The workflow will prompt for user input via user_interaction_manager
            initial_state = ARLabState(
                messages=[],
                current_node="",
                session_data={},
                user_response=""
            )
            
            if config.verbose:
                logger.info("Starting AR Lab Assistant workflow (will run until END)")
            
            # Run the graph - it will execute continuously, pausing at prompt_user_input() calls
            # until it reaches an END node (log_session or end_session)
            result = await graph.ainvoke(initial_state)
            
            # Get the final AI message
            final_message = ""
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage):
                    final_message = msg.content
                    break
            
            if not final_message:
                final_message = "Workflow completed."
            
            if config.verbose:
                logger.info("AR Lab Assistant workflow completed successfully")
            
            # Create usage statistics
            total_messages = len(result["messages"])
            prompt_tokens = sum(len(str(msg.content).split()) for msg in result["messages"] if isinstance(msg, HumanMessage))
            completion_tokens = len(final_message.split())
            total_tokens = prompt_tokens + completion_tokens
            usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)
            
            # Create response
            response = ChatResponse.from_string(final_message, usage=usage)
            
            if chat_request_or_message.is_string:
                return GlobalTypeConverter.get().convert(response, to_type=str)
            return response
            
        except Exception as ex:
            logger.error("AR Lab Assistant workflow failed with exception: %s", ex)
            error_msg = f"I encountered an error: {str(ex)}"
            
            # Create error response
            usage = Usage(prompt_tokens=0, completion_tokens=len(error_msg.split()), total_tokens=len(error_msg.split()))
            response = ChatResponse.from_string(error_msg, usage=usage)
            
            if chat_request_or_message.is_string:
                return GlobalTypeConverter.get().convert(response, to_type=str)
            return response
    
    try:
        yield FunctionInfo.from_fn(
            _run_workflow,
            description="AR Lab Assistant workflow for guiding students through the Kirby-Bauer disk diffusion assay experiment"
        )
    except GeneratorExit:
        logger.info("Workflow exited early!")
    finally:
        logger.info("Cleaning up AR Lab Assistant workflow")
