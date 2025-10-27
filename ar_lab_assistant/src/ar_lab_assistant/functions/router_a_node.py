"""
Router A Node for AR Lab Assistant workflow.
Determines the path based on user response before VPG.
"""

import logging
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class RouterANodeConfig(FunctionBaseConfig, name="router_a_node"):
    """Router A node configuration."""
    pass


@register_function(config_type=RouterANodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def router_a_node_function(config: RouterANodeConfig, builder: Builder):
    """Router A - determines path based on user response."""
    
    async def _router_a_node(state: dict) -> dict:
        """Router A - determines path based on user response."""
        if not state["messages"]:
            return state

        # Look for the most recent human message in the conversation
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if not human_messages:
            # No user input yet, default to reprompt
            state["current_node"] = "router_a->reprompt"
            logger.info("Router A: No user input found, routing to reprompt")
            return state
        
        # Get the most recent human message
        last_human_message = human_messages[-1]
        user_response = last_human_message.content.lower()

        # Check if VPG has been completed - if so, route to Router B instead
        if state.get("session_data", {}).get("vpg_completed", False):
            state["current_node"] = "router_a->router_b"
            logger.info("Router A: VPG completed, routing to Router B")
            return state

        # Simple routing logic - can be enhanced with LLM
        # Check for "end" first, as it should override everything
        if any(phrase in user_response for phrase in ["end", "finish", "done", "exit", "quit", "stop", "log session", "end session"]):
            state["current_node"] = "router_a->end"
        elif any(phrase in user_response for phrase in ["start", "begin", "guide", "procedure", "experiment", "let's start", "ready to start"]):
            state["current_node"] = "router_a->vpg"
        elif any(phrase in user_response for phrase in ["question", "ask", "what", "how", "why", "tell me"]):
            state["current_node"] = "router_a->qa"
        else:
            state["current_node"] = "router_a->reprompt"

        logger.info(f"Router A: Routing to {state['current_node']} based on: {user_response[:50]}...")

        return state
    
    yield FunctionInfo.from_fn(_router_a_node, description="Router A - determines path based on user response")
