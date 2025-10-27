"""
Router B Node for AR Lab Assistant workflow.
Determines the path based on user response after VPG completion.
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


class RouterBNodeConfig(FunctionBaseConfig, name="router_b_node"):
    """Router B node configuration."""
    pass


@register_function(config_type=RouterBNodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def router_b_node_function(config: RouterBNodeConfig, builder: Builder):
    """Router B - determines path after VPG completion."""
    
    async def _router_b_node(state: dict) -> dict:
        """Router B - determines path after VPG completion."""
        if not state["messages"]:
            return state
            
        # Look for the most recent human message in the conversation
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if not human_messages:
            # No user input yet, default to reprompt
            state["current_node"] = "router_b->reprompt"
            logger.info("Router B: No user input found, routing to reprompt")
            return state
        
        # Get the most recent human message
        last_human_message = human_messages[-1]
        user_response = last_human_message.content.lower()
        
        # Simple routing logic
        if any(phrase in user_response for phrase in ["end", "finish", "done", "exit", "quit", "stop", "log session", "end session"]):
            route = "end"
        elif any(phrase in user_response for phrase in ["question", "ask", "what", "how", "why", "tell me"]):
            route = "qa"
        elif any(phrase in user_response for phrase in ["log", "yes"]):
            route = "log"
        else:
            route = "reprompt"
        
        state["current_node"] = f"router_b->{route}"
        state["user_response"] = user_response
        
        logger.info(f"Router B: Routing to {route} based on: {user_response[:50]}...")
        
        return state
    
    yield FunctionInfo.from_fn(_router_b_node, description="Router B - determines path after VPG completion")
