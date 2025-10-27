"""
Router Node for AR Lab Assistant workflow.
Determines the path based on user response with configurable routing rules.
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


class RouterNodeConfig(FunctionBaseConfig, name="router_node"):
    """Router node configuration."""
    router_name: str = Field(..., description="Name of this router (for logging)")
    check_vpg_completion: bool = Field(default=False, description="Whether to check if VPG is completed")
    routing_rules: dict[str, list[str]] = Field(
        ...,
        description="Map of destination -> list of keywords. Keywords are matched against lowercased user input."
    )
    default_route: str = Field(default="reprompt", description="Default route if no keywords match")


@register_function(config_type=RouterNodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def router_node_function(config: RouterNodeConfig, builder: Builder):
    """Router - determines path based on user response."""
    
    async def _router_node(state: dict) -> dict:
        """Router - determines path based on user response."""
        if not state["messages"]:
            return state

        # Look for the most recent human message in the conversation
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if not human_messages:
            # No user input yet, default to reprompt
            state["current_node"] = f"{config.router_name}->reprompt"
            logger.info(f"{config.router_name}: No user input found, routing to reprompt")
            return state
        
        # Get the most recent human message
        last_human_message = human_messages[-1]
        user_response = last_human_message.content.lower()

        # Special case: Check if VPG has been completed (Router A only)
        if config.check_vpg_completion and state.get("session_data", {}).get("vpg_completed", False):
            state["current_node"] = f"{config.router_name}->router_b"
            logger.info(f"{config.router_name}: VPG completed, routing to Router B")
            return state

        # Apply routing rules - check in order
        route = None
        for destination, keywords in config.routing_rules.items():
            if any(phrase in user_response for phrase in keywords):
                route = destination
                break
        
        # Use default route if no match
        if route is None:
            route = config.default_route
        
        state["current_node"] = f"{config.router_name}->{route}"
        state["user_response"] = user_response
        
        logger.info(f"{config.router_name}: Routing to {route} based on: {user_response[:50]}...")
        
        return state
    
    yield FunctionInfo.from_fn(_router_node, description=f"Router - determines path based on user response")
