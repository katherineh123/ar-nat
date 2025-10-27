"""
Router Node for AR Lab Assistant workflow.
Determines the path based on user response using LLM-based routing.
"""

import logging
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import LLMRef

from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class RouterNodeConfig(FunctionBaseConfig, name="router_node"):
    """Router node configuration."""
    router_name: str = Field(..., description="Name of this router (for logging and routing identification)")
    llm_name: LLMRef = Field(..., description="LLM to use for routing decisions")
    routing_prompt: str = Field(
        ...,
        description="System prompt describing the routing rules and available destinations"
    )
    available_routes: list[str] = Field(..., description="List of valid route destinations")
    default_route: str = Field(default="reprompt", description="Default route if LLM response is unclear")


@register_function(config_type=RouterNodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def router_node_function(config: RouterNodeConfig, builder: Builder):
    """Router - determines path based on user response using LLM."""
    
    # Get the LLM for routing decisions
    llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _router_node(state: dict) -> dict:
        """Router - determines path based on user response using LLM."""
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
        user_response = last_human_message.content

        # Use LLM to determine routing
        try:
            routing_messages = [
                SystemMessage(content=config.routing_prompt),
                HumanMessage(content=f"User said: {user_response}\n\nWhich route should I take? Respond with ONLY the route name, nothing else.")
            ]
            
            llm_response = await llm.ainvoke(routing_messages)
            route = llm_response.content.strip().lower()
            
            # Validate the route is in available_routes
            if route not in [r.lower() for r in config.available_routes]:
                logger.warning(f"{config.router_name}: LLM returned invalid route '{route}', using default")
                route = config.default_route
            
            logger.info(f"{config.router_name}: LLM routed to '{route}' based on: {user_response[:50]}...")
            
        except Exception as e:
            logger.error(f"{config.router_name}: LLM routing failed with error: {e}, using default route")
            route = config.default_route
        
        state["current_node"] = f"{config.router_name}->{route}"
        state["user_response"] = user_response
        
        return state
    
    yield FunctionInfo.from_fn(_router_node, description=f"Router - determines path based on user response using LLM")
