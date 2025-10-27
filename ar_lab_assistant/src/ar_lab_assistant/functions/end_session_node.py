"""
End Session Node for AR Lab Assistant workflow.
Ends the session without logging.
"""

import logging
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from langchain_core.messages import AIMessage

logger = logging.getLogger(__name__)


class EndSessionNodeConfig(FunctionBaseConfig, name="end_session_node"):
    """End session node configuration."""
    end_message: str = Field(
        default=(
            "Session ended. Thank you for using the AR Lab Assistant! "
            "You can start a new session anytime by sending a new message."
        ),
        description="The message to display when ending the session"
    )


@register_function(config_type=EndSessionNodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def end_session_node_function(config: EndSessionNodeConfig, builder: Builder):
    """End Session node - ends the session without logging."""
    
    async def _end_session_node(state: dict) -> dict:
        """End Session node - ends the session without logging."""
        response = config.end_message
        
        state["messages"].append(AIMessage(content=response))
        state["current_node"] = "end_session"
        state["session_data"]["session_ended"] = True
        
        logger.info("End Session: Session ended without logging")
        
        return state
    
    yield FunctionInfo.from_fn(_end_session_node, description="End Session node - ends session without logging")
