"""
Reprompt Node for AR Lab Assistant workflow.
Guides user back on track with a configurable message.
"""

import logging
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class RepromptNodeConfig(FunctionBaseConfig, name="reprompt_node"):
    """Reprompt Node configuration."""
    reprompt_message: str = Field(
        ...,
        description="The reprompt message to guide the user"
    )


@register_function(config_type=RepromptNodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def reprompt_node_function(config: RepromptNodeConfig, builder: Builder):
    """Re-prompt Node - guides user back on track."""
    
    async def _reprompt_node(state: dict) -> dict:
        """Re-prompt Node - guides user back on track."""
        response = config.reprompt_message
        
        state["messages"].append(AIMessage(content=response))
        
        # Ask for user input
        from nat.builder.context import Context
        from nat.data_models.interactive import HumanPromptText
        
        context = Context.get()
        user_input_manager = context.user_interaction_manager
        
        prompt = HumanPromptText(
            text=response,
            required=True,
            placeholder="Type your response here..."
        )
        
        follow_up_response = await user_input_manager.prompt_user_input(prompt)
        user_response = follow_up_response.content.text
        
        # Add follow-up response to messages
        state["messages"].append(HumanMessage(content=user_response))
        state["user_response"] = user_response
        
        logger.info(f"Reprompt: Guided user and got response: {user_response[:50]}...")
        
        return state
    
    yield FunctionInfo.from_fn(_reprompt_node, description="Reprompt Node - guides user back on track")
