"""
Entry Node for AR Lab Assistant workflow.
Greets the user and requests initial input.
"""

import logging
from datetime import datetime
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class EntryNodeConfig(FunctionBaseConfig, name="entry_node"):
    """Entry node configuration."""
    greeting_message: str = Field(
        default=(
            "I am a tool for lab science students. Today, we will be following a procedure to perform "
            "the Kirby-Bauer disk diffusion assay experiment, which is a method used to determine the "
            "effectiveness of antibiotics against specific bacteria. Let's get started! We can begin by "
            "answering any initial questions you have for me, or we can jump right into it and I can "
            "guide you through the experiment."
        ),
        description="The greeting message to show to the user"
    )


@register_function(config_type=EntryNodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def entry_node_function(config: EntryNodeConfig, builder: Builder):
    """Entry node - greets the user and requests initial input."""
    
    async def _entry_node(state: dict) -> dict:
        """Entry node - greet the user and request initial input."""
        # Initialize session data
        state["session_data"] = {"start_time": datetime.now().isoformat()}
        
        # Send greeting
        state["messages"].append(AIMessage(content=config.greeting_message))
        
        # Request user input using NAT's UserInteractionManager
        from nat.builder.context import Context
        from nat.data_models.interactive import HumanPromptText
        
        context = Context.get()
        user_input_manager = context.user_interaction_manager
        
        prompt = HumanPromptText(
            text="How would you like to proceed? You can ask questions about the experiment or ask me to guide you through it.",
            required=True,
            placeholder="Type your response here..."
        )
        
        # This will pause the workflow until user responds via WebSocket
        response = await user_input_manager.prompt_user_input(prompt)
        user_response = response.content.text
        
        # Add user response to messages
        state["messages"].append(HumanMessage(content=user_response))
        state["user_response"] = user_response
        
        logger.info(f"Entry node: Got user response: {user_response[:50]}...")
        
        return state
    
    yield FunctionInfo.from_fn(_entry_node, description="Entry node - greets user and requests initial input")
