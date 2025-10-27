"""
VPG (Visual Process Guidance) Node for AR Lab Assistant workflow.
Guides students through the experiment with step-by-step instructions.
"""

import asyncio
import logging
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig

from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class VPGNodeConfig(FunctionBaseConfig, name="vpg_node"):
    """VPG node configuration."""
    experiment_type: str = Field(
        default="Kirby-Bauer disk diffusion assay",
        description="Type of experiment being performed"
    )


@register_function(config_type=VPGNodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def vpg_node_function(config: VPGNodeConfig, builder: Builder):
    """VPG node - guides through experiment."""
    
    async def _vpg_node(state: dict) -> dict:
        """Visual Process Guidance node - guides through experiment."""
        logger.info(f"Starting Visual Process Guidance for {config.experiment_type} experiment")
        
        # Mock the AR guidance process with step-by-step feedback
        steps = [
            "Step 1: Prepare your bacterial suspension by pipetting 4ml of saline.",
            "Step 2: Gently shake the petri dish back and forth.",
            "Congratulations! You have successfully completed the experiment."
        ]
        
        guidance_output = []
        for i, step in enumerate(steps, 1):
            guidance_output.append(f"Step {i}: {step}")
            await asyncio.sleep(0.5)  # Simulate processing time
        
        logger.info("Visual Process Guidance completed")
        result = "\n".join(guidance_output)
        response = f"🧪 **Experiment Guidance Complete!**\n\n{result}\n\n🎉 **Great work!** We are now done with the experiment."
        
        state["messages"].append(AIMessage(content=response))
        state["session_data"]["vpg_completed"] = True
        
        # Ask for follow-up input
        from nat.builder.context import Context
        from nat.data_models.interactive import HumanPromptText
        
        context = Context.get()
        user_input_manager = context.user_interaction_manager
        
        prompt = HumanPromptText(
            text="[Visual Process Guidance workflow happens] \n\nDo you have any follow-up questions for me, or would you like to log what we just did and exit?",
            required=True,
            placeholder="Type your response here..."
        )
        
        follow_up_response = await user_input_manager.prompt_user_input(prompt)
        user_response = follow_up_response.content.text
        
        # Add follow-up response to messages
        state["messages"].append(HumanMessage(content=user_response))
        state["user_response"] = user_response
        
        logger.info(f"VPG: Completed experiment guidance and got follow-up: {user_response[:50]}...")
        
        return state
    
    yield FunctionInfo.from_fn(_vpg_node, description="VPG node - guides students through experiment")
