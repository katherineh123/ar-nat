"""
Q&A Node for AR Lab Assistant workflow.
Handles questions using RAG tool with configurable prompts.
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


class QANodeConfig(FunctionBaseConfig, name="qa_node"):
    """Q&A Node configuration."""
    rag_tool_name: str = Field(default="rag_question_answering", description="Name of the RAG tool to use")
    response_prefix: str = Field(default="Based on the experiment:", description="Prefix for RAG responses")
    no_question_message: str = Field(default="I didn't receive a question. Could you please ask me something?", description="Message when no question is provided")
    error_message: str = Field(default="I can help answer questions. Could you be more specific?", description="Message when RAG fails")
    follow_up_prompt: str = Field(default="Do you have any other questions?", description="Follow-up prompt for user")


@register_function(config_type=QANodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def qa_node_function(config: QANodeConfig, builder: Builder):
    """Q&A Node - handles questions using RAG."""
    
    # Get the RAG tool
    rag_tool = await builder.get_tool(config.rag_tool_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _qa_node(state: dict) -> dict:
        """Q&A Node - handles questions."""
        # Use RAG tool to answer questions
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if not human_messages:
            response = config.no_question_message
            state["messages"].append(AIMessage(content=response))
            return state
            
        last_human_message = human_messages[-1]
        question = last_human_message.content
        
        # Call RAG tool
        try:
            answer = await rag_tool.ainvoke({"question": question})
            response = f"{config.response_prefix} {answer}"
        except Exception as e:
            response = config.error_message
            logger.error(f"RAG tool error: {e}")
        
        state["messages"].append(AIMessage(content=response))
        
        # Ask for follow-up input
        from nat.builder.context import Context
        from nat.data_models.interactive import HumanPromptText
        
        context = Context.get()
        user_input_manager = context.user_interaction_manager
        
        prompt = HumanPromptText(
            text=config.follow_up_prompt,
            required=True,
            placeholder="Type your response here..."
        )
        
        follow_up_response = await user_input_manager.prompt_user_input(prompt)
        user_response = follow_up_response.content.text
        
        # Add follow-up response to messages
        state["messages"].append(HumanMessage(content=user_response))
        state["user_response"] = user_response
        
        logger.info(f"Q&A: Answered question and got follow-up: {user_response[:50]}...")
        
        return state
    
    yield FunctionInfo.from_fn(_qa_node, description="Q&A Node - handles questions using RAG")
