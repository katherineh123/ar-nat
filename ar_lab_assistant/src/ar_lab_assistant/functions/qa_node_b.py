"""
Q&A Node B for AR Lab Assistant workflow.
Handles follow-up questions after VPG using RAG tool.
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


class QANodeBConfig(FunctionBaseConfig, name="qa_node_b"):
    """Q&A Node B configuration."""
    rag_tool_name: str = Field(default="rag_question_answering", description="Name of the RAG tool to use")


@register_function(config_type=QANodeBConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def qa_node_b_function(config: QANodeBConfig, builder: Builder):
    """Q&A Node B - handles follow-up questions after VPG using RAG."""
    
    # Get the RAG tool
    rag_tool = await builder.get_tool(config.rag_tool_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    
    async def _qa_node_b(state: dict) -> dict:
        """Q&A Node B - handles follow-up questions after VPG."""
        # Same as Q&A A but for follow-up questions
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if not human_messages:
            response = "I didn't receive a question. Could you please ask me something about the experiment we just performed?"
            state["messages"].append(AIMessage(content=response))
            return state
            
        last_human_message = human_messages[-1]
        question = last_human_message.content
        
        # Call RAG tool
        try:
            answer = await rag_tool.ainvoke({"question": question})
            response = f"Great question! Based on the experiment we just performed: {answer}"
        except Exception as e:
            response = "I can help answer questions about the experiment we just performed. Could you be more specific?"
            logger.error(f"RAG tool error: {e}")
        
        state["messages"].append(AIMessage(content=response))
        
        # Ask for follow-up input
        from nat.builder.context import Context
        from nat.data_models.interactive import HumanPromptText
        
        context = Context.get()
        user_input_manager = context.user_interaction_manager
        
        prompt = HumanPromptText(
            text="Do you have any other questions about the experiment, or would you like to log this session and finish?",
            required=True,
            placeholder="Type your response here..."
        )
        
        follow_up_response = await user_input_manager.prompt_user_input(prompt)
        user_response = follow_up_response.content.text
        
        # Add follow-up response to messages
        state["messages"].append(HumanMessage(content=user_response))
        state["user_response"] = user_response
        
        logger.info(f"Q&A B: Answered follow-up question and got response: {user_response[:50]}...")
        
        return state
    
    yield FunctionInfo.from_fn(_qa_node_b, description="Q&A Node B - handles follow-up questions after VPG using RAG")
