"""
Q&A Node for AR Lab Assistant workflow.
Uses NAT's built-in ReAct agent with RAG tool to answer questions naturally.
"""

import logging
from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import FunctionRef

from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)


class QANodeConfig(FunctionBaseConfig, name="qa_node"):
    """Q&A Node configuration."""
    qa_agent_name: FunctionRef = Field(..., description="Name of the ReAct agent function to use for Q&A")
    no_question_message: str = Field(default="I didn't receive a question. Could you please ask me something?", description="Message when no question is provided")
    follow_up_prompt: str = Field(default="Do you have any other questions?", description="Follow-up prompt for user")


@register_function(config_type=QANodeConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def qa_node_function(config: QANodeConfig, builder: Builder):
    """Q&A Node - handles questions using NAT's built-in ReAct agent."""
    
    # Get the ReAct agent function
    qa_agent = await builder.get_function(config.qa_agent_name)
    
    async def _qa_node(state: dict) -> dict:
        """Q&A Node - uses NAT's ReAct agent to answer questions with tool calling."""
        # Extract the user's question
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if not human_messages:
            response = config.no_question_message
            state["messages"].append(AIMessage(content=response))
            return state
            
        last_human_message = human_messages[-1]
        question = last_human_message.content
        
        # Call the ReAct agent to answer the question
        try:
            logger.info(f"Q&A: Calling ReAct agent for question: {question[:50]}...")
            
            # Invoke the ReAct agent with the question
            answer = await qa_agent.ainvoke(question)
            response = str(answer)
            
            logger.info(f"Q&A: Agent responded successfully")
            
        except Exception as e:
            response = "I encountered an error while trying to answer your question. Could you please rephrase it?"
            logger.error(f"ReAct agent error: {e}", exc_info=True)
        
        # Combine the answer with the follow-up prompt for display
        combined_prompt = f"{response}\n\n{config.follow_up_prompt}"
        
        # Add the answer to conversation history (without the follow-up prompt)
        state["messages"].append(AIMessage(content=response))
        
        # Ask for follow-up input (display includes answer + follow-up prompt together)
        from nat.builder.context import Context
        from nat.data_models.interactive import HumanPromptText
        
        context = Context.get()
        user_input_manager = context.user_interaction_manager
        
        prompt = HumanPromptText(
            text=combined_prompt,  # Show answer + follow-up together
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
    
    yield FunctionInfo.from_fn(_qa_node, description="Q&A Node - uses NAT's ReAct agent to answer questions with tool calling")
