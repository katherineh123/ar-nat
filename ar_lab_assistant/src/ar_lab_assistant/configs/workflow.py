"""
NAT-compatible wrapper for the AR Lab Assistant LangGraph workflow.
"""

import logging

from pydantic import Field

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.agent import AgentBaseConfig
from nat.data_models.api_server import ChatRequestOrMessage
from nat.data_models.api_server import ChatResponse
from nat.data_models.api_server import Usage
from nat.data_models.component_ref import FunctionRef
from nat.utils.type_converter import GlobalTypeConverter

from .langgraph_workflow import ARLabWorkflow

logger = logging.getLogger(__name__)


class ARLabWorkflowConfig(AgentBaseConfig, name="ar_lab_workflow"):
    """
    Configuration for the AR Lab Assistant workflow.
    Defines a NAT function that uses a LangGraph-based workflow for AR lab experiments.
    """
    description: str = Field(
        default="AR Lab Assistant Workflow",
        description="The description of this workflow's use."
    )
    tool_names: list[FunctionRef] = Field(
        default_factory=list,
        description="The list of tools to provide to the AR Lab Assistant."
    )
    verbose: bool = Field(
        default=True,
        description="Whether to enable verbose logging."
    )
    max_history: int = Field(
        default=15,
        description="Maximum number of messages to keep in the conversation history."
    )


@register_function(config_type=ARLabWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def ar_lab_workflow(config: ARLabWorkflowConfig, builder: Builder):
    """
    Registers the AR Lab Assistant workflow function.
    """

    async def _ar_lab_workflow(chat_request_or_message: ChatRequestOrMessage) -> ChatResponse | str:
        """
        Main workflow entry function for the AR Lab Assistant.

        This function invokes the LangGraph workflow and returns the response.

        Args:
            chat_request_or_message (ChatRequestOrMessage): The input message to process

        Returns:
            ChatResponse | str: The response from the workflow or error message
        """
        try:
            # Convert input to ChatRequest
            from nat.data_models.api_server import ChatRequest
            message = GlobalTypeConverter.get().convert(chat_request_or_message, to_type=ChatRequest)

            # Get the LLM
            llm = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)

            # Get the tools
            tools = await builder.get_tools(tool_names=config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
            if not tools:
                raise ValueError(f"No tools specified for AR Lab Assistant '{config.llm_name}'")

            # Create the workflow
            workflow = ARLabWorkflow(llm=llm, tools=tools, verbose=config.verbose)
            
            # Get the graph (not the run method)
            graph = workflow.graph

            # Process the input message
            if message.messages:
                # Get the last user message
                last_message = message.messages[-1]
                user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # Run the workflow
                result = await workflow.run(user_input)

                # Create usage statistics
                prompt_tokens = sum(len(str(msg.content).split()) for msg in message.messages)
                completion_tokens = len(result.split()) if result else 0
                total_tokens = prompt_tokens + completion_tokens
                usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)

                # Create response
                response = ChatResponse.from_string(result, usage=usage)

                if chat_request_or_message.is_string:
                    return GlobalTypeConverter.get().convert(response, to_type=str)
                return response
            else:
                # No messages, start with greeting
                result = await workflow.run("")

                prompt_tokens = 0
                completion_tokens = len(result.split()) if result else 0
                total_tokens = prompt_tokens + completion_tokens
                usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)

                response = ChatResponse.from_string(result, usage=usage)

                if chat_request_or_message.is_string:
                    return GlobalTypeConverter.get().convert(response, to_type=str)
                return response

        except Exception as ex:
            logger.error("AR Lab Assistant workflow failed with exception: %s", ex)
            error_msg = f"I encountered an error: {str(ex)}"

            # Create error response
            usage = Usage(prompt_tokens=0, completion_tokens=len(error_msg.split()), total_tokens=len(error_msg.split()))
            response = ChatResponse.from_string(error_msg, usage=usage)

            if chat_request_or_message.is_string:
                return GlobalTypeConverter.get().convert(response, to_type=str)
            return response

    yield FunctionInfo.from_fn(_ar_lab_workflow, description=config.description)
