"""
LangGraph implementation for AR Lab Assistant workflow.
This module defines the state machine for the AR Lab Science Assistant.
"""

import logging
from typing import Annotated, Literal, TypedDict
from datetime import datetime

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)


class ARLabState(TypedDict):
    """State for the AR Lab Assistant workflow."""
    messages: Annotated[list[BaseMessage], "The conversation messages"]
    current_node: str  # Track which node we're in
    session_data: dict  # Store session information
    user_response: str  # Current user response


class ARLabWorkflow:
    """AR Lab Assistant workflow using LangGraph."""
    
    def __init__(self, llm, tools, verbose: bool = True):
        self.llm = llm
        self.tools = tools
        self.verbose = verbose
        self.tool_node = ToolNode(tools)
        
        # Create the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(ARLabState)
        
        # Add nodes
        workflow.add_node("entry", self._entry_node)
        workflow.add_node("router_a", self._router_a_node)
        workflow.add_node("qa_a", self._qa_node_a)
        workflow.add_node("vpg", self._vpg_node)
        workflow.add_node("reprompt_a", self._reprompt_node_a)
        workflow.add_node("router_b", self._router_b_node)
        workflow.add_node("qa_b", self._qa_node_b)
        workflow.add_node("log_session", self._log_session_node)
        workflow.add_node("end_session", self._end_session_node)
        workflow.add_node("reprompt_b", self._reprompt_node_b)
        
        # Add edges
        workflow.set_entry_point("entry")
        workflow.add_edge("entry", "router_a")
        
        # Router A edges
        workflow.add_conditional_edges(
            "router_a",
            self._route_from_a,
            {
                "qa": "qa_a",
                "vpg": "vpg",
                "reprompt": "reprompt_a",
                "end": "end_session",
                "router_b": "router_b"
            }
        )
        
        # After Q&A A and Re-prompt A, loop back to Router A for follow-up
        workflow.add_edge("qa_a", "router_a")
        workflow.add_edge("reprompt_a", "router_a")
        
        # After VPG, go to Router B
        workflow.add_edge("vpg", "router_b")
        
        # Router B edges
        workflow.add_conditional_edges(
            "router_b",
            self._route_from_b,
            {
                "qa": "qa_b",
                "log": "log_session",
                "reprompt": "reprompt_b",
                "end": "end_session"
            }
        )
        
        # After Q&A B and Re-prompt B, loop back to Router B for follow-up
        workflow.add_edge("qa_b", "router_b")
        workflow.add_edge("reprompt_b", "router_b")
        
        # Log session and end session end the workflow
        workflow.add_edge("log_session", END)
        workflow.add_edge("end_session", END)
        
        # Compile the graph
        return workflow.compile()
    
    
    async def _entry_node(self, state: ARLabState) -> ARLabState:
        """Entry node - greet the user and request initial input."""
        # Initialize session data if not already set
        if not state.get("session_data"):
            state["session_data"] = {"start_time": datetime.now().isoformat()}
        
        # Check if we have a user response from the conversation history
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if human_messages:
            # We have user input, just greet and acknowledge
            greeting = (
                "I am a tool for lab science students. Today, we will be following a procedure to perform "
                "the Kirby-Bauer disk diffusion assay experiment, which is a method used to determine the "
                "effectiveness of antibiotics against specific bacteria. Let's get started! We can begin by "
                "answering any initial questions you have for me, or we can jump right into it and I can "
                "guide you through the experiment."
            )
            # Only add greeting if this is the first message (no AI messages yet)
            ai_messages = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
            if not ai_messages:
                state["messages"].insert(0, AIMessage(content=greeting))
            
            # Use the existing user message
            last_human_message = human_messages[-1]
            user_response = last_human_message.content
            state["user_response"] = user_response
            
            if self.verbose:
                logger.info(f"Entry node: Using user response from conversation: {user_response[:50]}...")
        else:
            # No user input yet - greet and request input
            greeting = (
                "I am a tool for lab science students. Today, we will be following a procedure to perform "
                "the Kirby-Bauer disk diffusion assay experiment, which is a method used to determine the "
                "effectiveness of antibiotics against specific bacteria. Let's get started! We can begin by "
                "answering any initial questions you have for me, or we can jump right into it and I can "
                "guide you through the experiment."
            )
            state["messages"].append(AIMessage(content=greeting))
            
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
            
            response = await user_input_manager.prompt_user_input(prompt)
            user_response = response.content.text
            
            # Add user response to messages
            state["messages"].append(HumanMessage(content=user_response))
            state["user_response"] = user_response
            
            if self.verbose:
                logger.info(f"Entry node: Got user response: {user_response[:50]}...")
        
        return state
    
    async def _router_a_node(self, state: ARLabState) -> ARLabState:
        """Router A - determines path based on user response."""
        if not state["messages"]:
            return state

        # Look for the most recent human message in the conversation
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if not human_messages:
            # No user input yet, default to reprompt
            state["current_node"] = "router_a->reprompt"
            if self.verbose:
                logger.info("Router A: No user input found, routing to reprompt")
            return state
        
        # Get the most recent human message
        last_human_message = human_messages[-1]
        user_response = last_human_message.content.lower()

        # Check if VPG has been completed - if so, route to Router B instead
        if state.get("session_data", {}).get("vpg_completed", False):
            state["current_node"] = "router_a->router_b"
            if self.verbose:
                logger.info("Router A: VPG completed, routing to Router B")
            return state

        # Simple routing logic - can be enhanced with LLM
        # Check for "end" first, as it should override everything
        if any(phrase in user_response for phrase in ["end", "finish", "done", "exit", "quit", "stop", "log session", "end session"]):
            state["current_node"] = "router_a->end"
        elif any(phrase in user_response for phrase in ["start", "begin", "guide", "procedure", "experiment", "let's start", "ready to start"]):
            state["current_node"] = "router_a->vpg"
        elif any(phrase in user_response for phrase in ["question", "ask", "what", "how", "why", "tell me"]):
            state["current_node"] = "router_a->qa"
        else:
            state["current_node"] = "router_a->reprompt"

        if self.verbose:
            logger.info(f"Router A: Routing to {state['current_node']} based on: {user_response[:50]}...")

        return state
    
    async def _qa_node_a(self, state: ARLabState) -> ARLabState:
        """Q&A Node A - handles initial questions."""
        # Use RAG tool to answer questions
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if not human_messages:
            response = "I didn't receive a question. Could you please ask me something about the experiment?"
            state["messages"].append(AIMessage(content=response))
            return state
            
        last_human_message = human_messages[-1]
        question = last_human_message.content
        
        # Call RAG tool
        rag_tool = next((tool for tool in self.tools if tool.name == "rag_question_answering"), None)
        if rag_tool:
            try:
                answer = await rag_tool.ainvoke({"question": question})
                response = f"Based on the Kirby-Bauer experiment: {answer}"
            except Exception as e:
                response = "I can help answer questions about the experiment. Could you be more specific about what you'd like to know?"
                logger.error(f"RAG tool error: {e}")
        else:
            response = "There was an error with the RAG tool. Would you like to move on to the next step?"
        
        state["messages"].append(AIMessage(content=response))
        
        # Ask for follow-up input
        from nat.builder.context import Context
        from nat.data_models.interactive import HumanPromptText
        
        context = Context.get()
        user_input_manager = context.user_interaction_manager
        
        prompt = HumanPromptText(
            text="Do you have any other questions about the experiment, or would you like me to guide you through it?",
            required=True,
            placeholder="Type your response here..."
        )
        
        follow_up_response = await user_input_manager.prompt_user_input(prompt)
        user_response = follow_up_response.content.text
        
        # Add follow-up response to messages
        state["messages"].append(HumanMessage(content=user_response))
        state["user_response"] = user_response
        
        if self.verbose:
            logger.info(f"Q&A A: Answered question and got follow-up: {user_response[:50]}...")
        
        return state
    
    async def _vpg_node(self, state: ARLabState) -> ARLabState:
        """Visual Process Guidance node - guides through experiment."""
        # Call VPG tool
        vpg_tool = next((tool for tool in self.tools if tool.name == "visual_process_guidance"), None)
        if vpg_tool:
            try:
                result = await vpg_tool.ainvoke({"trigger": "start"})
                response = f"🧪 **Experiment Guidance Complete!**\n\n{result}\n\n🎉 **Great work!** We are now done with the experiment."
            except Exception as e:
                response = "I encountered an issue with the visual guidance. Let me try to help you with questions instead."
                logger.error(f"VPG tool error: {e}")
        else:
            response = "There was an error with the VPG tool. Would you like to move on to the next step?"
        
        state["messages"].append(AIMessage(content=response))
        state["session_data"]["vpg_completed"] = True
        
        # Ask for follow-up input
        from nat.builder.context import Context
        from nat.data_models.interactive import HumanPromptText
        
        context = Context.get()
        user_input_manager = context.user_interaction_manager
        
        prompt = HumanPromptText(
            text="Do you have any follow-up questions for me, or would you like to log what we just did and exit?",
            required=True,
            placeholder="Type your response here..."
        )
        
        follow_up_response = await user_input_manager.prompt_user_input(prompt)
        user_response = follow_up_response.content.text
        
        # Add follow-up response to messages
        state["messages"].append(HumanMessage(content=user_response))
        state["user_response"] = user_response
        
        if self.verbose:
            logger.info(f"VPG: Completed experiment guidance and got follow-up: {user_response[:50]}...")
        
        return state
    
    async def _reprompt_node_a(self, state: ARLabState) -> ARLabState:
        """Re-prompt Node A - guides user back on track."""
        response = (
            "I'm here to help you with the Kirby-Bauer disk diffusion assay experiment. "
            "So would you like to ask initial questions about the experiment, or shall we dive into it?"
        )
        
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
        
        if self.verbose:
            logger.info(f"Re-prompt A: Guided user and got response: {user_response[:50]}...")
        
        return state
    
    async def _router_b_node(self, state: ARLabState) -> ARLabState:
        """Router B - determines path after VPG completion."""
        if not state["messages"]:
            return state
            
        # Look for the most recent human message in the conversation
        human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
        
        if not human_messages:
            # No user input yet, default to reprompt
            state["current_node"] = "router_b->reprompt"
            if self.verbose:
                logger.info("Router B: No user input found, routing to reprompt")
            return state
        
        # Get the most recent human message
        last_human_message = human_messages[-1]
        user_response = last_human_message.content.lower()
        
        # Simple routing logic
        if any(phrase in user_response for phrase in ["end", "finish", "done", "exit", "quit", "stop", "log session", "end session"]):
            route = "end"
        elif any(phrase in user_response for phrase in ["question", "ask", "what", "how", "why", "tell me"]):
            route = "qa"
        elif any(phrase in user_response for phrase in ["log", "yes"]):
            route = "log"
        else:
            route = "reprompt"
        
        state["current_node"] = f"router_b->{route}"
        state["user_response"] = user_response
        
        if self.verbose:
            logger.info(f"Router B: Routing to {route} based on: {user_response[:50]}...")
        
        return state
    
    async def _qa_node_b(self, state: ARLabState) -> ARLabState:
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
        rag_tool = next((tool for tool in self.tools if tool.name == "rag_question_answering"), None)
        if rag_tool:
            try:
                answer = await rag_tool.ainvoke({"question": question})
                response = f"Great question! Based on the experiment we just performed: {answer}"
            except Exception as e:
                response = "I can help answer questions about the experiment we just performed. Could you be more specific?"
                logger.error(f"RAG tool error: {e}")
        else:
            response = "There was an error with the RAG tool. Would you like to move on to the next step?"
        
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
        
        if self.verbose:
            logger.info(f"Q&A B: Answered follow-up question and got response: {user_response[:50]}...")
        
        return state
    
    async def _log_session_node(self, state: ARLabState) -> ARLabState:
        """Log Session node - logs the session and ends."""
        # Create conversation summary
        conversation_summary = "AR Lab Assistant session completed successfully. Student performed Kirby-Bauer disk diffusion assay experiment."
        
        # Call logging tool
        log_tool = next((tool for tool in self.tools if tool.name == "experiment_logging"), None)
        if log_tool:
            try:
                result = await log_tool.ainvoke({"conversation_summary": conversation_summary})
                response = f"{result}\n\nSession ended. Thank you for using the AR Lab Assistant!"
            except Exception as e:
                response = "Session logged successfully. Thank you for using the AR Lab Assistant!"
                logger.error(f"Logging tool error: {e}")
        else:
            response = "Thank you for using the AR Lab Assistant!"
        
        state["messages"].append(AIMessage(content=response))
        state["current_node"] = "log_session"
        state["session_data"]["session_ended"] = True
        
        if self.verbose:
            logger.info("Log Session: Session logged and ended")
        
        return state
    
    async def _end_session_node(self, state: ARLabState) -> ARLabState:
        """End Session node - ends the session without logging."""
        response = (
            "Session ended. Thank you for using the AR Lab Assistant! "
            "You can start a new session anytime by sending a new message."
        )
        
        state["messages"].append(AIMessage(content=response))
        state["current_node"] = "end_session"
        state["session_data"]["session_ended"] = True
        
        if self.verbose:
            logger.info("End Session: Session ended without logging")
        
        return state
    
    async def _reprompt_node_b(self, state: ARLabState) -> ARLabState:
        """Re-prompt Node B - guides user back on track after VPG."""
        response = (
            "I'm here to help you with questions about the experiment we just performed. "
            "So would you like to ask more questions about the experiment we just performed, "
            "or would you like to log this session and finish?"
        )
        
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
        
        if self.verbose:
            logger.info(f"Re-prompt B: Guided user and got response: {user_response[:50]}...")
        
        return state
    
    def _route_from_a(self, state: ARLabState) -> Literal["qa", "vpg", "reprompt", "end", "router_b"]:
        """Route from Router A based on user response."""
        current_node = state.get("current_node", "")
        if "router_a->qa" in current_node:
            return "qa"
        elif "router_a->vpg" in current_node:
            return "vpg"
        elif "router_a->end" in current_node:
            return "end"
        elif "router_a->router_b" in current_node:
            return "router_b"
        else:
            return "reprompt"
    
    def _route_from_b(self, state: ARLabState) -> Literal["qa", "log", "reprompt", "end"]:
        """Route from Router B based on user response."""
        current_node = state.get("current_node", "")
        if "router_b->qa" in current_node:
            return "qa"
        elif "router_b->log" in current_node:
            return "log"
        elif "router_b->end" in current_node:
            return "end"
        else:
            return "reprompt"
    
    async def run(self, initial_message: str) -> str:
        """Run the workflow with HITL support."""
        # Start with the initial message from the user
        messages = []
        if initial_message and initial_message.strip():
            # User provided an initial message, add it to the conversation
            messages.append(HumanMessage(content=initial_message))
        
        initial_state = ARLabState(
            messages=messages,
            current_node="",
            session_data={},
            user_response=initial_message if initial_message else ""
        )
        
        result = await self.graph.ainvoke(initial_state)
        
        # Return the last AI message
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage):
                return message.content
        
        return "Workflow completed."
