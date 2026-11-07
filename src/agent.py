"""Building the WIiTek agent and the evaluation judge model, and asking the agent a question.

The agent is a LangChain tool-calling agent: an OpenAI chat model, the
read-only database tools from ``database.repository`` and the system prompt
from ``templates``. It decides by itself which tools to call for a question.

The agent remembers conversations: every question is asked within a
conversation (``thread_id``), and the agent sees that conversation's earlier
questions, answers and tool results. A new ``thread_id`` starts from nothing.
"""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from config import AgentSettings, JudgeSettings
from database.repository import TOOLS
from templates import SYSTEM_PROMPT


def build_agent_model(settings: AgentSettings) -> ChatOpenAI:
    """Create the OpenAI chat model the agent thinks with.

    Args:
        settings: Model name, temperature and reasoning effort; this function does not read the environment itself.

    Returns:
        ChatOpenAI: The configured model. Settings left as None are not sent to the API at all.

    Raises:
        RuntimeError: OPENAI_API_KEY is not set.
    """
    return ChatOpenAI(
        model=settings.model,
        temperature=settings.temperature,
        reasoning_effort=settings.reasoning_effort,
        api_key=settings.require_api_key(),
    )


def build_judge_model(settings: JudgeSettings) -> ChatOpenAI:
    """Create the chat model that scores answers during evaluation.

    It has its own settings so it can be a different model from the agent: a judge built on the same model tends to miss
    the same mistakes.

    Args:
        settings: The judge's model name, temperature and reasoning effort.

    Returns:
        ChatOpenAI: The configured model.

    Raises:
        RuntimeError: OPENAI_API_KEY is not set.
    """
    return ChatOpenAI(
        model=settings.model,
        temperature=settings.temperature,
        reasoning_effort=settings.reasoning_effort,
        api_key=settings.require_api_key(),
    )


def build_agent(settings: AgentSettings) -> CompiledStateGraph:
    """Create the WIiTek agent: the chat model, the database tools and the system prompt.

    The agent keeps every conversation in process memory (``InMemorySaver``), so conversations are lost when the process
    exits.

    Args:
        settings: Settings for the agent's chat model.

    Returns:
        CompiledStateGraph: The agent; pass it to ``ask``.

    Raises:
        RuntimeError: OPENAI_API_KEY is not set.
    """
    return create_agent(
        model=build_agent_model(settings),
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def ask(agent: CompiledStateGraph, question: str, thread_id: str) -> list[BaseMessage]:
    """Ask the agent one question within a conversation and return this turn's messages.

    The agent sees everything said earlier in the same conversation, including tool results, so a follow-up like "a w
    2024?" is understood. Only this turn's messages are returned, not the whole conversation: the evaluation and the
    chat read the tool calls from them, and earlier turns' calls must not be counted again.

    Args:
        agent: The agent from ``build_agent``.
        question: The user's question, in Polish or English.
        thread_id: Which conversation the question belongs to; a new id starts a conversation with no history.

    Returns:
        list[BaseMessage]: This turn's messages in order, starting with the question; the last one is the final answer.
            Tool calls the agent made are in ``AIMessage.tool_calls``, and the rows each tool returned are
            in ``ToolMessage.artifact``.
    """
    conversation = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        {"configurable": {"thread_id": thread_id}},
    )["messages"]
    # invoke returns the whole conversation; every turn starts with the user's
    # question, so the current turn begins at the last HumanMessage.
    # E.g. [Human q1, AI, Tool, AI, Human q2, AI, Tool, AI] -> starts at index 4.
    turn_start = max(index for index, message in enumerate(conversation) if isinstance(message, HumanMessage))
    return conversation[turn_start:]
