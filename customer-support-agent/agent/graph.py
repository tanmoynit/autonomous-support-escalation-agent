"""
The LangGraph agent: a small state machine with two nodes.

  START -> agent -> (tool calls?) -> tools -> agent -> ... -> END

`agent` calls the LLM (with tools bound). If the LLM's response contains tool
calls, we route to `tools` to actually execute them (hitting the mock
database / vector store), then loop back to `agent` so the model can see the
results and decide what to do next - answer the user, or call another tool.
"""

import os
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from .tools import get_tools

CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = """You are an autonomous customer support agent for an online electronics store.

Your job is to resolve customer issues by using your tools - you must never
rely on memory or assumption for anything about a specific customer, order,
or company policy.

Hard rules (guardrails):
1. Never state or imply a policy fact (return windows, warranty terms,
   shipping rules, etc.) without first calling search_knowledge_base.
2. Never discuss, confirm, or act on a specific order without first calling
   verify_purchase to confirm the order ID and email actually match.
3. Before processing any refund, you must call check_return_eligibility.
   If it says NOT ELIGIBLE, you must NOT call process_refund - instead call
   create_escalation_ticket.
4. process_refund itself enforces the $200 auto-approval limit and the
   30-day return window. If it denies the refund, trust that result, explain
   it to the customer, and create an escalation ticket.
5. Never invent discounts, credits, goodwill refunds, or exceptions to
   policy that are not returned by search_knowledge_base. If a customer asks
   for something outside written policy, escalate to a human instead of
   agreeing to it.
6. If the customer is upset, insists on a human, or the situation involves
   warranty claims, fraud, damaged/missing items, or anything ambiguous,
   create an escalation ticket even if a refund isn't strictly involved.
7. Be transparent with the customer about what you're checking and why.
   Keep responses concise, empathetic, and professional.
"""


def build_agent():
    tools = get_tools()
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: MessagesState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.set_entry_point("agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,  # routes to "tools" if the last message has tool_calls, else END
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge("tools", "agent")

    checkpointer = MemorySaver()  # in-memory per-session conversation state
    return graph_builder.compile(checkpointer=checkpointer)
