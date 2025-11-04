

from dotenv import load_dotenv
from typing import Annotated, Optional
try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import AnyMessage, add_messages
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from agents.router_agent import router_system_prompt
from agents.music_agent import get_albums_by_artist, get_tracks_by_artist, check_for_songs, music_system_prompt
from agents.customer_agent import get_customer_info, edit_customer_info, customer_system_prompt
from agents.general_support import general_support_system_prompt
from utils.contexts import UserContext
from utils.model import model
from pydantic import BaseModel
from typing import Literal

load_dotenv()

class CustomState(TypedDict):
    """Custom state"""
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: Optional[int] #TODO: remove:
    router_choice: Optional[str]  # "account"/ "inventory"

#========AGENTS========

def _final_ai(out: dict) -> AIMessage | None:
    for m in reversed(out.get("messages", [])):
        if isinstance(m, AIMessage):
            return m
    return None
def _last_ai(state: CustomState) -> AIMessage | None:
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage):
            return m
    return None


class NextRoute(BaseModel):
    choice: Literal["account", "inventory", "other"]

router_agent = create_agent(model, tools=[], system_prompt=router_system_prompt, response_format=NextRoute)

# Account agent (non-sensitive tools)
account_agent = create_agent(
    model,
    tools=[get_customer_info],
    context_schema=UserContext,
    system_prompt=customer_system_prompt,
)

# Inventory agent
inventory_agent = create_agent(
    model,
    tools=[get_albums_by_artist, get_tracks_by_artist, check_for_songs],
    context_schema=UserContext,
    system_prompt=music_system_prompt,
)

# General support agent
general_agent = create_agent(
    model,
    tools=[],
    context_schema=UserContext,
    system_prompt=general_support_system_prompt,
)

# Edit agent (sensitive tool) — only after approval
edit_agent = create_agent(
    model,
    tools=[edit_customer_info],
    context_schema=UserContext,
    system_prompt=(
        "You may update customer info ONLY after explicit user approval. "
        "Use edit_customer_info(parameter, value) to change name/email/phone."
    ),
)

#========NODES========
def user_node(state: CustomState):
    if state.get("customer_id") is not None:
        return {}
    return {
        "messages": [AIMessage(content="Please provide your customer ID to continue.")],
        "customer_id": None,
    }

def router_node(state: CustomState):
    # 1) Call the router agent with the conversation so far
    out = router_agent.invoke({"messages": state["messages"]})

    # 2) Get the structured decision safely
    sr = out.get("structured_response")
    if sr is None:
        # You can raise, or fallback to a text heuristic using the last AI message
        raise RuntimeError("Router did not return a structured_response")

    # 3) Normalize to a dict (handles Pydantic v2 object or already-a-dict)
    if hasattr(sr, "model_dump"):
        sr = sr.model_dump()

    # 4) Read the choice from the normalized dict
    choice = sr.get("choice")

    # 5) Validate the value to keep your state clean
    if choice not in {"account", "inventory", "other"}:
        raise ValueError(f"Invalid choice: {choice!r}")

    # 6) Return a state patch that your graph expects
    return {"router_choice": choice}

def account_node(state: CustomState):
    ctx = UserContext(customer_id=state.get("customer_id"))
    out = account_agent.invoke({"messages": state["messages"]}, context=ctx)
    final_ai = _final_ai(out)
    if final_ai is None:
        raise RuntimeError("Account agent did not return a final AI message")
    return {"messages": [final_ai]}

def inventory_node(state: CustomState):
    # Inventory agent can also take context (not strictly needed)
    ctx = UserContext(customer_id=state.get("customer_id"))
    out = inventory_agent.invoke({"messages": state["messages"]}, context=ctx)
    final_ai = _final_ai(out)
    if final_ai is None:
        raise RuntimeError("Inventory agent did not return a final AI message")
    return {"messages": [final_ai]}

def general_node(state: CustomState):
    # General support agent responds normally to general questions
    ctx = UserContext(customer_id=state.get("customer_id"))
    out = general_agent.invoke({"messages": state["messages"]}, context=ctx)
    final_ai = _final_ai(out)
    if final_ai is None:
        raise RuntimeError("General agent did not return a final AI message")
    return {"messages": [final_ai]}


def detect_edit_intent(state: CustomState) -> str:
    # naive text heuristic — OK to tighten later
    ai = _last_ai(state)
    text = (ai.content or "").lower() if ai else ""
    if any(k in text for k in ["update my", "change my", "edit my", "modify my"]):
        return "approval_request"
    return "__end__"

def approval_request_node(state: CustomState):
    msg = AIMessage(
        content=("⚠️ Confirm updating your profile. Reply 'yes' to proceed or 'no' to cancel. "
                 "If proceeding, specify the change, e.g., `change my email to alex@example.com`.")
    )
    return {"messages": [msg]}

def human_approval_node(state: CustomState):
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    if not last_human:
        return {"messages": [AIMessage(content="Please reply 'yes' or 'no'.")]}
    t = last_human.content.strip().lower()
    if t in {"no", "n", "cancel"}:
        return {"messages": [AIMessage(content="Update cancelled. No changes made.")]}
    if t not in {"yes", "y", "confirm", "proceed"}:
        return {"messages": [AIMessage(content="Please reply 'yes' or 'no'.")]}
    # approved → run the sensitive tool through the edit agent
    ctx = UserContext(customer_id=state.get("customer_id"))
    out = edit_agent.invoke({"messages": state["messages"]}, context=ctx)
    final_ai = _final_ai(out)
    if final_ai is None:
        raise RuntimeError("Edit agent did not return a final AI message")
    return {"messages": [final_ai]}

# ---------- Routing helpers ----------
def entry_route(state: CustomState):
    return "user" if state.get("customer_id") is None else "router"

def route_from_user(state: CustomState):
    return "router" if state.get("customer_id") is not None else END

def route_from_router(state: CustomState):
    return state.get("router_choice", "other")

# ---------------- Graph ----------------
workflow = StateGraph(CustomState)
workflow.add_node("user", user_node)
workflow.add_node("router", router_node)
workflow.add_node("account", account_node)
workflow.add_node("inventory", inventory_node)
workflow.add_node("general", general_node)
workflow.add_node("approval_request", approval_request_node)
workflow.add_node("human_approval", human_approval_node)

workflow.set_conditional_entry_point(entry_route, {"user": "user", "router": "router"})
workflow.add_conditional_edges("user", route_from_user, {"router": "router", END: END})
workflow.add_conditional_edges("router", route_from_router, {"account": "account", "inventory": "inventory", "other": "general"})
workflow.add_edge("general", END)
workflow.add_conditional_edges("account", detect_edit_intent, {"approval_request": "approval_request", "__end__": END})
workflow.add_edge("approval_request", "human_approval")
workflow.add_edge("human_approval", "account")

graph = workflow.compile()