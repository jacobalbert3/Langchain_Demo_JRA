

from dotenv import load_dotenv
import os
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
from agents.music_agent import get_albums_by_artist, get_tracks_by_artist, music_system_prompt, get_info_about_track
from agents.customer_agent import get_customer_info, edit_customer_info, customer_system_prompt
from agents.general_support import general_support_system_prompt
from utils.contexts import UserContext
from utils.model import model
from pydantic import BaseModel
from typing import Literal
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langsmith import Client
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents.middleware import PIIMiddleware, ToolRetryMiddleware, before_agent
from langgraph.graph import START

memory = SqliteSaver.from_conn_string(":memory:")
load_dotenv()

ls_client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

class CustomState(TypedDict):
    """Custom state"""
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: Optional[int] #TODO: remove:
    username: Optional[str]
    router_choice: Optional[str]
    summary: str  # "account"/ "inventory"


@before_agent(can_jump_to=["end"])
def is_logged_in(state: CustomState, runtime):
    if state.get("customer_id") is None:
        return {
            "messages": [AIMessage(content="Please provide your customer ID to continue.")],
            "jump_to": "end"
        }
    return None

#handle tool errors w/ custom message
@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error!!({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

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
    choice: Literal["account", "account_sensitive", "inventory", "other"]


router_agent = create_agent(model, tools=[], system_prompt=router_system_prompt, response_format=NextRoute)

# Account agent (non-sensitive tools - read-only)
account_agent = create_agent(
    model,
    tools=[get_customer_info],
    context_schema=UserContext,
    system_prompt=customer_system_prompt,
    middleware=[
        ToolRetryMiddleware(
            max_retries=3,  # Retry up to 3 times
            backoff_factor=2.0,  # Exponential backoff multiplier
            initial_delay=1.0,  # Start with 1 second delay
            max_delay=60.0,  # Cap delays at 60 seconds
            jitter=True,  # Add random jitter to avoid thundering herd
        ),handle_tool_errors,
    ],
)

 

# Account sensitive agent (has edit capabilities)
account_sensitive_agent = create_agent(
    model,
    tools=[get_customer_info, edit_customer_info],
    context_schema=UserContext,
    system_prompt=(
        "You help users view and update their profile information. "
        "Use get_customer_info() to show current info. "
        "Use edit_customer_info(parameter, value) to update name, email, or phone when the user requests changes."
    ),
    middleware=[handle_tool_errors, PIIMiddleware("credit_card", strategy="mask")],
)

# Inventory agent
inventory_agent = create_agent(
    model,
    tools=[get_albums_by_artist, get_tracks_by_artist, get_info_about_track],
    context_schema=UserContext,
    system_prompt=music_system_prompt,
    middleware=[handle_tool_errors],
)

# General support agent
general_agent = create_agent(
    model,
    tools=[],
    context_schema=UserContext,
    system_prompt=general_support_system_prompt,
    middleware=[handle_tool_errors],
)


#========NODES========

def router_node(state: CustomState):
    if state.get("customer_id") is None:
        return {"messages": [AIMessage(content="Please provide your customer ID to continue.")], "jump_to": "end"}
    # 1) Call the router agent with the conversation so far
    config = RunnableConfig(tags=["router"], metadata={"customer_id_routed": state.get("customer_id")})
    out = router_agent.invoke({"messages": state["messages"]}, config=config)

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
    if choice not in {"account", "account_sensitive", "inventory", "other"}:
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

def account_sensitive_node(state: CustomState):
    ctx = UserContext(customer_id=state.get("customer_id"))
    out = account_sensitive_agent.invoke({"messages": state["messages"]}, context=ctx)
    final_ai = _final_ai(out)
    if final_ai is None:
        raise RuntimeError("Account sensitive agent did not return a final AI message")
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



def should_summarize_node(state: CustomState):
    '''Passthrough node - routing decision handled by should_summarize_route'''
    # Ensure the last AI message is in the output when routing to END
    # This ensures the response is displayed even when the graph ends here
    last_ai = _last_ai(state)
    if last_ai:
        # Return the last AI message to ensure it's in the final output
        return {"messages": [last_ai]}
    return {}

def should_summarize_route(state: CustomState):
    '''Routing function for should_summarize node'''
    messages = state["messages"]
    if len(messages) > 7:
        return "summarize"
    return END

def summarize(state: CustomState):
    summary = state.get("summary", "")
    if summary:
        sum_message = f"Here is the summary of the conversation so far: {summary} \n\n Please continue the summary including the new information."
    else:
        sum_message = "Create a summary of the conversation so far."
    messages = state["messages"] + [HumanMessage(content=sum_message)]
    response = model.invoke(messages)
    
    # Get the last AI message (the final response from the agent before summarize)
    # This is the message we want to display, not the summary
    last_ai = _last_ai(state)
    
    # Remove all old messages except keep the last 2 (last human + last AI message)
    # The summary is stored in state but not added as a message, so it won't be displayed
    del_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    
    # Return the summary (stored in state for future summaries) and remove old messages
    # The last AI message will remain since we're only removing messages[:-2]
    return {"summary": response.content, "messages": del_messages}
# ---------- Routing helpers ----------
def entry_route(state: CustomState):
    return "user" if state.get("customer_id") is None else "router"

def route_from_user(state: CustomState):
    return "router" if state.get("customer_id") is not None else END

def route_from_router(state: CustomState):
    return state.get("router_choice", "other")

# ---------------- Graph ----------------
workflow = StateGraph(CustomState)
workflow.add_node("router", router_node)
workflow.add_node("account", account_node)
workflow.add_node("account_sensitive", account_sensitive_node)
workflow.add_node("inventory", inventory_node)
workflow.add_node("general", general_node)
workflow.add_node("summarize", summarize)
workflow.add_node("should_summarize", should_summarize_node)
workflow.add_conditional_edges("router", route_from_router, {
    "account": "account", 
    "account_sensitive": "account_sensitive", 
    "inventory": "inventory", 
    "other": "general"
})
workflow.add_edge(START, "router")
workflow.add_edge("general", "should_summarize")
workflow.add_edge("account", "should_summarize")
workflow.add_edge("account_sensitive", "should_summarize")
workflow.add_edge("inventory", "should_summarize")
workflow.add_conditional_edges("should_summarize", should_summarize_route, {"summarize": "summarize", END: END})
workflow.add_edge("summarize", END)

graph = workflow.compile(
    checkpointer=memory,
    interrupt_before=["account_sensitive"],
)