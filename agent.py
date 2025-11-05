

from dotenv import load_dotenv
import os
import json
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
from utils.contexts import AccountState, InventoryState, GeneralState
from utils.model import model
from utils.prompt_injection import prompt_injection_guard
from pydantic import BaseModel
from typing import Literal
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langsmith import Client
from langchain_core.messages import RemoveMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents.middleware import PIIMiddleware, ToolRetryMiddleware, before_agent, HumanInTheLoopMiddleware
from langgraph.graph import START
from langchain.agents import AgentState
from langchain.tools import tool, ToolRuntime, InjectedToolCallId
from langgraph.types import Command
load_dotenv()

#memory for checkpointer (if not showing through studio)
memory = SqliteSaver.from_conn_string(":memory:")


ls_client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

_old_after_model = HumanInTheLoopMiddleware.after_model

def _patched_after_model(self, hitl_response, runtime, **kwargs):
    """Allow HITL response to be a raw JSON string (Studio bug workaround)."""
    if isinstance(hitl_response, str):
        try:
            hitl_response = json.loads(hitl_response)
        except Exception:
            pass
    return _old_after_model(self, hitl_response, runtime, **kwargs)

HumanInTheLoopMiddleware.after_model = _patched_after_model



#--------STATE DEFINITIONS---------------

class CustomState(TypedDict):
    """Custom state"""
    messages: Annotated[list[AnyMessage], add_messages] #reducer to add messages to state
    customer_id: Optional[int] 
    username: Optional[str]
    summary: str

#STATE for supervisor agent
class SupervisorState(AgentState):
    customer_id: Optional[int]
    username: Optional[str]
    summary: str | None = None
    router_choice: Literal["account", "inventory", "general"] | None = None
#handle tool error so it doesn't break the workflow
@wrap_tool_call
def handle_tool_errors(request, handler):
    try:
        return handler(request)
    except Exception as e:
        return ToolMessage(
            content=f"Tool error!!({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

def supervisor_node(state: CustomState):
    customer_id = state.get("customer_id")
    if not customer_id:
        return {"messages": [AIMessage(content="Please provide your customer ID to continue.")]}
    out = supervisor.invoke({
        "messages": state["messages"],
        "customer_id": customer_id,
        "username": state.get("username"),
    })
    final_ai = _final_ai(out)
    if final_ai is None:
        raise RuntimeError("Supervisor returned no final AI message")
    return {"messages": [final_ai]}

account_agent = create_agent(
    model,
    tools=[get_customer_info, edit_customer_info],
    system_prompt=customer_system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"edit_customer_info": True}
        ), 
        PIIMiddleware(
            "email",
            strategy="mask",
            apply_to_output=True,  # Enable PII detection on tool outputs for testing
        )],
    state_schema=AccountState, #defines what we can read at runtime
)

#Inventory agent
inventory_agent = create_agent(
    model,
    tools=[get_albums_by_artist, get_tracks_by_artist, get_info_about_track],
    system_prompt=music_system_prompt,
    middleware=[ToolRetryMiddleware(
            max_retries=3,  # Retry up to 3 times
            backoff_factor=2.0,  # Exponential backoff multiplier
            initial_delay=1.0,  # Start with 1 second delay
            max_delay=60.0,  # Cap delays at 60 seconds
            jitter=True,  # Add random jitter to avoid thundering herd (overloading the server)
        ), handle_tool_errors],
    state_schema=InventoryState,
)

general_agent = create_agent(
    model,
    tools=[],  # no tools
    system_prompt=general_support_system_prompt,
    middleware=[handle_tool_errors],
    state_schema=GeneralState,  # you already have this class
)
@tool(
    "account_agent_tool",
    description="Use for viewing or updating account/profile/order/billing (read & edit: name, email, phone)."
)
def call_account_agent_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime[None, SupervisorState],
) -> Command:
    cid = runtime.state.get("customer_id")

    #example for passing new data into sub agent (simplified)
    length = "long" if len(query) > 100 else "short"
    res = account_agent.invoke({
        "messages": [{"role": "user", "content": query}],
        "customer_id": cid,
        "length": length,
    })

    final_text = res["messages"][-1].content
    return Command(update={
        "messages": [
            ToolMessage(content=final_text, tool_call_id=tool_call_id)
        ]
    })

@tool("inventory_agent_tool", description="Use for music inventory lookups: albums, tracks, track details.")
def call_inventory_agent_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime[None, SupervisorState],
) -> Command:
    res = inventory_agent.invoke({
        "messages": [{"role": "user", "content": query}],
        "customer_id": runtime.state.get("customer_id"),
    })
    final_text = res["messages"][-1].content
    return Command(update={
        "messages": [ToolMessage(content=final_text, tool_call_id=tool_call_id)]
    })


@tool("general_agent_tool", description="Use for general support questions not about account or inventory.")
def call_general_agent_tool(
    query: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    runtime: ToolRuntime[None, SupervisorState],
) -> Command:
    res = general_agent.invoke({
        "messages": [{"role": "user", "content": query}],
        # if you want, you can pass customer_id along for consistency
        "customer_id": runtime.state.get("customer_id"),
    })
    final_text = res["messages"][-1].content
    return Command(update={
        "messages": [ToolMessage(content=final_text, tool_call_id=tool_call_id)]
    })

supervisor = create_agent(
    model,
    tools=[call_account_agent_tool, call_inventory_agent_tool, call_general_agent_tool],           # start with ONLY account tool
    system_prompt=router_system_prompt,
    middleware=[prompt_injection_guard],
    state_schema=SupervisorState,
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


def should_summarize_node(state: CustomState):
    '''Passthrough node - routing decision handled by should_summarize_route'''
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
    last_ai = _last_ai(state)
    del_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": del_messages}

# ---------------- Graph ----------------
workflow = StateGraph(CustomState)
workflow.add_node("supervisor", supervisor_node)
workflow.add_edge(START, "supervisor")
workflow.add_node("should_summarize", should_summarize_node)
workflow.add_node("summarize", summarize)
workflow.add_edge("supervisor", "should_summarize")
workflow.add_conditional_edges("should_summarize", should_summarize_route, {"summarize": "summarize", END: END})
workflow.add_edge("summarize", END)
graph = workflow.compile(checkpointer=memory)