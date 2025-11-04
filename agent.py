from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from pydantic import BaseModel, Field
from utils.database import db
from langgraph.graph import END
import json
import re
from langgraph.prebuilt import ToolNode, tools_condition
from utils.fallback import create_tool_node_with_fallback
from functools import partial
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.music_agent import song_recc_chain, get_albums_by_artist, get_tracks_by_artist, check_for_songs
from agents.customer_agent import customer_chain, get_customer_info, edit_customer_info
from agents.general_support import chain
from utils.assistant import Assistant
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated, Optional, TypedDict
load_dotenv()



class CustomState(TypedDict):
    """Custom state with messages and additional variables"""
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: Optional[int]
    router_choice: Optional[str]  # "account"/ "inventory"

#why is this necessary?? -> creates one argument message (i.e ready for pipeline)
def add_name(message, name): #TAGS message with node name (helps router know who created it)
    _dict = message.model_dump()
    _dict["name"] = name
    return AIMessage(**_dict)

#gets the last AI message
def _get_last_ai_message(messages):
    for m in messages[::-1]: #reverse order to get last message
        if isinstance(m, AIMessage):
            return m
    return None

def _is_tool_call(msg):
    return hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0


# Separate tools for each agent
account_tools = [get_customer_info, edit_customer_info]
# Use fallback wrapper to handle tool errors gracefully
account_tool_node_base = create_tool_node_with_fallback(account_tools)
inventory_tools = [get_albums_by_artist, get_tracks_by_artist, check_for_songs]
# Use fallback wrapper to handle tool errors gracefully
inventory_tool_node_base = create_tool_node_with_fallback(inventory_tools)

# Define sensitive vs non-sensitive tools
# Sensitive tools require human approval (human-in-the-loop)
SENSITIVE_TOOLS = [
    "edit_customer_info",  # Modifies customer data
]
NON_SENSITIVE_TOOLS = [
    "get_customer_info",
    "get_albums_by_artist",
    "get_tracks_by_artist",
    "check_for_songs",
]

# Account tools node - only customer tools available
def account_tools_node(state: CustomState, config=None):
    """Tools node for account agent - only customer tools available"""
    messages = state.get("messages", [])
    customer_id = state.get("customer_id")
    
    # Prepare config with customer_id for edit_customer_info (which uses config)
    tool_config = dict(config or {})
    tool_config["tool_runtime"] = 
    if customer_id is not None:
        if "configurable" not in tool_config:
            tool_config["configurable"] = {}
        tool_config["configurable"]["customer_id"] = customer_id
    
    result = account_tool_node_base.invoke(state)
    # ToolNode returns dict with "messages" key when given dict input
    if isinstance(result, dict) and "messages" in result:
        return {"messages": result["messages"]}
    # Fallback: if result is already a list (shouldn't happen but safe)
    return {"messages": result if isinstance(result, list) else [result]}

# Inventory tools node - only music tools available
def inventory_tools_node(state: CustomState):
    """Tools node for inventory agent - only music tools available"""
    messages = state.get("messages", [])
    result = inventory_tool_node_base.invoke(state)
    # ToolNode returns dict with "messages" key when given dict input
    if isinstance(result, dict) and "messages" in result:
        return {"messages": result["messages"]}
    # Fallback: if result is already a list (shouldn't happen but safe)
    return {"messages": result if isinstance(result, list) else [result]}

# Filter out routing tool calls to avoid confusion
def _filter_out_routes(messages):
    """Filter out routing tool calls (general/router) from messages."""
    ms = []
    for m in messages:
        if _is_tool_call(m):
            # Skip routing tool calls
            if getattr(m, "name", None) in ["general", "router"]:
                continue
        ms.append(m)
    return ms

# Node functions for StateGraph - accept state, return state updates
def general_node(state: CustomState):
    messages = state.get("messages", [])
    filtered = _filter_out_routes(messages)
    result = chain.invoke(filtered)
    named_result = add_name(result, "general")
    return {"messages": [named_result]}

def music_node(state: CustomState):
    messages = state.get("messages", [])
    filtered = _filter_out_routes(messages)
    result = song_recc_chain.invoke(filtered)
    named_result = add_name(result, "music")
    return {"messages": [named_result]}

def user_node(state: CustomState):
    """User node: extracts customer_id from messages or asks for it."""
    customer_id = state.get("customer_id")
    
    # If already set, keep it
    if customer_id is not None:
        return {"customer_id": customer_id}
     
    # No customer_id found - ask for it
    from langchain_core.messages import AIMessage
    ask_message = AIMessage(content="I need your customer ID to help you. Please provide your customer ID number if you would like information about your account.")
    return {"messages": [ask_message], "customer_id": None}

def router_node(state: CustomState):
    """Router node: uses LLM to decide whether to go to Account, Inventory, or respond directly."""
    messages = state.get("messages", [])
    
    # Filter out any routing tool calls to avoid confusion
    filtered_messages = [m for m in messages if not (_is_tool_call(m) and getattr(m, "name", None) == "router")]
    
    # Use Assistant-wrapped chain to ensure non-empty responses
    result = router_assistant.invoke(filtered_messages)
    
    # If LLM responded directly (no tool call), treat as "other" and return response
    if not _is_tool_call(result) and hasattr(result, 'content') and result.content:
        return {"messages": [result], "router_choice": "other"}
    
    # Extract the routing choice from tool calls
    choice = "other"  # default to other for general queries
    if _is_tool_call(result):
        tool_calls = getattr(result, "tool_calls", [])
        if tool_calls:
            choice = tool_calls[0].get("args", {}).get("choice", "other")
    
    # If choice is "other" but we got a tool call, respond directly
    if choice == "other":
        # Re-invoke without tool to get a direct response
        from agents.general_support import get_messages
        from utils.model import model
        from langchain_core.messages import SystemMessage
        response_chain = get_messages | model  # Chain without tool binding
        # Use Assistant to ensure non-empty response
        other_assistant = Assistant(response_chain)
        response = other_assistant.invoke(filtered_messages)
        return {"messages": [response], "router_choice": "other"}
    
    return {"router_choice": choice}

def account_node(state: CustomState):
    """Account node: uses customer_agent to handle account-related operations."""
    messages = state.get("messages", [])
    # Filter out routing tool calls
    filtered = _filter_out_routes(messages)
    # Use Assistant-wrapped chain to ensure non-empty responses
    result = account_assistant.invoke(filtered)
    named_result = add_name(result, "account")
    return {"messages": [named_result]}

def inventory_node(state: CustomState):
    """Inventory node: uses music_agent to handle inventory/music operations."""
    messages = state.get("messages", [])
    # Filter out routing tool calls
    filtered = _filter_out_routes(messages)
    # Use Assistant-wrapped chain to ensure non-empty responses
    result = inventory_assistant.invoke(filtered)
    named_result = add_name(result, "inventory")
    return {"messages": [named_result]}


def customer_node(state: CustomState):
    messages = state.get("messages", [])
    filtered = _filter_out_routes(messages)
    result = customer_chain.invoke(filtered)
    named_result = add_name(result, "customer")
    return {"messages": [named_result]}

#saves and retrieves checkpoints (state) 
memory = SqliteSaver.from_conn_string(":memory:") #external memory

# Router conditionally routes to Account, Inventory, or Other based on router_choice
def route_from_router(state: CustomState):
    """Route based on router's choice."""
    choice = state.get("router_choice", "other")
    if choice == "inventory":
        return "inventory"
    elif choice == "account":
        return "account"
    else:
        return "other"
# Note: Using LangGraph's prebuilt tools_condition instead of custom routing functions
# tools_condition automatically checks if the last AI message has tool_calls
# Returns "tools" if tool calls exist, "__end__" if not
# It works with StateGraph by checking state["messages"]

def route_sensitive_tools(state: CustomState) -> str:
    """
    Route based on whether the tools being called are sensitive or non-sensitive.
    Sensitive tools require human-in-the-loop approval.
    Returns: "sensitive" or "non_sensitive"
    """
    messages = state.get("messages", [])
    
    # Get the last AI message to check for tool calls
    last_ai_message = _get_last_ai_message(messages)
    
    if not last_ai_message or not _is_tool_call(last_ai_message):
        # No tool calls, shouldn't reach here if tools_condition worked correctly
        return "non_sensitive"
    
    # Check if any of the tool calls are sensitive
    tool_calls = getattr(last_ai_message, "tool_calls", [])
    for tool_call in tool_calls:
        tool_name = tool_call.get("name", "")
        if tool_name in SENSITIVE_TOOLS:
            return "sensitive"
    
    # All tool calls are non-sensitive
    return "non_sensitive"

def route_account_tools(state: CustomState) -> str:
    """
    Combined routing for account tools: first check if there are tool calls,
    then check if they're sensitive or non-sensitive.
    Returns: "sensitive", "non_sensitive", or "__end__"
    """
    # First check if there are tool calls at all
    tool_check = tools_condition(state)
    if tool_check == "__end__":
        return "__end__"
    
    # If we have tool calls, route based on sensitivity
    sensitivity = route_sensitive_tools(state)
    return sensitivity

def approval_request_node(state: CustomState):
    """
    Node that prepares the approval request by extracting tool call info
    and showing the new value that will be set.
    """
    messages = state.get("messages", [])
    customer_id = state.get("customer_id")
    
    # Get the last AI message with tool calls
    last_ai_message = _get_last_ai_message(messages)
    
    if not last_ai_message or not _is_tool_call(last_ai_message):
        # Shouldn't happen, but if it does, just proceed
        return {}
    
    tool_calls = getattr(last_ai_message, "tool_calls", [])
    if not tool_calls:
        return {}
    
    # Find the edit_customer_info tool call
    edit_call = None
    for tool_call in tool_calls:
        if tool_call.get("name") == "edit_customer_info":
            edit_call = tool_call
            break
    
    if not edit_call or not customer_id:
        return {}
    
    # Extract parameter and new value from tool call
    args = edit_call.get("args", {})
    parameter = args.get("parameter", "")
    new_value = args.get("value", "")
    
    # Create approval message with new value
    approval_message = AIMessage(
        content=f"""⚠️ **CONFIRMATION REQUIRED** ⚠️

You are about to update your customer information:

**Parameter:** {parameter}
**New Value:** {new_value}

Are you sure you want to proceed with this update?
Please respond with 'yes' or 'no'."""
    )
    
    return {"messages": [approval_message]}

def human_approval_node(state: CustomState, config=None):
    """
    Human-in-the-loop approval node for sensitive tools.
    Checks if user confirmed, then executes the tools if approved.
    """
    messages = state.get("messages", [])
    customer_id = state.get("customer_id")
    
    # Get the last human message to check for confirmation
    last_human_message = None
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            last_human_message = m
            break
    
    # Prepare config with customer_id for edit_customer_info (which uses config)
    tool_config = config or {}
    if customer_id is not None:
        if "configurable" not in tool_config:
            tool_config["configurable"] = {}
        tool_config["configurable"]["customer_id"] = customer_id
    
    # Check if user confirmed
    if last_human_message:
        content = last_human_message.content.lower().strip()
        if content in ['yes', 'y', 'confirm', 'proceed']:
            # User approved - execute the tools
            result = account_tool_node_base.invoke({"messages": messages}, config=tool_config)
            if isinstance(result, dict) and "messages" in result:
                return {"messages": result["messages"]}
            return {"messages": result if isinstance(result, list) else [result]}
        elif content in ['no', 'n', 'cancel', 'abort']:
            # User declined - return cancellation message
            cancel_message = AIMessage(
                content="Update cancelled. Your information has not been changed."
            )
            return {"messages": [cancel_message]}
    
    # No clear confirmation yet - ask again (this shouldn't happen if interrupt works)
    return {
        "messages": [
            AIMessage(content="Please confirm: Type 'yes' to proceed or 'no' to cancel.")
        ]
    }

def route_from_user(state: CustomState):
    """Route from user node: if customer_id set, go to router; otherwise END (wait for user)."""
    customer_id = state.get("customer_id")
    if customer_id is not None:
        return "router"  # Customer ID found, proceed to router
    else:
        return END  # Asked for customer_id, wait for user's next message

# Conditional entry point: check if customer_id is set
def entry_route(state: CustomState):
    """Entry point: if customer_id not set, go to user; otherwise go to router."""
    customer_id = state.get("customer_id")
    if customer_id is None:
        return "user"  # Need to set customer_id first
    else:
        return "router"  # Already authenticated, go straight to router



# Wrap chains with Assistant for reliable responses (handles empty responses)
router_assistant = Assistant(chain)
account_assistant = Assistant(customer_chain)
inventory_assistant = Assistant(song_recc_chain)

# Define Graph
workflow = StateGraph(CustomState)
workflow.add_node("user", user_node)
workflow.add_node("router", router_node)
workflow.add_node("account", account_node)
workflow.add_node("inventory", inventory_node)

# Set conditional entry point
workflow.set_conditional_entry_point(entry_route, {
    "user": "user",
    "router": "router"
})

# User node routes: if customer_id set → router, otherwise → END (wait for user)
workflow.add_conditional_edges("user", route_from_user, {
    "router": "router",
    END: END
})
workflow.add_node("account_tools", account_tools_node)
workflow.add_node("inventory_tools", inventory_tools_node)
workflow.add_node("approval_request", approval_request_node)
workflow.add_node("human_approval", human_approval_node)

workflow.add_conditional_edges("router", route_from_router, {
    "account": "account",
    "inventory": "inventory",
    "other": END  # "other" means respond directly and end
})

# Account node routing: check for tools, then route sensitive vs non-sensitive
workflow.add_conditional_edges("account", route_account_tools, {
    "sensitive": "approval_request",
    "non_sensitive": "account_tools",
    "__end__": END
})

# After approval request, go to human_approval (which will be interrupted)
workflow.add_edge("approval_request", "human_approval")

workflow.add_conditional_edges("inventory", tools_condition, {
    "tools": "inventory_tools",
    "__end__": END
})
workflow.add_edge("account_tools", "account")
workflow.add_edge("inventory_tools", "inventory")
workflow.add_edge("human_approval", "account")  # After approval, return to account node

# Compile graph with interrupt before human_approval node
# This will pause execution and wait for user input before executing sensitive tools
# NOTE: For Studio, interrupts work with Studio's built-in checkpointer
# For local testing without Studio, uncomment the memory checkpointer above
graph = workflow.compile(
    interrupt_before=["human_approval"]
)

