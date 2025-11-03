from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from utils.database import db
from langchain_core.messages import AIMessage
from langgraph.graph import END
import json
from langgraph.prebuilt import ToolNode
from functools import partial
from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.music_agent import song_recc_chain, get_albums_by_artist, get_tracks_by_artist, check_for_songs
from agents.customer_agent import customer_chain, get_customer_info, edit_customer_info
from agents.general_support import chain
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
account_tool_node_base = ToolNode(account_tools)
inventory_tools = [get_albums_by_artist, get_tracks_by_artist, check_for_songs]
inventory_tool_node_base = ToolNode(inventory_tools)

# Account tools node - only customer tools available
def account_tools_node(state: CustomState):
    """Tools node for account agent - only customer tools available"""
    messages = state.get("messages", [])
    result = account_tool_node_base.invoke(messages)
    return {"messages": result}

# Inventory tools node - only music tools available
def inventory_tools_node(state: CustomState):
    """Tools node for inventory agent - only music tools available"""
    messages = state.get("messages", [])
    result = inventory_tool_node_base.invoke(messages)
    return {"messages": result}

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
    """User node: sets/updates customer_id in state. For now, hardcoded to 1."""
    # TODO: Extract customer_id from user message, config, or authentication
    customer_id = 1
    
    return {"customer_id": customer_id}

def router_node(state: CustomState):
    """Router node: uses LLM to decide whether to go to Account or Inventory."""
    messages = state.get("messages", [])
    
    # Filter out any routing tool calls to avoid confusion
    filtered_messages = [m for m in messages if not (_is_tool_call(m) and getattr(m, "name", None) == "router")]
    
    # Use general_support chain (router agent) to get routing decision
    result = chain.invoke(filtered_messages)
    
    # Extract the routing choice from tool calls
    choice = "account"  # default
    if _is_tool_call(result):
        tool_calls = getattr(result, "tool_calls", [])
        if tool_calls:
            choice = tool_calls[0].get("args", {}).get("choice", "account")
    
    return {"router_choice": choice}

def account_node(state: CustomState):
    """Account node: uses customer_agent to handle account-related operations."""
    messages = state.get("messages", [])
    # Filter out routing tool calls
    filtered = _filter_out_routes(messages)
    # Use customer agent chain
    result = customer_chain.invoke(filtered)
    named_result = add_name(result, "account")
    return {"messages": [named_result]}

def inventory_node(state: CustomState):
    """Inventory node: uses music_agent to handle inventory/music operations."""
    messages = state.get("messages", [])
    # Filter out routing tool calls
    filtered = _filter_out_routes(messages)
    # Use music agent chain
    result = song_recc_chain.invoke(filtered)
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

# Router conditionally routes to Account or Inventory based on router_choice
def route_from_router(state: CustomState):
    """Route based on router's choice."""
    choice = state.get("router_choice", "account")
    if choice == "inventory":
        return "inventory"
    else:
        return "account"
# Account and Inventory nodes can call tools or respond
def route_from_account(state: CustomState):
    """Route from account node: if tool call, go to tools; otherwise END."""
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and _is_tool_call(last_message):
            return "tools"
    return END

def route_from_inventory(state: CustomState):
    """Route from inventory node: if tool call, go to tools; otherwise END."""
    messages = state.get("messages", [])
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, AIMessage) and _is_tool_call(last_message):
            return "tools"
    return END




# Define Graph
workflow = StateGraph(CustomState)
workflow.add_node("user", user_node)
workflow.add_node("router", router_node)
workflow.set_entry_point("user")  # Start at user node
workflow.add_node("account", account_node)
workflow.add_node("inventory", inventory_node)
workflow.add_edge("user", "router")  # User → Router
workflow.add_node("account_tools", account_tools_node)
workflow.add_node("inventory_tools", inventory_tools_node)

workflow.add_conditional_edges("router", route_from_router, {
    "account": "account",
    "inventory": "inventory",
    END: END
})

workflow.add_conditional_edges("account", route_from_account, {
    "tools": "account_tools",
    END: END
})

workflow.add_conditional_edges("inventory", route_from_inventory, {
    "tools": "inventory_tools",
    END: END
})
workflow.add_edge("account_tools", "account")
workflow.add_edge("inventory_tools", "inventory")
graph = workflow.compile()

#NOTE * if wanted to use memory
#graph = workflow.compile(checkpointer=memory)

