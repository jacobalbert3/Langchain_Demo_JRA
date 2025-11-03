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
from agents.music_agent import song_recc_chain
from agents.customer_agent import customer_chain, get_customer_info, edit_customer_info
from agents.general_support import chain
from agents.music_agent import get_albums_by_artist, get_tracks_by_artist, check_for_songs
from langgraph.graph.message import AnyMessage, add_messages
from typing import Annotated, Optional, TypedDict
load_dotenv()


class CustomState(TypedDict):
    """Custom state with messages and additional variables"""
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: Optional[int]

#why is this necessary?? -> creates one argument message (i.e ready for pipeline)
def add_name(message, name):
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

#route determines where to go next
def _route(state: CustomState):
    messages = state.get("messages", [])
    if not messages:
        return "general"
    last_message = messages[-1]
    if isinstance(last_message, AIMessage):
        #end the convo (i.e if no tool was called)
        if not _is_tool_call(last_message):
            return END
        else:
            if last_message.name == "general": #can expect that the tool call will be the router
                tool_calls = getattr(last_message, "tool_calls", [])
                if len(tool_calls) > 1:
                    raise ValueError #should never happen
                tool_call = tool_calls[0]
                args = tool_call.get("args")
                choice = args.get("choice")
                return choice
            else:
                return "tools"
    last_m = _get_last_ai_message(messages)
    if last_m is None:
        return "general"
    if last_m.name == "music":
        return "music"
    elif last_m.name == "customer":
        return "customer"
    else:
        return "general"


# List of tools we can use
tools = [get_albums_by_artist, get_tracks_by_artist, check_for_songs, get_customer_info, edit_customer_info]
tool_node_base = ToolNode(tools)

# Wrap ToolNode to work with StateGraph - extract messages, return state update
def tools_node(state: CustomState):
    """Tools node that extracts messages from state, runs tools, returns state update"""
    messages = state.get("messages", [])
    result = tool_node_base.invoke(messages)
    return {"messages": result}

#input = running list of messages
#including general can confuse the router, so we filter it out
def _filter_out_routes(messages):
    ms = []
    for m in messages:
        if _is_tool_call(m):
            if m.name == "general":
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

def get_customer_id_node(state: CustomState):
    """Get/set customer_id in state. For now, hardcoded to 1. Later will extract from messages/config."""
    # TODO: Extract customer_id from user message, config, or authentication
    customer_id = 1
    
    return {"customer_id": customer_id}

def customer_node(state: CustomState):
    messages = state.get("messages", [])
    filtered = _filter_out_routes(messages)
    result = customer_chain.invoke(filtered)
    named_result = add_name(result, "customer")
    return {"messages": [named_result]}

#saves and retrieves checkpoints (state) 
memory = SqliteSaver.from_conn_string(":memory:") #external memory

#dictionary of nodes and their names
nodes = {"general": "general", "music": "music", END: END, "tools": "tools", "customer": "customer", "get_customer_id": "get_customer_id"}
# Define a new graph with CustomState
workflow = StateGraph(CustomState)
workflow.add_node("general", general_node)
workflow.add_node("music", music_node)
workflow.add_node("customer", customer_node)
workflow.add_node("tools", tools_node)
workflow.add_node("get_customer_id", get_customer_id_node)
#route returns a key for where to go next
workflow.add_conditional_edges("general", _route, nodes)
workflow.add_conditional_edges("tools", _route, nodes)
workflow.add_conditional_edges("music", _route, nodes)
workflow.add_conditional_edges("customer", _route, nodes)
# Add edge from get_customer_id to customer node (so customer_id is set before customer operations)
workflow.add_edge("get_customer_id", "customer")
#?? why? TODO - so middle of convo can be saved and resumed??
workflow.set_conditional_entry_point(_route, nodes)
graph = workflow.compile()