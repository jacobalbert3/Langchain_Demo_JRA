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
from langgraph.graph import MessageGraph
from langgraph.checkpoint.sqlite import SqliteSaver
from agents.music_agent import song_recc_chain
from agents.customer_agent import customer_chain, get_customer_info
from agents.general_support import chain
from agents.music_agent import get_albums_by_artist, get_tracks_by_artist, check_for_songs
load_dotenv()


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



# OLD
# #checks if the message is a tool call
# def _is_tool_call(msg):
#     return hasattr(msg, "additional_kwargs") and 'tool_calls' in msg.additional_kwargs


def _is_tool_call(msg):
    return hasattr(msg, "tool_calls") and len(msg.tool_calls) > 0

#route determins where to go next
def _route(messages):
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


#list of tools we can use
tools = [get_albums_by_artist, get_tracks_by_artist, check_for_songs, get_customer_info]
tools_node = ToolNode(tools)

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

#NOTE** each node is RUNNABLE: -> pipeline of smaller steps (same as one function)
general_node = _filter_out_routes | chain | partial(add_name, name="general")
music_node = _filter_out_routes | song_recc_chain | partial(add_name, name="music")
customer_node = _filter_out_routes | customer_chain | partial(add_name, name="customer")

#saves and retrieves checkpoints (state) 
memory = SqliteSaver.from_conn_string(":memory:") #STORE IN RAM

#dictionary of ndoes and their names
nodes = {"general": "general", "music": "music", END: END, "tools": "tools", "customer": "customer"}
# Define a new graph
workflow = MessageGraph()
workflow.add_node("general", general_node)
workflow.add_node("music", music_node)
workflow.add_node("customer", customer_node)
workflow.add_node("tools", tools_node)
#route returns a key for where to go next
workflow.add_conditional_edges("general", _route, nodes)
workflow.add_conditional_edges("tools", _route, nodes)
workflow.add_conditional_edges("music", _route, nodes)
workflow.add_conditional_edges("customer", _route, nodes)
#?? why? TODO - so middle of convo can be saved and resumed??
workflow.set_conditional_entry_point(_route, nodes)
graph = workflow.compile()