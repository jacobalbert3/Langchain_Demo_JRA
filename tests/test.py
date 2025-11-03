from utils.database import db
from agents.music_agent import song_recc_chain
from langchain_core.messages import HumanMessage, AIMessage
from agents.general_support import chain
from agent import graph
from langgraph.graph import END
from langgraph.graph import START
import asyncio
#python -m tests.test
print(db.get_usable_table_names())

#TEST 2 - MUSIC AGENT
msgs = [HumanMessage(content="hi! can you help me find songs by amy whinehouse?")]
result = song_recc_chain.invoke(msgs)
print("\n=== Chain Result ===")
print(result)

#TEST 3 - GENERAL SUPPORT AGENT ROUTING TO MUSIC AGENT
msgs = [HumanMessage(content="hi! can you help me find a good song?")]
choice = chain.invoke(msgs).tool_calls[0]['args']['choice']
print("\n=== Choice ===")
print(choice)

#TEST 4 - GENERAL SUPPORT AGENT ROUTING TO CUSTOMER AGENT
msgs = [HumanMessage(content="hi! whats the email you have for me?")]
choice = chain.invoke(msgs).tool_calls[0]['args']['choice']
print("\n=== Choice ===")
print(choice)

#TEST 5 - RUN AGENT
history = []


async def run_graph_interactive():
    # StateGraph expects state dict format, not just messages list
    state = {"messages": []}
    
    while True:
        user = input('User (q/Q to quit): ')
        if user in {'q', 'Q'}:
            break
        
        # Add new message - pass as state dict
        new_message = HumanMessage(content=user)
        # StateGraph with add_messages reducer will merge messages
        new_state = {"messages": [new_message]}
        
        # StateGraph expects state dict format: {"messages": [...]}
        last_output = None
        async for output in graph.astream(new_state):
            if END in output or START in output:
                continue
            # stream() yields dictionaries with output keyed by node name
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                # value is state dict
                if isinstance(value, dict) and "messages" in value:
                    # Print latest messages
                    latest_messages = value["messages"][-1:] if value["messages"] else []
                    for msg in latest_messages:
                        if hasattr(msg, 'content') and msg.content:
                            print(msg.content)
                        else:
                            print(msg)
                else:
                    print(value)
                last_output = output
            print("\n---\n")
    
asyncio.run(run_graph_interactive())