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
    # Ask for customer_id before starting (optional - can be None to trigger user node)
    customer_id_input = input('Enter your customer ID (or press Enter to be asked later): ').strip()
    if customer_id_input:
        try:
            customer_id = int(customer_id_input)
            print(f"Authenticated as customer ID: {customer_id}\n")
        except ValueError:
            print("Invalid customer ID, will ask for it later\n")
            customer_id = None
    else:
        customer_id = None  # Will trigger user node to ask for it
    
    # Maintain conversation state (messages accumulate here)
    accumulated_messages = []
    
    while True:
        user = input('User (q/Q to quit): ')
        if user in {'q', 'Q'}:
            break
        
        # Add user message to accumulated messages
        accumulated_messages.append(HumanMessage(content=user))
        
        # Build state with all accumulated messages and customer_id (may be None)
        current_state = {"messages": accumulated_messages}
        if customer_id is not None:
            current_state["customer_id"] = customer_id
        
        # Stream graph execution and capture final state
        final_state = None
        async for output in graph.astream(current_state):
            if END in output or START in output:
                continue
            
            # Print output from each node
            for key, value in output.items():
                print(f"Output from node '{key}':")
                print("---")
                # value is state dict - print latest messages
                if isinstance(value, dict) and "messages" in value:
                    latest_messages = value["messages"][-1:] if value["messages"] else []
                    for msg in latest_messages:
                        if hasattr(msg, 'content') and msg.content:
                            print(msg.content)
                        else:
                            print(msg)
                else:
                    print(value)
            print("\n---\n")
            final_state = output  # Keep track of final output
        
        # Update accumulated messages and customer_id from final state (includes AI responses)
        if final_state:
            for value in final_state.values():
                if isinstance(value, dict):
                    if "messages" in value:
                        accumulated_messages = value["messages"]
                    # Update customer_id if it was set by user_node
                    if "customer_id" in value and value["customer_id"] is not None:
                        customer_id = value["customer_id"]
                        print(f"✓ Customer ID set to: {customer_id}\n")
                    break
    
asyncio.run(run_graph_interactive())