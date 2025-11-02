from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from utils.database import db
from utils.model import model


#PART 1 - DEFINITION FOR CUSTOMER AGENT
#STARTER CODE
@tool
def get_customer_info(customer_id: int):
    """Look up customer info given their ID. ALWAYS make sure you have the customer ID before invoking this."""
    return db.run(f"SELECT * FROM Customer WHERE CustomerID = {customer_id};")

#TODO: pull from prompt hub instead?
customer_prompt = """Your job is to help a user update their profile.

You only have certain tools you can use. These tools require specific input. If you don't know the required input, then ask the user for it.

If you are unable to help the user, you can """


#Get customer messages starting with the customer prompt
def get_customer_messages(messages):
    return [SystemMessage(content=customer_prompt)] + messages

#runs get_customer_messages, adding system prompt -> sends to model
customer_chain = get_customer_messages | model.bind_tools([get_customer_info])
