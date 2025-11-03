from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from utils.database import db
from utils.model import model
from langchain_core.runnables import RunnableConfig



#define the parameters that can be changed
editable_parameters = ["name", "email", "phone"]
search_parameters = ["name", "email", "phone"]


@tool
def edit_customer_info(config: RunnableConfig, parameter: str, value: str):
    """Update a customer's information"""
    #ensure that the parameter is one that can be changed (whitelist validation)
    if parameter not in editable_parameters:
        return f"The {parameter} parameter is not editable"
    #find customer id through the config
    configuration = config.get("configurable")
    customer_id = configuration.get("customer_id")
    if not customer_id:
        raise ValueError("Customer ID not found in config")
    # Validate customer_id is an integer
    try:
        customer_id = int(customer_id)
    except (ValueError, TypeError):
        raise ValueError("Customer ID must be a valid integer")
    
    conn = db._engine.raw_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(f"UPDATE Customer SET {parameter} = ? WHERE CustomerID = ?", (value, customer_id))
        conn.commit()
        cursor.close()
    finally:
        conn.close()
    
    return "Customer info updated"

#PART 1 - DEFINITION FOR CUSTOMER AGENT
#STARTER CODE
@tool
def get_customer_info(customer_id: int):
    """Look up customer info given their customer ID. ALWAYS make sure you have the customer ID before invoking this."""
    # Use parameterized query to prevent SQL injection
    conn = db._engine.raw_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Customer WHERE CustomerID = ?", (customer_id,))
        result = cursor.fetchall()
        # Get column names for formatted output
        columns = [description[0] for description in cursor.description]
        cursor.close()
        # Format as list of dicts similar to db.run() output
        return [dict(zip(columns, row)) for row in result]
    finally:
        conn.close()



#TODO: pull from prompt hub instead?
customer_prompt = """Your job is to help a user update their profile.

You only have certain tools you can use. These tools require specific input. If you don't know the required input, then ask the user for it.

If you are unable to help the user, you can """


#Get customer messages starting with the customer prompt
def get_customer_messages(messages):
    return [SystemMessage(content=customer_prompt)] + messages

#runs get_customer_messages, adding system prompt -> sends to model
customer_chain = get_customer_messages | model.bind_tools([get_customer_info, edit_customer_info])
