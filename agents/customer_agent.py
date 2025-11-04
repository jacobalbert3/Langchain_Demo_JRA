from langchain_core.messages import SystemMessage
from utils.database import db
from utils.model import model
from langchain.tools import tool, ToolRuntime
from utils.contexts import UserContext




#define the parameters that can be changed
editable_parameters = ["name", "email", "phone"]
search_parameters = ["name", "email", "phone"]


@tool
def edit_customer_info(runtime: ToolRuntime[UserContext], parameter: str, value: str):
    """Update a customer's information"""
    #ensure that the parameter is one that can be changed (whitelist validation)
    if parameter not in editable_parameters:
        return f"The {parameter} parameter is not editable"
    # Get customer_id from runtime context (state)
    customer_id = runtime.context.customer_id
    if not customer_id:
        raise ValueError("Customer ID not found in context")
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
def get_customer_info(runtime: ToolRuntime[UserContext]):
    """Look up customer info given their customer ID. The customer ID is automatically provided."""
    # Get customer_id from runtime context (state)
    customer_id = runtime.context.customer_id
    if not customer_id:
        raise ValueError("Customer ID not found in context")
    # Validate customer_id is an integer
    try:
        customer_id = int(customer_id)
    except (ValueError, TypeError):
        raise ValueError("Customer ID must be a valid integer")
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
customer_prompt = """Your job is to help a user view and update their profile.

You have access to tools that can look up customer information and update customer information.
- The get_customer_info tool automatically uses the authenticated customer ID. You do NOT need to pass customer_id as a parameter.
- The edit_customer_info tool requires a parameter name and new value. The customer_id is handled automatically.

When the user asks about their account information (like name, email, phone), use the get_customer_info tool (no parameters needed).
When the user wants to update their information, use the edit_customer_info tool with the parameter name and new value.

You only have certain tools you can use. If you don't know how to help with something, politely explain what you can help with."""


#Get customer messages starting with the customer prompt
def get_customer_messages(messages):
    return [SystemMessage(content=customer_prompt)] + messages

#runs get_customer_messages, adding system prompt -> sends to model
customer_chain = get_customer_messages | model.bind_tools([get_customer_info, edit_customer_info])
