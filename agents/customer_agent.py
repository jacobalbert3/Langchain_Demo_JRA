# agents/customer_agent.py
from langchain.tools import tool, ToolRuntime
from utils.database import db
from utils.contexts import UserContext
from typing import Literal
editable_parameters = ["name", "email", "phone"]

@tool
def edit_customer_info(runtime: ToolRuntime[UserContext], parameter: Literal["name", "email", "phone"], value: str) -> str:
    """Update a customer's information - parameter must be one of: name, email, phone"""
    
    if parameter not in editable_parameters:
        return f"The {parameter} parameter is not editable"

    customer_id = runtime.context.customer_id
    if not customer_id:
        raise ValueError("Customer ID not found in context")
    try:
        customer_id = int(customer_id)
    except (ValueError, TypeError):
        raise ValueError("Customer ID must be a valid integer")

    conn = db._engine.raw_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE Customer SET {parameter} = ? WHERE CustomerID = ?",
            (value, customer_id),
        )
        conn.commit()
        cursor.close()
    finally:
        conn.close()

    return "Customer info updated"

@tool
def get_customer_info(runtime: ToolRuntime[UserContext]):
    """Look up customer info (customer_id comes from context)."""

    customer_id = runtime.context.customer_id
    if not customer_id:
        raise ValueError("Customer ID not found in context")
    try:
        customer_id = int(customer_id)
    except (ValueError, TypeError):
        raise ValueError("Customer ID must be a valid integer")

    conn = db._engine.raw_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Customer WHERE CustomerID = ?", (customer_id,))
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        cursor.close()
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()

customer_system_prompt = """You help a user view/update their profile.

- Use get_customer_info (no params) to show current info.
- Use edit_customer_info(parameter, value) to update.
If the request is outside those, explain limits politely."""
