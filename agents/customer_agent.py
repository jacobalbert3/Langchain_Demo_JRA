# agents/customer_agent.py
from langchain.tools import tool, ToolRuntime
from utils.database import db
from utils.contexts import AccountState
from typing import Literal
editable_parameters = ["Address", "Phone", "Email"]



@tool
def past_invoices(runtime: ToolRuntime[None, AccountState]):
    """Look up past invoices for a customer."""
    customer_id = runtime.state.get("customer_id")
    if not customer_id:
        raise ValueError("Customer ID not found in context")
    try:
        customer_id = int(customer_id)
    except (ValueError, TypeError):
        raise ValueError("Customer ID must be a valid integer")
    
    conn = db._engine.raw_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                i.*,
                il.UnitPrice,
                il.Quantity,
                t.Name as TrackName
            FROM Invoice i
            INNER JOIN InvoiceLine il ON i.InvoiceId = il.InvoiceId
            INNER JOIN Track t ON il.TrackId = t.TrackId
            WHERE i.CustomerID = ?
            ORDER BY i.InvoiceDate DESC
            LIMIT 5
        """, (customer_id,))
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        cursor.close()
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


@tool
def edit_customer_info(runtime: ToolRuntime[None, AccountState], parameter: Literal["Address", "Phone", "Email"], value: str) -> str:
    """Update a customer's information - parameter must be one of: Address, Phone, Email"""
    
    if parameter not in editable_parameters:
        return f"The {parameter} parameter is not editable"

    customer_id = runtime.state.get("customer_id")
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

@tool(description="Look up customer information (any information is safe to show the user). Returns all customer fields including FirstName, LastName, Phone, Email, Address, City, State, Country, PostalCode, Company, Fax, etc.")
def get_customer_info(runtime: ToolRuntime[None, AccountState]):
    """Look up customer info (customer_id comes from context)."""

    customer_id = runtime.state.get("customer_id")
    
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

- Use get_customer_info (no params) to show current info. This returns ALL customer information.
- When user asks for specific information (like "name", "phone number", "email", etc.), extract that field from the returned data and present it clearly.
- Users can ask for ANY customer information - extract and show the requested field(s) from the full dataset.
- Use edit_customer_info(parameter, value) to update. Only Address, Phone, and Email can be edited.
- If the request is outside those limits, explain limits politely."""
