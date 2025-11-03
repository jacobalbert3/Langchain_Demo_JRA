from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from utils.database import db
from utils.model import model
from pydantic import BaseModel, Field

class Router(BaseModel):
    """Call this if you are able to route the user to the appropriate representative."""
    choice: str = Field(description="should be one of: account, inventory, other")

system_message = """Your job is to help as a customer service representative for a music store.

You should interact politely with customers to try to figure out how you can help. You can help in a few ways:

- Account management: if a customer wants to update their account information, view their order history, or manage their profile. Call the router with `account`
- Inventory/music inquiries: if a customer wants to find music, search for albums, or browse the store inventory. Call the router with `inventory`
- General inquiries: if the user's question doesn't fit into account or inventory categories (e.g., greetings, store hours, general questions). Call the router with `other`, or respond directly if it's a simple greeting.

If the user is asking or wants to ask about their account, orders, or profile, send them to `account`.
If the user is asking or wants to ask about music, albums, songs, or browsing inventory, send them to `inventory`.
If the user's question is not related to account or inventory, send them to `other`."""
def get_messages(messages):
    return [SystemMessage(content=system_message)] + messages
chain = get_messages | model.bind_tools([Router])