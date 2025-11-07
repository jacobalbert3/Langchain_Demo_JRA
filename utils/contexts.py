# utils/contexts.py
from dataclasses import dataclass
from typing import Optional
from langchain.agents import AgentState
from typing import Literal
#defines what we can read at runtime via ToolRuntime
class AccountState(AgentState):
    customer_id: Optional[int]
    example_string: Optional[str]
    example_context: Optional[str]

class InventoryState(AgentState):
    customer_id: Optional[int]

class GeneralState(AgentState):
    customer_id: Optional[int]