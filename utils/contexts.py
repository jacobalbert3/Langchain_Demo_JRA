# utils/contexts.py
from dataclasses import dataclass
from typing import Optional
from langchain.agents import AgentState
from typing import Literal
#defines what we can read at runtime via ToolRuntime
class AccountState(AgentState):
    customer_id: Optional[int]
    length: Literal["long", "short"] | None = None

class InventoryState(AgentState):
    customer_id: Optional[int]

class GeneralState(AgentState):
    customer_id: Optional[int]