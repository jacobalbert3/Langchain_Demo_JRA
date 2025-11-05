# utils/contexts.py
from dataclasses import dataclass
from typing import Optional
from langchain.agents import AgentState

@dataclass
class UserContext:
    # Per-run identity the tools can read via runtime.context
    customer_id: int | None = None
    username: str | None = None

#defines what we can read at runtime via ToolRuntime
class AccountState(AgentState):
    customer_id: Optional[int]

class InventoryState(AgentState):
    customer_id: Optional[int]

class GeneralState(AgentState):
    customer_id: Optional[int]