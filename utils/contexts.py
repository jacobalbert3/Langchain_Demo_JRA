# utils/contexts.py
from dataclasses import dataclass

@dataclass
class UserContext:
    # Per-run identity the tools can read via runtime.context
    customer_id: int | None = None