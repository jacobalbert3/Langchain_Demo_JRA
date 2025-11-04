from dataclasses import dataclass


@dataclass
class UserContext:
    customer_id: int | None = None