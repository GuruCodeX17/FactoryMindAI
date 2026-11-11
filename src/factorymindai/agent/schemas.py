
from typing import Any, Literal

from pydantic import BaseModel, Field


class SupervisorDecision(BaseModel):
    """Structured decision returned by the LLM supervisor."""

    intent: Literal[
        "rag",
        "prediction",
        "combined",
        "maintenance",
        "inventory",
        "unknown"
    ]

    mcp_server: Literal[
        "maintenance",
        "inventory",
        "none"
    ] = "none"

    mcp_tool: Literal[
    "get_machine_status",
    "get_maintenance_history",
    "check_stock_availability",
    "get_part_requirements",
    "search_spare_parts",
    "none"
] = "none"

    arguments: dict[str, Any] = Field(default_factory=dict)

    reasoning: str = Field(
        default="",
        max_length=300
    )
