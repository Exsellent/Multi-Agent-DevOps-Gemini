from datetime import datetime
from typing import Any, Optional, List, Dict

from pydantic import BaseModel, Field


class MCPRequest(BaseModel):
    method: str
    params: Dict[str, Any]
    id: Optional[int] = None


class MCPResponse(BaseModel):
    result: Optional[Any] = None
    error: Optional[str] = None
    reasoning: Optional[List["ReasoningStep"]] = None  # forward ref


class ReasoningStep(BaseModel):
    step_number: int
    description: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    input_data: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    agent: str | None = None


class Message(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
