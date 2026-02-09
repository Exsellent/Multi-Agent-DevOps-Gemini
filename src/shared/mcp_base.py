import inspect
import json
import logging
import os
from dataclasses import is_dataclass, asdict
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)


def finalize_output(obj: Any, agent_name: str) -> Any:
    """
    Recursively clean and normalize output for MCP/UI compatibility:
    - Convert Pydantic models using .model_dump(exclude_none=True)
    - Convert dataclasses using asdict
    - Remove all None/null values
    - Rename output_data → output if present
    - Add agent name if missing
    - Handle lists and nested structures
    """
    # 1. Convert Pydantic model → dict (exclude None)
    if isinstance(obj, BaseModel):
        return finalize_output(obj.model_dump(exclude_none=True), agent_name)

    # 2. Convert dataclass → dict
    if is_dataclass(obj) and not isinstance(obj, type):
        return finalize_output(asdict(obj), agent_name)

    # 3. Dictionary: clean keys, rename output_data → output, add agent
    if isinstance(obj, dict):
        new_dict = {}
        for k, v in obj.items():
            # Special handling for ReasoningStep-like structures
            if k == "output_data":
                new_dict["output"] = finalize_output(v, agent_name)
                continue
            if k == "agent" and v is None:
                new_dict[k] = agent_name
                continue
            # Skip None values
            if v is not None:
                new_dict[k] = finalize_output(v, agent_name)
        # Auto-add agent if this looks like a ReasoningStep
        if "step_number" in new_dict and "description" in new_dict:
            new_dict.setdefault("agent", agent_name)
        return new_dict

    # 4. List: filter None and recurse
    if isinstance(obj, list):
        return [finalize_output(item, agent_name) for item in obj if item is not None]

    # 5. Primitive types — return as is
    return obj


class MCPRequest(BaseModel):
    method: str
    params: Dict[str, Any]
    id: int | None = None


class MCPAgent:
    def __init__(self, name: str):
        self.name = name
        self.app = FastAPI()
        self.tools: Dict[str, Any] = {}

        # Enable CORS for web UI / frontend
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.get("/health")
        def health():
            return {"status": "ok", "agent": self.name}

        @self.app.get("/")
        def root():
            return {
                "message": "Multi-Agent DevOps Agent",
                "agent": self.name,
                "available_tools": list(self.tools.keys()),
                "docs": "/docs"
            }

        @self.app.post("/mcp")
        async def mcp(request: Request):
            try:
                body = await request.body()
                data = json.loads(body.decode('utf-8'))
                req = MCPRequest(**data)
            except json.JSONDecodeError as e:
                return {
                    "error": "Invalid JSON",
                    "details": str(e),
                    "hint": "Send valid JSON with 'method' and 'params'"
                }
            except ValidationError as e:
                return {
                    "error": "Invalid MCP request format",
                    "details": str(e),
                    "hint": "Required fields: method (str), params (dict)"
                }
            except Exception as e:
                return {
                    "error": "Request processing failed",
                    "details": str(e)
                }

            # Normalize tool name (remove "tools/" prefix if present)
            tool_name = req.method.replace("tools/", "")
            handler = self.tools.get(tool_name)

            if not handler:
                return {
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": list(self.tools.keys())
                }

            try:
                # Execute the tool (sync or async)
                if inspect.iscoroutinefunction(handler):
                    result = await handler(**req.params)
                else:
                    result = handler(**req.params)

                # Final clean-up: remove nulls, fix reasoning, add agent name
                cleaned_result = finalize_output(result, self.name)

                return cleaned_result

            except TypeError as e:
                # Better error for parameter mismatch
                sig = inspect.signature(handler)
                expected_params = [p for p in sig.parameters.keys() if p != "self"]
                return {
                    "error": f"Invalid parameters for tool '{tool_name}'",
                    "details": str(e),
                    "received_params": list(req.params.keys()),
                    "expected_params": expected_params
                }
            except Exception as e:
                return {
                    "error": f"Tool execution failed: {tool_name}",
                    "details": str(e)
                }

    def register_tool(self, name: str, func):
        """
        Register a tool function (sync or async) without wrapper/adapter
        The function will receive **kwargs directly from params
        """
        self.tools[name] = func
