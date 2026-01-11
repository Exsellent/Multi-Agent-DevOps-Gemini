import logging
from typing import List
from typing import Optional

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("metrics_agent")


def log_method(func):
    """Decorator for logging method calls"""

    async def wrapper(self, *args, **kwargs):
        logger.info(f"{func.__name__} called with args: {args}, kwargs: {kwargs}")
        try:
            result = await func(self, *args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            raise

    return wrapper


class MetricsAgent(MCPAgent):
    def __init__(self):
        super().__init__("Metrics-Agent")
        self.llm = LLMClient()  # For potential AI-based metrics analysis

        self.register_tool("get_metrics", self.get_metrics)

        logger.info("MetricsAgent initialized")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    @log_method
    @metric_counter("metrics_agent")
    async def get_metrics(self, agent_name: Optional[str] = None):
        """Get metrics for specific agent or all agents"""
        reasoning: List[ReasoningStep] = []

        if agent_name is None:
            agent_name = "all"

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Metrics request received",
            input_data={"agent_name": agent_name}
        ))

        try:

            mock_metrics = {
                "planner": {"tasks_processed": 45, "errors": 2},
                "risks": {"tasks_processed": 38, "errors": 1},
                "progress": {"tasks_processed": 30, "errors": 0},
                "digest": {"tasks_processed": 25, "errors": 0},
                "architecture_intelligence": {"tasks_processed": 15, "errors": 3},
                "health_monitor": {"tasks_processed": 60, "errors": 0},
                "metrics_agent": {"tasks_processed": 10, "errors": 0},
                "marathon": {"tasks_processed": 12, "errors": 1},
                "code_execution": {"tasks_processed": 20, "errors": 2}
            }

            if agent_name == "all":
                metrics_data = mock_metrics
            else:
                metrics_data = mock_metrics.get(agent_name, {"tasks_processed": 0, "errors": 0})

            reasoning.append(ReasoningStep(
                step_number=2,
                description="Metrics collected",
                output_data={"agent": agent_name, "data": metrics_data}
            ))

            logger.info("Metrics retrieved", extra={"agent_name": agent_name})

            return {
                "agent_name": agent_name,
                "metrics": metrics_data,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Metrics retrieval failed", extra={"error": str(e)})
            reasoning.append(ReasoningStep(
                step_number=3,
                description="Metrics retrieval failed",
                output_data={"error": str(e)}
            ))
            return {
                "agent_name": agent_name,
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Metrics unavailable"
            }
