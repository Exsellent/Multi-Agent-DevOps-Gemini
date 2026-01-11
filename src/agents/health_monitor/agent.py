import logging
from typing import Dict, Any, List

import httpx

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("health_monitor_agent")


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


class HealthMonitorAgent(MCPAgent):
    def __init__(self):
        super().__init__("Health-Monitor")
        self.llm = LLMClient()  # For potential AI-based anomaly detection

        self.register_tool("check_health", self.check_health)

        logger.info("HealthMonitorAgent initialized")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    @log_method
    @metric_counter("health_monitor")
    async def check_health(self, agents: Dict[str, str]) -> Dict[str, Any]:
        """Check health of all agents"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Received health check request",
            input_data={"agents": list(agents.keys())}
        ))

        prompt = (
            f"You are a health monitoring agent.\n"
            f"Analyze this agent status data for anomalies:\n"
            f"{agents}\n\n"
            f"Return summary of health issues if any."
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Generated prompt for LLM health analysis"
        ))

        health_status = {}

        for agent_name, url in agents.items():
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(url)
                    health_status[agent_name] = {
                        "status_code": resp.status_code,
                        "response": resp.json() if resp.status_code == 200 else str(resp.text)
                    }
            except Exception as e:
                health_status[agent_name] = {"error": str(e)}

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Checked health of all agents",
            output_data={"agents_checked": len(health_status)}
        ))

        try:
            analysis = await self.llm.chat(prompt)

            if self._is_invalid_response(analysis):
                analysis = "Health analysis unavailable — fallback to raw status."
                reasoning.append(ReasoningStep(
                    step_number=5,
                    description="LLM stub/error detected — using fallback analysis",
                    output_data={"fallback_used": True}
                ))

            reasoning.append(ReasoningStep(
                step_number=4,
                description="LLM health analysis completed",
                output_data={"analysis_length": len(analysis)}
            ))

            logger.info("Health check completed", extra={"agents": list(agents.keys())})

            return {
                "health_status": health_status,
                "analysis": analysis,
                "reasoning": reasoning
            }
        except Exception as e:
            logger.error("Health analysis failed", extra={"error": str(e)})
            reasoning.append(ReasoningStep(
                step_number=5,
                description="Health analysis failed",
                output_data={"error": str(e)}
            ))
            return {
                "health_status": health_status,
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Manual health check recommended"
            }
