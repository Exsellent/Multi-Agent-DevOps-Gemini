import logging
from typing import List

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("risks_agent")


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


class RisksAgent(MCPAgent):
    def __init__(self):
        super().__init__("Risks")
        self.llm = LLMClient()

        self.register_tool("analyze_risks", self.analyze_risks)

        logger.info("RisksAgent initialized")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    @log_method
    @metric_counter("risks")
    async def analyze_risks(self, feature: str):
        """Analyze risks for a feature"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Risk analysis requested",
            input_data={"feature": feature}
        ))

        prompt = (
            f"Analyze risks for implementing feature: {feature}\n"
            f"Consider security, compliance, performance, technical debt, team capacity.\n"
            f"List risks with mitigation."
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Generated risk analysis prompt"
        ))

        try:
            analysis = await self.llm.chat(prompt)

            if self._is_invalid_response(analysis):
                analysis = "Risk analysis unavailable — fallback to basic risks."
                reasoning.append(ReasoningStep(
                    step_number=4,
                    description="LLM stub/error detected — using fallback analysis",
                    output_data={"fallback_used": True}
                ))

            detected_risks = [
                line.strip().lstrip("-*• ")
                for line in analysis.split("\n")
                if line.strip().startswith(("- ", "* ", "• "))
            ]

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Risk analysis completed",
                output_data={"risks_count": len(detected_risks)}
            ))

            logger.info("Risk analysis completed", extra={"feature": feature})

            return {
                "feature": feature,
                "risk_analysis": analysis,
                "detected_risks": detected_risks,
                "reasoning": reasoning
            }
        except Exception as e:
            logger.error("Risk analysis failed", extra={"error": str(e)})
            reasoning.append(ReasoningStep(
                step_number=4,
                description="Risk analysis failed",
                output_data={"error": str(e)}
            ))
            return {
                "feature": feature,
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Manual risk assessment required"
            }
