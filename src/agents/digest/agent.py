import logging
from typing import List
from typing import Optional

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("digest_agent")


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


class DigestAgent(MCPAgent):
    def __init__(self):
        super().__init__("Digest")
        self.llm = LLMClient()

        self.register_tool("daily_digest", self.daily_digest)

        logger.info("DigestAgent initialized")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    @log_method
    @metric_counter("digest")
    async def daily_digest(self, date: Optional[str] = None):
        """Generate daily project digest. If date not provided — use current date"""
        reasoning: List[ReasoningStep] = []

        if date is None:
            from datetime import date
            date = date.today().isoformat()

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Daily digest requested",
            input_data={"date": date}
        ))

        prompt = (
            f"Generate a concise daily project digest for {date}.\n"
            f"Include key achievements, blockers, team mood.\n"
            f"Keep it positive and under 200 words."
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Generated digest prompt"
        ))

        try:
            digest = await self.llm.chat(prompt)

            if self._is_invalid_response(digest):
                digest = f"Daily digest for {date}: The team is making steady progress. No major blockers reported."
                reasoning.append(ReasoningStep(
                    step_number=4,
                    description="LLM stub/error detected — using fallback digest",
                    output_data={"fallback_used": True}
                ))

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Daily digest generated",
                output_data={"digest_length": len(digest)}
            ))

            logger.info("Daily digest completed", extra={"date": date})

            return {
                "date": date,
                "summary": digest,
                "reasoning": reasoning
            }
        except Exception as e:
            logger.error("Daily digest failed", extra={"error": str(e)})
            reasoning.append(ReasoningStep(
                step_number=4,
                description="Daily digest failed",
                output_data={"error": str(e)}
            ))
            return {
                "date": date,
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "No updates today"
            }
