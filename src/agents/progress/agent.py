import logging
from typing import List, Dict, Optional

from shared.jira import JiraClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("progress_agent")


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


class ProgressAgent(MCPAgent):
    def __init__(self):
        super().__init__("Progress")
        self.llm = LLMClient()
        self.jira = JiraClient()

        self.register_tool("analyze_progress", self.analyze_progress)
        self.register_tool("jira_velocity", self.jira_velocity)

        logger.info("ProgressAgent initialized with Jira integration")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    @log_method
    @metric_counter("progress")
    async def analyze_progress(self, commits: List[str]):
        """Analyze progress from commits"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Received commits for progress analysis",
            input_data={"commits_count": len(commits)}
        ))

        prompt = (
                "You are a progress tracking agent.\n"
                "Analyze these commits and summarize achievements and velocity:\n" +
                "\n".join(f"- {c}" for c in commits) +
                "\n\nBe concise and positive."
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Generated prompt for progress summary"
        ))

        try:
            summary = await self.llm.chat(prompt)

            if self._is_invalid_response(summary):
                summary = "Progress analysis unavailable — fallback to basic count."
                reasoning.append(ReasoningStep(
                    step_number=4,
                    description="LLM stub/error detected — using fallback summary",
                    output_data={"fallback_used": True}
                ))

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Received progress summary from LLM",
                output_data={"summary_length": len(summary)}
            ))

            logger.info("Progress analysis completed", extra={"commits_count": len(commits)})

            return {
                "commits_count": len(commits),
                "commits": commits,
                "summary": summary,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Progress analysis failed", extra={"error": str(e)})
            reasoning.append(ReasoningStep(
                step_number=4,
                description="Progress analysis failed",
                output_data={"error": str(e)}
            ))
            return {
                "commits_count": len(commits),
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Manual review needed"
            }

    @log_method
    @metric_counter("progress")
    async def jira_velocity(self, project_key: Optional[str] = None):
        """Analyze project velocity from Jira issues"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Jira velocity analysis requested",
            input_data={"project_key": project_key or self.jira.project_key}
        ))

        try:
            # Get dictionary with 'issues' list and 'mode' from JiraClient
            result = await self.jira.get_project_issues()
            issues = result.get("issues", [])
            jira_mode = result.get("mode", "unknown")

            reasoning.append(ReasoningStep(
                step_number=2,
                description="Retrieved issues from Jira",
                output_data={
                    "issues_count": len(issues),
                    "jira_mode": jira_mode
                }
            ))

            if not issues:
                reasoning.append(ReasoningStep(
                    step_number=3,
                    description="No issues found — returning fallback data"
                ))
                return {
                    "project": project_key or self.jira.project_key,
                    "total_issues": 0,
                    "completion_rate": 0.0,
                    "velocity_status": "no_data",
                    "jira_mode": jira_mode,
                    "reasoning": reasoning
                }

            # Count statuses
            status_counts: Dict[str, int] = {}
            for issue in issues:
                status = issue["fields"]["status"]["name"]
                status_counts[status] = status_counts.get(status, 0) + 1

            total = len(issues)
            done = sum(v for k, v in status_counts.items() if k.lower() in ["done", "closed"])
            completion_rate = round((done / total * 100) if total > 0 else 0, 1)
            velocity_status = "on_track" if completion_rate > 50 else "at_risk"

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Calculated project velocity",
                output_data={"completion_rate": completion_rate}
            ))

            logger.info("Jira velocity calculated", extra={"completion_rate": completion_rate})

            return {
                "project": project_key or self.jira.project_key,
                "total_issues": total,
                "status_breakdown": status_counts,
                "completion_rate": completion_rate,
                "velocity_status": velocity_status,
                "jira_mode": jira_mode,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Jira velocity failed", extra={"error": str(e)})
            reasoning.append(ReasoningStep(
                step_number=4,
                description="Jira velocity failed",
                output_data={"error": str(e)}
            ))
            return {
                "error": str(e),
                "reasoning": reasoning
            }
