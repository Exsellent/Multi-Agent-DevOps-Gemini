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
    """
    Progress Tracking and Velocity Analysis Agent

    Capabilities:
    1. Commit-based progress analysis (LLM-powered)
    2. Jira velocity tracking (deterministic calculations)
    3. Sprint velocity and completion rate metrics

    This agent demonstrates both:
    - LLM-enhanced analysis (analyze_progress)
    - Pure deterministic logic (jira_velocity)

    Proving the architecture works with AND without AI.
    """

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

    def _next_step(self, reasoning: List[ReasoningStep], description: str,
                   input_data: Optional[Dict] = None, output_data: Optional[Dict] = None):
        """Helper to add sequential reasoning steps"""
        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description=description,
            input_data=input_data or {},
            output_data=output_data or {}
        ))

    @log_method
    @metric_counter("progress")
    async def analyze_progress(self, commits: List[str]):
        """
        Analyze progress from commit messages using LLM

        This demonstrates LLM-powered analysis with proper fallback
        """
        reasoning: List[ReasoningStep] = []
        fallback_used = False

        # Step 1: Request received
        self._next_step(reasoning, "Received commits for progress analysis",
                        input_data={"commits_count": len(commits)})

        # Step 2: Generate prompt
        prompt = (
                "You are a progress tracking agent.\n"
                "Analyze these commits and summarize achievements and velocity:\n" +
                "\n".join(f"- {c}" for c in commits) +
                "\n\nProvide a concise, positive summary of progress made."
        )

        self._next_step(reasoning, "Generated prompt for progress summary",
                        output_data={"prompt_length": len(prompt)})

        try:
            # Attempt LLM analysis
            summary = await self.llm.chat(prompt)

            # Check if LLM response is valid
            if self._is_invalid_response(summary):
                fallback_used = True
                summary = (
                    f"Progress analysis unavailable. "
                    f"Processed {len(commits)} commit(s). "
                    f"Manual review recommended for detailed insights."
                )
                logger.warning("Progress Agent using fallback summary",
                               extra={"commits_count": len(commits)})

            # Step 3: Analysis completed
            self._next_step(reasoning, "Received progress summary from LLM",
                            output_data={
                                "summary_length": len(summary),
                                "fallback_used": fallback_used
                            })

            # Step 4 (optional): Fallback annotation
            if fallback_used:
                self._next_step(reasoning, "Fallback summary was used due to LLM unavailability",
                                output_data={"commits_analyzed": len(commits)})

            logger.info("Progress analysis completed",
                        extra={"commits_count": len(commits), "fallback": fallback_used})

            return {
                "commits_count": len(commits),
                "commits": commits,
                "summary": summary,
                "fallback_used": fallback_used,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Progress analysis failed", extra={"error": str(e)})

            # Use fallback even on exception
            self._next_step(reasoning, "Progress analysis failed with exception — using fallback",
                            output_data={"error": str(e), "fallback_used": True})

            self._next_step(reasoning, "Fallback summary generated",
                            output_data={"commits_count": len(commits)})

            return {
                "commits_count": len(commits),
                "commits": commits,
                "summary": f"Progress analysis failed. Processed {len(commits)} commit(s). Manual review needed.",
                "fallback_used": True,
                "reasoning": reasoning,
                "error": str(e)
            }

    @log_method
    @metric_counter("progress")
    async def jira_velocity(self, project_key: Optional[str] = None):
        """
        Analyze project velocity from Jira issues

        This is a DETERMINISTIC agent - no LLM required.
        Demonstrates that multi-agent architecture works for both AI and traditional logic.
        """
        reasoning: List[ReasoningStep] = []

        # Step 1: Request received
        self._next_step(reasoning, "Jira velocity analysis requested",
                        input_data={"project_key": project_key or self.jira.project_key})

        try:
            # Step 2: Fetch from Jira (with automatic fallback to mocks)
            result = await self.jira.get_project_issues()

            issues = result.get("issues", [])
            jira_mode = result.get("mode", "unknown")

            self._next_step(reasoning, "Retrieved issues from Jira",
                            output_data={
                                "issues_count": len(issues),
                                "jira_mode": jira_mode
                            })

            # Early exit if no data
            if not issues:
                self._next_step(reasoning, "No issues found — returning zero metrics")

                return {
                    "project": project_key or self.jira.project_key,
                    "total_issues": 0,
                    "completion_rate": 0.0,
                    "velocity_status": "no_data",
                    "jira_mode": jira_mode,
                    "reasoning": reasoning
                }

            # Step 3: Calculate velocity metrics (deterministic logic)
            status_counts: Dict[str, int] = {}
            for issue in issues:
                status = issue["fields"]["status"]["name"]
                status_counts[status] = status_counts.get(status, 0) + 1

            total = len(issues)
            done = sum(v for k, v in status_counts.items() if k.lower() in ["done", "closed"])
            completion_rate = round((done / total * 100) if total > 0 else 0, 1)

            # Velocity status classification
            if completion_rate >= 75:
                velocity_status = "excellent"
            elif completion_rate >= 50:
                velocity_status = "on_track"
            elif completion_rate >= 25:
                velocity_status = "at_risk"
            else:
                velocity_status = "critical"

            self._next_step(reasoning, "Calculated project velocity",
                            output_data={
                                "completion_rate": completion_rate,
                                "velocity_status": velocity_status,
                                "total_issues": total,
                                "done_issues": done
                            })

            logger.info("Jira velocity calculated",
                        extra={
                            "completion_rate": completion_rate,
                            "jira_mode": jira_mode,
                            "project": project_key or self.jira.project_key
                        })

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

            self._next_step(reasoning, "Jira velocity failed with exception",
                            output_data={"error": str(e)})

            return {
                "project": project_key or self.jira.project_key,
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Manual Jira review required"
            }
