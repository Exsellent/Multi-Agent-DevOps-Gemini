import logging
from datetime import datetime
from typing import List, Optional, Dict, Any

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

    Hybrid AI + Deterministic Architecture:
    - analyze_progress: LLM-powered commit analysis
    - jira_velocity: Deterministic metrics + LLM insights

    Both methods now use LLM for intelligent analysis!
    """

    def __init__(self):
        super().__init__("Progress")
        self.llm = LLMClient()
        self.jira = JiraClient()

        self.register_tool("analyze_progress", self.analyze_progress)
        self.register_tool("jira_velocity", self.jira_velocity)

        logger.info("ProgressAgent initialized with LLM-enhanced velocity analysis")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    def _next_step(
            self,
            reasoning: List[ReasoningStep],
            description: str,
            input_data: Optional[Dict] = None,
            output_data: Optional[Dict] = None
    ):
        """Helper to add sequential reasoning steps"""
        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description=description,
            timestamp=datetime.now().isoformat(),
            input_data=input_data or {},
            output=output_data or {},
            agent=self.name
        ))

    def _generate_baseline_analysis(
            self,
            completion_rate: float,
            velocity_status: str,
            total: int,
            done: int
    ) -> str:
        """
        Fallback analysis when LLM unavailable

        NEW: Provides intelligent baseline when LLM is down
        """
        if velocity_status == "excellent":
            return (
                f"Project showing excellent velocity with {completion_rate}% completion rate. "
                f"Team has completed {done} out of {total} issues, demonstrating strong execution. "
                f"Continue current execution plan and maintain momentum."
            )
        elif velocity_status == "on_track":
            return (
                f"Project velocity is healthy at {completion_rate}% completion. "
                f"With {done} issues done and {total - done} remaining, the team is on track. "
                f"Monitor progress closely to maintain current pace and address any emerging blockers."
            )
        elif velocity_status == "at_risk":
            return (
                f"Velocity below expectations at {completion_rate}% completion. "
                f"Only {done} of {total} issues completed. "
                f"Recommend immediate action: prioritize in-progress tasks, address blockers, "
                f"and consider scope adjustment if needed."
            )
        else:  # critical
            return (
                f"Critical velocity alert: {completion_rate}% completion is significantly below target. "
                f"With only {done} of {total} issues done, sprint goals are at high risk. "
                f"Immediate escalation required: conduct emergency standup, review scope, "
                f"reallocate resources, and identify critical blockers."
            )

    def _generate_headline(self, velocity_status: str, completion_rate: float) -> str:
        """Generate status headline based on velocity"""
        if velocity_status == "excellent":
            return f"Excellent sprint progress - {completion_rate}% complete"
        elif velocity_status == "on_track":
            return f"Sprint on track - {completion_rate}% complete"
        elif velocity_status == "at_risk":
            return f"Sprint at risk - only {completion_rate}% complete"
        else:
            return f"Critical sprint status - {completion_rate}% complete"

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
        self._next_step(
            reasoning,
            "Received commits for progress analysis",
            input_data={"commits_count": len(commits)}
        )

        # Step 2: Generate prompt
        prompt = (
                "You are a progress tracking agent.\n"
                "Analyze these commits and summarize achievements and velocity:\n" +
                "\n".join(f"- {c}" for c in commits) +
                "\n\nProvide a concise, positive summary of progress made."
        )

        self._next_step(
            reasoning,
            "Generated prompt for progress summary",
            output_data={"prompt_length": len(prompt)}
        )

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
                logger.warning(
                    "Progress Agent using fallback summary",
                    extra={"commits_count": len(commits)}
                )

            # Step 3: Analysis completed
            self._next_step(
                reasoning,
                "Received progress summary from LLM",
                output_data={
                    "summary_length": len(summary),
                    "fallback_used": fallback_used
                }
            )

            # Step 4 (optional): Fallback annotation
            if fallback_used:
                self._next_step(
                    reasoning,
                    "Fallback summary was used due to LLM unavailability",
                    output_data={"commits_analyzed": len(commits)}
                )

            logger.info(
                "Progress analysis completed",
                extra={"commits_count": len(commits), "fallback": fallback_used}
            )

            return {
                "commits_count": len(commits),
                "commits": commits,
                "summary": summary,
                "fallback_used": fallback_used,
                "reasoning": reasoning,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error("Progress analysis failed", extra={"error": str(e)})

            # Use fallback even on exception
            self._next_step(
                reasoning,
                "Progress analysis failed with exception - using fallback",
                output_data={"error": str(e), "fallback_used": True}
            )

            self._next_step(
                reasoning,
                "Fallback summary generated",
                output_data={"commits_count": len(commits)}
            )

            return {
                "commits_count": len(commits),
                "commits": commits,
                "summary": f"Progress analysis failed. Processed {len(commits)} commit(s). Manual review needed.",
                "fallback_used": True,
                "reasoning": reasoning,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    @log_method
    @metric_counter("progress")
    async def jira_velocity(self, project_key: Optional[str] = None):
        """
        Analyze project velocity with LLM-enhanced insights

        ENHANCED: Now combines deterministic calculations with LLM analysis!

        Architecture:
        1. Deterministic metrics (completion rate, status breakdown)
        2. LLM-powered interpretation (insights, risks, recommendations)
        3. Fallback to baseline when LLM unavailable
        """
        reasoning: List[ReasoningStep] = []

        # Step 1: Request received
        self._next_step(
            reasoning,
            "Jira velocity analysis requested",
            input_data={"project_key": project_key or self.jira.project_key}
        )

        try:
            # Step 2: Fetch from Jira (with automatic fallback to mocks)
            result = await self.jira.get_project_issues()

            issues = result.get("issues", [])
            jira_mode = result.get("mode", "unknown")

            self._next_step(
                reasoning,
                "Retrieved issues from Jira",
                output_data={
                    "issues_count": len(issues),
                    "jira_mode": jira_mode
                }
            )

            # Early exit if no data
            if not issues:
                self._next_step(reasoning, "No issues found - returning zero metrics")

                return {
                    "project": project_key or self.jira.project_key,
                    "executive_summary": {
                        "status": "NO_DATA",
                        "completion_rate": "0%",
                        "headline": "No Jira issues found",
                        "interpretation": "Unable to calculate velocity - no data available.",
                        "confidence": "low",
                        "data_source": f"jira ({jira_mode})",
                        "llm_enhanced": False
                    },
                    "metrics": {
                        "total_issues": 0,
                        "done": 0,
                        "completion_rate": 0.0,
                        "velocity_status": "no_data"
                    },
                    "status_breakdown": {},
                    "reasoning": reasoning,
                    "metadata": {
                        "jira_mode": jira_mode,
                        "agent": self.name,
                        "timestamp": datetime.now().isoformat()
                    }
                }

            # Step 3: Calculate deterministic metrics
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

            self._next_step(
                reasoning,
                "Calculated deterministic metrics",
                output_data={
                    "total_issues": total,
                    "done_issues": done,
                    "completion_rate": completion_rate,
                    "velocity_status": velocity_status
                }
            )

            # Step 4 - LLM-Enhanced Analysis (creates thinking pause!)
            self._next_step(
                reasoning,
                "Generating LLM-enhanced velocity insights",
                input_data={"velocity_status": velocity_status}
            )

            # Prepare context for LLM
            llm_prompt = f"""You are a project velocity analyst.

Project Velocity Data:
- Total Issues: {total}
- Completed: {done}
- In Progress: {status_counts.get('In Progress', 0)}
- To Do: {status_counts.get('To Do', 0)}
- Completion Rate: {completion_rate}%
- Velocity Classification: {velocity_status}
- Status Breakdown: {status_counts}

Analyze this data and provide:
1. Brief interpretation of current velocity (1-2 sentences)
2. Key risks or opportunities (1 sentence)
3. Recommended action (1 sentence)

Keep response under 100 words. Be specific and actionable."""

            llm_fallback = False

            try:

                llm_analysis = await self.llm.chat(llm_prompt)

                # Validate LLM response
                if self._is_invalid_response(llm_analysis):
                    llm_fallback = True
                    llm_analysis = self._generate_baseline_analysis(
                        completion_rate, velocity_status, total, done
                    )
                    logger.warning("LLM velocity analysis invalid - using baseline")

                self._next_step(
                    reasoning,
                    "LLM velocity analysis completed",
                    output_data={
                        "analysis_length": len(llm_analysis),
                        "llm_fallback": llm_fallback
                    }
                )

            except Exception as e:
                logger.error(f"LLM velocity analysis failed: {e}")
                llm_fallback = True
                llm_analysis = self._generate_baseline_analysis(
                    completion_rate, velocity_status, total, done
                )

                self._next_step(
                    reasoning,
                    "LLM analysis failed - using baseline",
                    output_data={"error": str(e), "fallback": True}
                )

            # Step 5: Generate executive summary
            executive_summary = {
                "status": velocity_status.upper(),
                "completion_rate": f"{completion_rate}%",
                "headline": self._generate_headline(velocity_status, completion_rate),
                "interpretation": llm_analysis,  # ✅ LLM-generated or baseline
                "confidence": "medium" if jira_mode == "mock" else "high",
                "data_source": f"jira ({jira_mode})",
                "llm_enhanced": not llm_fallback  # ✅ Transparency
            }

            self._next_step(
                reasoning,
                "Velocity analysis completed",
                output_data={
                    "completion_rate": completion_rate,
                    "velocity_status": velocity_status,
                    "llm_enhanced": not llm_fallback
                }
            )

            logger.info(
                "Jira velocity analysis completed",
                extra={
                    "completion_rate": completion_rate,
                    "velocity_status": velocity_status,
                    "llm_enhanced": not llm_fallback,
                    "jira_mode": jira_mode
                }
            )

            return {
                "project": project_key or self.jira.project_key,

                # Executive summary (enhanced with LLM insights!)
                "executive_summary": executive_summary,

                # Deterministic metrics
                "metrics": {
                    "total_issues": total,
                    "done": done,
                    "in_progress": status_counts.get("In Progress", 0),
                    "to_do": status_counts.get("To Do", 0),
                    "completion_rate": completion_rate,
                    "velocity_status": velocity_status
                },

                "status_breakdown": status_counts,

                "ai_analysis": {
                    "llm_insights": llm_analysis,
                    "llm_used": not llm_fallback,
                    "analysis_type": "hybrid",  # Deterministic + LLM
                    "fallback_reason": "llm_unavailable" if llm_fallback else None
                },

                "reasoning": reasoning,

                "metadata": {
                    "jira_mode": jira_mode,
                    "agent": self.name,
                    "timestamp": datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error("Jira velocity failed", extra={"error": str(e)})

            self._next_step(
                reasoning,
                "Jira velocity failed with exception",
                output_data={"error": str(e)}
            )

            return {
                "project": project_key or self.jira.project_key,
                "error": str(e),
                "reasoning": reasoning,
                "fallback": "Manual Jira review required",
                "metadata": {
                    "agent": self.name,
                    "timestamp": datetime.now().isoformat()
                }
            }


# Health check endpoint
def get_agent_status() -> Dict[str, Any]:
    """Returns agent status for HealthMonitor integration"""
    return {
        "agent_name": "progress",
        "status": "HEALTHY",
        "capabilities": [
            "analyze_progress",
            "jira_velocity"
        ],
        "features": [
            "llm_enhanced_velocity",
            "deterministic_calculations",
            "fallback_strategy"
        ],
        "agent_type": "hybrid",
        "llm_powered": True,
        "jira_integration": True,
        "timestamp": datetime.now().isoformat()
    }
