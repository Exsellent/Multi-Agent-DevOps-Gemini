import logging
from typing import List, Optional

from shared.jira import JiraClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("planner_agent")


def log_method(func):
    """Decorator for logging method calls"""

    async def wrapper(self, *args, **kwargs):
        logger.info(f"{func.__name__} called")
        try:
            result = await func(self, *args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            raise

    return wrapper


class PlannerAgent(MCPAgent):
    def __init__(self):
        super().__init__("Planner")
        self.llm = LLMClient()
        self.jira = JiraClient()

        self.register_tool("plan", self.plan)
        self.register_tool("plan_with_jira", self.plan_with_jira)

        logger.info("PlannerAgent initialized with JSON Mode support")

    @log_method
    @metric_counter("planner")
    async def plan(self, description: str):
        """
        Enhanced task planning with Gemini JSON Mode

        Uses structured output for reliable parsing
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Received task for planning",
            input_data={"description": description}
        ))

        # System instruction for JSON Mode
        system_instruction = (
            "You are a senior project planner. "
            "Analyze the task and break it down into 3-5 actionable subtasks. "
            "Return ONLY valid JSON in this exact format:\n"
            "{\n"
            "  \"subtasks\": [\"subtask 1\", \"subtask 2\", \"subtask 3\"],\n"
            "  \"complexity\": \"low|medium|high\",\n"
            "  \"estimated_days\": 3\n"
            "}\n"
            "No markdown, no explanations, just pure JSON."
        )

        prompt = f"Task: {description}\n\nBreak this down into concrete, actionable subtasks."

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Generated JSON Mode prompt for LLM"
        ))

        try:
            # Use JSON Mode for reliable parsing
            result = await self.llm.chat_json(prompt, system_instruction)

            # Extract subtasks from JSON response
            subtasks = result.get("subtasks", [])
            complexity = result.get("complexity", "medium")
            estimated_days = result.get("estimated_days", 5)

            if not subtasks:
                # Fallback to safe default
                logger.warning("Empty subtasks from LLM, using fallback")
                subtasks = self._get_fallback_subtasks(description)

                reasoning.append(ReasoningStep(
                    step_number=3,
                    description="Used fallback subtasks (empty response)",
                    output_data={"fallback_used": True}
                ))
            else:
                reasoning.append(ReasoningStep(
                    step_number=3,
                    description="Successfully parsed JSON response from LLM",
                    output_data={
                        "subtasks_count": len(subtasks),
                        "complexity": complexity,
                        "estimated_days": estimated_days
                    }
                ))

            logger.info(
                "Planning completed",
                extra={
                    "task": description,
                    "subtasks_count": len(subtasks),
                    "complexity": complexity
                }
            )

            return {
                "task": description,
                "subtasks": subtasks,
                "complexity": complexity,
                "estimated_days": estimated_days,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error(
                "Planning failed critically",
                extra={"task": description, "error": str(e)}
            )

            reasoning.append(ReasoningStep(
                step_number=4,
                description="Critical planning failure - using fallback",
                output_data={"error": str(e)}
            ))

            # Use safe fallback
            subtasks = self._get_fallback_subtasks(description)

            return {
                "task": description,
                "subtasks": subtasks,
                "complexity": "medium",
                "estimated_days": 5,
                "fallback": True,
                "reasoning": reasoning
            }

    def _get_fallback_subtasks(self, description: str) -> List[str]:
        """
        Generate safe fallback subtasks based on task description
        """
        # Generic subtasks that work for most tasks
        return [
            f"Research requirements and constraints for: {description[:50]}",
            "Design solution architecture and components",
            "Implement core functionality and features",
            "Add error handling and validation",
            "Write tests and documentation"
        ]

    @log_method
    @metric_counter("planner")
    async def plan_with_jira(
            self,
            description: str,
            project_key: Optional[str] = None
    ):
        """
        Planning + automatic task creation in Jira
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Received task for planning with Jira integration",
            input_data={"description": description, "project_key": project_key}
        ))

        # First, normal planning
        plan_result = await self.plan(description)
        subtasks = plan_result.get("subtasks", [])

        reasoning.extend(plan_result.get("reasoning", []))

        if "error" in plan_result and not plan_result.get("fallback"):
            return plan_result

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Planning phase completed, initiating Jira integration",
            output_data={"subtasks_count": len(subtasks)}
        ))

        jira_issues = []

        try:
            # Create Epic
            epic_result = await self.jira.create_task(
                summary=f"[Epic] {description}",
                description=f"Auto-generated by Multi-Agent DevOps Assistant\nSubtasks planned: {len(subtasks)}\nComplexity: {plan_result.get('complexity', 'medium')}"
            )
            jira_issues.append(epic_result)

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Created Epic in Jira",
                output_data={"epic": epic_result}
            ))

            # Create subtasks
            for i, subtask in enumerate(subtasks, 1):
                issue_result = await self.jira.create_task(
                    summary=f"[Subtask {i}] {subtask}",
                    description=f"Part of epic: {description}"
                )
                jira_issues.append(issue_result)

            reasoning.append(ReasoningStep(
                step_number=4,
                description="Successfully created all Jira issues",
                output_data={"total_issues_created": len(jira_issues)}
            ))

            logger.info(
                "Plan with Jira completed successfully",
                extra={
                    "task": description,
                    "issues_count": len(jira_issues)
                }
            )

        except Exception as e:
            logger.error("Jira integration failed", extra={"error": str(e)})
            reasoning.append(ReasoningStep(
                step_number=5,
                description="Jira task creation failed",
                output_data={"error": str(e)}
            ))
            jira_issues.append({"status": "error", "details": str(e)})

        return {
            "task": description,
            "subtasks": subtasks,
            "complexity": plan_result.get("complexity", "medium"),
            "estimated_days": plan_result.get("estimated_days", 5),
            "jira_issues": jira_issues,
            "jira_mode": "mock" if self.jira.mock_mode else "real",
            "reasoning": reasoning
        }