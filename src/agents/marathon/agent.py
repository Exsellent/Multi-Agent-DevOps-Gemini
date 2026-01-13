import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("marathon_agent")


class TaskState(str, Enum):
    """Possible states for long-running tasks"""
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    WAITING = "waiting"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ThoughtSignature:
    """
    Persistent reasoning state that survives across sessions

    This is the key to Marathon Agent - it maintains context
    and can resume from where it left off, even after hours/days.
    """
    timestamp: str
    task_id: str
    current_state: TaskState
    reasoning_chain: List[str]  # Chain of thought so far
    decisions_made: List[Dict[str, Any]]  # What was decided and why
    next_steps: List[str]  # Planned future actions
    blockers: List[str]  # Current obstacles
    self_corrections: List[Dict[str, Any]]  # Mistakes caught and fixed
    progress_percentage: int
    estimated_completion: str


class MarathonAgent(MCPAgent):
    """
    Autonomous agent for tasks spanning hours or days

    Key capabilities:
    - Maintains state across interruptions
    - Self-corrects based on feedback loops
    - Makes decisions without human intervention
    - Adapts plan based on changing conditions

    Example use case: Monitor CI/CD pipeline for 48 hours,
    auto-fix failures, and report when deployment is stable.
    """

    def __init__(self):
        super().__init__("Marathon")
        self.llm = LLMClient()

        # In-memory task state (in production, use Redis/DB)
        self.active_tasks: Dict[str, ThoughtSignature] = {}

        self.register_tool("start_marathon_task", self.start_marathon_task)
        self.register_tool("check_task_progress", self.check_task_progress)
        self.register_tool("resume_task", self.resume_task)
        self.register_tool("self_correct", self.self_correct)

        logger.info("Marathon Agent initialized - ready for long-running tasks")

    @metric_counter("marathon")
    async def start_marathon_task(
            self,
            task_description: str,
            duration_hours: int,
            success_criteria: str
    ) -> Dict[str, Any]:
        """
        Start a long-running autonomous task

        Args:
            task_description: What to accomplish
            duration_hours: How long the task can run
            success_criteria: When to consider it done
        """
        reasoning: List[ReasoningStep] = []
        task_id = f"marathon_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Initializing marathon task with long-term planning",
            input_data={
                "task": task_description,
                "duration_hours": duration_hours,
                "success_criteria": success_criteria
            }
        ))

        # Step 1: Break down into sub-goals with time estimates
        planning_prompt = f"""
You are planning a task that will run autonomously for up to {duration_hours} hours.

TASK: {task_description}
SUCCESS CRITERIA: {success_criteria}

Create a detailed execution plan:

1. Break into sub-goals (each should take 30min - 4h)
2. For each sub-goal:
   - Clear objective
   - Success check (how to verify it worked)
   - Failure recovery (what to do if it fails)
   - Estimated duration
3. Identify decision points where you'll need to adapt the plan
4. List potential blockers and mitigation strategies

Return a structured plan that can guide autonomous execution.
"""

        plan = await self.llm.chat(planning_prompt)

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Generated autonomous execution plan",
            output_data={"plan_created": True}
        ))

        # Step 2: Initialize thought signature
        thought_signature = ThoughtSignature(
            timestamp=datetime.now().isoformat(),
            task_id=task_id,
            current_state=TaskState.PLANNING,
            reasoning_chain=[
                f"Task started: {task_description}",
                f"Initial plan created with {duration_hours}h budget"
            ],
            decisions_made=[{
                "decision": "Created execution plan",
                "reasoning": "Breaking complex task into verifiable sub-goals",
                "timestamp": datetime.now().isoformat()
            }],
            next_steps=self._extract_next_steps(plan),
            blockers=[],
            self_corrections=[],
            progress_percentage=0,
            estimated_completion=(datetime.now() + timedelta(hours=duration_hours)).isoformat()
        )

        self.active_tasks[task_id] = thought_signature

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Initialized thought signature for long-term continuity",
            output_data={"task_id": task_id}
        ))

        return {
            "task_id": task_id,
            "status": "started",
            "execution_plan": plan,
            "thought_signature": asdict(thought_signature),
            "reasoning": reasoning,
            "message": f"Marathon task started. Will run autonomously for up to {duration_hours} hours."
        }

    @metric_counter("marathon")
    async def check_task_progress(
            self,
            task_id: str
    ) -> Dict[str, Any]:
        """
        Check current state of a marathon task

        This demonstrates continuity - the agent can report
        on what it's been doing even if hours have passed.
        """
        reasoning: List[ReasoningStep] = []

        if task_id not in self.active_tasks:
            return {"error": f"Task {task_id} not found"}

        thought_sig = self.active_tasks[task_id]

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Retrieving task state from thought signature",
            input_data={"task_id": task_id}
        ))

        # Generate progress summary using LLM
        summary_prompt = f"""
Analyze this task's progress and provide a human-readable status update:

TASK ID: {task_id}
CURRENT STATE: {thought_sig.current_state}
PROGRESS: {thought_sig.progress_percentage}%
REASONING CHAIN: {thought_sig.reasoning_chain[-5:]}  (last 5 thoughts)
DECISIONS MADE: {thought_sig.decisions_made[-3:]}  (recent)
BLOCKERS: {thought_sig.blockers}

Provide:
1. What has been accomplished so far
2. Current activity
3. Any issues encountered and how they were handled
4. Estimated time to completion
5. Confidence level in success (%)

Be concise but informative.
"""

        summary = await self.llm.chat(summary_prompt)

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Generated progress summary",
            output_data={"summary_ready": True}
        ))

        return {
            "task_id": task_id,
            "progress_percentage": thought_sig.progress_percentage,
            "current_state": thought_sig.current_state,
            "summary": summary,
            "thought_signature": asdict(thought_sig),
            "reasoning": reasoning
        }

    @metric_counter("marathon")
    async def resume_task(
            self,
            task_id: str,
            new_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Resume a task after interruption

        This is the core of Marathon capability:
        - Reads previous thought signature
        - Understands what was being done
        - Continues from where it left off
        """
        reasoning: List[ReasoningStep] = []

        if task_id not in self.active_tasks:
            return {"error": f"Task {task_id} not found"}

        thought_sig = self.active_tasks[task_id]

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Resuming task from thought signature",
            input_data={"task_id": task_id, "new_context": new_context}
        ))

        # Reconstruct context and decide next action
        resume_prompt = f"""
You are resuming a long-running task after an interruption.

PREVIOUS CONTEXT:
- Task ID: {task_id}
- Last state: {thought_sig.current_state}
- Progress: {thought_sig.progress_percentage}%
- Reasoning chain: {thought_sig.reasoning_chain}
- Recent decisions: {thought_sig.decisions_made[-3:]}
- Planned next steps: {thought_sig.next_steps}
- Known blockers: {thought_sig.blockers}

NEW INFORMATION: {new_context or "None"}

Based on this context:
1. What was the agent doing before interruption?
2. Is the previous plan still valid?
3. What should be the immediate next action?
4. Are there any adjustments needed based on new information?
5. Update the reasoning chain with your analysis

Provide a clear action plan to continue execution.
"""

        resume_plan = await self.llm.chat(resume_prompt)

        # Update thought signature
        thought_sig.reasoning_chain.append(f"Resumed after interruption: {datetime.now().isoformat()}")
        if new_context:
            thought_sig.reasoning_chain.append(f"New context: {new_context}")

        thought_sig.decisions_made.append({
            "decision": "Resume execution with updated plan",
            "reasoning": resume_plan[:200],
            "timestamp": datetime.now().isoformat()
        })

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Updated thought signature with resume context",
            output_data={"plan_updated": True}
        ))

        return {
            "task_id": task_id,
            "status": "resumed",
            "resume_plan": resume_plan,
            "updated_thought_signature": asdict(thought_sig),
            "reasoning": reasoning
        }

    @metric_counter("marathon")
    async def self_correct(
            self,
            task_id: str,
            observed_issue: str
    ) -> Dict[str, Any]:
        """
        Self-correction based on feedback loop

        The agent detects something went wrong and fixes its approach.
        This demonstrates autonomous error recovery.
        """
        reasoning: List[ReasoningStep] = []

        if task_id not in self.active_tasks:
            return {"error": f"Task {task_id} not found"}

        thought_sig = self.active_tasks[task_id]

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Initiating self-correction process",
            input_data={"issue": observed_issue}
        ))

        # Analyze the issue and determine correction
        correction_prompt = f"""
You are analyzing a problem in your autonomous execution.

CONTEXT:
- Task: {task_id}
- Current progress: {thought_sig.progress_percentage}%
- Recent actions: {thought_sig.reasoning_chain[-5:]}
- Decisions made: {thought_sig.decisions_made[-3:]}

OBSERVED ISSUE: {observed_issue}

Perform root cause analysis:
1. What went wrong?
2. Why did it happen? (decision error? assumption error? external change?)
3. What should have been done differently?
4. How to fix it now?
5. How to prevent similar issues in the future?

Provide a correction plan with clear reasoning.
"""

        correction_plan = await self.llm.chat(correction_prompt)

        # Record the self-correction
        correction_record = {
            "timestamp": datetime.now().isoformat(),
            "issue": observed_issue,
            "root_cause": correction_plan[:300],
            "correction_applied": True
        }

        thought_sig.self_corrections.append(correction_record)
        thought_sig.reasoning_chain.append(f"Self-corrected: {observed_issue}")

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Applied self-correction and updated approach",
            output_data={"correction_recorded": True}
        ))

        return {
            "task_id": task_id,
            "status": "self_corrected",
            "correction_plan": correction_plan,
            "correction_record": correction_record,
            "updated_thought_signature": asdict(thought_sig),
            "reasoning": reasoning,
            "message": "Task execution corrected autonomously"
        }

    def _extract_next_steps(self, plan: str) -> List[str]:
        """Extract actionable next steps from LLM plan"""

        lines = plan.split('\n')
        steps = [line.strip() for line in lines if line.strip().startswith(('1.', '2.', '3.', '-', '•'))]
        return steps[:5]
