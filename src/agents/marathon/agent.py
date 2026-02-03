import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Optional, Dict, Any, Protocol

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


class ThinkingLevel(str, Enum):
    """
    Hierarchical thinking levels for Marathon Agent
    Critical for Gemini 3 hackathon requirements
    """
    STRATEGIC = "strategic"  # Hours/days planning
    TACTICAL = "tactical"  # Next 3-5 steps
    OPERATIONAL = "operational"  # Individual tool calls
    REFLECTIVE = "reflective"  # Self-correction and learning


class SubGoalStatus(str, Enum):
    """Status of individual sub-goals"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubGoal:
    """
    Verifiable sub-goal with success criteria
    """
    name: str
    description: str
    status: SubGoalStatus
    success_check: str  # How to verify it worked
    failure_recovery: str  # What to do if it fails
    estimated_duration_mins: int
    actual_duration_mins: Optional[int] = None

    # References to external verification artifacts
    # (created by specialized agents like CodeExecutionAgent)
    verification_artifact_refs: List[str] = None

    def __post_init__(self):
        if self.verification_artifact_refs is None:
            self.verification_artifact_refs = []


@dataclass
class ThinkingRecord:
    """
    Record of reasoning at specific thinking level
    """
    level: ThinkingLevel
    timestamp: str
    thought: str
    decisions: List[str]
    confidence: float  # 0.0 - 1.0


@dataclass
class DelegationRecord:
    """
    Record of task delegation to another agent
    """
    timestamp: str
    delegated_to: str  # Agent name
    task_description: str
    result_summary: Optional[str] = None
    success: Optional[bool] = None
    artifact_refs: List[str] = None

    def __post_init__(self):
        if self.artifact_refs is None:
            self.artifact_refs = []


@dataclass
class ThoughtSignature:
    """
    Persistent reasoning state that survives across sessions

    This is the core of Marathon Agent - maintains full context
    and enables resumption after hours/days of interruption.
    """
    timestamp: str
    task_id: str
    current_state: TaskState
    current_thinking_level: ThinkingLevel

    # Multi-level reasoning chain
    strategic_plan: str  # High-level approach
    tactical_steps: List[str]  # Next 3-5 actions
    operational_log: List[str]  # Detailed execution log

    # Structured execution tracking
    sub_goals: List[SubGoal]
    current_sub_goal_idx: int

    # Decision trail
    thinking_records: List[ThinkingRecord]
    decisions_made: List[Dict[str, Any]]

    # Agent orchestration
    delegations: List[DelegationRecord]

    # Self-correction
    self_corrections: List[Dict[str, Any]]
    blockers: List[str]

    # Progress tracking
    progress_percentage: int
    estimated_completion: str
    actual_start_time: str

    # Execution metadata
    total_tool_calls: int
    total_delegations: int
    successful_delegations: int
    failed_delegations: int


class ThoughtSignatureStore(Protocol):
    """
    Protocol for persistent storage
    In production: Redis, PostgreSQL, or cloud storage
    """

    async def load(self, task_id: str) -> Optional[ThoughtSignature]:
        ...

    async def save(self, thought_signature: ThoughtSignature) -> bool:
        ...

    async def list_active(self) -> List[str]:
        ...


class InMemoryStore:
    """Simple in-memory implementation for demo/testing"""

    def __init__(self):
        self._store: Dict[str, ThoughtSignature] = {}

    async def load(self, task_id: str) -> Optional[ThoughtSignature]:
        return self._store.get(task_id)

    async def save(self, thought_signature: ThoughtSignature) -> bool:
        self._store[thought_signature.task_id] = thought_signature
        return True

    async def list_active(self) -> List[str]:
        return [
            tid for tid, ts in self._store.items()
            if ts.current_state not in [TaskState.COMPLETED, TaskState.FAILED]
        ]


class MarathonAgent(MCPAgent):
    """
    Pure Marathon Agent - Long-horizon Task Orchestrator

    Key responsibilities:
    1. Multi-level thinking (Strategic → Tactical → Operational → Reflective)
    2. Long-running task management (hours/days)
    3. Orchestration of specialized agents
    4. State persistence and recovery
    5. Self-correction and adaptation

    What Marathon DOES:
    - Plans and tracks multi-hour/day tasks
    - Orchestrates other agents (CodeExecution, ArchitectureIntelligence, etc.)
    - Maintains thought signatures across interruptions
    - Makes high-level decisions

    What Marathon DOESN'T DO:
    - Write or execute code (delegates to CodeExecutionAgent)
    - Perform specialized verification (delegates to specialists)
    - Domain-specific work (delegates appropriately)

    This is the gold standard for autonomous orchestration.
    """

    def __init__(self, store: Optional[ThoughtSignatureStore] = None):
        super().__init__("Marathon")
        self.llm = LLMClient()

        # Persistent storage
        self.store = store or InMemoryStore()

        # Execution control
        self._running_loops: Dict[str, bool] = {}

        # Reference to other agents (injected or discovered)
        # In production: service discovery or dependency injection
        self._available_agents: Dict[str, Any] = {}

        # Register tools
        self.register_tool("start_marathon_task", self.start_marathon_task)
        self.register_tool("check_task_progress", self.check_task_progress)
        self.register_tool("resume_task", self.resume_task)
        self.register_tool("autonomous_tick", self.autonomous_tick)
        self.register_tool("delegate_to_agent", self.delegate_to_agent)
        self.register_tool("self_correct", self.self_correct)
        self.register_tool("get_thinking_records", self.get_thinking_records)
        self.register_tool("stop_task", self.stop_task)

        logger.info("Marathon Agent initialized (PURE ORCHESTRATOR)")

    def register_agent(self, agent_name: str, agent_instance: Any):
        """
        Register a specialized agent for delegation

        Example:
            marathon.register_agent("CodeExecution", code_execution_agent)
            marathon.register_agent("ArchitectureIntelligence", arch_agent)
        """
        self._available_agents[agent_name] = agent_instance
        logger.info(f"Registered agent: {agent_name}")

    def _calculate_progress(self, sub_goals: List[SubGoal]) -> int:
        """INTERNAL DECISION: Calculate objective progress"""
        if not sub_goals:
            return 0

        done = sum(1 for sg in sub_goals if sg.status == SubGoalStatus.DONE)
        return int((done / len(sub_goals)) * 100)

    def _add_thinking_record(
            self,
            thought_sig: ThoughtSignature,
            level: ThinkingLevel,
            thought: str,
            decisions: List[str],
            confidence: float
    ):
        """Record thinking at specific level"""
        record = ThinkingRecord(
            level=level,
            timestamp=datetime.now().isoformat(),
            thought=thought,
            decisions=decisions,
            confidence=confidence
        )
        thought_sig.thinking_records.append(record)

    async def _think_strategic(
            self,
            task_description: str,
            duration_hours: int,
            success_criteria: str
    ) -> tuple[str, List[SubGoal]]:
        """
        STRATEGIC THINKING LEVEL
        Creates high-level plan and sub-goals
        """
        planning_prompt = f"""
You are planning an AUTONOMOUS task that will run for up to {duration_hours} hours.

TASK: {task_description}
SUCCESS CRITERIA: {success_criteria}

Create a strategic execution plan:

1. Break into 5-10 verifiable sub-goals
2. For EACH sub-goal provide:
   - Clear objective
   - Success check (how to verify programmatically)
   - Failure recovery strategy
   - Time estimate (in minutes)
3. Identify decision points and adaptation triggers
4. List potential blockers with mitigation

Output as JSON:
{{
  "strategic_plan": "overall approach...",
  "sub_goals": [
    {{
      "name": "goal name",
      "description": "what to do",
      "success_check": "verification method",
      "failure_recovery": "fallback plan",
      "estimated_duration_mins": 30
    }}
  ]
}}

CRITICAL: Each sub-goal must be VERIFIABLE through automated means.
"""

        response = await self.llm.chat(planning_prompt)

        # Parse response
        try:
            # Clean JSON
            clean = response.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

            data = json.loads(clean)
            strategic_plan = data.get("strategic_plan", "")

            sub_goals = []
            for sg_data in data.get("sub_goals", []):
                sub_goal = SubGoal(
                    name=sg_data.get("name", ""),
                    description=sg_data.get("description", ""),
                    status=SubGoalStatus.PENDING,
                    success_check=sg_data.get("success_check", ""),
                    failure_recovery=sg_data.get("failure_recovery", ""),
                    estimated_duration_mins=sg_data.get("estimated_duration_mins", 60),
                    verification_artifact_refs=[]
                )
                sub_goals.append(sub_goal)

            return strategic_plan, sub_goals

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategic plan: {e}")
            # Fallback
            return response, []

    async def _think_tactical(
            self,
            thought_sig: ThoughtSignature
    ) -> List[str]:
        """
        TACTICAL THINKING LEVEL
        Determines next 3-5 concrete actions
        """
        current_goal = None
        if thought_sig.current_sub_goal_idx < len(thought_sig.sub_goals):
            current_goal = thought_sig.sub_goals[thought_sig.current_sub_goal_idx]

        if not current_goal:
            return ["All goals completed or no goals available"]

        # Check available agents
        available_agents_str = ", ".join(self._available_agents.keys()) if self._available_agents else "None"

        tactical_prompt = f"""
Current sub-goal: {current_goal.name}
Description: {current_goal.description}
Success check: {current_goal.success_check}

Recent execution log:
{thought_sig.operational_log[-5:] if thought_sig.operational_log else "None"}

Available specialist agents: {available_agents_str}

Determine the next 3-5 CONCRETE actions to accomplish this sub-goal.

Each action should be:
- Specific (e.g., "delegate to CodeExecution: implement X")
- Executable without human input
- Verifiable

Return as JSON array:
["action 1", "action 2", "action 3"]
"""

        response = await self.llm.chat(tactical_prompt)

        try:
            clean = response.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

            actions = json.loads(clean)
            return actions[:5]  # Max 5 actions

        except json.JSONDecodeError:
            # Fallback: parse lines
            lines = [l.strip() for l in response.split('\n') if l.strip()]
            return lines[:5]

    async def _think_operational(
            self,
            action: str,
            thought_sig: ThoughtSignature
    ) -> Dict[str, Any]:
        """
        OPERATIONAL THINKING LEVEL
        Executes individual action (often by delegating)
        """
        # Log the action
        thought_sig.operational_log.append(
            f"[{datetime.now().isoformat()}] {action}"
        )
        thought_sig.total_tool_calls += 1

        # Check if this is a delegation
        if "delegate to" in action.lower() or "call" in action.lower():
            # Extract agent name and task
            # Simple heuristic - in production: more sophisticated parsing
            parts = action.split(":")
            if len(parts) >= 2:
                agent_part = parts[0].lower()
                task_desc = parts[1].strip()

                # Find matching agent
                for agent_name, agent_instance in self._available_agents.items():
                    if agent_name.lower() in agent_part:
                        # Delegate
                        return await self.delegate_to_agent(
                            task_id=thought_sig.task_id,
                            agent_name=agent_name,
                            task_description=task_desc
                        )

        # Generic action execution (monitoring, status checks, etc.)
        execution_result = {
            "action": action,
            "status": "executed",
            "timestamp": datetime.now().isoformat(),
            "details": f"Executed: {action}"
        }

        return execution_result

    async def _think_reflective(
            self,
            thought_sig: ThoughtSignature,
            observation: str
    ) -> Dict[str, Any]:
        """
        REFLECTIVE THINKING LEVEL
        Self-correction and learning
        """
        reflection_prompt = f"""
You are reflecting on autonomous execution progress.

CONTEXT:
- Current state: {thought_sig.current_state}
- Progress: {thought_sig.progress_percentage}%
- Recent actions: {thought_sig.operational_log[-5:]}
- Current blockers: {thought_sig.blockers}
- Delegations: {thought_sig.total_delegations} total, {thought_sig.successful_delegations} successful

OBSERVATION: {observation}

Perform reflective analysis:
1. Is the current approach working?
2. Are we making expected progress?
3. Do we need to adjust strategy or tactics?
4. Should we delegate differently?
5. What should change?

Return as JSON:
{{
  "assessment": "working|needs_adjustment|critical_issue",
  "confidence": 0.85,
  "recommended_changes": ["change 1", "change 2"],
  "reasoning": "why these changes"
}}
"""

        response = await self.llm.chat(reflection_prompt)

        try:
            clean = response.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()

            reflection = json.loads(clean)
            return reflection

        except json.JSONDecodeError:
            return {
                "assessment": "unknown",
                "confidence": 0.5,
                "recommended_changes": [],
                "reasoning": response
            }

    @metric_counter("marathon")
    async def start_marathon_task(
            self,
            task_description: str,
            duration_hours: int,
            success_criteria: str,
            auto_start_loop: bool = False
    ) -> Dict[str, Any]:
        """
        Start a long-running autonomous task

        Args:
            task_description: What to accomplish
            duration_hours: Maximum execution time
            success_criteria: When to consider it done
            auto_start_loop: If True, start autonomous execution immediately
        """
        reasoning: List[ReasoningStep] = []
        task_id = f"marathon_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Entering STRATEGIC thinking level",
            input_data={
                "task": task_description,
                "duration_hours": duration_hours,
                "thinking_level": ThinkingLevel.STRATEGIC
            }
        ))

        # STRATEGIC LEVEL: Create plan
        strategic_plan, sub_goals = await self._think_strategic(
            task_description=task_description,
            duration_hours=duration_hours,
            success_criteria=success_criteria
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Strategic plan created",
            output_data={
                "sub_goals_count": len(sub_goals),
                "total_estimated_mins": sum(sg.estimated_duration_mins for sg in sub_goals)
            }
        ))

        # Initialize thought signature
        thought_signature = ThoughtSignature(
            timestamp=datetime.now().isoformat(),
            task_id=task_id,
            current_state=TaskState.PLANNING,
            current_thinking_level=ThinkingLevel.STRATEGIC,
            strategic_plan=strategic_plan,
            tactical_steps=[],
            operational_log=[],
            sub_goals=sub_goals,
            current_sub_goal_idx=0,
            thinking_records=[],
            decisions_made=[{
                "decision": "Strategic plan created",
                "reasoning": f"Broke task into {len(sub_goals)} verifiable sub-goals",
                "timestamp": datetime.now().isoformat(),
                "thinking_level": ThinkingLevel.STRATEGIC
            }],
            delegations=[],
            self_corrections=[],
            blockers=[],
            progress_percentage=0,
            estimated_completion=(datetime.now() + timedelta(hours=duration_hours)).isoformat(),
            actual_start_time=datetime.now().isoformat(),
            total_tool_calls=0,
            total_delegations=0,
            successful_delegations=0,
            failed_delegations=0
        )

        # Add strategic thinking record
        self._add_thinking_record(
            thought_sig=thought_signature,
            level=ThinkingLevel.STRATEGIC,
            thought=f"Created execution plan with {len(sub_goals)} sub-goals",
            decisions=[f"Sub-goal {i + 1}: {sg.name}" for i, sg in enumerate(sub_goals)],
            confidence=0.8
        )

        # Save to persistent storage
        await self.store.save(thought_signature)

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Thought signature persisted",
            output_data={"task_id": task_id}
        ))

        result = {
            "task_id": task_id,
            "status": "initialized",
            "strategic_plan": strategic_plan,
            "sub_goals": [asdict(sg) for sg in sub_goals],
            "thought_signature": asdict(thought_signature),
            "reasoning": reasoning,
            "message": f"Marathon task initialized. {len(sub_goals)} sub-goals planned."
        }

        # Auto-start execution loop if requested
        if auto_start_loop:
            asyncio.create_task(self._autonomous_execution_loop(task_id))
            result["message"] += " Autonomous execution started."

        return result

    @metric_counter("marathon")
    async def delegate_to_agent(
            self,
            task_id: str,
            agent_name: str,
            task_description: str,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delegate task to a specialized agent

        This is how Marathon orchestrates other agents
        """
        reasoning: List[ReasoningStep] = []

        thought_sig = await self.store.load(task_id)
        if not thought_sig:
            return {"error": f"Task {task_id} not found"}

        reasoning.append(ReasoningStep(
            step_number=1,
            description=f"OPERATIONAL: Delegating to {agent_name}",
            input_data={
                "agent": agent_name,
                "task": task_description,
                "thinking_level": ThinkingLevel.OPERATIONAL
            }
        ))

        # Check if agent is available
        if agent_name not in self._available_agents:
            reasoning.append(ReasoningStep(
                step_number=2,
                description=f"Agent {agent_name} not available",
                output_data={"error": "agent_not_found"}
            ))

            thought_sig.failed_delegations += 1
            thought_sig.blockers.append(f"Agent {agent_name} not available")
            await self.store.save(thought_sig)

            return {
                "error": f"Agent {agent_name} not available",
                "available_agents": list(self._available_agents.keys()),
                "reasoning": reasoning
            }

        # Delegate
        agent = self._available_agents[agent_name]
        thought_sig.total_delegations += 1

        delegation_record = DelegationRecord(
            timestamp=datetime.now().isoformat(),
            delegated_to=agent_name,
            task_description=task_description
        )

        try:
            # Call agent (specific method depends on agent type)
            # This is a simplified interface - in production: standardized protocol

            result = None

            if agent_name == "CodeExecution":
                # Delegate to CodeExecutionAgent
                result = await agent.generate_and_test_code(
                    requirement=task_description,
                    context=context.get("context") if context else None
                )
            elif hasattr(agent, 'execute_task'):
                # Generic delegation interface
                result = await agent.execute_task(
                    description=task_description,
                    context=context
                )

            # Process result
            if result:
                # Check if delegation was successful
                success = result.get("production_ready", False) or result.get("status") == "success"

                delegation_record.success = success
                delegation_record.result_summary = str(result.get("summary", "Completed"))

                # Extract artifact references
                if "verification_artifact" in result:
                    artifact_id = result["verification_artifact"].get("artifact_id")
                    if artifact_id:
                        delegation_record.artifact_refs.append(artifact_id)

                if success:
                    thought_sig.successful_delegations += 1
                else:
                    thought_sig.failed_delegations += 1

                reasoning.append(ReasoningStep(
                    step_number=2,
                    description=f"Delegation to {agent_name} completed",
                    output_data={
                        "success": success,
                        "has_artifacts": len(delegation_record.artifact_refs) > 0
                    }
                ))
            else:
                delegation_record.success = False
                delegation_record.result_summary = "No result returned"
                thought_sig.failed_delegations += 1

                reasoning.append(ReasoningStep(
                    step_number=2,
                    description=f"Delegation to {agent_name} returned no result",
                    output_data={"success": False}
                ))

        except Exception as e:
            logger.exception(f"Delegation to {agent_name} failed")
            delegation_record.success = False
            delegation_record.result_summary = f"Error: {str(e)}"
            thought_sig.failed_delegations += 1

            reasoning.append(ReasoningStep(
                step_number=2,
                description=f"Delegation to {agent_name} failed with exception",
                output_data={"error": str(e)}
            ))

        # Record delegation
        thought_sig.delegations.append(delegation_record)

        # Log to operational log
        thought_sig.operational_log.append(
            f"[{datetime.now().isoformat()}] Delegated to {agent_name}: {delegation_record.result_summary}"
        )

        await self.store.save(thought_sig)

        return {
            "task_id": task_id,
            "delegation": asdict(delegation_record),
            "reasoning": reasoning
        }

    @metric_counter("marathon")
    async def autonomous_tick(
            self,
            task_id: str
    ) -> Dict[str, Any]:
        """
        Execute one autonomous tick of the Marathon loop

        This is the heart of Marathon Agent:
        Observe → Decide → Act → Verify → Update
        """
        reasoning: List[ReasoningStep] = []

        # Load state
        thought_sig = await self.store.load(task_id)
        if not thought_sig:
            return {"error": f"Task {task_id} not found"}

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Autonomous tick started",
            input_data={
                "task_id": task_id,
                "current_state": thought_sig.current_state,
                "progress": thought_sig.progress_percentage
            }
        ))

        # Check if completed
        if thought_sig.current_state in [TaskState.COMPLETED, TaskState.FAILED]:
            return {
                "task_id": task_id,
                "status": "finished",
                "final_state": thought_sig.current_state,
                "reasoning": reasoning
            }

        # TACTICAL LEVEL: Get next actions
        thought_sig.current_thinking_level = ThinkingLevel.TACTICAL
        tactical_steps = await self._think_tactical(thought_sig)
        thought_sig.tactical_steps = tactical_steps

        reasoning.append(ReasoningStep(
            step_number=2,
            description="TACTICAL: Determined next actions",
            output_data={"actions_count": len(tactical_steps)}
        ))

        # OPERATIONAL LEVEL: Execute first action
        if tactical_steps:
            thought_sig.current_thinking_level = ThinkingLevel.OPERATIONAL
            action = tactical_steps[0]

            execution_result = await self._think_operational(action, thought_sig)

            reasoning.append(ReasoningStep(
                step_number=3,
                description="OPERATIONAL: Executed action",
                output_data=execution_result
            ))

            # Check if action was a successful delegation
            if "delegation" in execution_result:
                delegation = execution_result["delegation"]
                if delegation.get("success"):
                    # Mark current sub-goal as done
                    current_goal = thought_sig.sub_goals[thought_sig.current_sub_goal_idx]
                    current_goal.status = SubGoalStatus.DONE

                    # Store artifact references
                    if delegation.get("artifact_refs"):
                        current_goal.verification_artifact_refs.extend(delegation["artifact_refs"])

                    thought_sig.current_sub_goal_idx += 1

        # Update progress
        thought_sig.progress_percentage = self._calculate_progress(thought_sig.sub_goals)

        # Check if all goals completed
        if thought_sig.current_sub_goal_idx >= len(thought_sig.sub_goals):
            thought_sig.current_state = TaskState.COMPLETED
        else:
            thought_sig.current_state = TaskState.IN_PROGRESS

        reasoning.append(ReasoningStep(
            step_number=4,
            description="State updated",
            output_data={
                "progress": thought_sig.progress_percentage,
                "state": thought_sig.current_state
            }
        ))

        # Save updated state
        await self.store.save(thought_sig)

        return {
            "task_id": task_id,
            "tick_completed": True,
            "current_state": thought_sig.current_state,
            "progress": thought_sig.progress_percentage,
            "reasoning": reasoning
        }

    async def _autonomous_execution_loop(self, task_id: str):
        """
        Continuous autonomous execution
        Runs until task completes or fails
        """
        self._running_loops[task_id] = True

        logger.info(f"Starting autonomous execution loop for {task_id}")

        try:
            while self._running_loops.get(task_id, False):
                result = await self.autonomous_tick(task_id)

                if result.get("status") == "finished":
                    logger.info(f"Task {task_id} finished: {result.get('final_state')}")
                    break

                # Wait between ticks (configurable)
                await asyncio.sleep(5)

        except Exception as e:
            logger.exception(f"Execution loop failed for {task_id}")
        finally:
            self._running_loops[task_id] = False

    @metric_counter("marathon")
    async def check_task_progress(self, task_id: str) -> Dict[str, Any]:
        """Check current state with thinking records"""
        reasoning: List[ReasoningStep] = []

        thought_sig = await self.store.load(task_id)
        if not thought_sig:
            return {"error": f"Task {task_id} not found"}

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Retrieved task state",
            input_data={"task_id": task_id}
        ))

        # Generate summary using LLM
        summary_prompt = f"""
Task progress summary:

STATE: {thought_sig.current_state}
PROGRESS: {thought_sig.progress_percentage}%
COMPLETED: {sum(1 for sg in thought_sig.sub_goals if sg.status == SubGoalStatus.DONE)}/{len(thought_sig.sub_goals)} sub-goals
DELEGATIONS: {thought_sig.successful_delegations}/{thought_sig.total_delegations} successful

Recent thinking records:
{[asdict(tr) for tr in thought_sig.thinking_records[-3:]]}

Provide concise status update (2-3 sentences).
"""

        summary = await self.llm.chat(summary_prompt)

        return {
            "task_id": task_id,
            "progress_percentage": thought_sig.progress_percentage,
            "current_state": thought_sig.current_state,
            "current_thinking_level": thought_sig.current_thinking_level.value,  # .value — чтобы была строка, а не enum
            "summary": summary,
            "sub_goals_status": [
                {
                    "name": sg.name,
                    "status": sg.status.value,  # тоже лучше .value
                    "artifact_refs": sg.verification_artifact_refs
                }
                for sg in thought_sig.sub_goals
            ],
            "delegation_stats": {
                "total": thought_sig.total_delegations,
                "successful": thought_sig.successful_delegations,
                "failed": thought_sig.failed_delegations
            },
            "reasoning": reasoning
        }
