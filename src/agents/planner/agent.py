import json
import logging
import re
import statistics
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from shared.jira import JiraClient
from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("planner_agent")


@dataclass
class HistoricalTask:
    """Historical task data for predictive estimation"""
    task_type: str
    estimated_hours: float
    actual_hours: float
    complexity: str
    blockers_encountered: List[str]
    team_size: int
    success: bool


@dataclass
class PredictiveEstimate:
    """ML-enhanced estimation based on historical data"""
    base_estimate_hours: float
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: float  # 0.0 to 1.0
    similar_tasks_analyzed: int
    accuracy_factors: Dict[str, float]


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
    """
    Strategic Planning Agent with Advanced Capabilities

    Key Features:
    1. Multi-Step Reasoning: Chain-of-thought planning with explicit decision trees
    2. Predictive Estimation: Data-driven estimation based on historical patterns
    3. Risk-Aware Planning: Proactive risk identification and mitigation strategies

    Proper sequential reasoning, fallback transparency, Jira composition
    """

    def __init__(self):
        super().__init__("Planner")
        self.llm = LLMClient()
        self.jira = JiraClient()

        # Historical data store (in production: use database)
        self.historical_tasks: List[HistoricalTask] = self._load_historical_data()

        # Register tools
        self.register_tool("plan", self.plan)
        self.register_tool("plan_with_reasoning", self.plan_with_reasoning)
        self.register_tool("predictive_estimate", self.predictive_estimate)
        self.register_tool("risk_aware_planning", self.risk_aware_planning)
        self.register_tool("plan_with_jira", self.plan_with_jira)

        logger.info("Planner Agent initialized with multi-step reasoning and predictive capabilities")

    def _load_historical_data(self) -> List[HistoricalTask]:
        """Load historical task data for predictive modeling"""
        return [
            HistoricalTask("api_development", 8, 12, "medium", ["dependency_delay"], 2, True),
            HistoricalTask("api_development", 16, 18, "high", ["scope_creep"], 3, True),
            HistoricalTask("database_migration", 12, 20, "high", ["data_quality"], 2, True),
            HistoricalTask("ui_component", 4, 5, "low", [], 1, True),
            HistoricalTask("integration", 10, 15, "high", ["api_changes"], 3, False),
        ]

    def _next_step(self, reasoning: List[ReasoningStep], description: str,
                   input_data: Optional[Dict] = None, output_data: Optional[Dict] = None):
        """
        Helper to add sequential reasoning steps

        Always uses len(reasoning) + 1 to ensure monotonic step numbers
        """
        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description=description,
            input_data=input_data or {},
            output_data=output_data or {}
        ))

    # ========================================================================
    # MAIN PLANNING TOOLS
    # ========================================================================

    @log_method
    @metric_counter("planner")
    async def plan(self, description: str):
        """
        Standard planning - delegates to plan_with_reasoning for enhanced capabilities
        """
        return await self.plan_with_reasoning(description)

    @log_method
    @metric_counter("planner")
    async def plan_with_reasoning(self, description: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhanced planning with explicit multi-step reasoning chain

        This is a SELF-CONTAINED reasoning process with proper sequential steps.
        All fallback scenarios are explicitly tracked in reasoning.
        """
        reasoning: List[ReasoningStep] = []
        fallback_used = False

        # Step 1: Initiating planning
        self._next_step(reasoning, "Initiating multi-step reasoning planning process",
                        input_data={"description": description, "context": context})

        # STEP 2: Task Classification
        classification_prompt = f"""
You are a senior technical planner analyzing a new task.

TASK: {description}
CONTEXT: {context or "General DevOps project"}

Analyze and determine:
1. Primary task type (api_development, database_migration, ui_component, integration, other)
2. Complexity (low/medium/high)
3. Technical uncertainty (low/medium/high)

Return ONLY valid JSON (no markdown, no backticks):
{{
  "task_type": "...",
  "complexity": "low|medium|high",
  "technical_uncertainty": "low|medium|high",
  "reasoning": "brief explanation"
}}
"""

        classification = None
        classification_fallback = False

        try:
            classification_response = await self.llm.chat(classification_prompt)
            classification = self._safe_parse_json(classification_response, None)

            # Check if fallback was used in JSON parsing
            if classification is None or classification.get("task_type") == "other":
                classification_fallback = True
                classification = {
                    "task_type": "other",
                    "complexity": "medium",
                    "technical_uncertainty": "medium",
                    "reasoning": "fallback classification due to LLM response parsing failure"
                }

            self._next_step(reasoning, "Classified task type and complexity",
                            output_data={
                                "task_type": classification.get("task_type"),
                                "complexity": classification.get("complexity"),
                                "fallback_used": classification_fallback
                            })

        except Exception as e:
            logger.error("Classification failed", extra={"error": str(e)})
            classification_fallback = True
            classification = {
                "task_type": "other",
                "complexity": "medium",
                "technical_uncertainty": "medium",
                "reasoning": f"fallback classification due to exception: {str(e)}"
            }

            # Explicit fallback step
            self._next_step(reasoning, "Classification failed - using fallback classification",
                            output_data={
                                "error": str(e),
                                "fallback_classification": classification
                            })

        # STEP 3: Decomposition
        decomposition_prompt = f"""
Break down this task into 4-6 actionable subtasks:

TASK: {description}
CLASSIFICATION: Type={classification.get('task_type')}, Complexity={classification.get('complexity')}

Return a numbered list of specific, actionable subtasks.
"""

        subtasks = []
        decomposition_fallback = False
        decomposition = ""

        try:
            decomposition = await self.llm.chat(decomposition_prompt)
            subtasks = self._extract_subtasks(decomposition)

            # Validate subtasks extraction
            if not subtasks or len(subtasks) < 3:
                decomposition_fallback = True
                subtasks = [
                    "Research requirements and constraints",
                    "Design solution architecture",
                    "Implement core functionality",
                    "Add testing and validation",
                    "Document solution and deployment"
                ]

                # Explicit fallback step
                self._next_step(reasoning, "Decomposition parsing failed - using baseline subtasks",
                                output_data={
                                    "fallback_used": True,
                                    "baseline_subtasks_count": len(subtasks)
                                })
            else:
                self._next_step(reasoning, "Generated task decomposition from LLM",
                                output_data={
                                    "subtasks_count": len(subtasks),
                                    "fallback_used": False
                                })

        except Exception as e:
            logger.error("Decomposition failed", extra={"error": str(e)})
            decomposition_fallback = True
            subtasks = [
                "Research requirements and constraints",
                "Design solution architecture",
                "Implement core functionality",
                "Add testing and validation",
                "Document solution and deployment"
            ]

            # Explicit exception fallback step
            self._next_step(reasoning, "Decomposition failed with exception - using baseline subtasks",
                            output_data={
                                "error": str(e),
                                "fallback_used": True,
                                "baseline_subtasks_count": len(subtasks)
                            })

        # STEP 4: Predictive Estimation
        predictive_estimate = None

        try:
            predictive_estimate = await self._generate_predictive_estimate(
                task_type=classification.get("task_type", "other"),
                complexity=classification.get("complexity", "medium"),
                subtasks_count=len(subtasks)
            )

            self._next_step(reasoning, "Generated predictive time estimate with confidence intervals",
                            output_data={
                                "base_estimate_hours": predictive_estimate.base_estimate_hours,
                                "confidence_level": predictive_estimate.confidence_level,
                                "similar_tasks_analyzed": predictive_estimate.similar_tasks_analyzed
                            })

        except Exception as e:
            logger.error("Predictive estimation failed", extra={"error": str(e)})

            # Fallback estimation
            predictive_estimate = PredictiveEstimate(
                base_estimate_hours=len(subtasks) * 8.0,
                confidence_interval_low=len(subtasks) * 6.0,
                confidence_interval_high=len(subtasks) * 12.0,
                confidence_level=0.3,
                similar_tasks_analyzed=0,
                accuracy_factors={"fallback": 1.0}
            )

            # Explicit fallback step
            self._next_step(reasoning, "Predictive estimation failed - using rule-based estimate",
                            output_data={
                                "error": str(e),
                                "fallback_estimate_hours": predictive_estimate.base_estimate_hours
                            })

        # Step 5 - Planning completed (ALWAYS present)
        overall_fallback = classification_fallback or decomposition_fallback

        self._next_step(reasoning, "Planning process completed successfully",
                        output_data={
                            "total_subtasks": len(subtasks),
                            "estimated_days": int(predictive_estimate.base_estimate_hours / 8) + 1,
                            "overall_fallback_used": overall_fallback,
                            "confidence_level": predictive_estimate.confidence_level
                        })

        logger.info("Planning completed",
                    extra={
                        "task": description,
                        "subtasks": len(subtasks),
                        "estimated_hours": predictive_estimate.base_estimate_hours,
                        "fallback_used": overall_fallback
                    })

        return {
            "task": description,
            "classification": classification,
            "decomposition": decomposition,
            "subtasks": subtasks,
            "complexity": classification.get("complexity", "medium"),
            "estimated_days": int(predictive_estimate.base_estimate_hours / 8) + 1,
            "predictive_estimate": asdict(predictive_estimate),
            "fallback_used": overall_fallback,
            "reasoning": reasoning,
            "confidence_level": predictive_estimate.confidence_level
        }

    @log_method
    @metric_counter("planner")
    async def predictive_estimate(
            self,
            task_description: str,
            task_type: Optional[str] = None,
            complexity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate predictive time estimate based on historical data

        Proper sequential reasoning with fallback transparency
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Starting predictive estimation analysis",
                        input_data={"task": task_description, "type": task_type})

        # If type/complexity not provided, classify first
        if not task_type or not complexity:
            classification_prompt = f"""
Classify this task:
{task_description}

Return ONLY valid JSON (no markdown):
{{
  "task_type": "api_development|database_migration|ui_component|integration|other",
  "complexity": "low|medium|high"
}}
"""
            try:
                classification_response = await self.llm.chat(classification_prompt)
                classification = self._safe_parse_json(classification_response, {
                    "task_type": "other",
                    "complexity": "medium"
                })
                task_type = classification.get("task_type", "other")
                complexity = classification.get("complexity", "medium")

                self._next_step(reasoning, "Classified task for estimation",
                                output_data={"task_type": task_type, "complexity": complexity})

            except Exception as e:
                task_type = "other"
                complexity = "medium"

                # Explicit fallback
                self._next_step(reasoning, "Classification failed - using fallback",
                                output_data={"error": str(e), "fallback_type": task_type})

        # Find similar historical tasks
        similar_tasks = self._find_similar_tasks(task_type, complexity)

        self._next_step(reasoning, f"Found {len(similar_tasks)} similar historical tasks",
                        output_data={"similar_tasks": len(similar_tasks)})

        # Generate predictive estimate
        estimate = await self._generate_predictive_estimate(task_type, complexity, 5)

        self._next_step(reasoning, "Generated predictive estimate with statistical analysis",
                        output_data={
                            "base_estimate": estimate.base_estimate_hours,
                            "confidence": estimate.confidence_level
                        })

        # Completion step
        self._next_step(reasoning, "Predictive estimation completed",
                        output_data={"total_steps": len(reasoning)})

        return {
            "task_description": task_description,
            "task_type": task_type,
            "complexity": complexity,
            "estimate": asdict(estimate),
            "recommendation": self._generate_estimate_recommendation(estimate),
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("planner")
    async def risk_aware_planning(
            self,
            task_description: str,
            deadline_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate plan with explicit risk assessment and mitigation

        Proper sequential reasoning throughout the risk analysis process
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Starting risk-aware planning",
                        input_data={"task": task_description, "deadline_days": deadline_days})

        # STEP 2: Identify planning-level risks
        risk_identification_prompt = f"""
You are a project planning expert analyzing risks.

TASK: {task_description}
DEADLINE: {deadline_days} days (if applicable)

Identify PLANNING-LEVEL risks (not detailed security vulnerabilities):

1. TIMELINE RISKS
2. RESOURCE RISKS
3. DEPENDENCY RISKS
4. TECHNICAL FEASIBILITY RISKS

For EACH risk, provide:
- Risk description
- Probability (low/medium/high)
- Impact (low/medium/high)
- Mitigation strategy
"""

        try:
            risk_identification = await self.llm.chat(risk_identification_prompt)

            self._next_step(reasoning, "Identified planning-level risk landscape",
                            output_data={"risks_identified": True})

        except Exception as e:
            risk_identification = "Risk identification failed - manual review recommended"

            # Explicit fallback
            self._next_step(reasoning, "Risk identification failed",
                            output_data={"error": str(e)})

        # STEP 3: Risk Prioritization
        risk_prioritization_prompt = f"""
Given these identified risks:
{risk_identification}

Prioritize them using a risk matrix (Probability × Impact).

For TOP 5 risks, create mitigation plans.
"""

        try:
            risk_prioritization = await self.llm.chat(risk_prioritization_prompt)

            self._next_step(reasoning, "Prioritized risks and created mitigation plans",
                            output_data={"top_risks_prioritized": True})

        except Exception as e:
            risk_prioritization = "Risk prioritization failed - manual review recommended"

            # Explicit fallback
            self._next_step(reasoning, "Risk prioritization failed",
                            output_data={"error": str(e)})

        # STEP 4: Timeline Adjustment
        timeline_adjustment_prompt = f"""
Based on the risk analysis:
{risk_prioritization}

Calculate risk-adjusted timeline:
1. Base estimate
2. Risk buffer for top 3 risks
3. Total realistic estimate
"""

        try:
            timeline_adjustment = await self.llm.chat(timeline_adjustment_prompt)

            self._next_step(reasoning, "Generated risk-adjusted timeline with buffers",
                            output_data={"timeline_adjusted": True})

        except Exception as e:
            timeline_adjustment = "Timeline adjustment failed - use conservative estimates"

            # Explicit fallback
            self._next_step(reasoning, "Timeline adjustment failed",
                            output_data={"error": str(e)})

        # Completion step
        self._next_step(reasoning, "Risk-aware planning completed",
                        output_data={"total_steps": len(reasoning)})

        return {
            "task_description": task_description,
            "risk_identification": risk_identification,
            "risk_prioritization": risk_prioritization,
            "timeline_adjustment": timeline_adjustment,
            "reasoning": reasoning,
            "recommendation": "Review risk mitigation plans before starting execution"
        }

    @log_method
    @metric_counter("planner")
    async def plan_with_jira(
            self,
            description: str,
            project_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Enhanced planning + Jira integration

        PROPER reasoning composition:
        - Planning phase results embedded as reference (NOT extended)
        - Jira phase has its own sequential reasoning
        """
        reasoning: List[ReasoningStep] = []

        # Step 1: Jira integration initiated
        self._next_step(reasoning, "Starting enhanced planning with Jira integration",
                        input_data={"description": description, "project_key": project_key})

        # PHASE 1: Planning (call plan_with_reasoning)
        plan_result = await self.plan_with_reasoning(description)

        # Step 2 - Planning phase completed (embedded reference, NOT extension)
        self._next_step(reasoning, "Strategic planning phase completed",
                        output_data={
                            "subtasks_count": len(plan_result.get("subtasks", [])),
                            "confidence_level": plan_result.get("confidence_level", 0.5),
                            "planning_steps_executed": len(plan_result.get("reasoning", [])),
                            "fallback_used": plan_result.get("fallback_used", False)
                        })

        # Extract data for Jira
        classification = plan_result.get("classification", {})
        estimate = plan_result.get("predictive_estimate", {})
        subtasks = plan_result.get("subtasks", [])

        # Early exit if planning failed
        if not subtasks:
            self._next_step(reasoning, "Planning failed - Jira integration skipped",
                            output_data={"reason": "no_subtasks"})

            return {
                **plan_result,
                "jira_issues": [],
                "jira_mode": self.jira.mode,
                "planning_reasoning": plan_result.get("reasoning", []),
                "reasoning": reasoning
            }

        # PHASE 2: Jira Integration
        jira_issues = []

        try:
            # Step 3: Create Epic
            epic_description = f"""
**Task Classification:**
- Type: {classification.get('task_type', 'N/A')}
- Complexity: {classification.get('complexity', 'N/A')}

**Predictive Estimate:**
- Base: {estimate.get('base_estimate_hours', 0):.1f}h
- Confidence: {estimate.get('confidence_level', 0) * 100:.0f}%

---
*Generated by Multi-Agent DevOps Assistant*
"""

            epic_result = await self.jira.create_task(
                summary=f"[EPIC] {description}",
                description=epic_description[:2000]
            )
            jira_issues.append(epic_result)

            self._next_step(reasoning, "Created Epic in Jira with planning metadata",
                            output_data={
                                "epic_key": epic_result.get("issue_key", "unknown"),
                                "jira_mode": epic_result.get("mode", "unknown")
                            })

            # Step 4: Create subtasks
            for idx, subtask in enumerate(subtasks[:10], 1):
                issue_result = await self.jira.create_task(
                    summary=f"[Subtask {idx}] {subtask[:80]}",
                    description=f"Part of: {description}"
                )
                jira_issues.append(issue_result)

            self._next_step(reasoning, "Successfully created all Jira subtasks",
                            output_data={
                                "total_issues_created": len(jira_issues),
                                "subtasks_created": len(subtasks)
                            })

        except Exception as e:
            logger.error("Jira integration failed", extra={"error": str(e)})

            # Explicit error step
            self._next_step(reasoning, "Jira task creation failed",
                            output_data={"error": str(e)})

        # Step 5 - Jira integration completed
        self._next_step(reasoning, "Jira integration completed",
                        output_data={
                            "total_jira_issues": len(jira_issues),
                            "success": len(jira_issues) > 0
                        })

        logger.info("Plan with Jira completed",
                    extra={"task": description, "issues": len(jira_issues)})

        return {
            "task": description,
            "plan": plan_result,  # Complete planning result
            "jira_issues": jira_issues,
            "jira_mode": self.jira.mode,
            "planning_reasoning": plan_result.get("reasoning", []),  # Separate for reference
            "reasoning": reasoning  # Main Jira integration reasoning
        }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _find_similar_tasks(self, task_type: str, complexity: str) -> List[HistoricalTask]:
        """Find historical tasks similar to current task"""
        similar = [
            task for task in self.historical_tasks
            if task.task_type == task_type and task.complexity == complexity
        ]
        if not similar:
            similar = [task for task in self.historical_tasks if task.task_type == task_type]
        return similar

    async def _generate_predictive_estimate(
            self, task_type: str, complexity: str, subtasks_count: int
    ) -> PredictiveEstimate:
        """Generate ML-like predictive estimate"""
        similar_tasks = self._find_similar_tasks(task_type, complexity)

        if not similar_tasks:
            complexity_multiplier = {"low": 1.0, "medium": 2.0, "high": 3.5}
            base_hours = subtasks_count * 4 * complexity_multiplier.get(complexity, 2.0)
            return PredictiveEstimate(
                base_estimate_hours=base_hours,
                confidence_interval_low=base_hours * 0.7,
                confidence_interval_high=base_hours * 1.5,
                confidence_level=0.5,
                similar_tasks_analyzed=0,
                accuracy_factors={"rule_based": 1.0}
            )

        actual_hours = [task.actual_hours for task in similar_tasks]
        avg_hours = statistics.mean(actual_hours)
        std_dev = statistics.stdev(actual_hours) if len(actual_hours) > 1 else avg_hours * 0.3

        avg_subtasks = 5
        scale_factor = subtasks_count / avg_subtasks
        adjusted_hours = avg_hours * scale_factor

        confidence = min(0.95, len(similar_tasks) / 10 * (1 - min(std_dev / avg_hours, 0.5)))

        return PredictiveEstimate(
            base_estimate_hours=adjusted_hours,
            confidence_interval_low=max(adjusted_hours - std_dev, adjusted_hours * 0.5),
            confidence_interval_high=adjusted_hours + std_dev * 1.5,
            confidence_level=confidence,
            similar_tasks_analyzed=len(similar_tasks),
            accuracy_factors={
                "historical_data_quality": len(similar_tasks) / 10,
                "variance_penalty": 1 - min(std_dev / avg_hours, 0.5)
            }
        )

    def _generate_estimate_recommendation(self, estimate: PredictiveEstimate) -> str:
        """Generate recommendation based on estimate confidence"""
        if estimate.confidence_level > 0.8:
            return f"High confidence estimate. Plan for {estimate.base_estimate_hours:.1f}h with {estimate.confidence_interval_high - estimate.base_estimate_hours:.1f}h buffer."
        elif estimate.confidence_level > 0.5:
            return f"Moderate confidence. Recommend {estimate.confidence_interval_high:.1f}h to account for uncertainty."
        else:
            return f"Low confidence estimate. Wide range: {estimate.confidence_interval_low:.1f}h - {estimate.confidence_interval_high:.1f}h"

    def _extract_subtasks(self, decomposition: str) -> List[str]:
        """Extract subtasks from decomposition text"""
        patterns = [
            r'\d+\.\s+(.+?)(?=\n\d+\.|\n\n|$)',
            r'[-•]\s+(.+?)(?=\n[-•]|\n\n|$)',
        ]

        subtasks = []
        for pattern in patterns:
            matches = re.findall(pattern, decomposition, re.MULTILINE | re.DOTALL)
            if matches:
                subtasks.extend([
                    match.strip().split('\n')[0].strip()
                    for match in matches if match.strip()
                ])
                break

        if not subtasks:
            lines = decomposition.split('\n')
            subtasks = [
                line.strip().lstrip('0123456789.-•* ')
                for line in lines
                if line.strip() and 20 < len(line.strip()) < 200
            ]

        return subtasks[:10]

    def _safe_parse_json(self, response: str, default: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Safely extract and parse JSON from LLM response

        Returns None if parsing fails and no default provided (for explicit fallback detection)
        """
        # 1. Try direct json.loads
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # 2. Extract from ```json block
        json_block = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        # 3. Find first { and last } (balanced braces)
        start = response.find('{')
        if start == -1:
            logger.warning("No JSON found in LLM response")
            return default

        brace_count = 0
        end = start
        for i in range(start, len(response)):
            if response[i] == '{':
                brace_count += 1
            elif response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i + 1
                    break

        if brace_count == 0:
            try:
                return json.loads(response[start:end])
            except json.JSONDecodeError:
                pass

        logger.warning("Failed to parse JSON from LLM response")
        return default
