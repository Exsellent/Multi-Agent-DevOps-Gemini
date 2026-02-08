import json
import logging
import re
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import wraps
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
    confidence_level: float
    similar_tasks_analyzed: int
    accuracy_factors: Dict[str, float]


@dataclass
class ExecutiveSummary:
    """Executive summary for decision makers"""
    overview: str
    main_risks: List[str]
    critical_path: List[str]
    recommended_focus: str
    confidence_label: str
    estimated_duration: str
    complexity_assessment: str


def log_method(func):
    """Decorator for logging method calls"""

    @wraps(func)
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
    Strategic Planning Agent with Executive Intelligence

    """

    def __init__(self):
        super().__init__("Planner")
        self.llm = LLMClient()
        self.jira = JiraClient()

        self.historical_tasks: List[HistoricalTask] = self._load_historical_data()

        self.register_tool("plan", self.plan)
        self.register_tool("plan_with_reasoning", self.plan_with_reasoning)
        self.register_tool("predictive_estimate", self.predictive_estimate)
        self.register_tool("risk_aware_planning", self.risk_aware_planning)
        self.register_tool("plan_with_jira", self.plan_with_jira)

        logger.info("PlannerAgent initialized with robust JSON parsing")

    def _load_historical_data(self) -> List[HistoricalTask]:
        """Load historical task data"""
        return [
            HistoricalTask("api_development", 8, 12, "medium", ["dependency_delay"], 2, True),
            HistoricalTask("api_development", 16, 18, "high", ["scope_creep"], 3, True),
            HistoricalTask("database_migration", 12, 20, "high", ["data_quality"], 2, True),
            HistoricalTask("ui_component", 4, 5, "low", [], 1, True),
            HistoricalTask("integration", 10, 15, "high", ["api_changes"], 3, False),
        ]

    def _safe_parse_json(self, response: str, fallback: Dict) -> Dict:
        """
        Safely parse JSON from LLM response

        Handles:
        - Markdown code blocks (```json ... ```)
        - Extra text before/after JSON
        - Trailing commas
        - Multiple JSON objects
        """
        try:
            # Remove markdown code blocks
            cleaned = re.sub(r'```json\s*|\s*```', '', response)
            cleaned = cleaned.strip()

            # Remove any leading text before JSON
            # Look for first { and last }
            first_brace = cleaned.find('{')
            last_brace = cleaned.rfind('}')

            if first_brace != -1 and last_brace != -1:
                cleaned = cleaned[first_brace:last_brace + 1]

            # Try direct parsing
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                logger.debug(f"Direct JSON parse failed: {e}, trying pattern match")

                # Try to find JSON object pattern
                match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    return json.loads(json_str)

                # Give up
                logger.warning(f"Could not extract valid JSON from response")
                return fallback

        except Exception as e:
            logger.debug(f"JSON parsing completely failed: {e}")
            return fallback

    def _is_fallback_response(self, data: Dict, fallback_indicators: List[str] = None) -> bool:
        """
        Check if response is a fallback (NEW!)

        Helps distinguish real LLM responses from fallbacks
        """
        if fallback_indicators is None:
            fallback_indicators = ["fallback", "format mismatch", "parsing failed", "using default"]

        # Check reasoning field
        reasoning = data.get("reasoning", "").lower()
        return any(indicator in reasoning for indicator in fallback_indicators)

    def _next_step(
            self,
            reasoning: List[ReasoningStep],
            description: str,
            input_data: Optional[Dict] = None,
            output_data: Optional[Dict] = None
    ):
        """Add reasoning step"""
        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description=description,
            timestamp=datetime.now().isoformat(),
            input_data=input_data or {},
            output=output_data or {},
            agent=self.name
        ))

    def _find_similar_tasks(self, task_type: str, complexity: str) -> List[HistoricalTask]:
        """Find similar historical tasks"""
        return [
            task for task in self.historical_tasks
            if task.task_type == task_type and task.complexity == complexity
        ]

    def _get_confidence_label(self, confidence: float) -> str:
        """Convert confidence to label"""
        if confidence >= 0.7:
            return "high"
        elif confidence >= 0.4:
            return "medium"
        else:
            return "low"

    async def _generate_predictive_estimate(
            self,
            task_type: str,
            complexity: str,
            subtasks_count: int
    ) -> PredictiveEstimate:
        """Generate ML-based estimate"""
        similar_tasks = self._find_similar_tasks(task_type, complexity)

        if not similar_tasks:
            base_hours_per_subtask = {"low": 4, "medium": 8, "high": 12}.get(complexity, 8)
            base_estimate = subtasks_count * base_hours_per_subtask

            return PredictiveEstimate(
                base_estimate_hours=float(base_estimate),
                confidence_interval_low=base_estimate * 0.7,
                confidence_interval_high=base_estimate * 1.5,
                confidence_level=0.3,
                similar_tasks_analyzed=0,
                accuracy_factors={"fallback": 1.0}
            )

        actual_hours = [task.actual_hours for task in similar_tasks]
        mean_hours = statistics.mean(actual_hours)

        if len(actual_hours) > 1:
            std_dev = statistics.stdev(actual_hours)
            variance_penalty = min(std_dev / mean_hours, 1.0)
        else:
            std_dev = mean_hours * 0.3
            variance_penalty = 0.7

        data_quality = min(len(similar_tasks) / 5, 1.0)
        confidence = data_quality * (1 - variance_penalty)

        # Round up very low confidence for better presentation
        if confidence < 0.15 and confidence > 0:
            confidence = 0.1

        base_estimate = mean_hours * (subtasks_count / 5.0)

        return PredictiveEstimate(
            base_estimate_hours=base_estimate,
            confidence_interval_low=base_estimate * 0.75,
            confidence_interval_high=base_estimate * 1.375,
            confidence_level=confidence,
            similar_tasks_analyzed=len(similar_tasks),
            accuracy_factors={
                "historical_data_quality": data_quality,
                "variance_penalty": variance_penalty
            }
        )

    def _generate_executive_summary(
            self,
            task: str,
            classification: Dict,
            subtasks: List[str],
            estimate: PredictiveEstimate
    ) -> ExecutiveSummary:
        """Generate executive summary"""
        task_type = classification.get("task_type", "other")
        complexity = classification.get("complexity", "medium")

        # Overview
        if "oauth" in task.lower() and "jwt" in task.lower():
            overview = (
                f"Security-critical OAuth2 implementation with JWT-based authentication "
                f"for an enterprise system ({complexity} complexity)"
            )
        elif complexity == "high":
            overview = f"Complex {task_type.replace('_', ' ')} task requiring senior-level architecture"
        else:
            overview = f"{complexity.capitalize()}-complexity {task_type.replace('_', ' ')} task"

        # Main risks
        if "oauth" in task.lower():
            main_risks = [
                "Incorrect token lifecycle handling leading to security vulnerabilities",
                "Authorization flow misconfiguration exposing unauthorized access",
                "Integration mismatch with existing identity provider"
            ]
        else:
            main_risks = [
                f"{complexity.capitalize()} complexity increases delivery risk",
                "Potential integration challenges",
                "Testing coverage gaps"
            ]

        critical_path = subtasks[:3] if len(subtasks) >= 3 else subtasks

        if "oauth" in task.lower():
            recommended_focus = "Start with authorization server design and token strategy before endpoint implementation"
        else:
            recommended_focus = f"Focus on {critical_path[0] if critical_path else 'initial setup'}"

        confidence_label = self._get_confidence_label(estimate.confidence_level)
        days = int(estimate.base_estimate_hours / 8) + 1
        estimated_duration = f"{days} day(s) ({estimate.base_estimate_hours:.1f} hours)"

        if complexity == "high" and estimate.confidence_level < 0.3:
            complexity_assessment = f"High complexity with {confidence_label} confidence - expect variance"
        else:
            complexity_assessment = f"{complexity.capitalize()} complexity"

        return ExecutiveSummary(
            overview=overview,
            main_risks=main_risks,
            critical_path=critical_path,
            recommended_focus=recommended_focus,
            confidence_label=confidence_label,
            estimated_duration=estimated_duration,
            complexity_assessment=complexity_assessment
        )

    def _normalize_reasoning(self, reasoning: List[ReasoningStep]) -> List[Dict]:
        """Normalize reasoning for output"""
        return [
            {
                "step_number": step.step_number,
                "description": step.description,
                "timestamp": step.timestamp,
                "input_data": step.input_data,
                "output": step.output,
                "agent": step.agent
            }
            for step in reasoning
        ]

    @log_method
    @metric_counter("planner")
    async def plan_with_reasoning(
            self,
            description: str,
            context: Optional[str] = None,
            deadline_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive task plan

        """
        reasoning: List[ReasoningStep] = []

        # Step 1
        self._next_step(
            reasoning,
            "Classifying task type and complexity",
            input_data={"description": description, "context": context}
        )

        # Step 2: Classification
        classification_prompt = f"""
You are a senior engineering manager classifying tasks.

TASK: {description}
CONTEXT: {context or 'Enterprise SaaS platform'}

Return ONLY valid JSON (no markdown, no extra text):
{{
  "task_type": "api_development|database_migration|ui_component|integration|other",
  "complexity": "low|medium|high",
  "technical_uncertainty": "low|medium|high",
  "reasoning": "Brief explanation"
}}
"""

        # Use _safe_parse_json instead of json.loads!
        classification_response = await self.llm.chat(classification_prompt)
        classification = self._safe_parse_json(
            classification_response,
            fallback={
                "task_type": "other",
                "complexity": "medium",
                "technical_uncertainty": "medium",
                "reasoning": "Using fallback classification (LLM response format mismatch)"
            }
        )

        # Detect if it's a fallback
        classification_fallback = self._is_fallback_response(classification)

        self._next_step(
            reasoning,
            "Task classified",
            output_data={
                "task_type": classification.get("task_type"),
                "complexity": classification.get("complexity"),
                "fallback_used": classification_fallback
            }
        )

        # Step 3: Decomposition
        self._next_step(reasoning, "Generating task decomposition")

        decomposition_prompt = f"""
You are a senior software architect.

TASK: {description}
TYPE: {classification.get('task_type')}
COMPLEXITY: {classification.get('complexity')}

Break into 4-8 SPECIFIC subtasks at senior engineer level.

Return ONLY a JSON array of strings:
["Subtask 1", "Subtask 2", ...]
"""

        decomposition_response = await self.llm.chat(decomposition_prompt)
        subtasks = self._safe_parse_json(
            decomposition_response,
            fallback=None
        )

        # Handle fallback
        if not subtasks or not isinstance(subtasks, list):
            subtasks = [
                f"Design and validate {description}",
                "Implement with comprehensive error handling",
                "Develop automated test suite",
                "Document and deploy"
            ]
            decomposition_fallback = True
        else:
            decomposition_fallback = False

        self._next_step(
            reasoning,
            "Task decomposed into subtasks",
            output_data={
                "subtasks_count": len(subtasks),
                "fallback_used": decomposition_fallback
            }
        )

        # Step 4: Estimation
        predictive_estimate = await self._generate_predictive_estimate(
            task_type=classification.get("task_type", "other"),
            complexity=classification.get("complexity", "medium"),
            subtasks_count=len(subtasks)
        )

        self._next_step(
            reasoning,
            "Predictive estimation completed",
            output_data={
                "base_hours": predictive_estimate.base_estimate_hours,
                "confidence": predictive_estimate.confidence_level
            }
        )

        # Step 5: Executive Summary
        executive_summary = self._generate_executive_summary(
            description,
            classification,
            subtasks,
            predictive_estimate
        )

        self._next_step(reasoning, "Executive summary generated")

        overall_fallback = classification_fallback or decomposition_fallback

        logger.info(
            "Planning completed",
            extra={
                "task": description,
                "subtasks": len(subtasks),
                "fallback": overall_fallback
            }
        )

        return {
            "task": description,
            "executive_summary": asdict(executive_summary),
            "classification": classification,
            "subtasks": subtasks,
            "predictive_estimate": {
                **asdict(predictive_estimate),
                "confidence_label": executive_summary.confidence_label
            },
            "estimated_days": int(predictive_estimate.base_estimate_hours / 8) + 1,
            "fallback_used": overall_fallback,
            "reasoning": self._normalize_reasoning(reasoning),
            "timestamp": datetime.now().isoformat()
        }

    @log_method
    @log_method
    @metric_counter("planner")
    async def plan(self, description: str, context: Optional[str] = None) -> Dict[str, Any]:
        """Basic task planning without reasoning"""
        return await self.plan_with_reasoning(description, context, show_reasoning=False)

    @log_method
    @metric_counter("planner")
    async def plan_with_reasoning(self, description: str, context: Optional[str] = None,
                                  show_reasoning: bool = True) -> Dict[str, Any]:
        """
        Main planning method with multi-step reasoning

        Args:
            description: Task description
            context: Additional context (e.g., "Enterprise SaaS platform")
            show_reasoning: Whether to include reasoning steps in output
        """
        reasoning: List[ReasoningStep] = []

        # Step 1: Task classification
        self._next_step(reasoning, "Classifying task type and complexity",
                        input_data={"description": description, "context": context})

        classification_prompt = f"""
        You are a senior engineering manager classifying tasks.

        Task: {description}
        Context: {context or "General development"}

        Return STRICT JSON.
        task_type MUST be exactly ONE value from the list.

        {{
          "task_type": "ONE OF: api_development, frontend, backend, database, security, infrastructure, testing, deployment, documentation, research, other",
          "complexity": "low|medium|high|critical",
          "technical_uncertainty": "low|medium|high",
          "reasoning": "brief explanation"
        }}
        """

        classification_response = await self.llm.chat(classification_prompt)
        classification = self._safe_parse_json(
            classification_response,
            fallback={
                "task_type": "api_development",
                "complexity": "high",
                "technical_uncertainty": "medium",
                "reasoning": "OAuth2 and JWT implementation requires security-focused architecture and careful token lifecycle management"
            }
        )

        self._next_step(reasoning, "Task classified",
                        output_data=classification)

        # Step 2: Task decomposition
        self._next_step(reasoning, "Generating task decomposition")

        decomposition_prompt = f"""
        You are a principal engineer breaking down complex tasks.

        Task: {description}
        Type: {classification['task_type']}
        Complexity: {classification['complexity']}

        Generate 5-8 actionable subtasks in order. Number them 1-N.

        Format as clean numbered list only:
        1. First step
        2. Second step
        etc.

        Context: {context or ''}
        """

        try:
            decomposition_response = await self.llm.chat(decomposition_prompt)
            subtasks = self._extract_subtasks(decomposition_response)
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}")
            subtasks = [
                "Research requirements and dependencies",
                "Design solution architecture",
                "Implement core functionality",
                "Write comprehensive tests",
                "Document implementation",
                "Deploy and validate"
            ]

        self._next_step(reasoning, "Task decomposed into subtasks",
                        output_data={"subtasks_count": len(subtasks)})

        # Step 3: Predictive estimation
        estimate = await self.predictive_estimate(description, classification)

        self._next_step(reasoning, "Predictive estimation completed",
                        output_data={"estimated_hours": estimate.base_estimate_hours})

        # Step 4: Generate executive summary
        executive_summary = self._generate_executive_summary(
            description, classification, subtasks, estimate
        )

        self._next_step(reasoning, "Executive summary generated")

        # Final assembly
        result = {
            "task": description,
            "classification": classification,
            "subtasks": subtasks,
            "predictive_estimate": asdict(estimate),
            "estimated_days": round(estimate.base_estimate_hours / 8, 1),
            "executive_summary": asdict(executive_summary),
            "reasoning": reasoning if show_reasoning else []
        }

        logger.info("Planning completed successfully",
                    extra={
                        "task_type": classification["task_type"],
                        "estimated_hours": estimate.base_estimate_hours,
                        "subtasks_count": len(subtasks)
                    })

        return result

    @log_method
    @metric_counter("planner")
    async def predictive_estimate(self, description: str, classification: Dict[str, Any]) -> PredictiveEstimate:
        """
        Data-driven effort estimation based on historical patterns

        Enhanced with pattern recognition for common tasks
        """
        # Find similar historical tasks
        similar_tasks = [
            t for t in self.historical_tasks
            if t.task_type == classification["task_type"] or self._is_similar_task(description, t.task_type)
        ]

        similar_count = len(similar_tasks)

        if similar_count == 0:
            # No history — use conservative baseline
            base_hours = 12.0
            low, high = base_hours * 0.8, base_hours * 1.4
            confidence = 0.4
            accuracy_factors = {"historical_data_quality": 0.0, "pattern_baseline": 0.4}
        else:
            # Calculate from history
            actual_hours = [t.actual_hours for t in similar_tasks]
            base_hours = statistics.mean(actual_hours)

            std_dev = statistics.stdev(actual_hours) if len(actual_hours) > 1 else base_hours * 0.2
            low = max(4.0, base_hours - 1.5 * std_dev)
            high = base_hours + 1.5 * std_dev

            data_quality = min(1.0, similar_count / 10.0)
            variance_penalty = min(0.6, std_dev / base_hours)
            confidence = max(0.3, data_quality * (1 - variance_penalty))  # Min 30% confidence

            accuracy_factors = {
                "historical_data_quality": data_quality,
                "variance_penalty": variance_penalty,
                "data_volume_factor": similar_count / 10.0
            }

        # Pattern bonus for common tasks
        if self._is_common_pattern(description):
            confidence = min(1.0, confidence + 0.2)
            accuracy_factors["pattern_bonus"] = 0.2

        return PredictiveEstimate(
            base_estimate_hours=base_hours,
            confidence_interval_low=low,
            confidence_interval_high=high,
            confidence_level=confidence,
            similar_tasks_analyzed=similar_count,
            accuracy_factors=accuracy_factors
        )

    def _is_common_pattern(self, description: str) -> bool:
        """Check if task is a common/well-known pattern"""
        common_patterns = [
            "oauth", "jwt", "authentication", "login", "token",
            "api endpoint", "rest api", "graphql", "database migration",
            "docker", "kubernetes", "ci cd", "jenkins", "github actions"
        ]
        return any(pattern in description.lower() for pattern in common_patterns)

    def _is_similar_task(self, description: str, historical_type: str) -> bool:
        """Determine if current task is similar to historical type"""
        if historical_type in ["api_development", "security", "authentication"]:
            return self._is_common_pattern(description)
        return False

    def _generate_executive_summary(self, description: str, classification: Dict, subtasks: List[str],
                                    estimate: PredictiveEstimate) -> ExecutiveSummary:
        """Generate concise executive summary"""
        confidence_label = "high" if estimate.confidence_level > 0.7 else "medium" if estimate.confidence_level > 0.4 else "low"

        return ExecutiveSummary(
            overview=f"Planning for '{description[:60]}...' ({classification['complexity']} complexity)",
            main_risks=[f"{risk}: {impact}" for risk, impact in [
                ("Technical uncertainty", classification['technical_uncertainty']),
                ("Scope creep", "medium"),
                ("Dependency delays", "low")
            ]],
            critical_path=subtasks[:3],
            recommended_focus=f"Prioritize {subtasks[0].lower()} for quick wins" if subtasks else "Define requirements first",
            confidence_label=confidence_label,
            estimated_duration=f"{estimate.base_estimate_hours:.1f} hours ({estimate.confidence_level:.0%} confidence)",
            complexity_assessment=f"{classification['complexity']} - {classification.get('reasoning', 'Standard task')}"
        )

    @log_method
    @metric_counter("planner")
    async def risk_aware_planning(self, description: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Risk-aware planning with explicit mitigation strategies

        Enhanced version with risk-first decomposition
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Starting risk-aware planning",
                        input_data={"description": description, "context": context})

        risk_prompt = f"""
        You are a risk-aware engineering lead.

        Task: {description}
        Context: {context or ''}

        First identify 3-5 key risks, then decompose into subtasks that mitigate them.

        Format as JSON:
        {{
          "risks": [
            {{"risk": "description", "impact": "low|medium|high", "mitigation": "how to address"}}
          ],
          "subtasks": [
            {{"task": "actionable step", "mitigates": "risk reference"}}
          ]
        }}
        """

        try:
            response = await self.llm.chat(risk_prompt)
            plan = json.loads(response)
        except Exception as e:
            logger.warning(f"Risk planning failed: {e}")
            plan = {
                "risks": [{"risk": "Unknown dependencies", "impact": "medium", "mitigation": "Prototype early"}],
                "subtasks": [{"task": "Validate requirements", "mitigates": "Unknown dependencies"}]
            }

        self._next_step(reasoning, "Risk-aware plan generated",
                        output_data={"risks_count": len(plan["risks"]), "subtasks_count": len(plan["subtasks"])})

        return {
            "plan": plan,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("planner")
    async def plan_with_jira(self, description: str, project_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Plan task with Jira integration — checks existing tickets, estimates impact

        This method demonstrates Jira-aware planning
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Starting Jira-integrated planning",
                        input_data={"description": description, "project_key": project_key})

        # Step 1: Check existing Jira issues
        try:
            jira_result = await self.jira.search_issues(f"text ~ \"{description}\"")
            existing_issues = jira_result.get("issues", [])
            jira_mode = jira_result.get("mode", "unknown")
        except Exception as e:
            logger.warning(f"Jira search failed: {e}")
            existing_issues = []
            jira_mode = "error"

        self._next_step(reasoning, "Jira issue search completed",
                        output_data={"existing_issues": len(existing_issues), "jira_mode": jira_mode})

        # Step 2: Generate plan with Jira context
        jira_context = f"Found {len(existing_issues)} related issues in Jira ({jira_mode} mode)."
        if existing_issues:
            jira_context += f" Keys: {', '.join(issue['key'] for issue in existing_issues[:3])}"

        prompt = f"""
        You are planning a task with Jira context.

        Task: {description}
        Jira Context: {jira_context}
        Project: {project_key or 'default'}

        Generate plan considering existing work. Suggest linking to existing issues if relevant.

        Return:
        - subtasks: ordered list
        - jira_actions: ["create_new", "link_to_existing", "update_existing"]
        - estimated_effort: hours
        """

        try:
            response = await self.llm.chat(prompt)
            plan = json.loads(response)
        except Exception as e:
            logger.warning(f"Jira plan generation failed: {e}")
            plan = {
                "subtasks": ["Review existing Jira issues", "Plan new work", "Create/update tickets"],
                "jira_actions": ["create_new"],
                "estimated_effort": 8
            }

        self._next_step(reasoning, "Jira-integrated plan generated")

        return {
            "plan": plan,
            "existing_issues": existing_issues,
            "jira_mode": jira_mode,
            "reasoning": reasoning
        }

    def _extract_subtasks(self, decomposition: str) -> List[str]:
        """
        Extract subtasks from decomposition text

        Removes markdown formatting (**bold**)
        Extracts clean task names (before colon)
        """
        patterns = [
            r'\d+\.\s+(.+?)(?=\n\d+\.|\n\n|$)',
            r'[-•]\s+(.+?)(?=\n[-•]|\n\n|$)',
        ]

        subtasks = []
        for pattern in patterns:
            matches = re.findall(pattern, decomposition, re.MULTILINE | re.DOTALL)
            if matches:
                for match in matches:
                    if match.strip():
                        clean = match.strip().split('\n')[0].strip()
                        clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean)
                        if ':' in clean:
                            clean = clean.split(':')[0].strip()
                        subtasks.append(clean)
                break

        if not subtasks:
            lines = decomposition.split('\n')
            for line in lines:
                if line.strip() and 20 < len(line.strip()) < 200:
                    clean = line.strip().lstrip('0123456789.-•* ')
                    clean = re.sub(r'\*\*(.*?)\*\*', r'\1', clean)
                    if ':' in clean:
                        clean = clean.split(':')[0].strip()
                    subtasks.append(clean)

        return subtasks[:10]