import base64
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep
from shared.vision import get_vision_provider


@dataclass
class ArchitecturalRisk:
    """Identified architectural risk with severity and confidence"""
    component: str
    risk_type: str  # "bottleneck", "spof", "security", "scalability"
    severity: str  # "critical", "high", "medium", "low"
    description: str
    mitigation: str
    affected_flows: List[str]
    confidence: float  # 0.0 - 1.0
    evidence: str  # What supports this conclusion


@dataclass
class DataFlow:
    """Identified data flow between components"""
    source: str
    destination: str
    data_type: str
    latency_impact: str  # "critical", "moderate", "low"
    failure_mode: str
    confidence: float


@dataclass
class SystemState:
    """Internal system understanding (agent's mental model)"""
    architecture_hash: str
    component_graph: Dict[str, List[str]]  # component -> dependencies
    critical_paths: List[List[str]]
    identified_risks: List[ArchitecturalRisk]
    data_flows: List[DataFlow]
    analysis_timestamp: str
    confidence_score: float


@dataclass
class DecisionRecord:
    """Records agent's internal decision-making process"""
    decision_type: str
    input_factors: Dict[str, Any]
    calculated_score: float
    threshold_used: float
    decision_result: str
    confidence: float
    explanation: str  # Human-readable why


logger = logging.getLogger("architecture_intelligence_agent")


class ArchitectureIntelligenceAgent(MCPAgent):
    """
    Gold Standard Architecture Intelligence Agent

    Key principles:
    1. Agent logic makes decisions, LLM explains them
    2. Explicit internal state and memory
    3. Confidence scoring at every level
    4. Separation: THINK (internal) -> DECIDE (structured) -> EXPLAIN (LLM)

    This is NOT a prompt engineering exercise.
    This is an architecture of thought.
    """

    def __init__(self):
        super().__init__("Architecture-Intelligence")
        self.llm = LLMClient()
        self.vision = get_vision_provider()

        # INTERNAL STATE - Agent's "memory"
        self._system_state: Optional[SystemState] = None
        self._decision_history: List[DecisionRecord] = []
        self._confidence_threshold = 0.7  # Decisions below this trigger warnings

        # Register tools
        self.register_tool("analyze_architecture_deep", self.analyze_architecture_deep)
        self.register_tool("analyze_local_file", self.analyze_local_file)
        self.register_tool("simulate_failure_propagation", self.simulate_failure_propagation)
        self.register_tool("analyze_temporal_dependencies", self.analyze_temporal_dependencies)
        self.register_tool("recommend_architecture_refactoring", self.recommend_architecture_refactoring)
        self.register_tool("get_system_state", self.get_system_state)

        logger.info("Architecture Intelligence Agent initialized (GOLD STANDARD MODE)")

    def _compute_architecture_hash(self, analysis: str) -> str:
        """Internal decision: create fingerprint of architecture"""
        return hashlib.sha256(analysis.encode()).hexdigest()[:16]

    def _calculate_risk_score(
            self,
            severity: str,
            blast_radius: int,
            mitigation_exists: bool
    ) -> float:
        """
        INTERNAL DECISION MODEL
        Agent calculates risk priority, not LLM

        Formula: (severity_weight * blast_radius) * mitigation_factor
        """
        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.2
        }

        severity_weight = severity_weights.get(severity.lower(), 0.5)
        mitigation_factor = 0.6 if mitigation_exists else 1.0

        score = (severity_weight * min(blast_radius / 10, 1.0)) * mitigation_factor
        return round(score, 3)

    def _prioritize_risks(
            self,
            risks: List[ArchitecturalRisk]
    ) -> List[Tuple[ArchitecturalRisk, float, DecisionRecord]]:
        """
        INTERNAL DECISION: Rank risks by calculated score
        LLM only explains the ranking later
        """
        prioritized = []

        for risk in risks:
            # Calculate blast radius estimate
            blast_radius = len(risk.affected_flows)
            mitigation_exists = bool(risk.mitigation and len(risk.mitigation) > 10)

            # AGENT DECIDES
            score = self._calculate_risk_score(
                severity=risk.severity,
                blast_radius=blast_radius,
                mitigation_exists=mitigation_exists
            )

            # Record decision process
            decision = DecisionRecord(
                decision_type="risk_prioritization",
                input_factors={
                    "severity": risk.severity,
                    "blast_radius": blast_radius,
                    "mitigation_exists": mitigation_exists,
                },
                calculated_score=score,
                threshold_used=self._confidence_threshold,
                decision_result=f"Priority score: {score}",
                confidence=risk.confidence,
                explanation=f"Risk '{risk.component}' scored {score} based on {risk.severity} severity affecting {blast_radius} flows"
            )

            prioritized.append((risk, score, decision))

        # Sort by score (descending)
        prioritized.sort(key=lambda x: x[1], reverse=True)
        return prioritized

    @metric_counter("arch_intelligence")
    async def analyze_architecture_deep(
            self,
            image_base64: str,
            context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deep architectural analysis with gold-standard reasoning

        Architecture:
        1. THINK (internal, hidden from user)
        2. DECIDE (structured, agent-driven)
        3. EXPLAIN (LLM-generated, human-facing)
        """
        reasoning: List[ReasoningStep] = []
        internal_decisions: List[DecisionRecord] = []

        # =================================================================
        # PHASE 1: SPATIAL ANALYSIS (What exists)
        # =================================================================

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Starting spatial analysis (component mapping)",
            input_data={"context": context, "has_image": True}
        ))

        spatial_prompt = f"""
You are analyzing a system architecture diagram.

TASK: SPATIAL COMPONENT MAPPING

Identify and map:
1. All components (services, databases, queues, caches, etc.)
2. Direct dependencies (A calls B)
3. Reverse dependencies (B is called by A)
4. Communication patterns (sync/async, protocols)

Output as structured text with clear sections.
Context: {context or "Multi-tier distributed system"}
"""

        spatial_analysis = await self.vision.analyze(
            prompt=spatial_prompt,
            image_base64=image_base64
        )

        # INTERNAL DECISION: Assess spatial analysis quality
        spatial_quality = len(spatial_analysis) / 1000  # Rough heuristic
        spatial_confidence = min(spatial_quality, 1.0)

        internal_decisions.append(DecisionRecord(
            decision_type="spatial_quality_assessment",
            input_factors={"analysis_length": len(spatial_analysis)},
            calculated_score=spatial_confidence,
            threshold_used=0.5,
            decision_result="proceed" if spatial_confidence > 0.5 else "low_quality_warning",
            confidence=spatial_confidence,
            explanation=f"Spatial analysis quality: {spatial_confidence:.2f}"
        ))

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Spatial component mapping completed",
            output_data={
                "analysis_length": len(spatial_analysis),
                "confidence": spatial_confidence
            }
        ))

        # =================================================================
        # PHASE 2: TEMPORAL ANALYSIS (How it behaves over time)
        # =================================================================

        temporal_prompt = f"""
Based on this spatial analysis:
{spatial_analysis}

TASK: TEMPORAL FLOW ANALYSIS

Analyze system behavior over time:
1. Request flows (user -> components -> response)
2. Async/background processes
3. Race conditions potential
4. Temporal coupling (must execute in sequence)
5. Circular dependencies

Focus on TIME-DEPENDENT behavior and causality.
"""

        temporal_analysis = await self.llm.chat(temporal_prompt)

        # INTERNAL DECISION: Detect temporal risks
        temporal_risk_indicators = [
            "race condition",
            "circular",
            "deadlock",
            "timeout",
            "eventual consistency"
        ]
        detected_risks = sum(
            1 for indicator in temporal_risk_indicators
            if indicator in temporal_analysis.lower()
        )

        temporal_confidence = 0.6 + (detected_risks * 0.08)  # More detected = better analysis

        internal_decisions.append(DecisionRecord(
            decision_type="temporal_risk_detection",
            input_factors={"detected_risk_patterns": detected_risks},
            calculated_score=temporal_confidence,
            threshold_used=0.6,
            decision_result=f"{detected_risks} temporal risk patterns identified",
            confidence=temporal_confidence,
            explanation=f"Temporal analysis detected {detected_risks} risk patterns"
        ))

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Temporal flow analysis completed",
            output_data={
                "detected_risk_patterns": detected_risks,
                "confidence": temporal_confidence
            }
        ))

        # =================================================================
        # PHASE 3: CAUSAL ANALYSIS (Failure propagation)
        # =================================================================

        failure_prompt = f"""
Given this system understanding:

SPATIAL: {spatial_analysis[:500]}...
TEMPORAL: {temporal_analysis[:500]}...

TASK: FAILURE PROPAGATION (CAUSE-EFFECT)

For each critical component:
1. What happens if it fails?
2. Which components cascade?
3. Blast radius estimate
4. Cascading failure paths
5. Circuit breakers present?

Identify Single Points of Failure (SPOF).
"""

        failure_analysis = await self.llm.chat(failure_prompt)

        reasoning.append(ReasoningStep(
            step_number=4,
            description="Causal failure propagation analysis completed",
            output_data={"failure_scenarios_analyzed": True}
        ))

        # =================================================================
        # PHASE 4: STRUCTURED RISK EXTRACTION
        # =================================================================

        risk_extraction_prompt = f"""
From this failure analysis:
{failure_analysis}

Extract architectural risks in JSON format:
[
  {{
    "component": "component name",
    "risk_type": "bottleneck|spof|security|scalability",
    "severity": "critical|high|medium|low",
    "description": "clear description",
    "mitigation": "specific recommendation",
    "affected_flows": ["flow1", "flow2"],
    "evidence": "what supports this conclusion"
  }}
]

Return ONLY valid JSON array. No markdown, no explanations.
"""

        risks_json_raw = await self.llm.chat(risk_extraction_prompt)

        # Parse and structure
        try:
            # Clean JSON response
            risks_json_clean = risks_json_raw.strip()
            if risks_json_clean.startswith("```json"):
                risks_json_clean = risks_json_clean[7:]
            if risks_json_clean.endswith("```"):
                risks_json_clean = risks_json_clean[:-3]
            risks_json_clean = risks_json_clean.strip()

            risks_data = json.loads(risks_json_clean)

            # Convert to structured risks with confidence
            structured_risks = []
            for risk_dict in risks_data:
                # INTERNAL DECISION: Calculate confidence for each risk
                evidence_quality = len(risk_dict.get("evidence", "")) / 100
                mitigation_quality = len(risk_dict.get("mitigation", "")) / 100
                risk_confidence = min((evidence_quality + mitigation_quality) / 2, 1.0)

                risk = ArchitecturalRisk(
                    component=risk_dict.get("component", "Unknown"),
                    risk_type=risk_dict.get("risk_type", "unknown"),
                    severity=risk_dict.get("severity", "medium"),
                    description=risk_dict.get("description", ""),
                    mitigation=risk_dict.get("mitigation", ""),
                    affected_flows=risk_dict.get("affected_flows", []),
                    confidence=risk_confidence,
                    evidence=risk_dict.get("evidence", "")
                )
                structured_risks.append(risk)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse risk JSON: {e}")
            structured_risks = []

        # INTERNAL DECISION: Prioritize risks
        prioritized_risks = self._prioritize_risks(structured_risks)

        reasoning.append(ReasoningStep(
            step_number=5,
            description="Risks extracted and prioritized by agent logic",
            output_data={
                "total_risks": len(structured_risks),
                "high_confidence_risks": sum(1 for r in structured_risks if r.confidence > 0.7)
            }
        ))

        # =================================================================
        # PHASE 5: ACTIONABLE RECOMMENDATIONS (LLM explains agent decisions)
        # =================================================================

        # Prepare risk summary for LLM
        top_risks_summary = "\n".join([
            f"- {risk.component} ({risk.risk_type}, severity: {risk.severity}, score: {score:.3f}, confidence: {risk.confidence:.2f})"
            for risk, score, _ in prioritized_risks[:5]
        ])

        recommendations_prompt = f"""
The agent has prioritized these top architectural risks:

{top_risks_summary}

TASK: Generate actionable recommendations

Provide:
1. IMMEDIATE ACTIONS (can be done today)
   - Focus on highest-priority risks
   - Specific, concrete steps

2. SHORT-TERM IMPROVEMENTS (1-2 weeks)
   - Address medium-priority risks
   - Include implementation guidance

3. STRATEGIC REFACTORING (long-term)
   - Fundamental architectural improvements
   - Trade-off analysis

For each recommendation:
- WHY (root cause addressed)
- IMPACT (expected improvement)
- STEPS (concrete implementation)
- EFFORT (estimated time)

Prioritize by ROI.
"""

        recommendations = await self.llm.chat(recommendations_prompt)

        reasoning.append(ReasoningStep(
            step_number=6,
            description="LLM generated human-facing recommendations from agent decisions",
            output_data={"recommendations_ready": True}
        ))

        # =================================================================
        # UPDATE INTERNAL STATE (MEMORY)
        # =================================================================

        arch_hash = self._compute_architecture_hash(spatial_analysis)

        # Build component graph from spatial analysis (simplified)
        component_graph = {}  # Would parse from spatial_analysis in production

        # Calculate overall confidence
        overall_confidence = (
                                     spatial_confidence +
                                     temporal_confidence +
                                     sum(r.confidence for r in structured_risks) / max(len(structured_risks), 1)
                             ) / 3

        self._system_state = SystemState(
            architecture_hash=arch_hash,
            component_graph=component_graph,
            critical_paths=[],  # Would be extracted from temporal analysis
            identified_risks=structured_risks,
            data_flows=[],  # Would be extracted from spatial analysis
            analysis_timestamp="2026-01-17T00:00:00Z",  # Use actual timestamp
            confidence_score=overall_confidence
        )

        # =================================================================
        # RETURN STRUCTURED RESULT
        # =================================================================

        return {
            "analysis_type": "gold_standard_architecture_intelligence",

            # LLM-generated content (explanations)
            "spatial_analysis": spatial_analysis,
            "temporal_analysis": temporal_analysis,
            "failure_propagation": failure_analysis,
            "recommendations": recommendations,

            # AGENT-DECIDED structured data
            "prioritized_risks": [
                {
                    "rank": idx + 1,
                    "risk": asdict(risk),
                    "priority_score": score,
                    "decision_record": asdict(decision)
                }
                for idx, (risk, score, decision) in enumerate(prioritized_risks)
            ],

            # Internal state snapshot
            "system_state": {
                "architecture_hash": arch_hash,
                "total_risks": len(structured_risks),
                "high_priority_risks": sum(1 for _, score, _ in prioritized_risks if score > 0.7),
                "overall_confidence": overall_confidence,
            },

            # Agent's decision trail
            "internal_decisions": [asdict(d) for d in internal_decisions],

            # Metadata
            "reasoning": reasoning,
            "methodology": "Spatial-Temporal-Causal with Agent-Driven Prioritization",
            "gold_standard_version": "1.0"
        }

    @metric_counter("arch_intelligence")
    async def analyze_local_file(
            self,
            file_path: str,
            context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze local image file with full gold-standard processing"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Analyzing local file",
            input_data={"file_path": file_path, "context": context}
        ))

        try:
            path = Path(file_path)
            if not path.exists():
                return {
                    "error": f"File not found: {file_path}",
                    "reasoning": reasoning
                }

            with open(path, "rb") as f:
                image_bytes = f.read()

            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            file_size_mb = len(image_bytes) / (1024 * 1024)

            logger.info(f"Read local file: {path.name}, size: {file_size_mb:.2f}MB")

            reasoning.append(ReasoningStep(
                step_number=2,
                description="File read and converted to base64",
                output_data={
                    "file_size": len(image_bytes),
                    "file_name": path.name,
                    "file_size_mb": f"{file_size_mb:.2f}"
                }
            ))

            # Use gold-standard analysis
            result = await self.analyze_architecture_deep(
                image_base64=image_base64,
                context=context or f"Analysis of {path.name}"
            )

            # Merge reasoning chains
            if "reasoning" in result:
                result["reasoning"] = reasoning + result["reasoning"]
            else:
                result["reasoning"] = reasoning

            result["source_file"] = str(file_path)

            return result

        except Exception as e:
            logger.exception("Local file analysis failed")
            reasoning.append(ReasoningStep(
                step_number=3,
                description="File analysis failed",
                output_data={"error": str(e)}
            ))
            return {
                "error": str(e),
                "reasoning": reasoning
            }

    @metric_counter("arch_intelligence")
    async def simulate_failure_propagation(
            self,
            component_name: str,
            architecture_analysis: str
    ) -> Dict[str, Any]:
        """
        Simulate temporal failure propagation with agent-driven timeline
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description=f"Simulating failure propagation for: {component_name}",
            input_data={"component": component_name}
        ))

        simulation_prompt = f"""
You are simulating a failure scenario.

ARCHITECTURE:
{architecture_analysis}

SCENARIO: Component "{component_name}" fails at T+0s

Simulate timeline:

T+0s: {component_name} stops responding
T+5s: Which components notice first? Why?
T+10s: What cascading failures begin?
T+30s: System state now?
T+60s: Full propagation? Circuit breakers?

For each step:
- WHAT fails
- WHY it fails (dependency/timeout/exhaustion)
- IMPACT (user-facing/data loss/degraded)

Be realistic and specific.
"""

        simulation = await self.llm.chat(simulation_prompt)

        # INTERNAL DECISION: Parse timeline and assess blast radius
        timeline_markers = ["T+0s", "T+5s", "T+10s", "T+30s", "T+60s"]
        detected_markers = sum(1 for marker in timeline_markers if marker in simulation)

        blast_radius_score = detected_markers / len(timeline_markers)

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Temporal simulation completed",
            output_data={
                "timeline_completeness": blast_radius_score,
                "detected_markers": detected_markers
            }
        ))

        action_items_prompt = f"""
Based on this failure simulation:
{simulation}

Generate concrete mitigation actions:
1. Circuit breakers to add (with thresholds)
2. Timeouts to configure (specific values)
3. Health checks to implement (intervals)
4. Fallback mechanisms (concrete design)
5. Monitoring alerts (specific conditions)

Be specific with numbers and implementation details.
"""

        action_items = await self.llm.chat(action_items_prompt)

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Mitigation actions generated",
            output_data={"actions_ready": True}
        ))

        return {
            "component": component_name,
            "failure_simulation": simulation,
            "mitigation_actions": action_items,
            "blast_radius_assessment": {
                "score": blast_radius_score,
                "timeline_completeness": f"{detected_markers}/{len(timeline_markers)}"
            },
            "reasoning": reasoning,
            "simulation_type": "temporal_propagation_with_agent_assessment"
        }

    @metric_counter("arch_intelligence")
    async def analyze_temporal_dependencies(
            self,
            image_base64: str
    ) -> Dict[str, Any]:
        """Analyze temporal coupling and timing-dependent risks"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Analyzing temporal dependencies"
        ))

        temporal_prompt = """
Analyze TIME-DEPENDENT architectural risks:

1. RACE CONDITIONS
   - Concurrent operation conflicts
   - Missing locks/synchronization
   - Read-modify-write cycles

2. EVENTUAL CONSISTENCY
   - Async operations
   - Message ordering guarantees
   - Out-of-order processing impact

3. TIMEOUT CASCADES
   - Network timeout chains
   - Stacked timeout risks
   - Missing timeout configs

4. ORDERING DEPENDENCIES
   - Required execution order
   - Implicit ordering assumptions
   - Reordering corruption risks

Be paranoid about edge cases.
"""

        analysis = await self.vision.analyze(
            prompt=temporal_prompt,
            image_base64=image_base64
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Temporal dependency analysis completed"
        ))

        return {
            "temporal_analysis": analysis,
            "focus": "race_conditions_and_timing_bugs",
            "reasoning": reasoning
        }

    @metric_counter("arch_intelligence")
    async def recommend_architecture_refactoring(
            self,
            current_architecture_analysis: str,
            business_requirements: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate refactoring proposals with explicit trade-off analysis"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Generating refactoring proposals"
        ))

        refactoring_prompt = f"""
Current Architecture:
{current_architecture_analysis}

Business Requirements: {business_requirements or "Scale 10x, improve reliability"}

Propose 3 alternative architecture improvements.

For EACH proposal:
1. WHAT: Specific changes
2. WHY: Root problem solved
3. TRADE-OFFS: Gains vs. losses
4. MIGRATION PATH: Step-by-step transition
5. EFFORT ESTIMATE: Dev weeks
6. RISK LEVEL: Low/Medium/High
7. EXPECTED IMPACT: Quantified improvements

Think like a principal architect presenting to leadership.
Be realistic about costs and benefits.
"""

        proposals = await self.llm.chat(refactoring_prompt)

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Refactoring proposals generated with trade-off analysis"
        ))

        return {
            "refactoring_proposals": proposals,
            "methodology": "trade_off_driven_design",
            "reasoning": reasoning
        }

    async def get_system_state(self) -> Dict[str, Any]:
        """
        Expose internal state for debugging/monitoring
        Shows what the agent "remembers"
        """
        if not self._system_state:
            return {
                "state": "no_analysis_performed",
                "message": "No architecture has been analyzed yet"
            }

        return {
            "architecture_hash": self._system_state.architecture_hash,
            "total_risks": len(self._system_state.identified_risks),
            "high_confidence_risks": sum(
                1 for r in self._system_state.identified_risks
                if r.confidence > self._confidence_threshold
            ),
            "overall_confidence": self._system_state.confidence_score,
            "analysis_timestamp": self._system_state.analysis_timestamp,
            "decision_history_size": len(self._decision_history),
        }
