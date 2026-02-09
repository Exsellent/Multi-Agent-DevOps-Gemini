import base64
import hashlib
import io
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

from PIL import Image

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

    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    def _resolve_local_image_path(self, file_path: str) -> Path:
        """
        Intelligently resolve the path to an image file by trying multiple possible locations.

        This method handles both absolute and relative paths, common container layouts,
        and typical directory structures (assets/, images/, etc.).

        Args:
            file_path (str): The input file path (absolute or relative)

        Returns:
            Path: Resolved absolute path to the existing image file

        Raises:
            ValueError: If the input path is empty
            FileNotFoundError: If no valid file is found after checking all candidates
        """
        logger.info(f"Resolving image path: '{file_path}' (cwd = {Path.cwd()})")

        original_path = file_path.strip()
        if not original_path:
            raise ValueError("File path cannot be empty")

        path = Path(original_path)

        # 1. If it's already an absolute path — check it directly
        if path.is_absolute():
            if path.exists() and path.is_file():
                logger.info(f"Absolute path found: {path}")
                return path
            else:
                raise FileNotFoundError(f"Absolute path not found: {path}")

        # 2. Application root (usually /app in Docker containers)
        app_root = Path("/app")

        # List of candidate paths to try
        candidates = []

        # Variant A: treat input as relative to /app
        candidates.append(app_root / path)

        # Variant B: input is just a filename → try in common directories
        file_name = path.name
        common_dirs = ["assets", "images", "static", "data", "public", ""]

        for directory in common_dirs:
            if directory:
                candidates.append(app_root / directory / file_name)
            else:
                candidates.append(app_root / file_name)

        # Variant C: relative to current working directory
        candidates.append(Path.cwd() / path)

        # 3. Try each candidate in order
        for candidate in candidates:
            try:
                if candidate.exists() and candidate.is_file():
                    logger.info(f"File found: {candidate}")
                    return candidate
                else:
                    logger.debug(f"Not found: {candidate}")
            except Exception as e:
                logger.warning(f"Error checking {candidate}: {e}")

        # 4. Nothing found → raise detailed error with all attempted paths
        checked_paths = "\n  ".join(f"→ {p}" for p in candidates)
        raise FileNotFoundError(
            f"Image file not found: '{original_path}'\n"
            f"Checked locations:\n  {checked_paths}\n"
            f"Current working directory: {Path.cwd()}"
        )

    def _load_and_encode_image(self, file_path: str) -> str:
        try:
            img_path = self._resolve_local_image_path(file_path)  # ← now exists

            img = Image.open(img_path)

            if img.mode not in ('RGB', 'RGBA'):
                img = img.convert('RGB')

            buffered = io.BytesIO()
            img.save(buffered, format="PNG")

            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        except Exception as e:
            logger.error(f"Failed to load image {file_path}: {e}")
            raise ValueError(f"Failed to load image: {str(e)}")

    def _validate_and_prepare_base64(self, base64_str: str) -> str:
        """
        Validate and prepare base64 string
        Removes data URL prefix if present
        """
        if not base64_str:
            raise ValueError("Empty base64 string provided")

        try:
            # Remove data URL prefix if present
            if ',' in base64_str and base64_str.startswith('data:'):
                base64_str = base64_str.split(',', 1)[1]

            # Remove whitespace and newlines
            base64_str = base64_str.strip().replace('\n', '').replace('\r', '').replace(' ', '')

            # Validate by decoding
            try:
                image_data = base64.b64decode(base64_str, validate=True)
            except Exception as decode_error:
                raise ValueError(f"Invalid base64 encoding: {str(decode_error)}")

            if len(image_data) < 100:
                raise ValueError(f"Image data too small ({len(image_data)} bytes), likely invalid")

            # Verify it's a valid image
            try:
                img = Image.open(io.BytesIO(image_data))
                img.verify()

                # Re-open for processing (verify closes the file)
                img = Image.open(io.BytesIO(image_data))

                # Convert to RGB if necessary
                if img.mode not in ('RGB', 'RGBA'):
                    logger.info(f"Converting image from {img.mode} to RGB")
                    img = img.convert('RGB')
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    base64_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

                logger.info(f"Base64 image validated and prepared ({img.size}, {img.mode})")
                return base64_str

            except Exception as img_error:
                raise ValueError(f"Invalid image format: {str(img_error)}")

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error validating base64: {e}")
            raise ValueError(f"Failed to process base64 image: {str(e)}")

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
            blast_radius = len(risk.affected_flows)
            mitigation_exists = bool(risk.mitigation and len(risk.mitigation) > 10)

            # AGENT DECIDES
            score = self._calculate_risk_score(
                severity=risk.severity,
                blast_radius=blast_radius,
                mitigation_exists=mitigation_exists
            )

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
            image_base64: Optional[str] = None,
            file_path: Optional[str] = None,
            context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deep architectural analysis with gold-standard reasoning

        Architecture:
        1. THINK (internal, hidden from user)
        2. DECIDE (structured, agent-driven)
        3. EXPLAIN (LLM-generated, human-facing)

        Accepts either:
        - file_path: path to image file (preferred for local files)
        - image_base64: base64 encoded image string or file path
        """
        reasoning: List[ReasoningStep] = []
        internal_decisions: List[DecisionRecord] = []

        # Prioritize file_path parameter if provided
        if file_path:
            try:
                logger.info(f"Loading image from file_path parameter: {file_path}")
                image_base64 = self._load_and_encode_image(file_path)
                reasoning.append(ReasoningStep(
                    step_number=0,
                    description=f"Loaded image from file_path parameter",
                    output_data={"source": "file_path", "path": file_path, "size": len(image_base64)}
                ))
            except Exception as e:
                logger.error(f"Failed to load file: {e}")
                return {
                    "error": f"Failed to load image file: {str(e)}",
                    "reasoning": [ReasoningStep(
                        step_number=0,
                        description="File loading failed",
                        output_data={"error": str(e), "file_path": file_path}
                    )]
                }
        elif not image_base64:
            error_msg = "No image provided. Please specify either 'file_path' or 'image_base64' parameter."
            logger.error(error_msg)
            return {
                "error": error_msg,
                "reasoning": [ReasoningStep(
                    step_number=0,
                    description="No image input",
                    output_data={}
                )]
            }
        else:
            # Determine input type and load image accordingly
            is_file_path = False
            is_placeholder = False

            # Check for placeholder values from UI
            if image_base64 in ['demo_placeholder', 'placeholder', '']:
                is_placeholder = True
                logger.warning(f"Received placeholder value: '{image_base64}' - checking context for file path")
                # Try to extract file path from context if present
                if context and ('/' in context or '\\' in context or context.endswith(
                        ('.png', '.jpg', '.jpeg', '.gif', '.webp'))):

                    potential_paths = [word for word in context.split() if
                                       '/' in word or word.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))]
                    if potential_paths:
                        image_base64 = potential_paths[0]
                        is_placeholder = False
                        is_file_path = True
                        logger.info(f"Extracted file path from context: {image_base64}")

            # Check if input is a file path
            if not is_placeholder and not image_base64.startswith('data:'):
                # Short strings that look like paths
                if len(image_base64) < 500:
                    # Check for file extensions
                    if any(image_base64.lower().endswith(ext) for ext in
                           ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']):
                        is_file_path = True
                    # Check for path separators
                    elif '/' in image_base64 or '\\' in image_base64:
                        is_file_path = True

            if is_placeholder:
                error_msg = "No valid image provided. Please specify either a file path (e.g., 'assets/architecture.png') or base64 encoded image data."
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "reasoning": [ReasoningStep(
                        step_number=0,
                        description="No valid image input",
                        output_data={"received": image_base64, "context": context}
                    )]
                }

            # If it's a file path, load it
            if is_file_path:
                try:
                    logger.info(f"Detected file path input: {image_base64}")
                    image_base64 = self._load_and_encode_image(image_base64)
                    reasoning.append(ReasoningStep(
                        step_number=0,
                        description=f"Loaded image from file path",
                        output_data={"source": "file_path", "size": len(image_base64)}
                    ))
                except Exception as e:
                    logger.error(f"Failed to load file: {e}")
                    return {
                        "error": f"Failed to load image file: {str(e)}",
                        "reasoning": [ReasoningStep(
                            step_number=0,
                            description="File loading failed",
                            output_data={"error": str(e), "file_path": image_base64}
                        )]
                    }
            else:
                # Validate and prepare base64
                try:
                    image_base64 = self._validate_and_prepare_base64(image_base64)
                    reasoning.append(ReasoningStep(
                        step_number=0,
                        description="Base64 image validated",
                        output_data={"source": "base64", "size": len(image_base64)}
                    ))
                except ValueError as e:
                    logger.error(f"Image validation failed: {e}")
                    return {
                        "error": str(e),
                        "reasoning": [ReasoningStep(
                            step_number=0,
                            description="Image validation failed",
                            output_data={"error": str(e)}
                        )]
                    }

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
        # PHASE 2: TEMPORAL ANALYSIS (How it changes over time)
        # =================================================================

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Starting temporal analysis (data flows over time)"
        ))

        temporal_prompt = f"""
Based on this architecture:

{spatial_analysis}

TASK: TEMPORAL FLOW ANALYSIS

Analyze the TIME dimension:

1. REQUEST FLOWS
   - User request → component chain → response
   - Critical paths (longest chains)
   - Async/background processes

2. RACE CONDITIONS
   - Concurrent access risks
   - State synchronization needs
   - Ordering dependencies

3. TEMPORAL COUPLING
   - Must-execute-in-order operations
   - Time-dependent failures
   - Circular dependencies

4. LATENCY PATHS
   - Network hops
   - Database roundtrips
   - Queue delays

Be specific about sequences and timing.
"""

        temporal_analysis = await self.llm.chat(temporal_prompt)

        temporal_confidence = min(len(temporal_analysis) / 1000, 1.0)

        reasoning.append(ReasoningStep(
            step_number=4,
            description="Temporal flow analysis completed",
            output_data={
                "analysis_length": len(temporal_analysis),
                "confidence": temporal_confidence
            }
        ))

        # =================================================================
        # PHASE 3: FAILURE PROPAGATION (What breaks when X fails)
        # =================================================================

        reasoning.append(ReasoningStep(
            step_number=5,
            description="Starting failure propagation analysis"
        ))

        failure_prompt = f"""
Based on this architecture understanding:

SPATIAL: {spatial_analysis[:500]}...
TEMPORAL: {temporal_analysis[:500]}...

TASK: FAILURE PROPAGATION ANALYSIS

For EACH major component, analyze:

1. FAILURE IMPACT
   - What stops working immediately?
   - What degrades?
   - What's unaffected?

2. CASCADE EFFECTS
   - Which other components fail next? Why?
   - Timeout cascades
   - Resource exhaustion chains

3. BLAST RADIUS
   - How many users affected?
   - Data loss potential?
   - Recovery time?

4. CIRCUIT BREAKERS
   - What prevents total collapse?
   - Missing protections?
   - Fallback mechanisms?

Be paranoid. Think like a chaos engineer.
Output in structured format with severity levels.
"""

        failure_analysis = await self.llm.chat(failure_prompt)

        failure_confidence = min(len(failure_analysis) / 1000, 1.0)

        reasoning.append(ReasoningStep(
            step_number=6,
            description="Failure propagation analysis completed",
            output_data={
                "analysis_length": len(failure_analysis),
                "confidence": failure_confidence
            }
        ))

        # =================================================================
        # PHASE 4: RISK EXTRACTION AND PRIORITIZATION (Agent logic)
        # =================================================================

        reasoning.append(ReasoningStep(
            step_number=7,
            description="Extracting and prioritizing architectural risks"
        ))

        risk_extraction_prompt = f"""
From this analysis:

{failure_analysis}

Extract concrete architectural risks in JSON format:

{{
  "risks": [
    {{
      "component": "ComponentName",
      "risk_type": "bottleneck|spof|security|scalability",
      "severity": "critical|high|medium|low",
      "description": "Specific issue",
      "mitigation": "Concrete fix",
      "affected_flows": ["flow1", "flow2"],
      "confidence": 0.85,
      "evidence": "What from the diagram supports this"
    }}
  ]
}}

Be specific. Every risk needs evidence.
"""

        risk_json = await self.llm.chat(risk_extraction_prompt)

        # INTERNAL DECISION: Parse and validate risks
        try:
            # Try to extract JSON from response
            if "```json" in risk_json:
                risk_json = risk_json.split("```json")[1].split("```")[0]
            elif "```" in risk_json:
                risk_json = risk_json.split("```")[1].split("```")[0]

            risk_data = json.loads(risk_json.strip())
            risks = [
                ArchitecturalRisk(**r)
                for r in risk_data.get("risks", [])
            ]
        except Exception as e:
            logger.warning(f"Failed to parse risk JSON: {e}")
            risks = []

        # AGENT DECISION: Prioritize risks
        prioritized_risks = self._prioritize_risks(risks)

        reasoning.append(ReasoningStep(
            step_number=8,
            description=f"Extracted {len(risks)} risks, prioritized by agent logic",
            output_data={
                "total_risks": len(risks),
                "high_priority": sum(1 for _, score, _ in prioritized_risks if score > 0.7)
            }
        ))

        # =================================================================
        # PHASE 5: SYSTEM STATE UPDATE (Agent memory)
        # =================================================================

        overall_confidence = (
                spatial_confidence * 0.4 +
                temporal_confidence * 0.3 +
                failure_confidence * 0.3
        )

        from datetime import datetime
        self._system_state = SystemState(
            architecture_hash=self._compute_architecture_hash(spatial_analysis),
            component_graph={},  # Would be populated by parsing spatial_analysis
            critical_paths=[],
            identified_risks=risks,
            data_flows=[],
            analysis_timestamp=datetime.now().isoformat(),
            confidence_score=overall_confidence
        )

        # Store decision history
        self._decision_history.extend([d for _, _, d in prioritized_risks])
        self._decision_history.extend(internal_decisions)

        reasoning.append(ReasoningStep(
            step_number=9,
            description="System state updated with analysis results",
            output_data={
                "architecture_hash": self._system_state.architecture_hash,
                "overall_confidence": overall_confidence
            }
        ))

        # =================================================================
        # FINAL OUTPUT
        # =================================================================

        return {
            "analysis_type": "deep_architecture_intelligence",
            "methodology": "spatial_temporal_causal_with_agent_reasoning",

            # Core analyses
            "spatial_analysis": spatial_analysis,
            "temporal_analysis": temporal_analysis,
            "failure_propagation": failure_analysis,

            # Agent decisions
            "identified_risks": [
                {
                    **asdict(risk),
                    "priority_score": score,
                    "decision_record": asdict(decision)
                }
                for risk, score, decision in prioritized_risks
            ],

            "confidence_scores": {
                "spatial": spatial_confidence,
                "temporal": temporal_confidence,
                "failure": failure_confidence,
                "overall": overall_confidence
            },
            "architecture_hash": self._system_state.architecture_hash,

            # Reasoning chain
            "reasoning": reasoning,
            "internal_decisions": [asdict(d) for d in internal_decisions]
        }

    @metric_counter("arch_intelligence")
    async def analyze_local_file(
            self,
            file_path: str,
            context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze architecture from a local file
        Loads image, converts to base64, then calls deep analysis
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description=f"Loading local file: {file_path}",
            input_data={"file_path": file_path}
        ))

        try:

            image_base64 = self._load_and_encode_image(file_path)

            reasoning.append(ReasoningStep(
                step_number=2,
                description="File loaded and encoded successfully",
                output_data={"encoded_size": len(image_base64)}
            ))

            # Perform deep analysis
            result = await self.analyze_architecture_deep(
                image_base64=image_base64,
                context=context
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

        # Determine input type
        is_file_path = False
        is_placeholder = image_base64 in ['demo_placeholder', 'placeholder', '']

        if is_placeholder:
            return {
                "error": "No valid image provided. Please specify either a file path or base64 encoded image data.",
                "reasoning": [ReasoningStep(
                    step_number=0,
                    description="No valid image input",
                    output_data={"received": image_base64}
                )]
            }

        # Check if input is a file path
        if not image_base64.startswith('data:') and len(image_base64) < 500:
            if any(image_base64.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']):
                is_file_path = True
            elif '/' in image_base64 or '\\' in image_base64:
                is_file_path = True

        if is_file_path:
            try:
                logger.info(f"Detected file path input: {image_base64}")
                image_base64 = self._load_and_encode_image(image_base64)
            except Exception as e:
                return {
                    "error": f"Failed to load image file: {str(e)}",
                    "reasoning": [ReasoningStep(
                        step_number=0,
                        description="File loading failed",
                        output_data={"error": str(e), "file_path": image_base64}
                    )]
                }
        else:

            try:
                image_base64 = self._validate_and_prepare_base64(image_base64)
            except ValueError as e:
                return {
                    "error": str(e),
                    "reasoning": [ReasoningStep(
                        step_number=0,
                        description="Image validation failed",
                        output_data={"error": str(e)}
                    )]
                }

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
def _load_and_encode_image(self, file_path: str) -> str:
    try:
        img_path = self._resolve_image_path(file_path)

        img = Image.open(img_path)

        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")

        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        logger.info(f"Image loaded and encoded: {img_path}")
        return img_base64

    except Exception as e:
        logger.error(f"Failed to load image {file_path}: {e}")
        raise ValueError(f"Failed to load image: {str(e)}")

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
