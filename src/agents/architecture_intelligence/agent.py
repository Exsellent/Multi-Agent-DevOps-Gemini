import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep
from shared.vision import get_vision_provider


@dataclass
class ArchitecturalRisk:
    """Identified architectural risk with severity"""
    component: str
    risk_type: str  # "bottleneck", "spof", "security", "scalability"
    severity: str  # "critical", "high", "medium", "low"
    description: str
    mitigation: str
    affected_flows: List[str]


@dataclass
class DataFlow:
    """Identified data flow between components"""
    source: str
    destination: str
    data_type: str
    latency_impact: str  # "critical", "moderate", "low"
    failure_mode: str


logger = logging.getLogger("architecture_intelligence_agent")


class ArchitectureIntelligenceAgent(MCPAgent):
    """
    Advanced architecture analysis agent with:
    - Spatial-temporal understanding (component relationships over time)
    - Cause-effect analysis (failure propagation)
    - Multi-step reasoning chains
    - Predictive risk assessment
    """

    def __init__(self):
        super().__init__("Architecture-Intelligence")
        self.llm = LLMClient()
        self.vision = get_vision_provider()

        # Register advanced tools
        self.register_tool("analyze_architecture_deep", self.analyze_architecture_deep)
        self.register_tool("simulate_failure_propagation", self.simulate_failure_propagation)
        self.register_tool("analyze_temporal_dependencies", self.analyze_temporal_dependencies)
        self.register_tool("recommend_architecture_refactoring", self.recommend_architecture_refactoring)

        logger.info("Architecture Intelligence Agent initialized with advanced reasoning")

    @metric_counter("arch_intelligence")
    async def analyze_architecture_deep(
            self,
            image_base64: str,
            context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deep architectural analysis with cause-effect reasoning

        Goes beyond simple component identification:
        - Maps data flows and dependencies
        - Identifies failure propagation paths
        - Analyzes temporal coupling
        - Predicts bottlenecks under load
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Starting deep architecture analysis with multi-step reasoning",
            input_data={"context": context, "has_image": True}
        ))

        # Step 1: Spatial Analysis - Identify components and their relationships
        spatial_prompt = f"""
You are a senior system architect analyzing a system diagram.

TASK 1 - SPATIAL ANALYSIS:
Identify all components, services, and their physical/logical relationships.
For each component, determine:
1. Type (API, database, queue, cache, load balancer, etc.)
2. Direct dependencies (what it calls/uses)
3. Dependents (what calls/uses it)
4. Communication patterns (sync/async, protocol)

Return structured analysis of the component graph.
Context: {context or "Multi-agent DevOps system"}
"""

        spatial_analysis = await self.vision.analyze(
            prompt=spatial_prompt,
            image_base64=image_base64
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Completed spatial component mapping",
            output_data={"analysis_length": len(spatial_analysis)}
        ))

        # Step 2: Temporal Analysis - Understand execution flows over time
        temporal_prompt = f"""
Based on this architecture analysis:
{spatial_analysis}

TASK 2 - TEMPORAL ANALYSIS:
Analyze the system's behavior over time:
1. Identify request flows (user → components → response)
2. Map async/background processes
3. Detect potential race conditions
4. Identify temporal coupling (components that must execute in sequence)
5. Find circular dependencies that could cause deadlocks

Focus on TIME-DEPENDENT risks and cause-effect chains.
"""

        temporal_analysis = await self.llm.chat(temporal_prompt)

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Completed temporal flow analysis",
            output_data={"temporal_risks_identified": True}
        ))

        # Step 3: Cause-Effect Analysis - Failure propagation
        failure_prompt = f"""
Given this system understanding:

SPATIAL: {spatial_analysis[:500]}...
TEMPORAL: {temporal_analysis[:500]}...

TASK 3 - FAILURE PROPAGATION ANALYSIS:
For each critical component, determine:
1. What happens if it fails?
2. Which other components are affected?
3. What's the blast radius?
4. Is there a cascading failure path?
5. Are there circuit breakers or fallback mechanisms?

Identify Single Points of Failure (SPOF) and rank by criticality.
"""

        failure_analysis = await self.llm.chat(failure_prompt)

        reasoning.append(ReasoningStep(
            step_number=4,
            description="Completed cause-effect failure analysis",
            output_data={"failure_scenarios_analyzed": True}
        ))

        # Step 4: Extract structured risks
        risk_extraction_prompt = f"""
From this failure analysis:
{failure_analysis}

Extract top 5 architectural risks in this EXACT JSON format:
[
  {{
    "component": "component name",
    "risk_type": "bottleneck|spof|security|scalability",
    "severity": "critical|high|medium|low",
    "description": "clear description",
    "mitigation": "specific recommendation",
    "affected_flows": ["flow1", "flow2"]
  }}
]

Return ONLY valid JSON, no markdown.
"""

        risks_json = await self.llm.chat(risk_extraction_prompt)

        reasoning.append(ReasoningStep(
            step_number=5,
            description="Extracted structured risk data",
            output_data={"risks_extracted": True}
        ))

        # Step 5: Generate actionable recommendations
        recommendations_prompt = f"""
Based on identified risks and architecture analysis, provide:

1. IMMEDIATE ACTIONS (can be done today)
2. SHORT-TERM IMPROVEMENTS (1-2 weeks)
3. STRATEGIC REFACTORING (long-term)

For each recommendation:
- Explain WHY (root cause)
- Explain IMPACT (what improves)
- Provide CONCRETE implementation steps

Be specific, actionable, and prioritize by ROI.
"""

        recommendations = await self.llm.chat(recommendations_prompt)

        reasoning.append(ReasoningStep(
            step_number=6,
            description="Generated prioritized recommendations",
            output_data={"recommendations_ready": True}
        ))

        return {
            "analysis_type": "deep_architecture_intelligence",
            "spatial_analysis": spatial_analysis,
            "temporal_analysis": temporal_analysis,
            "failure_propagation": failure_analysis,
            "structured_risks": risks_json,
            "recommendations": recommendations,
            "reasoning": reasoning,
            "methodology": "spatial-temporal-causal analysis with multi-step reasoning"
        }

    @metric_counter("arch_intelligence")
    async def simulate_failure_propagation(
            self,
            component_name: str,
            architecture_analysis: str
    ) -> Dict[str, Any]:
        """
        Simulate what happens when a specific component fails

        This demonstrates cause-effect understanding:
        - Maps the failure propagation graph
        - Estimates time-to-impact for each affected component
        - Identifies cascade triggers
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description=f"Simulating failure propagation for: {component_name}",
            input_data={"component": component_name}
        ))

        simulation_prompt = f"""
You are simulating a failure scenario in a distributed system.

ARCHITECTURE CONTEXT:
{architecture_analysis}

FAILURE SCENARIO: Component "{component_name}" just failed (stopped responding).

Simulate the next 60 seconds step-by-step:

T+0s: {component_name} fails
T+5s: Which components are affected first? Why?
T+10s: What cascading failures might occur?
T+30s: What's the system state now?
T+60s: Has the failure propagated fully? Is there circuit breaking?

For each time step, explain:
- WHAT fails
- WHY it fails (dependency/timeout/resource exhaustion)
- IMPACT (user-facing? data loss? degraded?)

Think like a chaos engineer. Be specific and realistic.
"""

        simulation = await self.llm.chat(simulation_prompt)

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Completed temporal failure simulation",
            output_data={"simulation_complete": True}
        ))

        # Extract action items
        action_items_prompt = f"""
Based on this failure simulation:
{simulation}

List concrete actions to prevent or mitigate this failure:
1. Circuit breakers to add
2. Timeouts to configure
3. Health checks to implement
4. Fallback mechanisms needed
5. Monitoring alerts to create

Be specific with thresholds and implementation details.
"""

        action_items = await self.llm.chat(action_items_prompt)

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Generated mitigation action items",
            output_data={"actions_ready": True}
        ))

        return {
            "component": component_name,
            "failure_simulation": simulation,
            "mitigation_actions": action_items,
            "reasoning": reasoning,
            "simulation_type": "temporal_propagation_with_cause_effect"
        }

    @metric_counter("arch_intelligence")
    async def analyze_temporal_dependencies(
            self,
            image_base64: str
    ) -> Dict[str, Any]:
        """
        Identify temporal coupling and timing-dependent behavior

        Focuses on:
        - Race conditions
        - Eventual consistency issues
        - Timing-dependent bugs
        - Order-dependent operations
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Analyzing temporal dependencies and timing risks"
        ))

        temporal_prompt = """
Analyze this architecture for TIME-DEPENDENT risks:

1. RACE CONDITIONS:
   - Where could concurrent operations conflict?
   - Which resources lack proper locking?
   - Are there read-modify-write cycles?

2. EVENTUAL CONSISTENCY:
   - Which operations are async?
   - How is consistency guaranteed?
   - What happens if messages are processed out of order?

3. TIMEOUT CASCADES:
   - Map all network calls and their timeouts
   - Identify timeout chains that could stack
   - Find missing timeout configurations

4. ORDERING DEPENDENCIES:
   - Which operations must happen in a specific order?
   - Are there implicit assumptions about execution order?
   - Could reordering cause data corruption?

Be paranoid. Think about edge cases and race conditions.
"""

        analysis = await self.vision.analyze(
            prompt=temporal_prompt,
            image_base64=image_base64
        )

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Completed temporal dependency analysis"
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
        """
        Propose architecture improvements based on deep analysis

        Unlike simple recommendations, this:
        - Considers trade-offs explicitly
        - Provides migration paths
        - Estimates implementation effort
        - Predicts impact on system behavior
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Generating architecture refactoring proposals"
        ))

        refactoring_prompt = f"""
Current Architecture Analysis:
{current_architecture_analysis}

Business Requirements: {business_requirements or "Scale to 10x load, improve reliability"}

Propose 3 alternative architecture improvements:

For EACH proposal:
1. WHAT: Specific changes to make
2. WHY: Root problem it solves
3. TRADE-OFFS: What you gain vs. what you lose
4. MIGRATION PATH: Step-by-step transition plan
5. EFFORT ESTIMATE: Dev weeks required
6. RISK LEVEL: Low/Medium/High
7. EXPECTED IMPACT: Quantify improvements

Think like a principal architect presenting to leadership.
Be realistic about costs and benefits.
"""

        proposals = await self.llm.chat(refactoring_prompt)

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Generated refactoring proposals with trade-off analysis"
        ))

        return {
            "refactoring_proposals": proposals,
            "methodology": "trade_off_driven_design",
            "reasoning": reasoning
        }