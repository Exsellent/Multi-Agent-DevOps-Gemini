import logging
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple

import httpx

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("health_monitor_agent")


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class AgentRole(Enum):
    """Agent roles with clear business vs observability separation"""
    BUSINESS_CORE = "business_core"  # Critical for business functionality
    OBSERVABILITY = "observability"  # Monitoring, metrics, health checks
    SPECIALIZED = "specialized"  # Domain-specific agents
    SUPPORT = "support"  # Auxiliary agents


@dataclass
class HealthCheck:
    """Single health check result"""
    agent_name: str
    status: HealthStatus
    error_rate: float
    response_time_ms: Optional[float]
    issues: List[str]
    timestamp: datetime
    is_reachable: bool


@dataclass
class SystemHealth:
    """Overall system health assessment"""
    overall_status: HealthStatus
    healthy_count: int
    degraded_count: int
    warning_count: int
    critical_count: int
    unknown_count: int
    systemic_risk: str
    cascade_risk: List[str]
    recommendations: List[str]
    timestamp: datetime


@dataclass
class HealthTrend:
    """Health trend analysis"""
    agent_name: str
    trend: str
    error_rate_change: float
    response_time_change: float
    forecast: str


class HealthHistory:
    """Maintains historical health data for trend analysis"""

    def __init__(self, max_history: int = 100):
        self.history: Dict[str, deque] = {}
        self.max_history = max_history

    def add(self, agent_name: str, check: HealthCheck):
        if agent_name not in self.history:
            self.history[agent_name] = deque(maxlen=self.max_history)
        self.history[agent_name].append(check)

    def get_recent(self, agent_name: str, count: int = 10) -> List[HealthCheck]:
        if agent_name not in self.history:
            return []
        return list(self.history[agent_name])[-count:]

    def calculate_trend(self, agent_name: str) -> Optional[HealthTrend]:
        recent = self.get_recent(agent_name, 10)
        if len(recent) < 2:
            return None

        error_rates = [c.error_rate for c in recent]
        error_rate_change = ((error_rates[-1] - error_rates[0]) / max(error_rates[0], 0.01)) * 100

        response_times = [c.response_time_ms for c in recent if c.response_time_ms is not None]
        rt_change = 0.0
        if response_times:
            rt_change = ((response_times[-1] - response_times[0]) / max(response_times[0], 1)) * 100

        if error_rate_change > 50 or rt_change > 100:
            trend = "CRITICAL"
            forecast = "CRITICAL within 30min"
        elif error_rate_change > 20 or rt_change > 50:
            trend = "DEGRADING"
            forecast = "WARNING within 1hr"
        elif error_rate_change < -10 and rt_change < -20:
            trend = "IMPROVING"
            forecast = "HEALTHY"
        else:
            trend = "STABLE"
            forecast = "STABLE"

        return HealthTrend(
            agent_name=agent_name,
            trend=trend,
            error_rate_change=error_rate_change,
            response_time_change=rt_change,
            forecast=forecast
        )


class HealthMonitorAgent(MCPAgent):
    """Advanced health monitoring with business-aware prioritization"""

    AGENT_ROLES = {
        "planner": AgentRole.BUSINESS_CORE,
        "progress": AgentRole.BUSINESS_CORE,
        "risks": AgentRole.BUSINESS_CORE,
        "digest": AgentRole.BUSINESS_CORE,
        "architecture_intelligence": AgentRole.BUSINESS_CORE,
        "code_execution": AgentRole.BUSINESS_CORE,
        "health_monitor": AgentRole.OBSERVABILITY,
        "health": AgentRole.OBSERVABILITY,
        "metrics": AgentRole.OBSERVABILITY,
        "marathon": AgentRole.SPECIALIZED,
    }

    DEPENDENCIES = {
        "progress": ["planner"],
        "risks": ["planner"],
        "digest": ["planner", "progress", "risks"],
        "architecture_intelligence": ["planner"],
        "code_execution": ["planner"],
        "marathon": ["planner"],
        "health_monitor": [],
        "health": [],
        "metrics": [],
    }

    DEFAULT_URLS = {
        "planner": "http://planner:8301/health",
        "progress": "http://progress:8302/health",
        "risks": "http://risks:8303/health",
        "digest": "http://digest:8304/health",
        "architecture_intelligence": "http://architecture_intelligence:8305/health",
        "health_monitor": "http://health_monitor:8306/health",
        "metrics": "http://metrics:8307/health",
        "marathon": "http://marathon:8308/health",
        "code_execution": "http://code_execution:8309/health",
    }

    def __init__(self):
        super().__init__("Health-Monitor")
        self.llm = LLMClient()
        self.health_history = HealthHistory()

        self.register_tool("check_health", self.check_health)
        self.register_tool("diagnose_agent", self.diagnose_agent)
        self.register_tool("predict_failures", self.predict_failures)
        self.register_tool("analyze_trends", self.analyze_trends)

        logger.info("HealthMonitorAgent initialized")

    def _next_step(
            self,
            reasoning: List[ReasoningStep],
            description: str,
            input_data: Optional[Dict[str, Any]] = None,
            output_data: Optional[Dict[str, Any]] = None
    ):
        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description=description,
            timestamp=datetime.now().isoformat(),
            input_data=input_data or {},
            output=output_data or {},
            agent=self.name
        ))

    async def _check_agent_reachability(
            self,
            agent_name: str,
            url: str,
            timeout: float = 5.0
    ) -> Tuple[bool, Optional[float], Optional[dict]]:
        try:
            start = time.time()
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(url)
            response_time = (time.time() - start) * 1000

            if response.status_code == 200:
                try:
                    data = response.json()
                except Exception:
                    data = {"raw": response.text}
                return True, response_time, data
            else:
                return False, response_time, None
        except Exception as e:
            logger.debug(f"{agent_name} unreachable: {e}")
            return False, None, None

    def _classify_agent(
            self,
            agent_name: str,
            is_reachable: bool,
            response_time: Optional[float],
            tasks_processed: int = 0,
            errors: int = 0
    ) -> HealthCheck:
        issues = []
        error_rate = errors / max(tasks_processed, 1) if tasks_processed > 0 else 0.0

        if not is_reachable:
            status = HealthStatus.CRITICAL
            issues.append("unreachable")
        elif tasks_processed == 0:
            status = HealthStatus.HEALTHY
            issues.append("idle_no_tasks_yet")
        elif error_rate > 0.1 or (response_time and response_time > 5000):
            status = HealthStatus.CRITICAL
            issues.append("high_error_or_latency")
        elif error_rate > 0.05 or (response_time and response_time > 2000):
            status = HealthStatus.WARNING
            issues.append("elevated_error_or_latency")
        else:
            status = HealthStatus.HEALTHY

        return HealthCheck(
            agent_name=agent_name,
            status=status,
            error_rate=error_rate,
            response_time_ms=response_time,
            issues=issues,
            timestamp=datetime.now(),
            is_reachable=is_reachable
        )

    def _detect_cascade_risks(self, checks: List[HealthCheck]) -> List[str]:
        unhealthy = {c.agent_name for c in checks if c.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]}
        risks = []
        for agent, deps in self.DEPENDENCIES.items():
            if agent in unhealthy:
                continue
            bad_deps = [d for d in deps if d in unhealthy]
            if bad_deps:
                risks.append(f"{agent} ← {', '.join(bad_deps)}")
        return risks

    @metric_counter("health_monitor")
    async def check_health(self, agents: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if agents is None:
            agents = self.DEFAULT_URLS

        reasoning: List[ReasoningStep] = []
        self._next_step(reasoning, "Starting system-wide health check", input_data={"agents_checked": len(agents)})

        health_checks: List[HealthCheck] = []

        for agent_name, url in agents.items():
            is_reachable, response_time, data = await self._check_agent_reachability(agent_name, url)

            tasks = data.get("tasks_processed", 0) if data else 0
            errors = data.get("errors", 0) if data else 0

            check = self._classify_agent(
                agent_name=agent_name,
                is_reachable=is_reachable,
                response_time=response_time,
                tasks_processed=tasks,
                errors=errors
            )

            health_checks.append(check)
            self.health_history.add(agent_name, check)

        self._next_step(reasoning, "Completed individual checks",
                        output_data={"reachable": sum(c.is_reachable for c in health_checks)})

        system_health = self._assess_system_health(health_checks)

        ai_analysis = await self._generate_ai_analysis(health_checks, system_health)

        trends = {
            name: asdict(trend)
            for name in agents
            if (trend := self.health_history.calculate_trend(name))
        }

        self._next_step(reasoning, "Health check completed",
                        output_data={"overall_status": system_health.overall_status.value})

        return {
            "system_health": asdict(system_health),
            "agent_health": [asdict(c) for c in health_checks],
            "trends": trends,
            "ai_analysis": ai_analysis,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }

    def _assess_system_health(self, checks: List[HealthCheck]) -> SystemHealth:
        counts = {s: 0 for s in HealthStatus}
        for c in checks:
            counts[c.status] += 1

        business_core = [c for c in checks if self.AGENT_ROLES.get(c.agent_name) == AgentRole.BUSINESS_CORE]
        observability = [c for c in checks if self.AGENT_ROLES.get(c.agent_name) == AgentRole.OBSERVABILITY]

        business_critical = [c for c in business_core if c.status == HealthStatus.CRITICAL]
        observability_critical = [c for c in observability if c.status == HealthStatus.CRITICAL]

        if business_critical:
            overall = HealthStatus.CRITICAL
            risk = "CRITICAL"
        elif observability_critical:
            overall = HealthStatus.WARNING
            risk = "MEDIUM"
        elif any(c.status == HealthStatus.WARNING for c in business_core):
            overall = HealthStatus.WARNING
            risk = "HIGH"
        elif counts[HealthStatus.WARNING] > 2:
            overall = HealthStatus.WARNING
            risk = "LOW"
        elif counts[HealthStatus.HEALTHY] == len(checks):
            overall = HealthStatus.HEALTHY
            risk = "LOW"
        else:
            overall = HealthStatus.DEGRADED
            risk = "LOW"

        cascade_risk = self._detect_cascade_risks(checks)
        recommendations = self._generate_recommendations(checks, business_critical, observability_critical,
                                                         cascade_risk)

        return SystemHealth(
            overall_status=overall,
            healthy_count=counts[HealthStatus.HEALTHY],
            degraded_count=counts[HealthStatus.DEGRADED],
            warning_count=counts[HealthStatus.WARNING],
            critical_count=counts[HealthStatus.CRITICAL],
            unknown_count=counts[HealthStatus.UNKNOWN],
            systemic_risk=risk,
            cascade_risk=cascade_risk,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _generate_recommendations(
            self,
            checks: List[HealthCheck],
            business_critical: List[HealthCheck],
            observability_critical: List[HealthCheck],
            cascade_risk: List[str]
    ) -> List[str]:
        recs = []

        if business_critical:
            recs.append(
                f"🚨 CRITICAL: Business core failing ({', '.join(c.agent_name for c in business_critical)}). "
                "Immediate fix required — business functions impacted."
            )
            return recs

        if observability_critical:
            names = ', '.join(c.agent_name for c in observability_critical)
            recs.append(
                f"⚠️ WARNING: Observability impaired ({names}). "
                "Business functions operational, but visibility lost. Restore monitoring ASAP."
            )
            healthy_business = len([c for c in checks if self.AGENT_ROLES.get(
                c.agent_name) == AgentRole.BUSINESS_CORE and c.status == HealthStatus.HEALTHY])
            recs.append(f"📊 Business core: {healthy_business} agents healthy and operational.")

        if cascade_risk and not recs:
            recs.append(f"🔗 CASCADE RISK: Potential propagation: {', '.join(cascade_risk[:3])}")

        idle = [c for c in checks if "idle" in " ".join(c.issues).lower()]
        if idle and not recs:
            recs.append(f"🟢 System idle: {len(idle)} agents healthy but awaiting tasks.")

        if not recs:
            recs.append("✅ System fully operational and healthy.")

        return recs

    async def _generate_ai_analysis(self, checks: List[HealthCheck], system_health: SystemHealth) -> str:
        problematic = [c for c in checks if c.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]]

        if not problematic:
            return "No issues detected. System healthy."

        business_critical = [c.agent_name for c in problematic if
                             self.AGENT_ROLES.get(c.agent_name) == AgentRole.BUSINESS_CORE]
        observability_issues = [c.agent_name for c in problematic if
                                self.AGENT_ROLES.get(c.agent_name) == AgentRole.OBSERVABILITY]

        prompt = f"""System reliability analysis:

STATUS: {system_health.overall_status.value} | Risk: {system_health.systemic_risk}

BUSINESS IMPACT: {'CRITICAL' if business_critical else 'NONE'}
OBSERVABILITY: {'IMPAIRED' if observability_issues else 'HEALTHY'}

Affected agents:
{business_critical and '🚨 Business: ' + ', '.join(business_critical) or ''}
{observability_issues and '⚠️ Monitoring: ' + ', '.join(observability_issues) or ''}

Answer briefly:
1. Primary impact domain? (Business / Observability / None)
2. Recommended first action?
3. Can business continue? (YES / PARTIALLY / NO)

Max 100 words."""

        try:
            return (await self.llm.chat(prompt)).strip()
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return "AI analysis unavailable."

    @metric_counter("health_monitor")
    async def diagnose_agent(self, agent_name: str, url: str) -> Dict[str, Any]:
        reasoning: List[ReasoningStep] = []
        self._next_step(reasoning, f"Diagnosing {agent_name}")

        is_reachable, response_time, data = await self._check_agent_reachability(agent_name, url)
        trend = self.health_history.calculate_trend(agent_name)

        return {
            "agent_name": agent_name,
            "is_reachable": is_reachable,
            "response_time_ms": response_time,
            "current_status": data if is_reachable else {"error": "unreachable"},
            "trend": asdict(trend) if trend else None,
            "role": self.AGENT_ROLES.get(agent_name, AgentRole.SPECIALIZED).value,
            "dependencies": self.DEPENDENCIES.get(agent_name, []),
            "reasoning": reasoning
        }

    @metric_counter("health_monitor")
    async def analyze_trends(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        if agent_name:
            trend = self.health_history.calculate_trend(agent_name)
            return {"agent": agent_name, "trend": asdict(trend) if trend else None}

        trends = {
            name: asdict(trend)
            for name in self.health_history.history
            if (trend := self.health_history.calculate_trend(name))
        }
        return {"trends": trends}

    @metric_counter("health_monitor")
    async def predict_failures(self, time_horizon_minutes: int = 60) -> Dict[str, Any]:
        reasoning: List[ReasoningStep] = []
        self._next_step(reasoning, "Predicting failures", input_data={"horizon": time_horizon_minutes})

        predictions = []
        for name in self.health_history.history:
            trend = self.health_history.calculate_trend(name)
            if trend and trend.trend in ["DEGRADING", "CRITICAL"]:
                predictions.append({
                    "agent": name,
                    "risk_level": trend.trend,
                    "forecast": trend.forecast,
                    "confidence": "HIGH" if trend.trend == "CRITICAL" else "MEDIUM"
                })

        self._next_step(reasoning, "Prediction complete", output_data={"at_risk": len(predictions)})

        return {
            "predictions": predictions,
            "horizon_minutes": time_horizon_minutes,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }
