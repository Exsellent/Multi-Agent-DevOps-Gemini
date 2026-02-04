import logging
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from statistics import mean
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
    """Agent roles for dependency analysis"""
    CORE = "core"  # Critical agents (planner, orchestrator)
    SPECIALIZED = "specialized"  # Domain-specific agents
    SUPPORT = "support"  # Monitoring, logging agents


@dataclass
class AgentMetrics:
    """Detailed agent metrics"""
    agent_name: str
    tasks_processed: int
    errors: int
    avg_response_time: float  # milliseconds
    p95_response_time: float
    cpu_usage: float  # percentage
    memory_usage: float  # percentage
    last_seen: datetime
    uptime_seconds: int


@dataclass
class HealthCheck:
    """Single health check result"""
    agent_name: str
    status: HealthStatus
    error_rate: float
    response_time_ms: float
    issues: List[str]
    metrics: Optional[AgentMetrics]
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
    systemic_risk: str  # LOW, MEDIUM, HIGH, CRITICAL
    cascade_risk: List[str]  # Agents at risk of cascade failure
    recommendations: List[str]
    timestamp: datetime


@dataclass
class HealthTrend:
    """Health trend analysis"""
    agent_name: str
    trend: str  # IMPROVING, STABLE, DEGRADING, CRITICAL
    error_rate_change: float  # percentage change
    response_time_change: float  # percentage change
    forecast: str  # Expected status in next hour


class HealthHistory:
    """Maintains historical health data for trend analysis"""

    def __init__(self, max_history: int = 100):
        self.history: Dict[str, deque] = {}
        self.max_history = max_history

    def add(self, agent_name: str, check: HealthCheck):
        """Add health check to history"""
        if agent_name not in self.history:
            self.history[agent_name] = deque(maxlen=self.max_history)
        self.history[agent_name].append(check)

    def get_recent(self, agent_name: str, count: int = 10) -> List[HealthCheck]:
        """Get recent health checks"""
        if agent_name not in self.history:
            return []
        return list(self.history[agent_name])[-count:]

    def calculate_trend(self, agent_name: str) -> Optional[HealthTrend]:
        """Calculate health trend for agent"""
        recent = self.get_recent(agent_name, 10)
        if len(recent) < 2:
            return None

        # Calculate error rate trend
        error_rates = [c.error_rate for c in recent]
        error_rate_change = ((error_rates[-1] - error_rates[0]) / max(error_rates[0], 0.01)) * 100

        # Calculate response time trend
        response_times = [c.response_time_ms for c in recent if c.response_time_ms > 0]
        if response_times:
            rt_change = ((response_times[-1] - response_times[0]) / max(response_times[0], 1)) * 100
        else:
            rt_change = 0.0

        # Determine trend
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


class HealthMonitorAgent(MCPAgent):
    """Advanced health monitoring agent with predictive capabilities"""

    # Agent dependency graph (who depends on whom)
    DEPENDENCIES = {
        "planner": [],
        "orchestrator": ["planner"],
        "progress": ["planner"],
        "risks": ["planner"],
        "digest": ["planner", "progress", "risks"],
        "architecture_intelligence": ["planner"],
        "image": [],
        "health_monitor": [],
        "llm_router": []
    }

    # Agent role classification
    AGENT_ROLES = {
        "planner": AgentRole.CORE,
        "orchestrator": AgentRole.CORE,
        "progress": AgentRole.SPECIALIZED,
        "risks": AgentRole.SPECIALIZED,
        "digest": AgentRole.SPECIALIZED,
        "architecture_intelligence": AgentRole.SPECIALIZED,
        "image": AgentRole.SPECIALIZED,
        "health_monitor": AgentRole.SUPPORT,
        "llm_router": AgentRole.SUPPORT
    }

    # Health thresholds
    ERROR_RATE_WARNING = 0.05  # 5%
    ERROR_RATE_CRITICAL = 0.15  # 15%
    RESPONSE_TIME_WARNING = 2000  # 2 seconds
    RESPONSE_TIME_CRITICAL = 5000  # 5 seconds

    def __init__(self):
        super().__init__("Health-Monitor")
        self.llm = LLMClient()
        self.health_history = HealthHistory()

        # Register tools
        self.register_tool("check_health", self.check_health)
        self.register_tool("diagnose_agent", self.diagnose_agent)
        self.register_tool("get_system_status", self.get_system_status)
        self.register_tool("analyze_trends", self.analyze_trends)
        self.register_tool("predict_failures", self.predict_failures)

        logger.info("HealthMonitorAgent initialized with advanced features")

    def _next_step(
            self,
            reasoning: List[ReasoningStep],
            description: str,
            input_data: Optional[Dict[str, Any]] = None,
            output_data: Optional[Dict[str, Any]] = None
    ):
        """Helper to add reasoning step with safe defaults"""
        reasoning.append(
            ReasoningStep(
                step_number=len(reasoning) + 1,
                description=description,
                input_data=input_data or {},
                output_data=output_data or {},
                agent="HealthMonitor"
            )
        )

    async def _check_agent_reachability(
            self,
            agent_name: str,
            url: str,
            timeout: float = 3.0
    ) -> Tuple[bool, Optional[float], Optional[Dict]]:
        """
        Check if agent is reachable and measure response time

        Returns: (is_reachable, response_time_ms, response_data)
        """
        try:
            start_time = time.time()
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.get(url)
                response_time = (time.time() - start_time) * 1000  # Convert to ms

                if resp.status_code == 200:
                    return True, response_time, resp.json()
                else:
                    return False, response_time, {"error": f"Status {resp.status_code}"}
        except Exception as e:
            return False, None, {"error": str(e)}

    def _classify_agent_health(
            self,
            agent_name: str,
            tasks_processed: int,
            errors: int,
            response_time: Optional[float] = None,
            is_reachable: bool = True
    ) -> HealthCheck:
        """
        Classify agent health based on metrics

        Uses multiple factors: error rate, response time, reachability
        """
        # Calculate error rate
        error_rate = errors / max(tasks_processed, 1)

        issues = []

        # Determine status based on multiple factors
        if not is_reachable:
            status = HealthStatus.CRITICAL
            issues.append("Agent unreachable")
        elif tasks_processed == 0:
            status = HealthStatus.UNKNOWN
            issues.append("No tasks processed yet")
        elif error_rate >= self.ERROR_RATE_CRITICAL:
            status = HealthStatus.CRITICAL
            issues.append(f"Critical error rate: {error_rate * 100:.1f}%")
        elif error_rate >= self.ERROR_RATE_WARNING:
            status = HealthStatus.WARNING
            issues.append(f"Elevated error rate: {error_rate * 100:.1f}%")
        elif response_time and response_time >= self.RESPONSE_TIME_CRITICAL:
            status = HealthStatus.CRITICAL
            issues.append(f"Critical response time: {response_time:.0f}ms")
        elif response_time and response_time >= self.RESPONSE_TIME_WARNING:
            status = HealthStatus.DEGRADED
            issues.append(f"Slow response time: {response_time:.0f}ms")
        else:
            status = HealthStatus.HEALTHY

        return HealthCheck(
            agent_name=agent_name,
            status=status,
            error_rate=error_rate,
            response_time_ms=response_time or 0.0,
            issues=issues,
            metrics=None,  # Will be populated if available
            timestamp=datetime.now(),
            is_reachable=is_reachable
        )

    def _detect_cascade_failures(
            self,
            health_checks: List[HealthCheck]
    ) -> List[str]:
        """
        Detect potential cascade failures based on dependency graph

        If a core agent fails, downstream agents may fail too
        """
        at_risk = []
        failed_agents = {
            check.agent_name for check in health_checks
            if check.status in [HealthStatus.CRITICAL, HealthStatus.WARNING]
        }

        # Check if any failed agent has dependents
        for agent_name in failed_agents:
            # Find agents that depend on this one
            for dependent, deps in self.DEPENDENCIES.items():
                if agent_name in deps and dependent not in failed_agents:
                    at_risk.append(dependent)

        return list(set(at_risk))

    def _assess_system_health(
            self,
            health_checks: List[HealthCheck]
    ) -> SystemHealth:
        """
        Assess overall system health

        Considers: individual agent status, cascade risks, core agent health
        """
        status_counts = {
            HealthStatus.HEALTHY: 0,
            HealthStatus.DEGRADED: 0,
            HealthStatus.WARNING: 0,
            HealthStatus.CRITICAL: 0,
            HealthStatus.UNKNOWN: 0
        }

        core_agents_critical = []

        for check in health_checks:
            status_counts[check.status] += 1

            # Track critical core agents
            role = self.AGENT_ROLES.get(check.agent_name, AgentRole.SPECIALIZED)
            if role == AgentRole.CORE and check.status == HealthStatus.CRITICAL:
                core_agents_critical.append(check.agent_name)

        # Detect cascade risks
        cascade_risk = self._detect_cascade_failures(health_checks)

        # Determine overall status
        if status_counts[HealthStatus.CRITICAL] > 0:
            if core_agents_critical:
                overall = HealthStatus.CRITICAL
                systemic_risk = "CRITICAL"
            else:
                overall = HealthStatus.WARNING
                systemic_risk = "HIGH"
        elif status_counts[HealthStatus.WARNING] > 2:
            overall = HealthStatus.WARNING
            systemic_risk = "MEDIUM"
        elif status_counts[HealthStatus.DEGRADED] > 0:
            overall = HealthStatus.DEGRADED
            systemic_risk = "LOW"
        else:
            overall = HealthStatus.HEALTHY
            systemic_risk = "LOW"

        # Generate recommendations
        recommendations = self._generate_system_recommendations(
            health_checks,
            core_agents_critical,
            cascade_risk
        )

        return SystemHealth(
            overall_status=overall,
            healthy_count=status_counts[HealthStatus.HEALTHY],
            degraded_count=status_counts[HealthStatus.DEGRADED],
            warning_count=status_counts[HealthStatus.WARNING],
            critical_count=status_counts[HealthStatus.CRITICAL],
            unknown_count=status_counts[HealthStatus.UNKNOWN],
            systemic_risk=systemic_risk,
            cascade_risk=cascade_risk,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _generate_system_recommendations(
            self,
            health_checks: List[HealthCheck],
            core_agents_critical: List[str],
            cascade_risk: List[str]
    ) -> List[str]:
        """Generate actionable system-level recommendations"""
        recommendations = []

        # Critical core agent failures
        if core_agents_critical:
            recommendations.append(
                f"IMMEDIATE: Core agents failing ({', '.join(core_agents_critical)}). "
                "System functionality severely impacted. Priority: restart/rollback."
            )

        # Cascade failure risk
        if cascade_risk:
            recommendations.append(
                f"WARNING: Cascade failure risk detected. Agents at risk: {', '.join(cascade_risk)}. "
                "Monitor closely and consider preemptive scaling."
            )

        # Multiple warnings
        warning_agents = [c.agent_name for c in health_checks if c.status == HealthStatus.WARNING]
        if len(warning_agents) > 2:
            recommendations.append(
                f"Multiple agents showing warnings ({len(warning_agents)} total). "
                "Possible systemic issue - check infrastructure (network, DB, resources)."
            )

        # Performance degradation
        slow_agents = [c.agent_name for c in health_checks
                       if c.response_time_ms > self.RESPONSE_TIME_WARNING]
        if len(slow_agents) > 1:
            recommendations.append(
                f"Performance degradation detected in {len(slow_agents)} agents. "
                "Check resource utilization and external dependencies."
            )

        if not recommendations:
            recommendations.append("System operating normally. Continue standard monitoring.")

        return recommendations

    @log_method
    @metric_counter("health_monitor")
    async def check_health(
            self,
            agents: Dict[str, str],
            include_metrics: bool = True
    ) -> Dict[str, Any]:

        reasoning: List[ReasoningStep] = []

        self._next_step(
            reasoning,
            "Initiated comprehensive health check",
            input_data={"agent_count": len(agents), "include_metrics": include_metrics}
        )

        # Step 1: Check reachability for all agents
        health_checks = []

        for agent_name, url in agents.items():
            is_reachable, response_time, data = await self._check_agent_reachability(
                agent_name, url
            )

            # Extract metrics if available
            tasks = data.get("tasks_processed", 0) if isinstance(data, dict) else 0
            errors = data.get("errors", 0) if isinstance(data, dict) else 0

            # Classify health
            check = self._classify_agent_health(
                agent_name=agent_name,
                tasks_processed=tasks,
                errors=errors,
                response_time=response_time,
                is_reachable=is_reachable
            )

            health_checks.append(check)

            # Add to history for trend analysis
            self.health_history.add(agent_name, check)

        self._next_step(
            reasoning,
            "Completed reachability checks",
            output_data={
                "agents_checked": len(health_checks),
                "reachable": sum(1 for c in health_checks if c.is_reachable)
            }
        )

        # Step 2: Assess overall system health
        system_health = self._assess_system_health(health_checks)

        self._next_step(
            reasoning,
            "Assessed system health",
            output_data={
                "overall_status": system_health.overall_status.value,
                "systemic_risk": system_health.systemic_risk
            }
        )

        # Step 3: Analyze trends (if enough history)
        trends = {}
        for check in health_checks:
            trend = self.health_history.calculate_trend(check.agent_name)
            if trend:
                trends[check.agent_name] = asdict(trend)

        if trends:
            self._next_step(
                reasoning,
                "Analyzed health trends",
                output_data={
                    "agents_with_trends": len(trends),
                    "degrading": sum(1 for t in trends.values() if t["trend"] == "DEGRADING")
                }
            )

        # Step 4: AI-powered analysis (only if issues detected)
        ai_analysis = None
        if system_health.overall_status != HealthStatus.HEALTHY:
            critical_data = {
                "system_status": system_health.overall_status.value,
                "critical_agents": [c.agent_name for c in health_checks
                                    if c.status == HealthStatus.CRITICAL],
                "warnings": [c.agent_name for c in health_checks
                             if c.status == HealthStatus.WARNING],
                "cascade_risk": system_health.cascade_risk,
                "trends": {name: data["trend"] for name, data in trends.items()
                           if data["trend"] in ["DEGRADING", "CRITICAL"]}
            }

            prompt = f"""Analyze this multi-agent system health crisis:

System Status: {critical_data['system_status']}
Critical Agents: {critical_data['critical_agents']}
Warning Agents: {critical_data['warnings']}
Cascade Risk: {critical_data['cascade_risk']}
Degrading Trends: {list(critical_data['trends'].keys())}

Provide:
1. Most likely root cause (infrastructure, code, or cascade)
2. Immediate action priority (which agent to fix first)
3. Risk of total system failure (LOW/MEDIUM/HIGH)

Keep under 150 words."""

            try:
                ai_analysis = await self.llm.chat(prompt)

                self._next_step(
                    reasoning,
                    "Generated AI root cause analysis",
                    output_data={"analysis_length": len(ai_analysis)}
                )
            except Exception as e:
                logger.error(f"AI analysis failed: {e}")
                self._next_step(
                    reasoning,
                    "AI analysis failed - using rule-based recommendations",
                    output_data={"error": str(e)}
                )

        # Step 5: Final compilation
        self._next_step(
            reasoning,
            "Health check completed",
            output_data={
                "total_agents": len(health_checks),
                "status": system_health.overall_status.value,
                "recommendations": len(system_health.recommendations)
            }
        )

        logger.info(
            "Health check completed",
            extra={
                "status": system_health.overall_status.value,
                "critical": system_health.critical_count
            }
        )

        return {
            "system_health": asdict(system_health),
            "agent_health_checks": [asdict(check) for check in health_checks],
            "trends": trends,
            "ai_analysis": ai_analysis,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }

    @log_method
    @metric_counter("health_monitor")
    async def diagnose_agent(
            self,
            agent_name: str,
            deep_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Deep diagnostics for a specific agent

        Args:
            agent_name: Agent to diagnose
            deep_analysis: Whether to include AI-powered analysis
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(
            reasoning,
            "Agent diagnostics initiated",
            input_data={"agent": agent_name, "deep_analysis": deep_analysis}
        )

        # Get recent history
        recent_checks = self.health_history.get_recent(agent_name, 20)

        if not recent_checks:
            return {
                "agent_name": agent_name,
                "error": "No health history available",
                "recommendation": "Agent may be new or hasn't been monitored yet",
                "reasoning": reasoning
            }

        self._next_step(
            reasoning,
            "Retrieved health history",
            output_data={"history_items": len(recent_checks)}
        )

        # Analyze trends
        trend = self.health_history.calculate_trend(agent_name)

        # Calculate statistics
        error_rates = [c.error_rate for c in recent_checks]
        response_times = [c.response_time_ms for c in recent_checks if c.response_time_ms > 0]

        stats = {
            "avg_error_rate": mean(error_rates) if error_rates else 0,
            "max_error_rate": max(error_rates) if error_rates else 0,
            "avg_response_time": mean(response_times) if response_times else 0,
            "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)]
            if response_times else 0
        }

        self._next_step(
            reasoning,
            "Calculated performance statistics",
            output_data=stats
        )

        # Generate specific recommendations
        recommendations = []
        current_status = recent_checks[-1].status

        if current_status == HealthStatus.CRITICAL:
            recommendations.extend([
                "IMMEDIATE: Restart agent or rollback to last stable version",
                "Check error logs for stack traces and exceptions",
                "Verify external dependencies (database, APIs, message queues)",
                "Consider circuit breaker to prevent cascading failures"
            ])
        elif current_status == HealthStatus.WARNING:
            recommendations.extend([
                "Increase monitoring frequency to every 30 seconds",
                "Review recent deployments or configuration changes",
                "Check for gradual resource exhaustion (memory leaks, file handles)",
                "Prepare rollback plan in case of further degradation"
            ])
        elif trend and trend.trend == "DEGRADING":
            recommendations.extend([
                "Trend analysis shows degradation - investigate before critical",
                "Check resource utilization patterns",
                "Review recent code changes",
                f"Forecast: {trend.forecast}"
            ])
        else:
            recommendations.append("Agent operating normally - maintain current monitoring")

        # Add role-specific recommendations
        role = self.AGENT_ROLES.get(agent_name, AgentRole.SPECIALIZED)
        if role == AgentRole.CORE and current_status != HealthStatus.HEALTHY:
            recommendations.insert(
                0,
                f"⚠️ CORE AGENT: {agent_name} failure impacts entire system. Priority: HIGH"
            )

        self._next_step(
            reasoning,
            "Generated recommendations",
            output_data={"recommendation_count": len(recommendations)}
        )

        # AI analysis if requested
        ai_diagnosis = None
        if deep_analysis and current_status != HealthStatus.HEALTHY:
            prompt = f"""Deep diagnostic analysis for agent: {agent_name}

Current Status: {current_status.value}
Error Rate: {stats['avg_error_rate'] * 100:.1f}% (max: {stats['max_error_rate'] * 100:.1f}%)
Response Time: {stats['avg_response_time']:.0f}ms (p95: {stats['p95_response_time']:.0f}ms)
Trend: {trend.trend if trend else 'UNKNOWN'}
Recent Issues: {', '.join(recent_checks[-1].issues)}

Provide specific diagnostic insights:
1. Most likely root cause category (code/infrastructure/dependency)
2. Specific areas to investigate
3. Estimated recovery time

Keep under 100 words."""

            try:
                ai_diagnosis = await self.llm.chat(prompt)
                self._next_step(
                    reasoning,
                    "Generated AI diagnostic analysis",
                    output_data={"analysis_length": len(ai_diagnosis)}
                )
            except Exception as e:
                logger.error(f"AI diagnosis failed: {e}")

        self._next_step(
            reasoning,
            "Diagnostics completed",
            output_data={"total_steps": len(reasoning)}
        )

        return {
            "agent_name": agent_name,
            "current_status": current_status.value,
            "statistics": stats,
            "trend": asdict(trend) if trend else None,
            "recommendations": recommendations,
            "ai_diagnosis": ai_diagnosis,
            "recent_history": [asdict(check) for check in recent_checks[-5:]],
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("health_monitor")
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Quick system status overview (lightweight)

        Returns summary without deep analysis
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(
            reasoning,
            "System status check initiated",
            input_data={"mode": "lightweight"}
        )

        # Get most recent checks for all agents
        all_agents = set()
        for agent_name in self.health_history.history.keys():
            all_agents.add(agent_name)

        if not all_agents:
            return {
                "overall_status": "UNKNOWN",
                "message": "No health data available yet",
                "reasoning": reasoning
            }

        recent_checks = []
        for agent_name in all_agents:
            checks = self.health_history.get_recent(agent_name, 1)
            if checks:
                recent_checks.append(checks[0])

        # Assess system health
        system_health = self._assess_system_health(recent_checks)

        self._next_step(
            reasoning,
            "System status assessed",
            output_data={
                "overall_status": system_health.overall_status.value,
                "agents_checked": len(recent_checks)
            }
        )

        return {
            "overall_status": system_health.overall_status.value,
            "healthy_agents": system_health.healthy_count,
            "degraded_agents": system_health.degraded_count,
            "warning_agents": system_health.warning_count,
            "critical_agents": system_health.critical_count,
            "systemic_risk": system_health.systemic_risk,
            "cascade_risk": system_health.cascade_risk,
            "top_recommendations": system_health.recommendations[:3],
            "timestamp": system_health.timestamp.isoformat(),
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("health_monitor")
    async def analyze_trends(
            self,
            time_window_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Analyze health trends across all agents

        Args:
            time_window_minutes: Time window for analysis
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(
            reasoning,
            "Trend analysis initiated",
            input_data={"time_window_minutes": time_window_minutes}
        )

        cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

        trends = {}
        degrading_agents = []
        improving_agents = []

        for agent_name in self.health_history.history.keys():
            trend = self.health_history.calculate_trend(agent_name)
            if trend:
                trends[agent_name] = asdict(trend)

                if trend.trend in ["DEGRADING", "CRITICAL"]:
                    degrading_agents.append(agent_name)
                elif trend.trend == "IMPROVING":
                    improving_agents.append(agent_name)

        self._next_step(
            reasoning,
            "Trend analysis completed",
            output_data={
                "agents_analyzed": len(trends),
                "degrading": len(degrading_agents),
                "improving": len(improving_agents)
            }
        )

        # Generate insights
        insights = []
        if degrading_agents:
            insights.append(
                f"⚠️ {len(degrading_agents)} agents showing degradation: "
                f"{', '.join(degrading_agents)}"
            )
        if improving_agents:
            insights.append(
                f"✓ {len(improving_agents)} agents improving: "
                f"{', '.join(improving_agents)}"
            )
        if not insights:
            insights.append("All monitored agents showing stable trends")

        return {
            "time_window_minutes": time_window_minutes,
            "trends": trends,
            "degrading_agents": degrading_agents,
            "improving_agents": improving_agents,
            "insights": insights,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("health_monitor")
    async def predict_failures(
            self,
            forecast_minutes: int = 60
    ) -> Dict[str, Any]:
        """
        Predict potential failures based on current trends

        Args:
            forecast_minutes: How far ahead to forecast
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(
            reasoning,
            "Failure prediction initiated",
            input_data={"forecast_minutes": forecast_minutes}
        )

        at_risk_agents = []
        predictions = {}

        for agent_name in self.health_history.history.keys():
            trend = self.health_history.calculate_trend(agent_name)

            if trend and trend.trend in ["DEGRADING", "CRITICAL"]:
                # Calculate time to failure based on error rate change
                recent = self.health_history.get_recent(agent_name, 10)
                if recent:
                    current_error_rate = recent[-1].error_rate

                    # Estimate time to critical threshold
                    if trend.error_rate_change > 0:
                        error_rate_per_min = trend.error_rate_change / 60  # Per minute
                        remaining = (self.ERROR_RATE_CRITICAL - current_error_rate) * 100
                        time_to_critical = remaining / max(error_rate_per_min, 0.1)
                    else:
                        time_to_critical = float('inf')

                    if time_to_critical < forecast_minutes:
                        at_risk_agents.append(agent_name)
                        predictions[agent_name] = {
                            "estimated_time_to_failure_minutes": int(time_to_critical),
                            "current_error_rate": f"{current_error_rate * 100:.1f}%",
                            "trend": trend.trend,
                            "confidence": "HIGH" if len(recent) >= 5 else "MEDIUM"
                        }

        self._next_step(
            reasoning,
            "Failure prediction completed",
            output_data={
                "at_risk_agents": len(at_risk_agents),
                "forecast_minutes": forecast_minutes
            }
        )

        # Generate recommendations
        recommendations = []
        if at_risk_agents:
            recommendations.append(
                f"ALERT: {len(at_risk_agents)} agents predicted to fail within "
                f"{forecast_minutes} minutes"
            )
            for agent in at_risk_agents:
                pred = predictions[agent]
                recommendations.append(
                    f"- {agent}: ~{pred['estimated_time_to_failure_minutes']}min to failure "
                    f"(confidence: {pred['confidence']})"
                )
        else:
            recommendations.append(
                f"No failures predicted within {forecast_minutes} minutes"
            )

        return {
            "forecast_minutes": forecast_minutes,
            "at_risk_agents": at_risk_agents,
            "predictions": predictions,
            "recommendations": recommendations,
            "reasoning": reasoning
        }
