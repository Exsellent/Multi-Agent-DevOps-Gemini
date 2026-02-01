import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("risks_agent")


@dataclass
class SecurityRisk:
    """Detailed security risk assessment"""
    risk_id: str
    category: str  # "authentication", "authorization", "data_protection", "injection", "configuration"
    severity: str  # "critical", "high", "medium", "low"
    cwe_id: Optional[str]  # Common Weakness Enumeration ID
    description: str
    attack_vector: str
    potential_impact: str
    affected_components: List[str]
    remediation: str
    references: List[str]


@dataclass
class ComplianceIssue:
    """Compliance/regulatory concern"""
    framework: str  # "GDPR", "HIPAA", "PCI-DSS", "SOC2", etc.
    requirement: str
    current_gap: str
    risk_level: str
    remediation_steps: List[str]


def log_method(func):
    """Decorator for logging method calls"""

    async def wrapper(self, *args, **kwargs):
        logger.info(f"{func.__name__} called with args: {args}, kwargs: {kwargs}")
        try:
            result = await func(self, *args, **kwargs)
            logger.info(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            raise

    return wrapper


class RisksAgent(MCPAgent):
    """
    Specialized Security & Technical Risk Analysis Agent

    Focus: Deep security audits, vulnerability assessments, compliance checks

    Key Capabilities:
    1. Security vulnerability analysis (OWASP Top 10, CVE)
    2. Compliance assessment (GDPR, HIPAA, PCI-DSS, SOC2)
    3. Performance risk identification
    4. Technical debt assessment
    5. Dependency vulnerability scanning
    """

    # Conservative baseline risks for OAuth2 + JWT when LLM is unavailable
    OAUTH2_BASELINE_RISKS = [
        "Token leakage due to improper JWT storage (localStorage vulnerable to XSS)",
        "Insufficient token expiration and rotation policies",
        "Misconfigured OAuth scopes leading to over-privileged access",
        "Lack of refresh token revocation strategy on logout/compromise",
        "Improper validation of JWT signature or issuer (iss claim)",
        "Missing rate limiting on token endpoints (brute force attacks)",
        "Inadequate secret key management for JWT signing",
        "CSRF vulnerabilities in OAuth callback handling",
        "Token replay attacks without proper nonce/jti validation",
        "Exposure of sensitive data in JWT payload (not encrypted)"
    ]

    def __init__(self):
        super().__init__("Risks")
        self.llm = LLMClient()

        # Register specialized risk analysis tools
        self.register_tool("analyze_risks", self.analyze_risks)
        self.register_tool("security_audit", self.security_audit)
        self.register_tool("compliance_check", self.compliance_check)
        self.register_tool("performance_risk_analysis", self.performance_risk_analysis)
        self.register_tool("dependency_vulnerability_scan", self.dependency_vulnerability_scan)

        logger.info("Risks Agent initialized - specialized security & compliance analysis")

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    def _next_step(self, reasoning: List[ReasoningStep], description: str,
                   input_data: Optional[Dict] = None, output_data: Optional[Dict] = None):
        """Helper to add sequential reasoning steps"""
        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description=description,
            input_data=input_data or {},
            output_data=output_data or {}
        ))

    def _get_baseline_risks(self, feature: str) -> List[str]:
        """
        Return conservative baseline risks based on feature keywords.
        This ensures Risk Agent ALWAYS provides value, even without LLM.
        """
        feature_lower = feature.lower()

        # OAuth2/JWT specific risks
        if any(kw in feature_lower for kw in ["oauth", "jwt", "token", "auth"]):
            return self.OAUTH2_BASELINE_RISKS

        # Generic security baseline for unknown features
        return [
            "Insufficient input validation and sanitization",
            "Inadequate error handling and information disclosure",
            "Missing security testing and code review",
            "Lack of logging and monitoring for security events",
            "Potential dependency vulnerabilities in third-party libraries"
        ]

    # ========================================================================
    # MAIN RISK ANALYSIS TOOLS
    # ========================================================================

    @log_method
    @metric_counter("risks")
    async def analyze_risks(self, feature: str) -> Dict[str, Any]:
        """
        Comprehensive risk analysis for a feature

        Analyzes:
        - Security vulnerabilities
        - Compliance concerns
        - Performance risks
        - Technical debt impact
        - Operational risks
        """
        reasoning: List[ReasoningStep] = []
        fallback_used = False

        # Step 1: Request received
        self._next_step(reasoning, "Starting comprehensive risk analysis",
                        input_data={"feature": feature})

        # Step 2: Generate prompt
        prompt = f"""
You are a senior security engineer and risk analyst conducting a comprehensive risk assessment.

FEATURE TO ANALYZE: {feature}

Perform a DETAILED risk analysis covering:

1. SECURITY RISKS:
   - Authentication/Authorization vulnerabilities
   - Data protection concerns
   - Input validation issues
   - Injection vulnerabilities (SQL, XSS, CSRF, etc.)
   - Session management risks
   - API security concerns

2. COMPLIANCE RISKS:
   - GDPR implications (if handling EU user data)
   - Data privacy concerns
   - Audit trail requirements
   - Data retention policies

3. PERFORMANCE RISKS:
   - Scalability bottlenecks
   - Database query performance
   - Caching strategies
   - Resource exhaustion risks

4. TECHNICAL DEBT:
   - Code maintainability concerns
   - Dependency management
   - Testing coverage gaps
   - Documentation needs

5. OPERATIONAL RISKS:
   - Deployment complexity
   - Rollback strategy
   - Monitoring requirements
   - Incident response readiness

For EACH identified risk:
- Risk name and category
- Severity (Critical/High/Medium/Low)
- Detailed description
- Potential impact
- Specific remediation steps
- Priority (1-5, where 1 is highest)

Format: Use bullet points starting with '- ' for each risk.
Be thorough and specific. Cite security frameworks (OWASP, CWE) where applicable.
"""

        self._next_step(reasoning, "Generated comprehensive risk analysis prompt",
                        output_data={"prompt_length": len(prompt)})

        try:
            # Attempt LLM analysis
            analysis = await self.llm.chat(prompt)

            # Check if LLM response is valid
            if self._is_invalid_response(analysis):
                fallback_used = True
                detected_risks = self._get_baseline_risks(feature)
                analysis = (
                        f"⚠️ LLM analysis unavailable. Applied conservative baseline risk model.\n\n"
                        f"Baseline risks for {feature}:\n" +
                        "\n".join(f"- {risk}" for risk in detected_risks)
                )
                logger.warning("Risk Agent using baseline model",
                               extra={"feature": feature, "risks_count": len(detected_risks)})
            else:
                # Extract structured risks from analysis
                detected_risks = self._extract_risks_from_analysis(analysis)

                # Fallback if parsing failed
                if not detected_risks:
                    fallback_used = True
                    detected_risks = self._get_baseline_risks(feature)
                    analysis += f"\n\n⚠️ Supplemented with baseline risks:\n" + \
                                "\n".join(f"- {risk}" for risk in detected_risks)

            # Step 3: Analysis completed
            self._next_step(reasoning, "Completed comprehensive risk analysis",
                            output_data={
                                "risks_count": len(detected_risks),
                                "fallback_used": fallback_used
                            })

            # Step 4 (optional): Fallback annotation
            if fallback_used:
                self._next_step(reasoning, "Baseline risk model applied due to LLM unavailability",
                                output_data={"baseline_risks_count": len(detected_risks)})

            logger.info("Risk analysis completed",
                        extra={"feature": feature, "risks": len(detected_risks), "fallback": fallback_used})

            return {
                "feature": feature,
                "risk_analysis": analysis,
                "detected_risks": detected_risks,
                "fallback_used": fallback_used,
                "risk_summary": self._generate_risk_summary(detected_risks),
                "reasoning": reasoning,
                "recommendations": self._generate_recommendations(detected_risks)
            }

        except Exception as e:
            logger.error("Risk analysis failed", extra={"error": str(e)})

            # Use baseline risks even on exception
            baseline_risks = self._get_baseline_risks(feature)

            self._next_step(reasoning, "Risk analysis failed with exception — using baseline model",
                            output_data={"error": str(e), "fallback_used": True})

            self._next_step(reasoning, "Baseline risk model applied",
                            output_data={"risks_count": len(baseline_risks)})

            return {
                "feature": feature,
                "risk_analysis": f"⚠️ LLM analysis failed: {str(e)}\n\nBaseline risks applied:\n" +
                                 "\n".join(f"- {risk}" for risk in baseline_risks),
                "detected_risks": baseline_risks,
                "fallback_used": True,
                "reasoning": reasoning,
                "error": str(e)
            }

    @log_method
    @metric_counter("risks")
    async def security_audit(self, component: str, code_snippet: Optional[str] = None) -> Dict[str, Any]:
        """
        Deep security audit of a specific component

        Focuses on OWASP Top 10 and common vulnerabilities
        """
        reasoning: List[ReasoningStep] = []
        fallback_used = False

        self._next_step(reasoning, "Starting OWASP-based security audit",
                        input_data={"component": component, "has_code": code_snippet is not None})

        code_context = f"\n\nCODE SNIPPET:\n```\n{code_snippet}\n```" if code_snippet else ""

        audit_prompt = f"""
You are a security auditor performing an OWASP Top 10 vulnerability assessment.

COMPONENT: {component}{code_context}

Perform a detailed security audit checking for:

**A01 - Broken Access Control:**
- Missing authorization checks
- Insecure direct object references (IDOR)
- Privilege escalation vulnerabilities
- CORS misconfiguration

**A02 - Cryptographic Failures:**
- Weak encryption algorithms
- Hardcoded secrets/keys
- Insecure password storage
- Missing encryption for sensitive data

**A03 - Injection:**
- SQL injection vulnerabilities
- NoSQL injection
- Command injection
- XPath injection

**A04 - Insecure Design:**
- Missing rate limiting
- Lack of input validation
- Business logic flaws

**A05 - Security Misconfiguration:**
- Default credentials
- Unnecessary features enabled
- Missing security headers

For EACH vulnerability found:
- OWASP category
- Severity (Critical/High/Medium/Low)
- CWE ID (if applicable)
- Attack vector description
- Potential impact
- Specific remediation steps

Format: Use bullet points starting with '- ' for each vulnerability.
"""

        self._next_step(reasoning, "Generated OWASP Top 10 audit prompt",
                        output_data={"prompt_length": len(audit_prompt)})

        try:
            audit_result = await self.llm.chat(audit_prompt)

            if self._is_invalid_response(audit_result):
                fallback_used = True
                audit_result = "⚠️ Security audit unavailable – manual code review required."
                vulnerabilities = []
            else:
                vulnerabilities = self._extract_vulnerabilities(audit_result)

            self._next_step(reasoning, "Completed OWASP security audit",
                            output_data={
                                "vulnerabilities_found": len(vulnerabilities),
                                "fallback_used": fallback_used
                            })

            if fallback_used:
                self._next_step(reasoning, "Manual security audit recommended",
                                output_data={"reason": "LLM unavailable"})

            return {
                "component": component,
                "audit_result": audit_result,
                "vulnerabilities": vulnerabilities,
                "fallback_used": fallback_used,
                "severity_summary": self._categorize_by_severity(vulnerabilities),
                "remediation_priority": self._prioritize_remediation(vulnerabilities),
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Security audit failed", extra={"error": str(e)})

            self._next_step(reasoning, "Security audit failed with exception",
                            output_data={"error": str(e), "fallback_used": True})

            return {
                "component": component,
                "error": str(e),
                "fallback_used": True,
                "reasoning": reasoning
            }

    @log_method
    @metric_counter("risks")
    async def compliance_check(self, feature: str, frameworks: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compliance assessment against regulatory frameworks

        Supported frameworks: GDPR, HIPAA, PCI-DSS, SOC2, ISO 27001
        """
        reasoning: List[ReasoningStep] = []
        fallback_used = False

        if frameworks is None:
            frameworks = ["GDPR", "SOC2"]

        self._next_step(reasoning, "Starting compliance assessment",
                        input_data={"feature": feature, "frameworks": frameworks})

        compliance_prompt = f"""
You are a compliance officer assessing regulatory requirements.

FEATURE: {feature}
FRAMEWORKS TO CHECK: {', '.join(frameworks)}

For EACH framework, assess compliance requirements and current gaps.

Format: Use bullet points starting with '- ' for each compliance issue.
Include requirement, gap description, risk level, and remediation steps.
"""

        self._next_step(reasoning, "Generated compliance assessment prompt",
                        output_data={"frameworks_count": len(frameworks)})

        try:
            compliance_result = await self.llm.chat(compliance_prompt)

            if self._is_invalid_response(compliance_result):
                fallback_used = True
                compliance_result = "⚠️ Compliance check unavailable – manual review required."
                issues = []
            else:
                issues = self._extract_compliance_issues(compliance_result)

            self._next_step(reasoning, "Completed compliance assessment",
                            output_data={
                                "issues_found": len(issues),
                                "fallback_used": fallback_used
                            })

            if fallback_used:
                self._next_step(reasoning, "Manual compliance review recommended",
                                output_data={"frameworks": frameworks})

            return {
                "feature": feature,
                "frameworks": frameworks,
                "compliance_result": compliance_result,
                "issues": issues,
                "fallback_used": fallback_used,
                "compliance_score": self._calculate_compliance_score(issues),
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Compliance check failed", extra={"error": str(e)})

            self._next_step(reasoning, "Compliance check failed with exception",
                            output_data={"error": str(e), "fallback_used": True})

            return {
                "feature": feature,
                "error": str(e),
                "fallback_used": True,
                "reasoning": reasoning
            }

    @log_method
    @metric_counter("risks")
    async def performance_risk_analysis(self, feature: str, expected_load: Optional[str] = None) -> Dict[str, Any]:
        """
        Performance and scalability risk assessment
        """
        reasoning: List[ReasoningStep] = []
        fallback_used = False

        self._next_step(reasoning, "Starting performance risk analysis",
                        input_data={"feature": feature, "expected_load": expected_load})

        performance_prompt = f"""
You are a performance engineer analyzing potential bottlenecks.

FEATURE: {feature}
EXPECTED LOAD: {expected_load or "Standard production load"}

Analyze performance risks in these areas:

1. Database Performance (query optimization, N+1 problems, indexing)
2. Caching Strategy (invalidation, memory usage, cache stampede)
3. Resource Exhaustion (memory leaks, CPU, file handles)
4. Scalability Concerns (horizontal scaling, stateful components)
5. API Performance (response times, timeouts, retry logic)

Format: Use bullet points starting with '- ' for each risk.
Include severity, impact, and optimization recommendations.
"""

        self._next_step(reasoning, "Generated performance analysis prompt",
                        output_data={"expected_load": expected_load or "standard"})

        try:
            performance_result = await self.llm.chat(performance_prompt)

            if self._is_invalid_response(performance_result):
                fallback_used = True
                performance_result = "⚠️ Performance analysis unavailable – load testing recommended."
                perf_risks = []
            else:
                perf_risks = self._extract_risks_from_analysis(performance_result)

            self._next_step(reasoning, "Completed performance risk analysis",
                            output_data={
                                "performance_risks": len(perf_risks),
                                "fallback_used": fallback_used
                            })

            if fallback_used:
                self._next_step(reasoning, "Load testing and profiling recommended",
                                output_data={"reason": "LLM unavailable"})

            return {
                "feature": feature,
                "expected_load": expected_load,
                "performance_analysis": performance_result,
                "performance_risks": perf_risks,
                "fallback_used": fallback_used,
                "optimization_recommendations": self._extract_optimizations(performance_result),
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Performance risk analysis failed", extra={"error": str(e)})

            self._next_step(reasoning, "Performance analysis failed with exception",
                            output_data={"error": str(e), "fallback_used": True})

            return {
                "feature": feature,
                "error": str(e),
                "fallback_used": True,
                "reasoning": reasoning
            }

    @log_method
    @metric_counter("risks")
    async def dependency_vulnerability_scan(self, dependencies: Optional[List[str]] = None,
                                            package_manager: str = "npm") -> Dict[str, Any]:
        """
        Dependency vulnerability scanning simulation
        """
        reasoning: List[ReasoningStep] = []
        fallback_used = False

        if dependencies is None:
            dependencies = ["example-package@1.0.0"]

        self._next_step(reasoning, "Starting dependency vulnerability scan",
                        input_data={
                            "package_manager": package_manager,
                            "deps_count": len(dependencies)
                        })

        scan_prompt = f"""
You are a security scanner analyzing software dependencies.

PACKAGE MANAGER: {package_manager}
DEPENDENCIES: {', '.join(dependencies)}

For EACH dependency, assess:
1. Known Vulnerabilities (CVE identifiers, severity, fixed versions)
2. Security Advisories (GitHub, npm/PyPI notices)
3. License Risks (type, compatibility, legal implications)
4. Maintenance Status (last update, active maintainers, deprecation)
5. Transitive Dependencies (vulnerable sub-dependencies)

Format: Use bullet points starting with '- ' for each vulnerability.
Include CVE ID, severity, and remediation.
"""

        self._next_step(reasoning, "Generated dependency scan prompt",
                        output_data={"dependencies_count": len(dependencies)})

        try:
            scan_result = await self.llm.chat(scan_prompt)

            if self._is_invalid_response(scan_result):
                fallback_used = True
                scan_result = "⚠️ Dependency scan unavailable – use npm audit or Snyk."
                vulnerabilities = []
            else:
                vulnerabilities = self._extract_vulnerabilities(scan_result)

            self._next_step(reasoning, "Completed dependency vulnerability scan",
                            output_data={
                                "vulnerabilities_found": len(vulnerabilities),
                                "fallback_used": fallback_used
                            })

            if fallback_used:
                self._next_step(reasoning, "Manual dependency audit recommended",
                                output_data={"tools": ["npm audit", "Snyk", "Dependabot"]})

            return {
                "package_manager": package_manager,
                "dependencies_scanned": len(dependencies),
                "scan_result": scan_result,
                "vulnerabilities": vulnerabilities,
                "fallback_used": fallback_used,
                "action_plan": self._generate_dependency_action_plan(vulnerabilities),
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("Dependency scan failed", extra={"error": str(e)})

            self._next_step(reasoning, "Dependency scan failed with exception",
                            output_data={"error": str(e), "fallback_used": True})

            return {
                "package_manager": package_manager,
                "error": str(e),
                "fallback_used": True,
                "reasoning": reasoning
            }

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _extract_risks_from_analysis(self, analysis: str) -> List[str]:
        """Extract risk descriptions from analysis text"""
        risks = []
        for line in analysis.splitlines():
            line = line.strip()
            if line.startswith("- ") or line.startswith("* ") or line.startswith("• "):
                risks.append(line.lstrip("-*• ").strip())
        return risks[:20]  # Limit to top 20 risks

    def _extract_vulnerabilities(self, audit_result: str) -> List[str]:
        """Extract vulnerability descriptions from audit"""
        return self._extract_risks_from_analysis(audit_result)

    def _extract_compliance_issues(self, compliance_result: str) -> List[str]:
        """Extract compliance issues from assessment"""
        return self._extract_risks_from_analysis(compliance_result)

    def _extract_optimizations(self, performance_result: str) -> List[str]:
        """Extract performance optimization recommendations"""
        return self._extract_risks_from_analysis(performance_result)

    def _generate_risk_summary(self, risks: List[str]) -> str:
        """Generate executive summary of risks"""
        if not risks:
            return "No significant risks identified."
        return f"Identified {len(risks)} risk areas requiring attention. Priority review recommended for security and compliance concerns."

    def _generate_recommendations(self, risks: List[str]) -> List[str]:
        """Generate actionable recommendations"""
        if not risks:
            return ["Continue with current approach", "Maintain security best practices"]
        return [
            "Address critical and high severity risks before deployment",
            "Implement recommended security controls",
            "Schedule follow-up security audit after remediation",
            "Document all security decisions in ADRs"
        ]

    def _categorize_by_severity(self, vulnerabilities: List[str]) -> Dict[str, int]:
        """Categorize vulnerabilities by severity"""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for vuln in vulnerabilities:
            vuln_lower = vuln.lower()
            if any(word in vuln_lower for word in ["critical", "severe", "dangerous"]):
                severity_counts["critical"] += 1
            elif any(word in vuln_lower for word in ["high", "important", "major"]):
                severity_counts["high"] += 1
            elif any(word in vuln_lower for word in ["medium", "moderate"]):
                severity_counts["medium"] += 1
            else:
                severity_counts["low"] += 1
        return severity_counts

    def _prioritize_remediation(self, vulnerabilities: List[str]) -> List[str]:
        """Prioritize remediation actions"""
        if not vulnerabilities:
            return []
        return [
            "1. Fix critical vulnerabilities immediately",
            "2. Address high severity issues within 1 week",
            "3. Plan medium severity fixes in next sprint",
            "4. Review low severity items for future iterations"
        ]

    def _calculate_compliance_score(self, issues: List[str]) -> float:
        """Calculate compliance score (0-100)"""
        if not issues:
            return 100.0
        base_score = 100.0
        deduction_per_issue = 100.0 / max(len(issues), 1)
        return max(0.0, base_score - (len(issues) * deduction_per_issue * 0.5))

    def _generate_dependency_action_plan(self, vulnerabilities: List[str]) -> List[str]:
        """Generate action plan for dependency vulnerabilities"""
        if not vulnerabilities:
            return ["Dependencies are up to date", "Continue monitoring for new advisories"]
        return [
            "Update vulnerable dependencies to patched versions",
            "Review transitive dependencies",
            "Enable automated dependency scanning (Dependabot/Snyk)",
            "Establish dependency update policy"
        ]
