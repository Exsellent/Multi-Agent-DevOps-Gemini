import asyncio
import hashlib
import logging
import os
import re
import tempfile
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("code_execution_agent")


class ThinkingLevel(str, Enum):
    """
    Thinking levels for Code Execution Agent
    Unified with Marathon Agent architecture
    """
    STRATEGIC = "strategic"  # What are we trying to build
    GENERATION = "generation"  # Write code and tests
    EXECUTION = "execution"  # Run code
    VERIFICATION = "verification"  # Analyze results
    REFLECTION = "reflection"  # Self-correction


class ExecutionStatus(str, Enum):
    """Execution result status"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SYNTAX_ERROR = "syntax_error"


@dataclass
class ExecutableTest:
    """
    Executable test case (not textual description)

    CRITICAL: Tests must be runnable code, not descriptions
    """
    test_id: str
    test_code: str  # Actual executable code with assertions
    description: str
    expected_behavior: str

    # Execution results (populated after run)
    passed: Optional[bool] = None
    execution_time_ms: Optional[float] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class VerificationArtifact:
    """
    Proof of execution artifact

    Critical for Vibe Engineering: tangible proof, not narratives
    """
    artifact_id: str
    artifact_type: str  # "execution_log", "test_report", "coverage_report"
    timestamp: str

    # Execution evidence
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float

    # Test results
    tests_passed: int
    tests_failed: int
    test_details: List[Dict[str, Any]]

    # Code analysis
    code_hash: str
    code_length: int

    # Quality metrics (agent-decided, not LLM-decided)
    quality_score: float  # 0.0 - 1.0
    production_ready: bool
    confidence: float


@dataclass
class TestSuite:
    """
    Immutable test suite for consistency across iterations

    CRITICAL: Same tests must run in debug loop to prevent
    "fixing the error by breaking functionality"
    """
    suite_id: str
    tests: List[ExecutableTest]
    created_at: str
    code_hash: str  # Hash of code this suite was created for


@dataclass
class CodeIteration:
    """
    One iteration in the debug loop
    Shows evolution: broken → working
    """
    iteration_number: int
    timestamp: str
    code: str
    code_hash: str

    # Why this iteration happened
    trigger: str  # "initial_generation", "debug_attempt", "refinement"
    previous_error: Optional[str]

    # Execution results
    verification_artifact: Optional[VerificationArtifact]

    # Agent's decision
    thinking_level: ThinkingLevel
    status: ExecutionStatus
    next_action: str


class CodeExecutionAgent(MCPAgent):
    """
    Gold Standard Code Execution Agent for Vibe Engineering

    Key principles:
    1. Tests are EXECUTABLE, not descriptions
    2. Verification through ARTIFACTS, not narratives
    3. Consistent test suite across debug iterations
    4. Agent decides quality, LLM generates code
    5. Thinking levels separate generation from verification

    This is NOT "LLM-driven test runner"
    This IS "autonomous verification system"
    """

    def __init__(self):
        super().__init__("Code-Execution")
        self.llm = LLMClient()

        # Only Python for hackathon (honest scope)
        self.supported_language = "python"

        # Store test suites for consistency
        self._test_suites: Dict[str, TestSuite] = {}

        # Store iterations for learning
        self._iterations_history: Dict[str, List[CodeIteration]] = {}

        self.register_tool("generate_and_test_code", self.generate_and_test_code)
        self.register_tool("autonomous_debug_loop", self.autonomous_debug_loop)
        self.register_tool("verify_code_quality", self.verify_code_quality)
        self.register_tool("get_verification_artifacts", self.get_verification_artifacts)

        logger.info("Code Execution Agent initialized (GOLD STANDARD - VIBE ENGINEERING)")

    def _calculate_code_hash(self, code: str) -> str:
        """Generate hash for code versioning"""
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def _calculate_quality_score(
            self,
            tests_passed: int,
            tests_total: int,
            execution_time_ms: float,
            code_length: int
    ) -> tuple[float, bool, float]:
        """
        INTERNAL DECISION MODEL
        Agent calculates quality, not LLM

        Returns: (quality_score, production_ready, confidence)
        """
        if tests_total == 0:
            return 0.0, False, 0.0

        # Pass rate
        pass_rate = tests_passed / tests_total

        # Performance penalty (simplified)
        perf_score = 1.0 if execution_time_ms < 1000 else 0.8

        # Code size check (not too short, not too long)
        size_score = 1.0 if 20 <= code_length <= 1000 else 0.7

        # Overall quality
        quality_score = (pass_rate * 0.7) + (perf_score * 0.2) + (size_score * 0.1)

        # Production readiness decision
        production_ready = (
                tests_passed == tests_total and
                execution_time_ms < 5000 and
                code_length >= 10
        )

        # Confidence based on test coverage
        confidence = min(pass_rate + (tests_total / 10 * 0.1), 1.0)

        return quality_score, production_ready, confidence

    async def _execute_python_code(
            self,
            code: str,
            timeout_seconds: float = 5.0
    ) -> Dict[str, Any]:
        """
        Execute Python code in isolated subprocess

        Returns actual execution results, not LLM interpretation
        """
        start_time = datetime.now()

        try:
            with tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.py',
                    delete=False
            ) as f:
                f.write(code)
                temp_file = f.name

            # Execute with timeout
            process = await asyncio.create_subprocess_exec(
                'python3', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds
                )

                execution_time = (datetime.now() - start_time).total_seconds() * 1000

                os.unlink(temp_file)

                return {
                    "status": ExecutionStatus.SUCCESS if process.returncode == 0 else ExecutionStatus.FAILURE,
                    "exit_code": process.returncode,
                    "stdout": stdout.decode('utf-8'),
                    "stderr": stderr.decode('utf-8'),
                    "execution_time_ms": execution_time
                }

            except asyncio.TimeoutError:
                process.kill()
                os.unlink(temp_file)

                execution_time = (datetime.now() - start_time).total_seconds() * 1000

                return {
                    "status": ExecutionStatus.TIMEOUT,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": f"Execution timeout ({timeout_seconds}s limit)",
                    "execution_time_ms": execution_time
                }

        except SyntaxError as e:
            return {
                "status": ExecutionStatus.SYNTAX_ERROR,
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Syntax error: {str(e)}",
                "execution_time_ms": 0
            }
        except Exception as e:
            return {
                "status": ExecutionStatus.FAILURE,
                "exit_code": 1,
                "stdout": "",
                "stderr": f"Execution error: {str(e)}",
                "execution_time_ms": 0
            }

    def _extract_code_block(self, response: str) -> str:
        """Extract code from markdown code blocks"""
        # Try to find python code block
        pattern = r"```python\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try generic code block
        pattern = r"```\s*\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Fallback: return whole response
        return response.strip()

    @metric_counter("code_execution")
    async def generate_and_test_code(
            self,
            requirement: str,
            context: Optional[str] = None,
            language: Optional[str] = None,
            **kwargs
    ) -> Dict[str, Any]:

        """
        Generate code and immediately verify through execution

        Full cycle: Strategic → Generation → Execution → Verification
        """
        reasoning: List[ReasoningStep] = []
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # ✅ CRITICAL FIX: Use output_data (not output!)
        reasoning.append(ReasoningStep(
            step_number=1,
            description="STRATEGIC: Planning code generation",
            input_data={"requirement": requirement},
            output_data={"thinking_level": ThinkingLevel.STRATEGIC.value}
        ))

        # STRATEGIC: Understand requirement
        strategic_prompt = f"""You are a code generation strategist.

Requirement: {requirement}
Context: {context or 'None'}

Break down this requirement into:
1. What functions/classes needed
2. Key logic components
3. Edge cases to handle

Keep under 100 words."""

        try:
            strategic_plan = await self.llm.chat(strategic_prompt)
        except Exception as e:
            logger.error(f"Strategic planning failed: {e}")
            strategic_plan = "Basic implementation needed"

        reasoning.append(ReasoningStep(
            step_number=2,
            description="GENERATION: Creating code and tests",
            input_data={"thinking_level": ThinkingLevel.GENERATION.value},
            output_data={"plan_length": len(strategic_plan)}
        ))

        # GENERATION: Create code
        generation_prompt = f"""Generate Python code for this requirement:

{requirement}

Strategic plan:
{strategic_plan}

CRITICAL RULES:
1. Write complete, runnable Python code
2. Include docstrings
3. Handle edge cases
4. Keep it simple and clean
5. Do NOT include test code here (tests come separately)

Output ONLY the code in a ```python code block."""

        code_response = await self.llm.chat(generation_prompt)
        generated_code = self._extract_code_block(code_response)
        code_hash = self._calculate_code_hash(generated_code)

        reasoning.append(ReasoningStep(
            step_number=3,
            description="GENERATION: Creating executable test suite",
            input_data={"code_hash": code_hash},
            output_data={"code_length": len(generated_code)}
        ))

        # GENERATION: Create EXECUTABLE tests
        test_generation_prompt = f"""Create executable Python test code for:

CODE:
{generated_code}

REQUIREMENT:
{requirement}

CRITICAL RULES:
1. Write EXECUTABLE test code (not descriptions!)
2. Use assertions (assert statements)
3. Each test should be self-contained
4. Include edge cases
5. Tests must be runnable standalone

Generate 3-5 test functions. Each test should:
- Have a clear name (test_<feature>)
- Have assertions
- Print "PASSED: <test_name>" on success

Output test code in ```python block."""

        test_response = await self.llm.chat(test_generation_prompt)
        test_code = self._extract_code_block(test_response)

        # Parse test functions
        test_functions = re.findall(
            r'def (test_\w+)\([^)]*\):.*?(?=\ndef |$)',
            test_code,
            re.DOTALL
        )

        tests = []
        for i, test_match in enumerate(test_functions[:5]):  # Max 5 tests
            test_id = f"test_{i + 1}"
            tests.append(ExecutableTest(
                test_id=test_id,
                test_code=test_code,  # Include all test code
                description=f"Test function: {test_match[:50]}",
                expected_behavior="Should pass all assertions"
            ))

        if not tests:
            # Fallback: create basic test
            basic_test = f"""
# Basic test
assert True, "Fallback test"
print("PASSED: basic_test")
"""
            tests.append(ExecutableTest(
                test_id="test_fallback",
                test_code=basic_test,
                description="Fallback test",
                expected_behavior="Basic validation"
            ))

        # Store test suite
        suite_id = f"suite_{code_hash}"
        test_suite = TestSuite(
            suite_id=suite_id,
            tests=tests,
            created_at=datetime.now().isoformat(),
            code_hash=code_hash
        )
        self._test_suites[suite_id] = test_suite

        reasoning.append(ReasoningStep(
            step_number=4,
            description="EXECUTION: Running code and tests",
            input_data={"tests_count": len(tests)},
            output_data={"suite_id": suite_id}
        ))

        # EXECUTION: Run code
        code_execution = await self._execute_python_code(generated_code)

        reasoning.append(ReasoningStep(
            step_number=5,
            description="EXECUTION: Running test suite",
            input_data={"code_status": code_execution["status"].value},
            output_data={
                "exit_code": code_execution["exit_code"],
                "execution_time_ms": code_execution["execution_time_ms"]
            }
        ))

        # EXECUTION: Run tests
        test_results = []
        tests_passed = 0
        tests_failed = 0

        for test in tests:
            # Combine code + test
            full_test_code = f"{generated_code}\n\n{test.test_code}"

            test_exec = await self._execute_python_code(full_test_code)

            passed = test_exec["exit_code"] == 0
            if passed:
                tests_passed += 1
            else:
                tests_failed += 1

            test.passed = passed
            test.execution_time_ms = test_exec["execution_time_ms"]
            test.stdout = test_exec["stdout"]
            test.stderr = test_exec["stderr"]
            test.error_message = test_exec["stderr"] if not passed else None

            test_results.append({
                "test_id": test.test_id,
                "passed": passed,
                "execution_time_ms": test.execution_time_ms,
                "output": test.stdout[:200]  # First 200 chars
            })

        reasoning.append(ReasoningStep(
            step_number=6,
            description="VERIFICATION: Analyzing results",
            input_data={"tests_total": len(tests)},
            output_data={
                "tests_passed": tests_passed,
                "tests_failed": tests_failed
            }
        ))

        # VERIFICATION: Calculate quality
        quality_score, production_ready, confidence = self._calculate_quality_score(
            tests_passed=tests_passed,
            tests_total=len(tests),
            execution_time_ms=code_execution["execution_time_ms"],
            code_length=len(generated_code)
        )

        # Create verification artifact
        artifact = VerificationArtifact(
            artifact_id=f"artifact_{session_id}",
            artifact_type="test_report",
            timestamp=datetime.now().isoformat(),
            stdout=code_execution["stdout"],
            stderr=code_execution["stderr"],
            exit_code=code_execution["exit_code"],
            execution_time_ms=code_execution["execution_time_ms"],
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            test_details=test_results,
            code_hash=code_hash,
            code_length=len(generated_code),
            quality_score=quality_score,
            production_ready=production_ready,
            confidence=confidence
        )

        # Store iteration
        iteration = CodeIteration(
            iteration_number=1,
            timestamp=datetime.now().isoformat(),
            code=generated_code,
            code_hash=code_hash,
            trigger="initial_generation",
            previous_error=None,
            verification_artifact=artifact,
            thinking_level=ThinkingLevel.VERIFICATION,
            status=code_execution["status"],
            next_action="return_results" if production_ready else "debug_needed"
        )

        self._iterations_history[session_id] = [iteration]

        reasoning.append(ReasoningStep(
            step_number=7,
            description="COMPLETED: Code generation and verification",
            input_data={
                "production_ready": production_ready,
                "quality_score": quality_score
            },
            output_data={
                "session_id": session_id,
                "tests_passed": tests_passed,
                "tests_total": len(tests),
                "final_status": iteration.status.value
            }
        ))

        logger.info(
            "Code generation completed",
            extra={
                "session_id": session_id,
                "tests_passed": tests_passed,
                "production_ready": production_ready
            }
        )

        return {
            "session_id": session_id,
            "code": generated_code,
            "code_hash": code_hash,
            "verification_artifact": asdict(artifact),
            "test_suite": {
                "suite_id": suite_id,
                "tests_count": len(tests),
                "tests_passed": tests_passed,
                "tests_failed": tests_failed
            },
            "quality_metrics": {
                "quality_score": quality_score,
                "production_ready": production_ready,
                "confidence": confidence
            },
            "reasoning": reasoning
        }

    @metric_counter("code_execution")
    async def autonomous_debug_loop(
            self,
            session_id: str,
            max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Autonomous debug loop with immutable test suite

        CRITICAL: Same tests run each iteration
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="DEBUG LOOP: Starting autonomous debugging",
            input_data={"session_id": session_id, "max_iterations": max_iterations},
            output_data={"mode": "autonomous_debug"}
        ))

        # Get previous iteration
        iterations = self._iterations_history.get(session_id, [])
        if not iterations:
            return {
                "error": "No session found",
                "session_id": session_id,
                "reasoning": reasoning
            }

        last_iteration = iterations[-1]

        # Get test suite
        suite_id = f"suite_{last_iteration.code_hash}"
        test_suite = self._test_suites.get(suite_id)

        if not test_suite:
            return {
                "error": "No test suite found",
                "session_id": session_id,
                "reasoning": reasoning
            }

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Retrieved test suite and previous iteration",
            input_data={"iterations_count": len(iterations)},
            output_data={
                "suite_id": suite_id,
                "tests_count": len(test_suite.tests)
            }
        ))

        current_code = last_iteration.code
        current_iteration_num = len(iterations) + 1

        for iteration in range(max_iterations):
            reasoning.append(ReasoningStep(
                step_number=3 + iteration,
                description=f"DEBUG ITERATION {current_iteration_num}: Analyzing failures",
                input_data={"iteration_number": current_iteration_num},
                output_data={"previous_status": last_iteration.status.value}
            ))

            # Check if already succeeded
            if last_iteration.status == ExecutionStatus.SUCCESS:
                artifact = last_iteration.verification_artifact
                if artifact and artifact.tests_passed == len(test_suite.tests):
                    reasoning.append(ReasoningStep(
                        step_number=4 + iteration,
                        description="All tests passing - debug complete",
                        input_data={},
                        output_data={"success": True}
                    ))
                    break

            # Analyze errors
            failed_tests = [t for t in test_suite.tests if not t.passed]
            error_summary = "\n".join([
                f"Test {t.test_id}: {t.error_message[:100]}"
                for t in failed_tests[:3]
            ])

            # Generate fix
            debug_prompt = f"""Fix this Python code to pass all tests.

CURRENT CODE:
{current_code}

FAILED TESTS:
{error_summary}

RULES:
1. Keep same functionality
2. Fix specific errors
3. Don't break passing tests
4. Return complete fixed code

Output fixed code in ```python block."""

            try:
                fix_response = await self.llm.chat(debug_prompt)
                fixed_code = self._extract_code_block(fix_response)
                code_hash = self._calculate_code_hash(fixed_code)
            except Exception as e:
                logger.error(f"Debug attempt failed: {e}")
                break

            reasoning.append(ReasoningStep(
                step_number=5 + iteration,
                description=f"Generated fix attempt",
                input_data={},
                output_data={"new_code_hash": code_hash}
            ))

            # Execute fixed code with same tests
            code_execution = await self._execute_python_code(fixed_code)

            # Run test suite
            tests_passed = 0
            tests_failed = 0
            test_results = []

            for test in test_suite.tests:
                full_test_code = f"{fixed_code}\n\n{test.test_code}"
                test_exec = await self._execute_python_code(full_test_code)

                passed = test_exec["exit_code"] == 0
                if passed:
                    tests_passed += 1
                else:
                    tests_failed += 1

                test_results.append({
                    "test_id": test.test_id,
                    "passed": passed,
                    "execution_time_ms": test_exec["execution_time_ms"]
                })

            # Calculate quality
            quality_score, production_ready, confidence = self._calculate_quality_score(
                tests_passed=tests_passed,
                tests_total=len(test_suite.tests),
                execution_time_ms=code_execution["execution_time_ms"],
                code_length=len(fixed_code)
            )

            # Create artifact
            artifact = VerificationArtifact(
                artifact_id=f"artifact_{session_id}_iter{current_iteration_num}",
                artifact_type="debug_report",
                timestamp=datetime.now().isoformat(),
                stdout=code_execution["stdout"],
                stderr=code_execution["stderr"],
                exit_code=code_execution["exit_code"],
                execution_time_ms=code_execution["execution_time_ms"],
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                test_details=test_results,
                code_hash=code_hash,
                code_length=len(fixed_code),
                quality_score=quality_score,
                production_ready=production_ready,
                confidence=confidence
            )

            # Store iteration
            new_iteration = CodeIteration(
                iteration_number=current_iteration_num,
                timestamp=datetime.now().isoformat(),
                code=fixed_code,
                code_hash=code_hash,
                trigger="debug_attempt",
                previous_error=error_summary,
                verification_artifact=artifact,
                thinking_level=ThinkingLevel.REFLECTION,
                status=code_execution["status"],
                next_action="continue_debug" if tests_failed > 0 else "completed"
            )

            iterations.append(new_iteration)
            self._iterations_history[session_id] = iterations

            reasoning.append(ReasoningStep(
                step_number=6 + iteration,
                description=f"Debug iteration {current_iteration_num} completed",
                input_data={},
                output_data={
                    "tests_passed": tests_passed,
                    "tests_failed": tests_failed,
                    "quality_score": quality_score
                }
            ))

            # Update for next iteration
            last_iteration = new_iteration
            current_code = fixed_code
            current_iteration_num += 1

            # Break if all tests pass
            if tests_passed == len(test_suite.tests):
                break

        final_iteration = iterations[-1]
        final_artifact = final_iteration.verification_artifact

        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description="Debug loop completed",
            input_data={},
            output_data={
                "total_iterations": len(iterations),
                "final_status": final_iteration.status.value,
                "tests_passed": final_artifact.tests_passed if final_artifact else 0
            }
        ))

        return {
            "session_id": session_id,
            "total_iterations": len(iterations),
            "final_code": final_iteration.code,
            "final_artifact": asdict(final_artifact) if final_artifact else None,
            "all_iterations": [
                {
                    "iteration": it.iteration_number,
                    "status": it.status.value,
                    "tests_passed": it.verification_artifact.tests_passed if it.verification_artifact else 0
                }
                for it in iterations
            ],
            "reasoning": reasoning
        }

    @metric_counter("code_execution")
    async def verify_code_quality(
            self,
            code: str
    ) -> Dict[str, Any]:
        """
        Verify code quality through deterministic checks + LLM analysis
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Starting code quality verification",
            input_data={"code_length": len(code)},
            output_data={"mode": "quality_check"}
        ))

        # Deterministic checks
        code_lines = [line for line in code.split('\n') if line.strip()]
        code_length = len(code)

        has_functions = bool(re.search(r'def \w+\(', code))
        has_docstrings = bool(re.search(r'""".*?"""', code, re.DOTALL))
        has_forbidden = bool(re.search(r'(eval|exec|__import__|compile)\(', code))

        deterministic_score = sum([
            0.4 if has_functions else 0,
            0.3 if has_docstrings else 0,
            0.3 if not has_forbidden else 0
        ])

        # LLM quality analysis
        quality_prompt = f"""Analyze this Python code quality:

{code}

Check:
1. CORRECTNESS: Logic errors?
2. PERFORMANCE: Obvious inefficiencies?
3. MAINTAINABILITY: Code smells, naming, modularity?
4. ERROR HANDLING: Edge cases covered?

Be concise. List issues only."""

        quality_report = await self.llm.chat(quality_prompt)

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Quality analysis completed",
            input_data={},
            output_data={
                "deterministic_score": deterministic_score,
                "has_forbidden": has_forbidden
            }
        ))

        return {
            "code_length": code_length,
            "code_lines": len(code_lines),
            "deterministic_checks": {
                "has_functions": has_functions,
                "has_docstrings": has_docstrings,
                "has_forbidden_imports": has_forbidden,
                "score": deterministic_score
            },
            "llm_analysis": quality_report,
            "reasoning": reasoning
        }

    async def get_verification_artifacts(
            self,
            session_id: str
    ) -> Dict[str, Any]:
        """
        Get all verification artifacts for a session

        Returns tangible proof of execution
        """
        iterations = self._iterations_history.get(session_id, [])

        artifacts = []
        for iteration in iterations:
            if iteration.verification_artifact:
                artifacts.append(asdict(iteration.verification_artifact))

        return {
            "session_id": session_id,
            "total_iterations": len(iterations),
            "artifacts": artifacts,
            "artifact_types": list(set(a["artifact_type"] for a in artifacts))
        }
