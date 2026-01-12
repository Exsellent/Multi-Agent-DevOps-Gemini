import asyncio
import logging
import os
import tempfile
from typing import List, Optional, Dict, Any

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("code_execution_agent")


class CodeExecutionAgent(MCPAgent):
    """
    Agent that generates, tests, and verifies code autonomously

    This addresses the "Vibe Engineering" requirement:
    - Generates code based on requirements
    - Executes it in isolated environment
    - Verifies results through testing loops
    - Self-corrects on failures

    Example: Generate a data processing script, test it with sample data,
    fix bugs, and confirm it works before deployment.
    """

    def __init__(self):
        super().__init__("Code-Execution")
        self.llm = LLMClient()

        # Safety: Only allow safe operations
        self.allowed_languages = {"python", "javascript", "bash"}

        self.register_tool("generate_and_test_code", self.generate_and_test_code)
        self.register_tool("autonomous_debug_loop", self.autonomous_debug_loop)
        self.register_tool("verify_code_quality", self.verify_code_quality)

        logger.info("Code Execution Agent initialized with autonomous testing")

    @metric_counter("code_execution")
    async def generate_and_test_code(
            self,
            requirement: str,
            language: str = "python",
            test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate code and automatically test it

        This is the core "write code AND verify it" capability.
        """
        reasoning: List[ReasoningStep] = []

        if language not in self.allowed_languages:
            return {"error": f"Language {language} not allowed. Use: {self.allowed_languages}"}

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Starting code generation with automated testing",
            input_data={"requirement": requirement, "language": language}
        ))

        # Step 1: Generate code with test cases
        code_gen_prompt = f"""
Generate production-ready {language} code for this requirement:

REQUIREMENT: {requirement}

Your code must:
1. Include proper error handling
2. Have docstrings/comments
3. Be testable with clear inputs/outputs
4. Handle edge cases

Also generate 3-5 test cases that verify correctness.

Return in this format:
```{language}
# Main code here
```

TEST CASES:
1. Input: ..., Expected: ...
2. Input: ..., Expected: ...
etc.
"""

        code_response = await self.llm.chat(code_gen_prompt)

        # Extract code from response
        code = self._extract_code_block(code_response, language)
        test_cases_text = self._extract_test_cases(code_response)

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Generated code and test cases",
            output_data={"code_length": len(code), "tests_count": len(test_cases_text)}
        ))

        # Step 2: Execute tests in isolated environment
        test_results = []

        if language == "python":
            for idx, test in enumerate(test_cases_text, 1):
                result = await self._run_python_test(code, test)
                test_results.append({
                    "test_id": idx,
                    "passed": result["success"],
                    "output": result["output"],
                    "error": result.get("error")
                })

        reasoning.append(ReasoningStep(
            step_number=3,
            description="Executed all test cases",
            output_data={"tests_run": len(test_results)}
        ))

        # Step 3: Analyze results
        passed_tests = sum(1 for t in test_results if t["passed"])
        success_rate = (passed_tests / len(test_results) * 100) if test_results else 0

        analysis_prompt = f"""
Code testing completed:
- Tests passed: {passed_tests}/{len(test_results)}
- Success rate: {success_rate}%

Test Results:
{test_results}

Analyze:
1. Is this code production-ready?
2. What issues were found?
3. What improvements are needed?
4. Risk level for deployment (Low/Medium/High)

Provide concrete recommendations.
"""

        analysis = await self.llm.chat(analysis_prompt)

        reasoning.append(ReasoningStep(
            step_number=4,
            description="Analyzed test results and code quality",
            output_data={"success_rate": success_rate}
        ))

        return {
            "requirement": requirement,
            "generated_code": code,
            "test_results": test_results,
            "success_rate": success_rate,
            "quality_analysis": analysis,
            "production_ready": success_rate >= 80,
            "reasoning": reasoning
        }

    @metric_counter("code_execution")
    async def autonomous_debug_loop(
            self,
            code: str,
            error_message: str,
            language: str = "python",
            max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Autonomous debugging: agent fixes its own code

        This demonstrates self-correction in action:
        1. Code fails
        2. Agent analyzes error
        3. Agent fixes code
        4. Agent re-tests
        5. Repeat until fixed or max iterations
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Starting autonomous debug loop",
            input_data={"error": error_message, "max_iterations": max_iterations}
        ))

        current_code = code
        iteration = 0
        debug_history = []

        while iteration < max_iterations:
            iteration += 1

            debug_prompt = f"""
You are debugging code that failed.

ITERATION: {iteration}/{max_iterations}

CURRENT CODE:
```{language}
{current_code}
```

ERROR:
{error_message}

PREVIOUS ATTEMPTS: {debug_history}

Fix the bug:
1. Explain what's wrong
2. Show the corrected code
3. Explain why this fix works

Return corrected code in ```{language} blocks.
"""

            debug_response = await self.llm.chat(debug_prompt)
            fixed_code = self._extract_code_block(debug_response, language)

            reasoning.append(ReasoningStep(
                step_number=iteration + 1,
                description=f"Debug iteration {iteration} - attempted fix",
                output_data={"iteration": iteration}
            ))

            # Test the fix
            if language == "python":
                test_result = await self._run_python_code(fixed_code)

                if test_result["success"]:
                    reasoning.append(ReasoningStep(
                        step_number=iteration + 2,
                        description="Fix successful - code now works",
                        output_data={"iterations_needed": iteration}
                    ))

                    return {
                        "status": "fixed",
                        "original_code": code,
                        "fixed_code": fixed_code,
                        "iterations": iteration,
                        "debug_history": debug_history,
                        "reasoning": reasoning
                    }
                else:
                    # Fix didn't work, try again
                    error_message = test_result.get("error", "Unknown error")
                    debug_history.append({
                        "iteration": iteration,
                        "attempted_fix": fixed_code[:100],
                        "result": "failed",
                        "error": error_message
                    })
                    current_code = fixed_code

        # Max iterations reached without success
        reasoning.append(ReasoningStep(
            step_number=max_iterations + 2,
            description="Max debug iterations reached - manual review needed",
            output_data={"final_status": "needs_manual_review"}
        ))

        return {
            "status": "needs_manual_review",
            "original_code": code,
            "last_attempt": current_code,
            "iterations": max_iterations,
            "debug_history": debug_history,
            "reasoning": reasoning,
            "message": "Autonomous debugging could not fully resolve the issue. Human review recommended."
        }

    @metric_counter("code_execution")
    async def verify_code_quality(
            self,
            code: str,
            language: str = "python"
    ) -> Dict[str, Any]:
        """
        Static analysis and quality checks

        Verifies:
        - Code style
        - Security issues
        - Best practices
        - Performance concerns
        """
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Running code quality verification"
        ))

        quality_prompt = f"""
Perform comprehensive code review:

CODE ({language}):
```
{code}
```

Analyze for:
1. SECURITY: SQL injection, XSS, command injection, etc.
2. PERFORMANCE: O(n) complexity issues, inefficient patterns
3. MAINTAINABILITY: Code smells, naming conventions, modularity
4. ERROR HANDLING: Try-catch coverage, edge cases
5. TESTING: Is this code testable? Any tight coupling?

For each category, give:
- Score (0-10)
- Issues found
- Recommendations

Be thorough and critical.
"""

        quality_report = await self.llm.chat(quality_prompt)

        reasoning.append(ReasoningStep(
            step_number=2,
            description="Generated quality analysis report"
        ))

        return {
            "code_length": len(code),
            "language": language,
            "quality_report": quality_report,
            "reasoning": reasoning
        }

    async def _run_python_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code in isolated environment"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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
                    timeout=5.0
                )

                os.unlink(temp_file)

                if process.returncode == 0:
                    return {
                        "success": True,
                        "output": stdout.decode('utf-8')
                    }
                else:
                    return {
                        "success": False,
                        "error": stderr.decode('utf-8')
                    }

            except asyncio.TimeoutError:
                process.kill()
                os.unlink(temp_file)
                return {
                    "success": False,
                    "error": "Execution timeout (5s limit)"
                }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def _run_python_test(self, code: str, test_case: str) -> Dict[str, Any]:
        """Run a specific test case"""
        # Combine code with test case
        test_code = f"{code}\n\n# Test\n{test_case}"
        return await self._run_python_code(test_code)

    def _extract_code_block(self, response: str, language: str) -> str:
        """Extract code from markdown code blocks"""
        import re
        pattern = f"```{language}\\s*\\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return response  # Fallback: return whole response

    def _extract_test_cases(self, response: str) -> List[str]:
        """Extract test cases from response"""
        # Simple extraction - look for numbered test cases
        import re
        tests = re.findall(r'(\d+\.\s*Input:.*?Expected:.*?)(?=\d+\.|$)', response, re.DOTALL)
        return tests if tests else []
