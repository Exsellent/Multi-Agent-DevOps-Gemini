import logging
from dataclasses import dataclass, asdict
from datetime import date as date_module
from typing import List, Optional, Dict, Any

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep

logger = logging.getLogger("digest_agent")


@dataclass
class DigestValidation:
    """Validation results for digest quality"""
    word_count: int
    under_limit: bool
    has_achievements: bool
    has_blockers: bool
    tone_positive: bool
    confidence: float


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


class DigestAgent(MCPAgent):
    """
    Advanced Daily Project Digest Agent

    Key Features:
    1. Deterministic Validation: Word count, structure, tone analysis
    2. Confidence Scoring: Based on data quality and validation
    3. Section Extraction: Achievements, blockers, mood analysis
    4. Fallback Strategy: Graceful degradation with transparency

    Proper sequential reasoning, validation as reasoning steps
    """

    # Validation thresholds
    MAX_WORD_COUNT = 200
    MIN_WORD_COUNT = 50
    REQUIRED_SECTIONS = ["achievement", "blocker", "mood"]

    def __init__(self):
        super().__init__("Digest")
        self.llm = LLMClient()

        self.register_tool("daily_digest", self.daily_digest)
        self.register_tool("validate_digest", self.validate_digest)
        self.register_tool("extract_key_points", self.extract_key_points)

        logger.info("DigestAgent initialized with validation and extraction capabilities")

    def _next_step(self, reasoning: List[ReasoningStep], description: str,
                   input_data: Optional[Dict] = None, output_data: Optional[Dict] = None):
        """Helper to add sequential reasoning steps"""
        reasoning.append(ReasoningStep(
            step_number=len(reasoning) + 1,
            description=description,
            input_data=input_data or {},
            output_data=output_data or {}
        ))

    def _is_invalid_response(self, response: str) -> bool:
        """Check if LLM response is stub or error"""
        text = response.lower()
        indicators = [
            "[stub]", "[llm error]", "unauthorized", "401", "client error",
            "for more information check", "status/401", "connection error", "timeout"
        ]
        return any(indicator in text for indicator in indicators)

    def _validate_digest_quality(self, digest: str) -> DigestValidation:
        """
        Deterministic validation of digest quality

        Checks:
        - Word count within limits
        - Required sections present
        - Positive tone keywords
        """
        words = digest.split()
        word_count = len(words)

        # Check for required sections
        digest_lower = digest.lower()
        has_achievements = any(keyword in digest_lower for keyword in [
            "achieve", "complete", "deliver", "success", "progress", "milestone"
        ])
        has_blockers = any(keyword in digest_lower for keyword in [
            "blocker", "issue", "challenge", "delay", "pending", "waiting"
        ])
        tone_positive = any(keyword in digest_lower for keyword in [
            "good", "great", "excellent", "positive", "motivated", "productive"
        ])

        # Calculate confidence
        confidence = 0.5  # Base confidence

        if self.MIN_WORD_COUNT <= word_count <= self.MAX_WORD_COUNT:
            confidence += 0.2
        if has_achievements:
            confidence += 0.1
        if has_blockers:
            confidence += 0.1
        if tone_positive:
            confidence += 0.1

        return DigestValidation(
            word_count=word_count,
            under_limit=word_count <= self.MAX_WORD_COUNT,
            has_achievements=has_achievements,
            has_blockers=has_blockers,
            tone_positive=tone_positive,
            confidence=min(confidence, 0.95)
        )

    def _extract_sections(self, digest: str) -> Dict[str, str]:
        """

        Extract key sections from digest

        Returns structured data for better observability
        """
        sections = {
            "achievements": "",
            "blockers": "",
            "mood": "",
            "full_text": digest
        }

        # Simple heuristic extraction (in production: use LLM)
        lines = digest.split('\n')
        current_section = None

        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ["achieve", "complete", "deliver"]):
                current_section = "achievements"
            elif any(keyword in line_lower for keyword in ["blocker", "issue", "challenge"]):
                current_section = "blockers"
            elif any(keyword in line_lower for keyword in ["mood", "team", "spirit"]):
                current_section = "mood"

            if current_section and line.strip():
                sections[current_section] += line + "\n"

        return sections

    @log_method
    @metric_counter("digest")
    async def daily_digest(self, date: Optional[str] = None, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate daily project digest with validation

        Proper sequential reasoning with validation steps and quality scoring
        """
        reasoning: List[ReasoningStep] = []

        # Handle date
        if date is None:
            date = date_module.today().isoformat()

        # Step 1: Request received
        self._next_step(reasoning, "Daily digest generation requested",
                        input_data={
                            "date": date,
                            "context_provided": bool(context)
                        })

        # Prepare context-aware prompt
        context_section = ""
        if context:
            context_section = f"\nProject context:\n{context}\n"

        prompt = f"""Generate a concise daily project digest for {date}.
{context_section}
Include:
1. Key achievements (2-3 items)
2. Current blockers (if any)
3. Team mood/morale

Requirements:
- Keep it positive and actionable
- Under 200 words
- Professional tone suitable for stakeholders

Format as natural prose, not bullet points."""

        # Step 2: LLM request initiated
        digest = None
        llm_fallback = False

        try:
            digest = await self.llm.chat(prompt)

            #  Check for LLM errors
            if self._is_invalid_response(digest):
                llm_fallback = True

                #  Explicit LLM error fallback step
                self._next_step(reasoning, "LLM response invalid - using baseline digest",
                                output_data={"fallback_reason": "invalid_response"})
            else:
                # Successful LLM response
                self._next_step(reasoning, "LLM digest generated successfully",
                                output_data={
                                    "initial_length": len(digest),
                                    "initial_word_count": len(digest.split())
                                })

        except Exception as e:
            logger.error("LLM digest generation failed", extra={"error": str(e)})
            llm_fallback = True

            # Explicit exception fallback step
            self._next_step(reasoning, "LLM request failed - using baseline digest",
                            output_data={
                                "error": str(e),
                                "fallback_reason": "exception"
                            })

        # Generate fallback if needed
        if llm_fallback or not digest:
            digest = f"""Daily Project Digest - {date}

The team made steady progress today. Key development tasks are on track, and collaboration remains strong. 

While no major blockers are currently impacting delivery, we're monitoring external dependencies closely. The team maintains a positive and productive atmosphere, focused on delivering quality results.

Overall, it's been a solid day of progress toward our project goals."""

        # Step 3 - Validate digest quality
        validation = self._validate_digest_quality(digest)

        self._next_step(reasoning, "Digest quality validation completed",
                        output_data={
                            "word_count": validation.word_count,
                            "under_limit": validation.under_limit,
                            "has_achievements": validation.has_achievements,
                            "has_blockers": validation.has_blockers,
                            "confidence": validation.confidence
                        })

        # Step 4 - Extract sections
        sections = self._extract_sections(digest)

        self._next_step(reasoning, "Key sections extracted from digest",
                        output_data={
                            "sections_found": sum(1 for v in sections.values() if v and v != digest),
                            "structured_data_available": True
                        })

        # Step 5 - Digest finalized (ALWAYS present)
        self._next_step(reasoning, "Daily digest generation completed",
                        output_data={
                            "final_word_count": validation.word_count,
                            "validation_passed": validation.under_limit and validation.has_achievements,
                            "confidence_level": validation.confidence,
                            "llm_fallback_used": llm_fallback
                        })

        logger.info("Daily digest completed",
                    extra={
                        "date": date,
                        "word_count": validation.word_count,
                        "confidence": validation.confidence,
                        "fallback": llm_fallback
                    })

        return {
            "date": date,
            "summary": digest,
            "sections": sections,  # Structured data
            "validation": asdict(validation),
            "fallback_used": llm_fallback,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("digest")
    async def validate_digest(self, digest: str) -> Dict[str, Any]:
        """
        Standalone digest validation tool

        Allows external validation of digest quality
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Digest validation requested",
                        input_data={"digest_length": len(digest)})

        validation = self._validate_digest_quality(digest)

        self._next_step(reasoning, "Validation completed",
                        output_data=asdict(validation))

        return {
            "validation": asdict(validation),
            "passed": validation.under_limit and validation.confidence > 0.6,
            "reasoning": reasoning
        }

    @log_method
    @metric_counter("digest")
    async def extract_key_points(self, digest: str) -> Dict[str, Any]:
        """
        Extract structured key points from digest

        Useful for downstream agents or reporting
        """
        reasoning: List[ReasoningStep] = []

        self._next_step(reasoning, "Key points extraction requested",
                        input_data={"digest_length": len(digest)})

        sections = self._extract_sections(digest)

        self._next_step(reasoning, "Sections extracted and structured",
                        output_data={
                            "achievements_found": bool(sections["achievements"]),
                            "blockers_found": bool(sections["blockers"]),
                            "mood_found": bool(sections["mood"])
                        })

        # LLM-based extraction for better quality (optional)
        if sections["achievements"] or sections["blockers"]:
            self._next_step(reasoning, "Deterministic extraction successful",
                            output_data={"method": "heuristic"})
        else:
            self._next_step(reasoning, "Minimal structure detected - full text returned",
                            output_data={"method": "fallback"})

        self._next_step(reasoning, "Key points extraction completed",
                        output_data={"total_sections": len(sections)})

        return {
            "sections": sections,
            "reasoning": reasoning
        }
