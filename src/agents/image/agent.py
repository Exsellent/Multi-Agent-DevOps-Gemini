import logging
import base64
from pathlib import Path
from typing import List, Optional

import httpx

from shared.llm_client import LLMClient
from shared.mcp_base import MCPAgent
from shared.metrics import metric_counter
from shared.models import ReasoningStep
from shared.vision import get_vision_provider

logger = logging.getLogger("image_agent")


class ImageAgent(MCPAgent):
    def __init__(self):
        super().__init__("Image")

        self.llm = LLMClient()
        self.vision = get_vision_provider()

        logger.info(
            "ImageAgent initialized with vision provider: %s",
            self.vision.__class__.__name__,
        )

        self.register_tool("analyze_image", self.analyze_image)
        self.register_tool("analyze_architecture", self.analyze_architecture)
        self.register_tool("analyze_local_file", self.analyze_local_file)

    def _is_invalid_response(self, response: str) -> bool:
        """Check if response indicates an API error (not content about errors)"""
        if not response:
            return True

        text = response.lower()
        error_indicators = [
            "unauthorized",
            "invalid api key",
            "vision unavailable",
            "vision is unavailable",
            "[vision fallback]",
            "api error",
            "failed to",
            "could not access",
            "service unavailable",
            "unexpected keyword argument",
        ]
        return any(indicator in text for indicator in error_indicators)

    async def _image_url_to_base64(self, image_url: str) -> Optional[str]:
        """Download image from localhost URL and convert to base64"""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(image_url)
                resp.raise_for_status()
                image_bytes = resp.content
                return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to download image from {image_url}: {e}")
            return None

    @metric_counter("image")
    async def analyze_image(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        context: Optional[str] = None,
    ):
        """Analyze image using Gemini Vision (via OpenRouter) or fallback to LLM"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Image analysis requested",
            input_data={
                "image_url": image_url,
                "has_base64": image_base64 is not None,
                "context": context,
            }
        ))

        if not image_url and not image_base64:
            return {
                "error": "image_url or image_base64 required",
                "reasoning": reasoning,
            }

        # Convert localhost URLs to base64
        if image_url and ("localhost" in image_url or "127.0.0.1" in image_url):
            logger.info("Localhost URL detected — converting to base64")
            image_base64 = await self._image_url_to_base64(image_url)
            if not image_base64:
                return {
                    "error": f"Failed to download image from {image_url}",
                    "reasoning": reasoning,
                }
            reasoning.append(ReasoningStep(
                step_number=2,
                description="Converted localhost URL to base64",
                output_data={"base64_length": len(image_base64)}
            ))
            image_url = None  # Clear URL after conversion

        # Vision provider requires ONLY base64
        if not image_base64:
            return {
                "error": "Vision provider requires image_base64 (after conversion if needed)",
                "reasoning": reasoning,
            }

        # Enhanced prompt for architecture/UI analysis
        prompt = f"""
        You are a senior software architect analyzing a diagram.
        Context: {context or "General system architecture / workflow diagram"}

        Provide a detailed technical analysis including:
        1. Architecture overview and main components
        2. Data flow and interactions between services
        3. Potential bottlenecks or single points of failure
        4. Security, scalability, and reliability considerations
        5. Specific recommendations for improvement

        Be precise and reference visible elements in the diagram.
        """

        try:
            logger.info("Calling Gemini Vision provider")
            analysis = await self.vision.analyze(
                prompt=prompt,
                image_base64=image_base64,
            )

            if self._is_invalid_response(analysis):
                raise RuntimeError("Invalid response from vision provider")

            reasoning.append(ReasoningStep(
                step_number=3,
                description="Vision analysis completed successfully",
                output_data={
                    "provider": self.vision.__class__.__name__,
                    "analysis_length": len(analysis),
                }
            ))

            return {
                "image_source": "base64" if image_base64 else "url",
                "analysis": analysis,
                "vision_provider": self.vision.__class__.__name__,
                "reasoning": reasoning,
            }

        except Exception as e:
            logger.exception("Vision analysis failed — using LLM fallback")

            fallback_prompt = f"""
            Vision analysis is temporarily unavailable.
            Context: {context or "Architecture/workflow diagram"}

            Based on typical patterns for this type of diagram, provide a best-practice analysis covering:
            1. Common components and services
            2. Standard architecture patterns
            3. Typical risks and concerns
            4. General recommendations for improvement

            Assume it's a modern multi-agent DevOps system with orchestration.
            """

            text = await self.llm.chat(fallback_prompt)

            reasoning.append(ReasoningStep(
                step_number=4,
                description="LLM fallback analysis used (vision unavailable)",
                output_data={"error": str(e)},
            ))

            return {
                "analysis": text,
                "fallback": True,
                "error": str(e),
                "reasoning": reasoning,
            }

    async def analyze_architecture(
        self,
        image_url: str,
        project_context: Optional[str] = None
    ):
        """Specialized tool for architecture diagram analysis"""
        return await self.analyze_image(
            image_url=image_url,
            context=project_context or "Multi-agent DevOps architecture with n8n orchestrator"
        )

    @metric_counter("image")
    async def analyze_local_file(
        self,
        file_path: str,
        context: Optional[str] = None
    ):
        """Analyze image from mounted assets folder"""
        reasoning: List[ReasoningStep] = []

        reasoning.append(ReasoningStep(
            step_number=1,
            description="Local file analysis requested",
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

            result = await self.analyze_image(
                image_base64=image_base64,
                context=context or f"Analysis of {path.name}"
            )

            if "reasoning" in result:
                result["reasoning"] = reasoning + result["reasoning"]

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