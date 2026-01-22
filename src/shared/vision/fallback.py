from typing import Optional

from .base import VisionProvider


class FallbackVisionProvider(VisionProvider):
    async def analyze(
            self,
            prompt: str,
            image_url: Optional[str] = None,
            image_base64: Optional[str] = None
    ) -> str:
        return (
            "Vision is unavailable.\n\n"
            "Based on context, typical architecture diagrams include:\n"
            "- API Gateway\n"
            "- Services\n"
            "- Databases\n"
            "- Observability stack\n"
        )
