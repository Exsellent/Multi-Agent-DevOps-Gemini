import logging
import os
from typing import Any

logger = logging.getLogger("vision")


def get_vision_provider() -> Any:
    """
    Return vision provider based on VISION_PROVIDER env var.

    """
    provider = os.getenv("VISION_PROVIDER", "gemini").lower()

    if provider == "gemini":
        from .gemini import GeminiVisionProvider
        logger.info("✅ Gemini Vision Provider initialized")
        return GeminiVisionProvider()

    # Fallback for demo/testing
    from .fallback import FallbackVisionProvider
    logger.info("⚠️ Using fallback vision provider")
    return FallbackVisionProvider()
