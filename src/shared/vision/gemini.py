import base64
import binascii
import logging
import os
import re
from typing import Tuple

import httpx

logger = logging.getLogger("gemini_vision")


def prepare_image_for_gemini(image_input: str) -> Tuple[str, str]:
    """
    Prepare image data for Gemini API

    Returns: (mime_type, cleaned_base64)
    """
    mime_type = "image/jpeg"
    cleaned_base64 = image_input.strip()

    # Check if it's a data URI
    if image_input.startswith("data:"):
        match = re.match(
            r"data:(?P<mime>[\w/]+);base64,(?P<data>.*)",
            image_input,
        )
        if match:
            mime_type = match.group("mime")
            cleaned_base64 = match.group("data")
    else:
        # Try to detect mime type from base64 header
        try:
            header = base64.b64decode(cleaned_base64[:32])
            if header.startswith(b"\xff\xd8\xff"):
                mime_type = "image/jpeg"
            elif header.startswith(b"\x89PNG\r\n\x1a\n"):
                mime_type = "image/png"
            elif header.startswith(b"GIF87a") or header.startswith(b"GIF89a"):
                mime_type = "image/gif"
            elif header.startswith(b"RIFF") and header[8:12] == b"WEBP":
                mime_type = "image/webp"
        except (binascii.Error, Exception):
            pass

    return mime_type, cleaned_base64


class GeminiVisionProvider:
    """
    Vision provider Gemini models

    """

    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")

        # Try vision-specific model first, fallback to general model
        self.model = os.getenv("GEMINI_VISION_MODEL") or os.getenv("GEMINI_MODEL", "google/gemini-2.0-flash-exp:free")

        # Try vision-specific base URL first, fallback to general
        self.base_url = os.getenv("GEMINI_VISION_BASE_URL") or os.getenv(
            "GEMINI_BASE_URL"
        )

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        logger.info(
            "✅ Gemini Vision initialized",
            extra={
                "model": self.model,
                "base_url": self.base_url,
            },
        )

    async def analyze(
            self,
            prompt: str,
            image_base64: str,
    ) -> str:
        """
        Analyze image using Gemini Vision
        """
        mime_type, cleaned_b64 = prepare_image_for_gemini(image_base64)


        data_uri = f"data:{mime_type};base64,{cleaned_b64}"

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_uri
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.4
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost",
            "X-Title": "multi-agent-devops",
        }

        logger.info(f"Sending vision request: {self.model}")
        logger.info(f"Image size: {len(cleaned_b64)} bytes (base64)")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                )

                if response.status_code != 200:
                    error_text = response.text
                    logger.error(
                        "Gemini Vision API error",
                        extra={
                            "status": response.status_code,
                            "body": error_text[:500],
                            "model": self.model
                        },
                    )

                    # Provide helpful error message
                    if response.status_code == 400:
                        if "model" in error_text.lower():
                            raise RuntimeError(
                                f"Model '{self.model}' may not support vision. "
                                f"Try: 'google/gemini-2.0-flash-exp:free' or 'google/gemini-pro-vision'"
                            )
                        elif "image" in error_text.lower() or "content" in error_text.lower():
                            raise RuntimeError(
                                f"Image format error. Image size: {len(cleaned_b64)} bytes. "
                                f"Model may have size limits. Try a smaller image."
                            )

                    response.raise_for_status()

                data = response.json()


                if "choices" in data and len(data["choices"]) > 0:
                    message = data["choices"][0].get("message", {})
                    content = message.get("content", "")

                    if content:
                        logger.info("Successfully received vision analysis")
                        return content
                    else:
                        raise RuntimeError("Empty content in response")
                else:
                    raise RuntimeError(f"Unexpected response format: {data}")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e}")
            raise
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            raise


# Fallback provider for when vision is unavailable
class FallbackVisionProvider:
    """Fallback when real vision is unavailable"""

    async def analyze(self, prompt: str, image_base64: str) -> str:
        return f"""
[Vision analysis unavailable - fallback mode]

Based on the prompt: "{prompt[:100]}..."

This appears to be an architecture diagram. General observations:
- Multiple interconnected components
- Service-oriented architecture pattern
- Consider: scalability, reliability, security
- Recommendation: Manual review for production decisions

Note: Enable Gemini Vision API for detailed analysis.
"""


def get_vision_provider():
    """Get the appropriate vision provider"""
    try:
        provider = GeminiVisionProvider()
        return provider
    except Exception as e:
        logger.warning(f"Gemini Vision unavailable: {e}. Using fallback.")
        return FallbackVisionProvider()
