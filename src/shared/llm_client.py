import asyncio
import json
import logging
import os
from typing import Dict, Any

import httpx

logger = logging.getLogger("llm_client")


class LLMClient:
    """
    Unified LLM client for Google Gemini API.
    Configured via environment variables (GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL).
    """

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "stub").lower()

        if self.provider == "gemini":
            self.api_key = os.getenv("GEMINI_API_KEY")
            self.base_url = os.getenv("GEMINI_BASE_URL")
            self.model = os.getenv("GEMINI_MODEL", "gemini-3.0-flash")

            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY is not set — get it from Google AI Studio")

            # Load optional extra headers from env (JSON string)
            extra_headers_str = os.getenv("GEMINI_EXTRA_HEADERS_JSON", "{}")
            try:
                self.extra_headers = json.loads(extra_headers_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid GEMINI_EXTRA_HEADERS_JSON: {e} — ignoring")
                self.extra_headers = {}

            logger.info(
                "Google Gemini API enabled",
                extra={
                    "model": self.model,
                    "base_url": self.base_url,
                    "extra_headers_count": len(self.extra_headers),
                }
            )
        else:
            logger.warning("Using stub LLM provider")

    async def chat(self, prompt: str) -> str:
        if self.provider != "gemini":
            return f"[stub] {prompt}"

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Add extra headers if defined (harmless for official API)
        headers.update(self.extra_headers)

        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                # Simple retry for rate limits (429)
                for attempt in range(3):
                    resp = await client.post(self.base_url, json=payload, headers=headers)
                    if resp.status_code == 429:
                        logger.warning(f"Rate limit hit — retry {attempt + 1}/3")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    return data["choices"][0]["message"]["content"].strip()

                return "[Gemini error] Rate limit exceeded after retries"

        except httpx.HTTPStatusError as e:
            logger.error("Gemini API HTTP error", extra={"status": e.response.status_code})
            return f"[HTTP {e.response.status_code}] {e.response.text[:200]}"
        except Exception as e:
            logger.error("Gemini API request failed", extra={"error": str(e)})
            return f"[Gemini error] {str(e)}"

    async def chat_structured(self, prompt: str) -> Dict[str, Any]:
        text = await self.chat(prompt)
        return {"raw": text}
