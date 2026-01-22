from abc import ABC, abstractmethod
from typing import Optional


class VisionProvider(ABC):
    @abstractmethod
    async def analyze(
            self,
            prompt: str,
            image_url: Optional[str] = None,
            image_base64: Optional[str] = None
    ) -> str:
        pass
