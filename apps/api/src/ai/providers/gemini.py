"""Google Gemini Provider (Free Tier: 1,500 req/day on Google AI Studio)."""
from typing import Optional
import httpx
from apps.api.src.ai.providers.base import BaseAIProvider


class GeminiProvider(BaseAIProvider):
    """Google Gemini AI Studio Free-Tier provider."""

    def __init__(self, api_key: str = "", model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": f"System Instructions:\n{system_prompt}"}]})
            contents.append({"role": "model", "parts": [{"text": "Understood. I will strictly follow these instructions."}]})
        
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            raise RuntimeError(f"Gemini API generation failed: {str(e)}")
