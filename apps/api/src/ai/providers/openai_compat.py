"""OpenAI-Compatible AI Provider (Works with Groq, vLLM, DeepSeek, LocalAI)."""
from typing import Optional
import httpx
from apps.api.src.ai.providers.base import BaseAIProvider
from packages.analytics_core.src.security.ssrf import resolve_pinned_target, SSRFSecurityError


class OpenAICompatibleProvider(BaseAIProvider):
    """Provider for OpenAI-compatible endpoints."""

    def __init__(
        self,
        api_key: str = "mock-key",
        base_url: str = "https://api.groq.com/openai/v1",
        model: str = "llama-3.3-70b-versatile",
        allow_loopback: bool = False,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.allow_loopback = allow_loopback

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        try:
            target = resolve_pinned_target(self.base_url, allow_loopback=self.allow_loopback, label="OpenAI-compatible provider base URL")
        except SSRFSecurityError as e:
            raise RuntimeError(f"OpenAI-compatible generation refused: {e}") from e

        url = f"{target.url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Host": target.host_header,
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, json=payload, extensions={"sni_hostname": target.sni_hostname})
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise RuntimeError(f"OpenAI-compatible generation failed: {str(e)}")
