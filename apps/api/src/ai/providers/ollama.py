"""Ollama Local AI Provider (100% Free / Offline)."""
from typing import Optional
import httpx
from apps.api.src.ai.providers.base import BaseAIProvider
from packages.analytics_core.src.security.ssrf import resolve_pinned_target, SSRFSecurityError


class OllamaProvider(BaseAIProvider):
    """Local Ollama AI provider for zero-cost, private, offline execution."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2.5-coder:7b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Ollama is, by design, a local/self-hosted server -- loopback addresses
        # (127.0.0.1, ::1, localhost) are its normal, expected deployment, not an
        # SSRF risk to guard against here the way a cloud provider's base_url
        # would be. Non-loopback private-LAN addresses still go through the same
        # RFC 1918 check as any other provider.
        self.allow_loopback = True

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        try:
            target = resolve_pinned_target(self.base_url, allow_loopback=self.allow_loopback, label="Ollama base URL")
        except SSRFSecurityError as e:
            raise RuntimeError(f"Ollama generation refused: {e}") from e

        url = f"{target.url}/api/generate"
        headers = {"Host": target.host_header}
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, json=payload, extensions={"sni_hostname": target.sni_hostname})
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
        except Exception as e:
            raise RuntimeError(f"Ollama local generation failed: {str(e)}. Ensure Ollama is running at {self.base_url}")
