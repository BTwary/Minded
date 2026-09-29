"""Anthropic Claude AI Provider implementation."""
import json
from typing import Any, Dict, Optional
import httpx
from apps.api.src.ai.providers.base import BaseAIProvider
from packages.analytics_core.src.security.ssrf import resolve_pinned_target, SSRFSecurityError


class ClaudeProvider(BaseAIProvider):
    """Anthropic Claude provider using Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-7-sonnet-20250219",
        base_url: str = "https://api.anthropic.com/v1",
        allow_loopback: bool = False,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.allow_loopback = allow_loopback

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Call Anthropic Messages API."""
        try:
            target = resolve_pinned_target(self.base_url, allow_loopback=self.allow_loopback, label="Claude AI provider base URL")
        except SSRFSecurityError as e:
            return f'{{"error": "Claude provider error: {e}"}}'

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "Host": target.host_header,
        }

        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt

        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(
                    f"{target.url}/messages",
                    headers=headers,
                    json=body,
                    extensions={"sni_hostname": target.sni_hostname},
                )
                res.raise_for_status()
                data = res.json()
                content = ""
                for block in data.get("content", []):
                    if block.get("type") == "text":
                        content += block.get("text", "")
                return content.strip()
        except Exception as e:
            return f'{{"error": "Claude provider error: {str(e)}"}}'
