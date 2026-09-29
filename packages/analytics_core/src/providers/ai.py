"""BYOM Pluggable AI Model Provider Abstraction Layer for AA-OS."""
from abc import ABC, abstractmethod
import json
import os
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


class BaseAIProvider(ABC):
    """Abstract interface for AI reasoning and hypothesis generation engines."""

    @property
    @abstractmethod
    def provider_type(self) -> str:
        pass

    @property
    @abstractmethod
    def is_local(self) -> bool:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate textual response from the model."""
        pass

    @abstractmethod
    def test_connection(self) -> Tuple[bool, str]:
        """Verify API key, connectivity, and model availability."""
        pass


class NoneAIProvider(BaseAIProvider):
    """Explicit no-AI provider used by default in local deterministic mode."""

    @property
    def provider_type(self) -> str:
        return "none"

    @property
    def is_local(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return "none"

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        raise RuntimeError("AI is disabled. Deterministic AA-OS analysis does not require an AI provider.")

    def test_connection(self) -> Tuple[bool, str]:
        return True, "AI augmentation disabled; deterministic engine is authoritative."


class MockAIProvider(BaseAIProvider):
    """Offline deterministic baseline provider for Zero-Cost Community Edition."""

    def __init__(self, model_name: str = "deterministic-mock"):
        self._model_name = model_name

    @property
    def provider_type(self) -> str:
        return "mock"

    @property
    def is_local(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return "Analysis generated via deterministic local engines."

    def test_connection(self) -> Tuple[bool, str]:
        return True, "Deterministic Offline Engine active (100% Free / Local)"


class OllamaAIProvider(BaseAIProvider):
    """Local LLM Provider (Ollama, llama.cpp, vLLM, LM Studio) for 100% Private / Offline inference."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen2.5-coder:7b",
    ):
        self.base_url = base_url.rstrip("/")
        self._model_name = model_name

    @property
    def provider_type(self) -> str:
        return "ollama"

    @property
    def is_local(self) -> bool:
        return True

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        payload = {
            "model": self._model_name,
            "prompt": prompt,
            "system": system_prompt or "You are an expert autonomous data analyst.",
            "stream": False,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                res = json.loads(resp.read().decode())
                return res.get("response", "")
        except Exception as e:
            return f"Ollama execution error: {str(e)}"

    def test_connection(self) -> Tuple[bool, str]:
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
                models = [m.get("name") for m in data.get("models", [])]
                return True, f"Connected to Ollama ({len(models)} models available, active: {self._model_name})"
        except Exception as e:
            return False, f"Could not connect to Ollama at {self.base_url}: {str(e)}"


class OpenAIProvider(BaseAIProvider):
    """BYOK OpenAI / Groq / OpenAI-Compatible Cloud Provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._model_name = model_name
        self.base_url = base_url.rstrip("/")

    @property
    def provider_type(self) -> str:
        return "openai"

    @property
    def is_local(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            return "Error: OpenAI API key is not configured."
        payload = {
            "model": self._model_name,
            "messages": [
                {"role": "system", "content": system_prompt or "You are an expert autonomous data analyst."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode())
                return res["choices"][0]["message"]["content"]
        except Exception as e:
            return f"OpenAI API error: {str(e)}"

    def test_connection(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "OpenAI API Key is missing."
        return True, f"OpenAI configured ({self._model_name} via {self.base_url})"


class GeminiProvider(BaseAIProvider):
    """BYOK Google Gemini Cloud Provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self._model_name = model_name

    @property
    def provider_type(self) -> str:
        return "gemini"

    @property
    def is_local(self) -> bool:
        return False

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            return "Error: Gemini API key is not configured."
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"{system_prompt or ''}\n\n{prompt}"}]}]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                res = json.loads(resp.read().decode())
                return res["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"Gemini API error: {str(e)}"

    def test_connection(self) -> Tuple[bool, str]:
        if not self.api_key:
            return False, "Gemini API Key is missing."
        return True, f"Google Gemini configured ({self._model_name})"
