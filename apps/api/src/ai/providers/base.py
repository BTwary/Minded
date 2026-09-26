"""Base AI Provider Abstraction."""
from abc import ABC, abstractmethod
import json
import re
from typing import Any, Dict, Optional, Type, TypeVar
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseAIProvider(ABC):
    """Abstract Base Class for AI model providers."""

    @property
    def is_ai_enabled(self) -> bool:
        """Whether this provider represents an active external AI service."""
        return True

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """Generate unstructured text completion."""
        pass

    def generate_structured(
        self,
        prompt: str,
        schema_cls: Type[T],
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> T:
        """Generate structured JSON response parsed into a Pydantic model."""
        schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
        augmented_prompt = (
            f"{prompt}\n\n"
            f"CRITICAL: You must return ONLY a single valid JSON object matching this schema:\n"
            f"```json\n{schema_json}\n```\n"
            f"Do not include any Markdown text outside the JSON code block."
        )

        response_text = self.generate(
            prompt=augmented_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
        )

        # Extract JSON from potential code blocks
        clean_json = response_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_json)
        if match:
            clean_json = match.group(1).strip()

        try:
            parsed_dict = json.loads(clean_json)
            return schema_cls.model_validate(parsed_dict)
        except Exception as e:
            # Fallback: Attempt line-by-line or regex repair
            try:
                # If trailing comma or minor format issue
                clean_json_fixed = re.sub(r",\s*([\]}])", r"\1", clean_json)
                parsed_dict = json.loads(clean_json_fixed)
                return schema_cls.model_validate(parsed_dict)
            except Exception:
                raise ValueError(f"Failed to parse structured output into {schema_cls.__name__}: {str(e)}\nRaw Response:\n{response_text}")
