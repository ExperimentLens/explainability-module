from typing import Any, Optional

from openai import OpenAI
from llmSHAP.llm.llm_interface import LLMInterface


class OllamaInterface(LLMInterface):
    """LLMInterface backed by Ollama's OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ) -> None:
        # Run params store LiteLLM-style "ollama/<model>" identifiers (e.g. from the
        # experiment script's MODEL_NAME="ollama/llama3.2"), but Ollama's own
        # OpenAI-compatible endpoint expects the bare tag ("llama3.2"), not the
        # provider-prefixed form - strip it if present.
        self.model_name = model_name.removeprefix("ollama/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(
        self,
        prompt: Any,
        tools: Optional[list[Any]] = None,
        images: Optional[list[Any]] = None,
    ) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": prompt,
            "temperature": self.temperature,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
