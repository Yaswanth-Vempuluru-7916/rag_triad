from typing import Protocol

import anthropic
import ollama

from ragscope.config import active_models


class LLM(Protocol):
    model_name: str

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str: ...


class OllamaLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.Client(timeout=120)

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        res = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature},
        )
        return res.message.content


class ClaudeLLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = anthropic.Anthropic(max_retries=3, timeout=60)  # reads ANTHROPIC_API_KEY

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        res = self.client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return res.content[0].text


PROVIDERS = {"ollama": OllamaLLM, "anthropic": ClaudeLLM}


def make_llm() -> LLM:
    provider, model_name = active_models()["llm"]
    return PROVIDERS[provider](model_name)
