import os
import re

from dotenv import load_dotenv

load_dotenv()

# (provider, model_name) per role. Providers are resolved in embed.py and llm.py.
MODELS = {
    "local": {
        "embedder": ("sentence_transformers", "BAAI/bge-small-en-v1.5"),
        "llm": ("ollama", "llama3.1:8b"),
    },
    "premium": {
        "embedder": ("openai", "text-embedding-3-small"),
        "llm": ("anthropic", "claude-haiku-5-5"),
    },
}


def use_premium() -> bool:
    return os.getenv("USE_PREMIUM_MODELS", "false").strip().lower() in ("1", "true", "yes")


def active_models() -> dict:
    return MODELS["premium" if use_premium() else "local"]


def collection_name(chunk_size: int, overlap: int, embedder_model: str) -> str:
    # One collection per (chunking, embedder): vectors from different models must never mix.
    safe_model = re.sub(r"[^a-zA-Z0-9._-]", "-", embedder_model)
    return f"chunks_{chunk_size}_{overlap}_{safe_model}"
