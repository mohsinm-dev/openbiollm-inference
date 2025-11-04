"""FastAPI proxy for vLLM OpenAI-compatible server.

Endpoints:
- GET /health
- POST /generate  { "input": "..." }
"""
from __future__ import annotations
import os
import time
import logging

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from openbiollm_inference.common.logging_setup import setup_logging
from openbiollm_inference.common.templates import load_template

LOGGER = logging.getLogger("openbiollm.gpu.app")
setup_logging()

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "not-required")
MODEL_ID = os.getenv("VLLM_MODEL_ID", "aaditya/Llama3-OpenBioLLM-8B")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

class GenerateIn(BaseModel):
    input: str

class GenerateOut(BaseModel):
    text: str
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

app = FastAPI()

@app.on_event("startup")
def _validate_template_on_startup() -> None:
    """Validate prompt template shape on startup and warn if malformed."""
    try:
        template = load_template()
        has_sys = "<|system|>" in template
        has_user = "<|user|>" in template
        if not (has_sys and has_user):
            LOGGER.warning(
                "Prompt template missing expected tags; found system=%s user=%s",
                has_sys,
                has_user,
            )
    except Exception as e:
        LOGGER.warning("Failed to read prompt template: %s", e)

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}

def _extract_system(template: str) -> str:
    """Extract system prompt from the template between <|system|> and <|user|>.

    Falls back to a concise default if tags are missing.
    """
    sys_tag, user_tag = "<|system|>", "<|user|>"
    try:
        if sys_tag in template and user_tag in template:
            start = template.index(sys_tag) + len(sys_tag)
            end = template.index(user_tag, start)
            return template[start:end].strip()
    except Exception:
        pass
    return "You are a careful clinical assistant. Be concise."


@app.post("/generate", response_model=GenerateOut)
def generate(body: GenerateIn) -> GenerateOut:
    template = load_template()
    system_prompt = _extract_system(template)

    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": body.input},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    start = time.time()
    try:
        with httpx.Client(timeout=120.0) as client:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        LOGGER.error("vLLM request failed: %s", e)
        raise HTTPException(status_code=502, detail="Upstream vLLM error")

    latency = int((time.time() - start) * 1000)
    try:
        choice = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return GenerateOut(
            text=choice,
            latency_ms=latency,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
    except Exception as e:
        LOGGER.error("Malformed response: %s", e)
        raise HTTPException(status_code=500, detail="Malformed vLLM response")
