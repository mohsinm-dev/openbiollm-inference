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
from openbiollm_inference.common.templates import load_template, render_prompt

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

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID}

@app.post("/generate", response_model=GenerateOut)
def generate(body: GenerateIn) -> GenerateOut:
    template = load_template()
    prompt = render_prompt(template, body.input)

    url = f"{VLLM_BASE_URL}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": "You are a careful clinical assistant. Be concise."},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "stream": False
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
