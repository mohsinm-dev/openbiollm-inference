"""Local inference for OpenBioLLM-8B (GGUF) on Apple Silicon via llama.cpp (Metal).

Adds chat-style prompting: extracts system prompt from the shared template
and sends system/user messages to llama.cpp's chat API for consistency with GPU path.
Falls back to plain prompt template if chat API is unavailable.
"""
from __future__ import annotations
import argparse
import logging
import time
from pathlib import Path
from typing import Any

import yaml
from llama_cpp import Llama  # type: ignore

from openbiollm_inference.common.logging_setup import setup_logging
from openbiollm_inference.common.schema import GenResponse
from openbiollm_inference.common.templates import load_template, render_prompt

LOGGER = logging.getLogger("openbiollm.local_m1.gguf")

def load_cfg(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_system(template: str) -> str:
    """Extract system prompt between <|system|> and <|user|>; fallback to default."""
    sys_tag, user_tag = "<|system|>", "<|user|>"
    try:
        if sys_tag in template and user_tag in template:
            start = template.index(sys_tag) + len(sys_tag)
            end = template.index(user_tag, start)
            return template[start:end].strip()
    except Exception:
        pass
    return "You are a careful clinical assistant. Be concise."

def generate(text: str, cfg_path: str = "configs/local_gguf.yaml") -> GenResponse:
    """
    Generate text using llama.cpp with a GGUF model.

    Args:
        text: User input text to inject into the chat template.
        cfg_path: YAML config path for model and sampling params.
    """
    cfg = load_cfg(cfg_path)
    model_path = cfg["model_path"]
    if not Path(model_path).exists():
        raise FileNotFoundError(f"GGUF model not found at {model_path}")

    llm = Llama(
        model_path=model_path,
        n_ctx=int(cfg.get("n_ctx", 8192)),
        n_gpu_layers=int(cfg.get("n_gpu_layers", -1)),
        logits_all=False,
        embedding=False,
        verbose=False,
    )

    params = dict(
        max_tokens=int(cfg.get("max_tokens", 512)),
        temperature=float(cfg.get("temperature", 0.2)),
        top_p=float(cfg.get("top_p", 0.9)),
        stop=list(cfg.get("stop", []) or []),
        repeat_penalty=float(cfg.get("repeat_penalty", 1.1)),
    )

    template = load_template()

    # Prefer chat-style completion for parity with GPU serving
    try:
        system_prompt = _extract_system(template)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        start = time.time()
        out = llm.create_chat_completion(
            messages=messages,
            temperature=params["temperature"],
            top_p=params["top_p"],
            max_tokens=params["max_tokens"],
            stop=params["stop"],
            repeat_penalty=params["repeat_penalty"],
        )
        latency_ms = int((time.time() - start) * 1000)

        choice = out["choices"][0]["message"]["content"]
        usage = out.get("usage", {})
        return GenResponse(
            text=str(choice).strip(),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
        )
    except Exception:
        # Fallback: render full prompt and call completion API
        prompt = render_prompt(template, text)
        start = time.time()
        out = llm(
            prompt,
            **params,
        )
        latency_ms = int((time.time() - start) * 1000)

        choice = out["choices"][0]["text"]
        usage = out.get("usage", {})
        return GenResponse(
            text=str(choice).strip(),
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
        )

def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Run GGUF model locally (llama.cpp)")
    ap.add_argument("--text", required=True, help="User input text")
    ap.add_argument("--cfg", default="configs/local_gguf.yaml", help="Config path")
    args = ap.parse_args()

    resp = generate(args.text, args.cfg)
    LOGGER.info("Latency: %sms | in=%s out=%s", resp.latency_ms, resp.input_tokens, resp.output_tokens)
    print(resp.text)

if __name__ == "__main__":
    main()
