"""Local inference for Apple Silicon via MLX (mlx-lm)."""
from __future__ import annotations
import argparse
import logging
import time

from mlx_lm import load, generate  # type: ignore
from openbiollm_inference.common.logging_setup import setup_logging
from openbiollm_inference.common.schema import GenResponse
from openbiollm_inference.common.templates import load_template, render_prompt

LOGGER = logging.getLogger("openbiollm.local_m1.mlx")

def run_mlx(model_id: str, text: str, max_tokens: int = 512, temperature: float = 0.2, top_p: float = 0.9) -> GenResponse:
    """
    Run generation using MLX models (converted or native MLX format).

    Args:
        model_id: MLX-compatible model path or id.
        text: User input.
    """
    template = load_template()
    prompt = render_prompt(template, text)

    model, tokenizer = load(model_id)
    start = time.time()
    out = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        top_p=top_p,
        verbose=False,
    )
    latency_ms = int((time.time() - start) * 1000)
    # mlx-lm does not return token usage; set -1 to indicate N/A.
    return GenResponse(text=out, input_tokens=-1, output_tokens=-1, latency_ms=latency_ms)

def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="Run MLX model locally")
    ap.add_argument("--model", required=True, help="MLX model id/path")
    ap.add_argument("--text", required=True, help="User input text")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    resp = run_mlx(args.model, args.text, args.max_tokens, args.temperature, args.top_p)
    LOGGER.info("Latency: %sms", resp.latency_ms)
    print(resp.text)

if __name__ == "__main__":
    main()
