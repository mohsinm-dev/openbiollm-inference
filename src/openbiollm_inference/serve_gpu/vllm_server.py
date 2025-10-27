"""Helper to launch vLLM OpenAI-compatible server from Python (optional)."""
from __future__ import annotations
import os
import subprocess

def main() -> None:
    model = os.getenv("VLLM_MODEL_ID", "aaditya/Llama3-OpenBioLLM-8B")
    port = os.getenv("VLLM_PORT", "8001")
    max_len = os.getenv("VLLM_MAX_MODEL_LEN", "8192")
    tp = os.getenv("VLLM_TP_SIZE", "1")

    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--max-model-len", str(max_len),
        "--tensor-parallel-size", str(tp),
        "--port", str(port),
        "--dtype", "bfloat16",
    ]
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
