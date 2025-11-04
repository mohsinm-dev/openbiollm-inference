# OpenBioLLM Inference (Apple Silicon + GPU)

OpenBioLLM-8B locally on Apple Silicon (M1/M2) with quantized GGUF
or MLX, and deploy on a GPU server using vLLM + FastAPI.

## Features
- Mac local: llama.cpp (Metal) or MLX runners
- GPU: vLLM server + FastAPI proxy (OpenAI-style)
- Clean prompt templating and consistent interface
- Structured logging, health checks, simple config in YAML
- Uses uv (Python package manager) for fast, reproducible installs

---

## 0) Prereqs

- uv: https://docs.astral.sh/uv/  (install once: `pipx install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Apple Silicon (macOS 13+), or Linux GPU box (CUDA 12+ for vLLM image)
- Model files:
  - GGUF: place into `models/gguf/` (e.g., `OpenBioLLM-Llama3-8B-Q4_K_M.gguf`)
  - HF model id for GPU (default: `aaditya/Llama3-OpenBioLLM-8B`)

Note: You are responsible for model licenses and usage compliance.

---

## 1) Create env

```bash
uv venv .venv
source .venv/bin/activate
```

### Apple Silicon (llama.cpp route)
```bash
uv pip install -e ".[mac-gguf]"
```

### Apple Silicon (MLX route)
```bash
uv pip install -e ".[mac-mlx]"
```

### GPU Server
```bash
uv pip install -e ".[gpu]"
```

---

## 2) Local run on Mac (GGUF via llama.cpp)

1) Put your `.gguf` at `models/gguf/OpenBioLLM-Llama3-8B-Q4_K_M.gguf`
2) Edit `configs/local_gguf.yaml` if needed
3) Run:

```bash
python -m openbiollm_inference.local_m1.run_gguf --text "Summarize: patient with fever and cough."
```

## 3) Local run on Mac (MLX)

```bash
python -m openbiollm_inference.local_m1.run_mlx --text "Summarize: patient with fever and cough."
```

---

## 4) GPU deployment (vLLM + FastAPI)

### Compose
Set a proper NVIDIA host with CUDA drivers. Then:

```bash
cp .env.example .env
# edit .env if needed
docker compose -f docker/docker-compose.gpu.yml up --build
```

This starts:
- vllm serving OpenAI-compatible API
- app FastAPI proxy at http://localhost:8000

Notes:
- The `app` container installs only the `.[app]` extra (FastAPI, Uvicorn, httpx, etc.); the `vllm` dependency lives only in the `vllm` container.
- The proxy reads the system prompt from `configs/prompt_template.txt`; on startup it warns if the template lacks `<|system|>`/`<|user|>` tags.

Test:
```bash
curl -X POST http://localhost:8000/generate -H "Content-Type: application/json"   -d '{"input":"Give two differentials for fever in adults."}'
```

---

## 5) Project layout

```
openbiollm inference/
├─ README.md
├─ pyproject.toml
├─ .gitignore
├─ .env.example
├─ configs/
│  ├─ prompt_template.txt
│  ├─ local_gguf.yaml
│  └─ gpu_vllm.yaml
├─ models/
│  └─ gguf/
│     └─ .gitkeep
├─ docker/
│  ├─ Dockerfile.vllm
│  └─ docker-compose.gpu.yml
└─ src/
   └─ openbiollm_inference/
      ├─ __init__.py
      ├─ common/
      │  ├─ logging_setup.py
      │  ├─ schema.py
      │  └─ templates.py
      ├─ local_m1/
      │  ├─ run_gguf.py
      │  └─ run_mlx.py
      └─ serve_gpu/
         ├─ fastapi_app.py
         └─ vllm_server.py
```

---

## 6) Notes

- Keep your prompt template stable between local and GPU serving for consistent behavior.
- For production: add authentication in FastAPI, rate-limits at NGINX, and observability stack (Loki/Grafana).
- To extend: add a RAG service and a prompt registry with Redis caching.

---

## 7) Tests

Install dev deps and run tests:

```bash
uv pip install -e ".[app]" -e ".[dev]"
pytest -q
```
