"""Pydantic models and dataclasses for request/response types."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class GenResponse:
    """Text generation response metadata."""
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
