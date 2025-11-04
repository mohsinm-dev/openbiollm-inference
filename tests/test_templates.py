from __future__ import annotations

from openbiollm_inference.common.templates import render_prompt, load_template
from openbiollm_inference.serve_gpu.fastapi_app import _extract_system


def test_render_prompt_substitution() -> None:
    tpl = "Hello {{input}}!"
    out = render_prompt(tpl, "world")
    assert out == "Hello world!"


def test_extract_system_from_repo_template() -> None:
    tpl = load_template()
    sys_prompt = _extract_system(tpl)
    assert isinstance(sys_prompt, str)
    assert "clinical assistant" in sys_prompt

