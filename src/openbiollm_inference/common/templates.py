"""Prompt templating helpers."""
from __future__ import annotations
from pathlib import Path

def load_template(path: str = "configs/prompt_template.txt") -> str:
    """
    Load a prompt template file.

    Args:
        path: Path to template.
    """
    return Path(path).read_text(encoding="utf-8")

def render_prompt(template: str, user_input: str) -> str:
    """
    Render user input into the template.

    Args:
        template: Template content containing {{input}}.
        user_input: Input string.

    Returns:
        Rendered prompt.
    """
    return template.replace("{{input}}", user_input)
