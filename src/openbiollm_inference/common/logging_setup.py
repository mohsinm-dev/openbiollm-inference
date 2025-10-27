"""Central logging setup for the project."""
from __future__ import annotations
import logging
import sys

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with sane defaults.

    Args:
        level: Logging level.
    """
    handler = logging.StreamHandler(sys.stdout)
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
