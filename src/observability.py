from __future__ import annotations
import logging
import time
from contextlib import contextmanager

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("multi-agent")


@contextmanager
def span(name: str):
    attrs: dict = {}
    start = time.perf_counter()
    try:
        yield attrs
    finally:
        ms = (time.perf_counter() - start) * 1000
        extra = " ".join(f"{k}={v}" for k, v in attrs.items())
        log.info("node=%-12s duration_ms=%-6.0f %s", name, ms, extra)
