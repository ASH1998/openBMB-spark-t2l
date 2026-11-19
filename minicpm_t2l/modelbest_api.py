"""Client for the ModelBest hosted MiniCPM API (OpenAI-compatible).

Optional: for quick experiments and sanity checks only. The plan's comparisons
run the model locally/on Modal, so results from this API are not comparable to
those runs (unknown quantisation, decoding defaults and serving stack).

Settings come from .env: MODELBEST_API_KEY, MODELBEST_BASE_URL, MODELBEST_MODEL.

    python -m minicpm_t2l.modelbest_api "Hello"
    python -m minicpm_t2l.modelbest_api "What is in this image?" --image page.png
"""

from __future__ import annotations

import base64
import mimetypes
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_ROOT / ".env")

DEFAULT_BASE_URL = "https://api.modelbest.ai/v1"
DEFAULT_MODEL = "MiniCPM-V-4.7-1B"


def get_client() -> OpenAI:
    key = os.environ.get("MODELBEST_API_KEY")
    if not key:
        raise RuntimeError("MODELBEST_API_KEY is not set (add it to .env)")
    return OpenAI(api_key=key, base_url=os.environ.get("MODELBEST_BASE_URL", DEFAULT_BASE_URL))


def default_model() -> str:
    return os.environ.get("MODELBEST_MODEL", DEFAULT_MODEL)


def image_part(path: str | Path) -> dict:
    """Build an OpenAI-style image content part from a local file."""
    path = Path(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def chat(prompt: str, images: list[str | Path] | None = None, model: str | None = None, **kwargs) -> str:
    """Send one user turn (text plus optional images) and return the reply text."""
    content: str | list[dict] = prompt
    if images:
        content = [*(image_part(p) for p in images), {"type": "text", "text": prompt}]
    resp = get_client().chat.completions.create(
        model=model or default_model(),
        messages=[{"role": "user", "content": content}],
        **kwargs,
    )
    return resp.choices[0].message.content or ""


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("prompt", nargs="?", default="Hello")
    ap.add_argument("--image", action="append", default=[], help="local image path (repeatable)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    print(chat(args.prompt, images=args.image, model=args.model))
    sys.exit(0)
