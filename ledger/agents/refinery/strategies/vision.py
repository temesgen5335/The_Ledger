"""Vision-augmented extraction via configurable VLMs (Gemini, OpenAI, Anthropic)."""

import os
from pathlib import Path
from typing import Callable

import pymupdf

from ledger.agents.refinery.models.document_profile import DocumentProfile
from ledger.agents.refinery.models.extracted_document import (
    ExtractedDocument,
    ExtractedPage,
    ExtractedTable,
    ExtractedFigure,
)
from ledger.agents.refinery.strategies.base import BaseExtractor

# Fallback model IDs used when the corresponding env var is not set.
# Override these via GEMINI_VISION_MODEL / OPENAI_VISION_MODEL / ANTHROPIC_VISION_MODEL in .env
_FALLBACK_GEMINI_VISION_MODEL = "gemini-3.0-flash"
_FALLBACK_OPENAI_VISION_MODEL = "gpt-4o"
_FALLBACK_ANTHROPIC_VISION_MODEL = "claude-3-5-sonnet-20241022"

EXTRACTION_PROMPT = (
    "Extract all text from this document page. "
    "Preserve structure (paragraphs, lists). Output plain text only."
)


def _load_dotenv_if_available() -> None:
    """Load .env from project root when python-dotenv is installed (no-op otherwise)."""
    try:
        from dotenv import load_dotenv
        base = Path(__file__).resolve().parent.parent.parent
        load_dotenv(base / ".env")
    except ImportError:
        pass


def _vision_config_from_env() -> tuple[list[str], dict[str, str], dict[str, str]]:
    """Load provider order, model names, and API keys from environment.
    Returns (provider_order, models_by_provider, api_keys_by_provider)."""
    _load_dotenv_if_available()
    order_raw = os.environ.get("VISION_PROVIDER_ORDER", "gemini,openai,anthropic")
    provider_order = [p.strip().lower() for p in order_raw.split(",") if p.strip()]

    models = {
        "gemini": os.environ.get("GEMINI_VISION_MODEL", _FALLBACK_GEMINI_VISION_MODEL).strip(),
        "openai": os.environ.get("OPENAI_VISION_MODEL", _FALLBACK_OPENAI_VISION_MODEL).strip(),
        "anthropic": os.environ.get("ANTHROPIC_VISION_MODEL", _FALLBACK_ANTHROPIC_VISION_MODEL).strip(),
    }

    keys = {
        "gemini": (os.environ.get("GEMINI_API_KEY") or "").strip(),
        "openai": (os.environ.get("OPENAI_API_KEY") or "").strip(),
        "anthropic": (os.environ.get("ANTHROPIC_API_KEY") or "").strip(),
    }

    return provider_order, models, keys


def _make_gemini_caller(model: str, api_key: str) -> Callable[[bytes], str] | None:
    if not api_key:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)

        def call(image_bytes: bytes) -> str:
            response = client.models.generate_content(
                model=model,
                contents=[
                    EXTRACTION_PROMPT,
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                ],
            )
            return (response.text or "").strip()

        return call
    except Exception:
        return None


def _make_openai_caller(model: str, api_key: str) -> Callable[[bytes], str] | None:
    if not api_key:
        return None
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        def call(image_bytes: bytes) -> str:
            import base64
            b64 = base64.standard_b64encode(image_bytes).decode("ascii")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": EXTRACTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ]},
                ],
                max_tokens=4096,
            )
            return (response.choices[0].message.content or "").strip()
        return call
    except Exception:
        return None


def _make_anthropic_caller(model: str, api_key: str) -> Callable[[bytes], str] | None:
    if not api_key:
        return None
    try:
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

        def call(image_bytes: bytes) -> str:
            import base64
            b64 = base64.standard_b64encode(image_bytes).decode("ascii")
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": EXTRACTION_PROMPT},
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                        ],
                    }
                ],
            )
            text = response.content[0].text if response.content else ""
            return text.strip()
        return call
    except Exception:
        return None


def _build_available_providers(
    provider_order: list[str],
    models: dict[str, str],
    keys: dict[str, str],
    api_key_override: str | None,
) -> list[tuple[str, Callable[[bytes], str]]]:
    """Build list of (provider_name, call_fn) in priority order. Uses api_key_override for Gemini when provided."""
    builders = {
        "gemini": lambda: _make_gemini_caller(models["gemini"], api_key_override or keys["gemini"]),
        "openai": lambda: _make_openai_caller(models["openai"], keys["openai"]),
        "anthropic": lambda: _make_anthropic_caller(models["anthropic"], keys["anthropic"]),
    }
    out: list[tuple[str, Callable[[bytes], str]]] = []
    for name in provider_order:
        if name not in builders:
            continue
        fn = builders[name]()
        if fn is not None:
            out.append((name, fn))
    return out


def _vlm_page_confidence(text: str | None) -> float:
    """Score a VLM-extracted page based on text yield and content quality.

    VLMs generally produce high-quality output when they return text at all,
    so the main failure mode is an empty or very short response (API error,
    blank/decorative page, image the model cannot read).
    """
    text_len = len((text or "").strip())
    if text_len >= 300:
        return 0.95
    if text_len >= 100:
        return 0.75 + 0.20 * (text_len - 100) / 200
    if text_len >= 20:
        return 0.45 + 0.30 * (text_len - 20) / 80
    if text_len > 0:
        return 0.2 + 0.25 * (text_len / 20)
    return 0.0


class VisionExtractor(BaseExtractor):
    """Extract via configurable VLMs (Gemini, OpenAI, Anthropic) with primary/fallback; enforces budget_guard."""

    def __init__(
        self,
        max_cost_per_document_usd: float = 1.0,
        cost_per_image_usd: float = 0.00035,
        api_key: str | None = None,
        provider_order: list[str] | None = None,
        models: dict[str, str] | None = None,
        api_keys: dict[str, str] | None = None,
    ):
        self.max_cost_per_document_usd = max_cost_per_document_usd
        self.cost_per_image_usd = cost_per_image_usd
        # Load from env if not passed
        if provider_order is None or models is None or api_keys is None:
            _order, _models, _keys = _vision_config_from_env()
            provider_order = provider_order or _order
            models = models or _models
            api_keys = api_keys or _keys
        self._providers = _build_available_providers(provider_order, models, api_keys, api_key)

    def extract(
        self,
        doc_path: Path | str,
        profile: DocumentProfile,
        page_numbers: set[int] | None = None,
    ) -> ExtractedDocument:
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        if not self._providers:
            return ExtractedDocument(
                doc_id=profile.doc_id,
                profile=profile,
                pages=[ExtractedPage(page_number=1, text="", blocks=[], strategy_used="vision")],
                tables=[],
                figures=[],
                strategy_used="vision",
                confidence_score=0.0,
            )

        pages: list[ExtractedPage] = []
        cost_estimate = 0.0

        doc = pymupdf.open(path)
        try:
            for i in range(len(doc)):
                page_num = i + 1
                if page_numbers is not None and page_num not in page_numbers:
                    continue
                if cost_estimate >= self.max_cost_per_document_usd:
                    break
                page = doc[i]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                text = self._call_vlm_for_page(img_bytes, page_num)
                page_conf = _vlm_page_confidence(text)
                pages.append(
                    ExtractedPage(
                        page_number=page_num, text=text or "", blocks=[],
                        strategy_used="vision", confidence=page_conf,
                    )
                )
                cost_estimate += self.cost_per_image_usd
        finally:
            doc.close()

        if not pages:
            pages.append(ExtractedPage(page_number=1, text="", blocks=[], strategy_used="vision"))

        page_confs = [p.confidence for p in pages if p.confidence is not None]
        doc_confidence = sum(page_confs) / len(page_confs) if page_confs else 0.0

        return ExtractedDocument(
            doc_id=profile.doc_id,
            profile=profile,
            pages=pages,
            tables=[],
            figures=[],
            strategy_used="vision",
            confidence_score=round(doc_confidence, 4),
        )

    def _call_vlm_for_page(self, image_bytes: bytes, page_no: int) -> str:
        """Try primary VLM then fallbacks; return extracted text or empty string."""
        last_error: Exception | None = None
        for provider_name, call_fn in self._providers:
            try:
                text = call_fn(image_bytes)
                if text:
                    return text
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            raise last_error
        return ""
