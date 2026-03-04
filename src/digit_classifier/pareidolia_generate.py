"""Generate a dataset of AI-generated images where digits 0-9 are implied by
out-of-distribution real-world objects (pareidolia).

Config files (required, in this package directory):
  - pareidolia_variety.txt: ~100 options per category (capture_medium, setting, etc.)
  - pareidolia_digits.txt: digit shape variations
  Regenerate variety: python scripts/generate_pareidolia_variety.py

Pipeline:
1) LLM (OpenAI Responses API or Gemini) writes a single image-generator prompt per sample.
2) Image model (OpenAI Images API or Gemini) generates an image from that prompt.
3) Save images + JSONL metadata.

Batch mode: OpenAI Batch API (prompts+images) or Gemini Batch API (prompts+images).
  - 50%% cost reduction, ~24h turnaround.
  - Use --batch with --llm-provider/--provider to enable.

Requirements:
  pip install digit-classifier[pareidolia]
  # Includes: openai, pillow, tqdm, google-genai

Env:
  # For OpenAI (prompts and/or images):
  export OPENAI_API_KEY="..."
  # For Gemini (prompts and/or images):
  export GEMINI_API_KEY="..."   # or GOOGLE_API_KEY

Optional knobs:
  export LLM_MODEL="gpt-5.2" | "gemini-3.1-flash-lite-preview"
  export IMAGE_MODEL="gpt-image-1" | "gemini-3.1-flash-image-preview"
"""

from __future__ import annotations

import base64
import io
import json
import os
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
from PIL import Image
from tqdm import tqdm


# -------- Resolution options per provider --------

OPENAI_SIZE_OPTIONS = ("auto", "1024x1024", "1536x1024", "1024x1536")
GEMINI_IMAGE_SIZE_OPTIONS = ("1K", "2K", "4K")


def list_resolution_options() -> None:
    """Print available resolution options per provider."""
    print("Resolution options by provider:\n")
    print("  OpenAI (--size):")
    for s in OPENAI_SIZE_OPTIONS:
        print(f"    {s}")
    print("\n  Gemini / Imagen (--image-size):")
    for s in GEMINI_IMAGE_SIZE_OPTIONS:
        print(f"    {s}")
    print()


# -------- Prompt template (LLM generates the *image prompt*) --------

_DATA_DIR = Path(__file__).resolve().parent


def _load_variety_config() -> Dict[str, Tuple[str, ...]]:
    """Load variety categories from pareidolia_variety.txt (~100 options per category)."""
    path = _DATA_DIR / "pareidolia_variety.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Pareidolia variety config not found: {path}\n"
            "Generate it with: python scripts/generate_pareidolia_variety.py"
        )
    result: Dict[str, List[str]] = {}
    current: Optional[str] = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current = line[1:-1].strip()
                if current:
                    result[current] = []
                continue
            if current is not None and line:
                result[current].append(line)
    return {k: tuple(v) for k, v in result.items() if v}


def _load_digit_variations() -> Dict[int, Tuple[str, ...]]:
    """Load digit shape variations from pareidolia_digits.txt."""
    path = _DATA_DIR / "pareidolia_digits.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Pareidolia digit variations not found: {path}\n"
            "The file should be committed with the package."
        )
    result: Dict[int, List[str]] = {}
    current: Optional[int] = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[digit_") and "]" in line:
                try:
                    current = int(line[7:line.index("]")])
                    result[current] = []
                except (ValueError, IndexError):
                    current = None
                continue
            if current is not None and line:
                result[current].append(line)
    return {k: tuple(v) for k, v in result.items() if v}


# Load config from text files at module init (pareidolia_variety.txt, pareidolia_digits.txt)
_VARIETY_RAW = _load_variety_config()
_DIGIT_VARIATIONS = _load_digit_variations()
_UNIQUENESS_AVOID: Tuple[str, ...] = _VARIETY_RAW.get(
    "uniqueness_avoid",
    (),  # Required in pareidolia_variety.txt
)
VARIETY_CATEGORIES: Dict[str, Tuple[str, ...]] = {
    k: v for k, v in _VARIETY_RAW.items() if k != "uniqueness_avoid"
}


def sample_variety_hint() -> str:
    """Sample one option from each category and combine into a structured variety hint."""
    parts: List[str] = []
    for category, options in VARIETY_CATEGORIES.items():
        choice = random.choice(options)
        label = category.replace("_", " ").title()
        parts.append(f"{label}: {choice}")
    return "; ".join(parts)


def sample_uniqueness_avoid() -> str:
    """Sample one style to explicitly avoid for this image."""
    opts = _UNIQUENESS_AVOID or ("do NOT use dark moody underground aesthetic",)
    return random.choice(opts)


def get_digit_hint(digit: int) -> str:
    """Get digit-specific shape guidance for the LLM (samples from variations)."""
    variations = _DIGIT_VARIATIONS.get(digit, ())
    if not variations:
        return ""
    variation = random.choice(variations)
    return f"The digit {digit} shape: {variation}."


def _sample_discrete_params() -> str:
    """Sample object count, variety, and spacing for discrete formations."""
    count = random.choice([
        "3–5 objects",
        "5–8 objects",
        "8–12 objects",
    ])
    variety = random.choice([
        "single type (e.g. all coins, all pills, all mugs)",
        "mixed types (2–3 kinds, e.g. pills and coins, or mugs and fruit)",
    ])
    spacing = random.choice([
        "tightly clustered",
        "loosely spaced",
        "varied (some tight, some loose)",
    ])
    return f"{count}, {variety}, {spacing}"


def sample_formation_type() -> str:
    """Sample formation type: continuous (varied) or discrete (with randomized params)."""
    if random.random() < 0.5:
        # Continuous
        return random.choice([
            "continuous stroke: folded fabric, paper crease, bent metal sheet, rust streak, or paint line—NOT cables, wires, rope, or hose",
            "continuous stroke: shadow, crack, groove, or similar (e.g. shadow on wall, crack in pavement, groove in wood)",
            "continuous stroke: cable, wire, tape, rope, hose, liquid trail, or paint ribbon",
        ])
    # Discrete: randomize object count, variety, spacing
    params = _sample_discrete_params()
    return (
        f"discrete objects: {params}; e.g. pills, mugs, coins, candies, stones, books, fruit, toys, Lego bricks, tools; "
        "AVOID cables, wires, rope, hose, tape, liquid trails"
    )


def _format_llm_prompt(digit: int) -> str:
    """Build the full LLM prompt with variety, uniqueness, and digit-specific hints."""
    hint = get_digit_hint(digit)
    digit_hint_block = f"\n- {hint}\n" if hint else ""
    return LLM_PROMPT_TEMPLATE.format(
        digit=digit,
        digit_hint=digit_hint_block,
        formation_type=sample_formation_type(),
        variety_hint=sample_variety_hint(),
        uniqueness_avoid=sample_uniqueness_avoid(),
    )

LLM_PROMPT_TEMPLATE = r"""
You are an expert prompt-writer for AI image generators. Create ONE final image-generation prompt that will produce a PHOTOREALISTIC image where the digit "{digit}" (0–9) is strongly implied by real-world objects or natural arrangements, but NOT shown as printed text, signage, typed numbers, digital overlays, or any explicit numeral glyph.

CRITICAL: The output must look like a real photograph—documentary, candid, or snapshot quality. NOT stylized, NOT surreal, NOT AI-art aesthetic. A viewer should believe it could have been taken by a real camera in the real world.

Core concept:
- The digit "{digit}" must emerge via pareidolia: objects in the scene happen to form the digit's shape.
- For this image you MUST use: {formation_type}
- Use varied materials and contexts appropriate to that formation type. Variety comes from setting and materials, not from surreal styling.
- The result must be physically plausible and believable for the chosen capture medium.
{digit_hint}

Hard constraints (must follow):
- Do NOT include any literal digits, printed numbers, text labels, logos, watermarks, UI overlays, or captions.
- Do NOT say "looks like a {digit}" or "shaped like {digit}" in the prompt. Describe the scene so the shape is implied naturally.
- The digit must be readable from the main viewpoint (clear gestalt). Avoid ambiguity.
- The digit must be UPRIGHT as the viewer would see it (like on a clock or display). Explicitly specify "upright", "right-side up", or "oriented for normal reading" in the prompt. Do NOT use tilted, rotated, Dutch-angle, or overhead views that make the digit hard to read.
- For digits 6 and 9, specify orientation clearly (e.g., "upright" or "upside-down") so the viewer knows which digit is intended.
- Use a plausible attention anchor: natural colors and materials (e.g., yellow caution tape, orange traffic cone, red cable, rust streak, bright fabric) that make the digit-forming elements stand out. AVOID neon, fluorescent, or artificially saturated colors.
- Make sure sufficient specification is made in the prompt that the image generator will produce the correct digit shape, even without explicitly naming the digit.

Variety requirements (critical):
- You MUST use this exact combination of attributes for your image. Build the entire scene around ALL of these (do not omit any):
{variety_hint}
- For this image: {uniqueness_avoid}.
- AVOID overused pareidolia aesthetics: underground, endoscope, pipeline, tunnel, sewer, drain, circular vignette, black circular frame, dark void, ring-LED in darkness, perfectly centered subject in void.
- AVOID: vines/roots on walls, cracks in pavement, spilled liquids, abstract patterns, driftwood, seaweed, shadows on flat surfaces.
- Style: use 2–3 subtle, plausible capture attributes (e.g., "overcast daylight, slight lens softness at edges" or "dashcam, natural street lighting"). Real photos have imperfections but they are SUBTLE. Do NOT stack 6–10 exaggerated artifacts (no "sickly" color casts, no heavy chromatic aberration, no aggressive vignetting, no neon false-color).
- Color and lighting: use natural, believable descriptions (overcast daylight, warm tungsten, cool shade, mixed fluorescent). NO "sickly", "aggressive", "neon", "glaring", or artificially saturated language.
- Your style choices must be testable and visual, but grounded in real-world photography.

Process you must follow internally:
1) Brainstorm 4–6 candidate concepts that could imply "{digit}" using the required formation type ({formation_type}). Use varied materials and settings.
2) Select the most visually clear concept that would look believable as a real photograph.
3) Pair it with a capture style that a real camera would produce—subtle, plausible, not stylized.

Final prompt output requirements:
- Output ONE image-generator prompt (one paragraph is fine) that includes:
  - The scene and the digit-implying objects (what they are, where they are, how arranged)
  - Explicit statement that the digit is upright/right-side-up for normal viewing
  - Environment/context details (surface, surrounding items, time/weather if relevant)
  - Camera/viewpoint and composition details (avoid tilted or rotated views that obscure the digit)
  - Lighting details (natural, plausible)
  - 2–3 subtle capture/optics attributes (not a long list of exaggerated artifacts)
  - Explicit negative constraints at the end: "no text, no numbers, no logos, no watermark, no UI overlay"
- Add "photorealistic, documentary photograph, real-world lighting" or similar to reinforce believability.
- Keep it generator-friendly: concrete nouns, physical details, minimal abstraction.
- Return ONLY the final image-generator prompt text. No headings, no bullet points, no analysis.
""".strip()

# Suffix appended to the LLM-generated prompt before sending to the image model.
_IMAGE_PROMPT_DIGIT_SUFFIX = (
    "\n\nCritical: The arrangement of objects must form the digit {digit} clearly and legibly. "
    "No printed text, numbers, signage, or digital overlays. "
    "Photorealistic, documentary-style photograph with natural lighting and believable materials."
)
_IMAGE_PROMPT_DIGIT_4_EXTRA = (
    " The digit 4 specifically needs: a vertical element on the right, a horizontal element across the top, "
    "and a diagonal from the upper-left meeting the vertical. Sharp, clean angles (whether strokes or discrete objects)."
)


def _build_image_prompt(llm_prompt: str, digit: int) -> str:
    """Augment the LLM prompt with the target digit so the image model knows the shape."""
    suffix = _IMAGE_PROMPT_DIGIT_SUFFIX.format(digit=digit)
    if digit == 4:
        suffix += _IMAGE_PROMPT_DIGIT_4_EXTRA
    return llm_prompt + suffix


# -------- Cost estimation (approximate; pricing changes) --------
# Sources: platform.openai.com/docs/pricing, ai.google.dev/gemini-api/docs/pricing

# Per-sample token estimates
_PROMPT_INPUT_TOKENS = 800  # LLM template
_PROMPT_OUTPUT_TOKENS = 200  # image prompt

# Extra output tokens when thinking is enabled (thinking tokens billed as output)
_THINKING_EXTRA_TOKENS: Dict[str, int] = {
    "low": 2000,
    "high": 8000,
}

# $ per 1M tokens (input, output) - sync pricing
_LLM_PRICING: Dict[str, Tuple[float, float]] = {
    # OpenAI
    "gpt-5.2": (1.75, 14.0),
    "gpt-5.1": (1.25, 10.0),
    "gpt-5-mini": (0.25, 2.0),
    "gpt-5-nano": (0.05, 0.40),
    # Gemini
    "gemini-3.1-flash-lite-preview": (0.25, 1.50),
    "gemini-3.1-flash-preview": (0.25, 1.50),
    "gemini-3-flash-preview": (0.25, 1.50),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-flash-lite": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-3.1-pro-preview": (2.0, 12.0),
    "gemini-2.5-pro": (1.25, 5.0),
}

# $ per image - sync pricing (base; OpenAI uses quality multiplier)
_IMAGE_PRICING: Dict[str, float] = {
    # OpenAI (gpt-image-1: medium=0.04, high=0.17, low=0.01; auto≈medium)
    "gpt-image-1": 0.04,
    "gpt-image-1-mini": 0.04,
    "gpt-image-1.5": 0.04,
    # Gemini native
    "gemini-3.1-flash-image-preview": 0.067,
    "gemini-2.5-flash-image": 0.039,
    "gemini-2.5-flash-preview-image": 0.05,
    # Imagen
    "imagen-4.0-generate-001": 0.04,
    "imagen-4.0-fast-generate-001": 0.02,
    "imagen-4.0-ultra-generate-001": 0.06,
    "imagen-3.0-generate-001": 0.03,
}

# OpenAI quality multiplier (gpt-image-1: low $0.01, medium $0.04, high $0.17)
_OPENAI_QUALITY_MULTIPLIER: Dict[str, float] = {
    "low": 0.25,
    "medium": 1.0,
    "high": 4.25,
    "auto": 1.0,
}

# Gemini/Imagen: resolution multiplier (1K base; OpenAI size is quality-only, not resolution)
# Gemini 3.1 Flash Image: 1K $0.067, 2K ~$0.10, 4K $0.151
_GEMINI_RESOLUTION_MULTIPLIER: Dict[str, float] = {
    "1K": 1.0,
    "2K": 1.49,   # ~$0.10 / $0.067
    "4K": 2.25,   # $0.151 / $0.067
}

_BATCH_DISCOUNT = 0.5  # 50% off in batch mode


def _estimate_cost(
    total: int,
    llm_provider: str,
    llm_model: str,
    provider: str,
    image_model: str,
    use_batch: bool,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[str] = None,
    quality: str = "auto",
    image_size: Optional[str] = None,
) -> Tuple[float, str]:
    """Estimate cost in USD. Returns (cost, breakdown)."""
    prompt_cost = 0.0
    image_cost = 0.0

    # Prompt cost
    in_price, out_price = _LLM_PRICING.get(
        llm_model,
        _LLM_PRICING.get(
            "gemini-3.1-flash-lite-preview" if llm_provider == "gemini" else "gpt-5.2",
            (0.5, 2.0),
        ),
    )
    if use_batch:
        in_price *= _BATCH_DISCOUNT
        out_price *= _BATCH_DISCOUNT

    # Extra output tokens from thinking (billed as output)
    extra_output = 0
    if thinking_budget is not None and thinking_budget > 0:
        extra_output = min(thinking_budget, 32768)  # cap at max
    elif thinking_level is not None:
        extra_output = _THINKING_EXTRA_TOKENS.get(
            thinking_level.lower(), _THINKING_EXTRA_TOKENS["low"]
        )

    prompt_cost = total * (
        (in_price * _PROMPT_INPUT_TOKENS + out_price * (_PROMPT_OUTPUT_TOKENS + extra_output))
        / 1_000_000
    )

    # Image cost
    _img_fallback = (
        "gemini-3.1-flash-image-preview"
        if provider == "gemini"
        else "imagen-4.0-generate-001"
        if provider == "gemini-imagen"
        else "gpt-image-1"
    )
    img_price = _IMAGE_PRICING.get(image_model, _IMAGE_PRICING.get(_img_fallback, 0.05))
    if provider == "openai":
        q_mult = _OPENAI_QUALITY_MULTIPLIER.get(
            quality.lower(), _OPENAI_QUALITY_MULTIPLIER["auto"]
        )
        img_price *= q_mult
    elif provider in ("gemini", "gemini-imagen") and image_size is not None:
        res_mult = _GEMINI_RESOLUTION_MULTIPLIER.get(
            image_size.upper(), _GEMINI_RESOLUTION_MULTIPLIER["1K"]
        )
        img_price *= res_mult
    if use_batch:
        img_price *= _BATCH_DISCOUNT
    image_cost = total * img_price

    total_cost = prompt_cost + image_cost
    thinking_note = ""
    if extra_output > 0:
        thinking_note = f" (incl. ~{extra_output} thinking tokens/sample)"
    res_note = f" @{image_size}" if (provider in ("gemini", "gemini-imagen") and image_size) else ""
    breakdown = (
        f"  Prompts ({llm_model}){thinking_note}: ~${prompt_cost:.2f}\n"
        f"  Images ({image_model}{res_note}): ~${image_cost:.2f}\n"
        f"  Total: ~${total_cost:.2f}"
    )
    return total_cost, breakdown


def _confirm_cost(estimate: str, total: int, skip_confirm: bool) -> bool:
    """Prompt user to confirm. Returns True to proceed."""
    print(f"\nEstimated cost for {total} samples:\n{estimate}")
    print("\n(Pricing is approximate; actual costs may vary.)")
    if skip_confirm:
        return True
    try:
        resp = input("\nProceed? [y/N]: ").strip().lower()
        return resp in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# -------- Metadata --------


@dataclass
class SampleMeta:
    sample_id: str
    digit: int
    llm_model: str
    image_model: str
    provider: str
    created_utc: str
    prompt_for_image: str
    image_path: str
    size: str
    quality: str
    output_format: str
    prompt_to_llm: Optional[str] = None  # Input prompt sent to the LLM (prompt-generating model)


# -------- Batch helpers (OpenAI Batch API) --------


def _extract_responses_text(body: Dict[str, Any]) -> str:
    """Best-effort extraction of text from a Responses API JSON body."""
    # Newer Responses API returns `output` as a list of items; each item has `content`.
    out = body.get("output")
    parts: List[str] = []
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        t = c.get("text")
                        if isinstance(t, str):
                            parts.append(t)
            # Some variants nest text differently
            t2 = item.get("text")
            if isinstance(t2, str):
                parts.append(t2)
    # Fallback: some SDK helpers expose `output_text`, but batch outputs are raw JSON.
    if not parts:
        maybe = body.get("output_text")
        if isinstance(maybe, str):
            parts.append(maybe)
    text = "\n".join(p.strip() for p in parts if p and p.strip()).strip()
    return text


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _openai_upload_batch_file(client: OpenAI, jsonl_path: Path) -> str:
    """Upload a JSONL file for the OpenAI Batch API and return file_id."""
    with jsonl_path.open("rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    return uploaded.id


def _openai_run_batch(
    client: OpenAI,
    input_file_id: str,
    endpoint: str,
    poll_seconds: float = 10.0,
) -> Tuple[str, str]:
    """Create a batch, poll until completion, and return (output_file_id, batch_id)."""
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint=endpoint,
        completion_window="24h",
    )
    batch_id = batch.id

    while True:
        b = client.batches.retrieve(batch_id)
        status = getattr(b, "status", "unknown")
        if status in ("completed", "failed", "cancelled", "expired"):
            if status != "completed":
                raise RuntimeError(f"Batch {batch_id} ended with status={status}")
            out_id = getattr(b, "output_file_id", None)
            if not out_id:
                raise RuntimeError(f"Batch {batch_id} completed but output_file_id is empty")
            return out_id, batch_id
        time.sleep(poll_seconds)


def _openai_download_file_bytes(client: OpenAI, file_id: str) -> bytes:
    """Download a file from OpenAI Files API."""
    # The Python SDK returns a binary stream; we normalize to bytes.
    content = client.files.content(file_id)
    data = content.read() if hasattr(content, "read") else bytes(content)
    return data


def _parse_openai_batch_output(jsonl_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    """Parse OpenAI batch output JSONL into {custom_id: line_obj}."""
    out: Dict[str, Dict[str, Any]] = {}
    for raw_line in jsonl_bytes.splitlines():
        if not raw_line.strip():
            continue
        line = json.loads(raw_line.decode("utf-8"))
        cid = line.get("custom_id")
        if isinstance(cid, str):
            out[cid] = line
    return out


# -------- Gemini Batch API helpers --------


def _gemini_run_batch(
    model: str,
    requests: List[Dict[str, Any]],
    poll_seconds: float = 10.0,
    response_modalities: Optional[List[str]] = None,
    thinking_config: Optional[Any] = None,
    image_size: Optional[str] = None,
    temperature: Optional[float] = None,
) -> List[Any]:
    """Run a Gemini batch job and return inlined_responses.

    requests: list of InlinedRequest dicts with contents, optional metadata.
    response_modalities: e.g. [Modality.IMAGE, Modality.TEXT] for image gen.
    thinking_config: optional ThinkingConfig or dict for thinking (e.g. thinking_budget=1024).
    Returns: list of InlinedResponse objects (order may not match input).
    """
    from google import genai
    from google.genai import types

    client = genai.Client()

    inlined: List[Dict[str, Any]] = []
    for req in requests:
        r: Dict[str, Any] = {"contents": req["contents"]}
        if "metadata" in req:
            r["metadata"] = req["metadata"]
        config: Dict[str, Any] = {}
        if response_modalities is not None:
            config["response_modalities"] = response_modalities
            img_cfg: Dict[str, Any] = {"aspectRatio": "1:1"}
            if image_size is not None:
                img_cfg["imageSize"] = image_size
            config["image_config"] = img_cfg
        if thinking_config is not None:
            config["thinking_config"] = thinking_config
        if temperature is not None:
            config["temperature"] = temperature
        if config:
            r["config"] = config
        inlined.append(r)

    job = client.batches.create(model=model, src=inlined)

    completed = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}
    while True:
        job = client.batches.get(name=job.name)
        state = getattr(getattr(job, "state", None), "name", None) or str(getattr(job, "state", ""))
        if state in completed:
            if state != "JOB_STATE_SUCCEEDED":
                err = getattr(job, "error", None)
                raise RuntimeError(f"Gemini batch failed: state={state}, error={err}")
            break
        time.sleep(poll_seconds)

    responses = getattr(job, "inlined_responses", None) or []
    return list(responses)


def _extract_gemini_text(response: Any) -> str:
    """Extract text from a Gemini GenerateContentResponse."""
    parts = getattr(response, "candidates", None) or []
    text_parts: List[str] = []
    for c in parts:
        content = getattr(c, "content", None)
        if content is None:
            continue
        for p in getattr(content, "parts", []) or []:
            t = getattr(p, "text", None)
            if isinstance(t, str) and t.strip():
                text_parts.append(t.strip())
    return "\n".join(text_parts).strip()


def _extract_gemini_image(response: Any) -> Tuple[bytes, str]:
    """Extract image bytes from a Gemini GenerateContentResponse. Returns (bytes, ext)."""
    parts = getattr(response, "candidates", None) or []
    for c in parts:
        content = getattr(c, "content", None)
        if content is None:
            continue
        for p in getattr(content, "parts", []) or []:
            inline = getattr(p, "inline_data", None)
            if inline is None:
                continue
            data = getattr(inline, "data", None)
            mime = getattr(inline, "mime_type", None) or "image/png"
            if data is not None:
                ext = "png"
                if "jpeg" in mime or "jpg" in mime:
                    ext = "jpeg"
                elif "webp" in mime:
                    ext = "webp"
                return bytes(data), ext
    raise RuntimeError("Gemini response did not include inline image data.")


# -------- Core functions --------


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_image_prompt_openai(
    client: OpenAI,
    llm_model: str,
    digit: int,
    temperature: Optional[float] = None,
) -> Tuple[str, str]:
    """Use OpenAI Responses API to produce a single image-generation prompt.
    Returns (prompt_to_llm, prompt_for_image).
    """
    prompt_to_llm = _format_llm_prompt(digit)
    kwargs: Dict[str, Any] = {
        "model": llm_model,
        "input": prompt_to_llm,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = client.responses.create(**kwargs)
    text = (resp.output_text or "").strip()
    if not text:
        raise RuntimeError("LLM returned empty prompt text.")
    return prompt_to_llm, text


def generate_image_prompt_gemini(
    llm_model: str,
    digit: int,
    max_retries: int = 5,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[str] = None,
    temperature: Optional[float] = None,
) -> Tuple[str, str]:
    """Use Gemini to produce a single image-generation prompt.
    Returns (prompt_to_llm, prompt_for_image).
    """
    from google import genai
    from google.genai import types

    prompt_to_llm = _format_llm_prompt(digit)
    client = genai.Client()

    cfg_kw: Dict[str, Any] = {}
    if thinking_budget is not None or thinking_level is not None:
        tc_kw: Dict[str, Any] = {}
        if thinking_budget is not None:
            tc_kw["thinking_budget"] = thinking_budget
        if thinking_level is not None:
            tc_kw["thinking_level"] = thinking_level.upper()
        cfg_kw["thinking_config"] = types.ThinkingConfig(**tc_kw)
    if temperature is not None:
        cfg_kw["temperature"] = temperature
    cfg = types.GenerateContentConfig(**cfg_kw) if cfg_kw else None

    def _call() -> Any:
        return client.models.generate_content(
            model=llm_model,
            contents=[prompt_to_llm],
            config=cfg,
        )

    resp = _call_with_retry_on_429(_call, max_retries=max_retries)
    text = _extract_gemini_text(resp)
    if not text:
        raise RuntimeError("Gemini returned empty prompt text.")
    return prompt_to_llm, text


def openai_generate_image_bytes(
    client: OpenAI,
    image_model: str,
    prompt: str,
    size: str,
    quality: str,
    output_format: str,
) -> bytes:
    """Generate an image via OpenAI Images API and return raw bytes."""
    img_resp = client.images.generate(
        model=image_model,
        prompt=prompt,
        size=size,
        quality=quality,
        output_format=output_format,
        n=1,
    )
    if not img_resp.data or not getattr(img_resp.data[0], "b64_json", None):
        raise RuntimeError("Image API response missing b64_json image data.")
    return base64.b64decode(img_resp.data[0].b64_json)


def gemini_generate_image_bytes(
    prompt: str,
    model: str,
    aspect_ratio: Optional[str] = None,
    image_size: Optional[str] = None,
    max_retries: int = 5,
) -> tuple[bytes, str]:
    """Generate an image using Gemini native image generation (Nano Banana).

    Requires: `pip install google-genai`
    Auth: set GEMINI_API_KEY (or GOOGLE_API_KEY).

    Returns: (image_bytes, file_ext)
    """
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError(
            "Gemini provider requires `google-genai`. Install with: pip install google-genai"
        ) from e

    client = genai.Client()

    cfg = None
    # Config options are only supported by some Gemini image models.
    if aspect_ratio or image_size:
        ic_kw: Dict[str, Any] = {}
        if aspect_ratio is not None:
            ic_kw["aspect_ratio"] = aspect_ratio
        if image_size is not None:
            ic_kw["image_size"] = image_size
        cfg = types.GenerateContentConfig(
            image_config=types.ImageConfig(**ic_kw)
        )

    def _call() -> Any:
        return client.models.generate_content(
            model=model,
            contents=[prompt],
            config=cfg,
        )

    resp = _call_with_retry_on_429(_call, max_retries=max_retries)

    # Response contains parts; image bytes are in inline_data.
    for part in getattr(resp, "parts", []) or []:
        if getattr(part, "inline_data", None) is not None:
            inline = part.inline_data
            data = getattr(inline, "data", None)
            mime = getattr(inline, "mime_type", None) or "image/png"
            if data is None:
                continue
            # `data` is bytes in the SDK.
            ext = "png"
            if "jpeg" in mime or "jpg" in mime:
                ext = "jpeg"
            elif "webp" in mime:
                ext = "webp"
            return bytes(data), ext

    raise RuntimeError("Gemini native image response did not include inline image data.")


def gemini_imagen_generate_image_bytes(
    prompt: str,
    model: str,
    number_of_images: int = 1,
    aspect_ratio: str = "1:1",
    image_size: Optional[str] = None,
    max_retries: int = 5,
) -> tuple[bytes, str]:
    """Generate an image using Imagen via the Gemini API.

    Requires: `pip install google-genai`
    Auth: set GEMINI_API_KEY (or GOOGLE_API_KEY).

    Returns: (image_bytes, file_ext)
    """
    if number_of_images != 1:
        raise ValueError("This dataset generator currently supports number_of_images=1")

    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise RuntimeError(
            "Gemini Imagen provider requires `google-genai`. Install with: pip install google-genai"
        ) from e

    client = genai.Client()

    def _call() -> Any:
        cfg_kw: Dict[str, Any] = {"number_of_images": 1, "aspect_ratio": aspect_ratio}
        if image_size is not None:
            cfg_kw["image_size"] = image_size
        return client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(**cfg_kw),
        )

    resp = _call_with_retry_on_429(_call, max_retries=max_retries)
    imgs = getattr(resp, "generated_images", None) or []
    if not imgs:
        raise RuntimeError("Imagen response did not include generated_images.")

    img0 = imgs[0]
    # The SDK exposes PIL images via .image; we serialize to PNG for a stable dataset.
    pil = getattr(img0, "image", None)
    if pil is None:
        raise RuntimeError("Imagen generated image missing .image")

    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue(), "png"


def _parse_retry_delay_seconds(e: Exception) -> Optional[float]:
    """Extract retry delay from 429 error (e.g. '16s' or '16.558919116s'). Returns seconds or None."""
    import re
    msg = str(e)
    # Match "retry in X.XXs" or "retryDelay": "16s"
    m = re.search(r"retry in ([\d.]+)s", msg, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r'"retryDelay"\s*:\s*"([^"]+)"', msg)
    if m:
        val = m.group(1).strip()
        if val.endswith("s"):
            try:
                return float(val[:-1])
            except ValueError:
                pass
    return None


def _call_with_retry_on_429(
    fn: Any,
    *args: Any,
    max_retries: int = 5,
    **kwargs: Any,
) -> Any:
    """Call fn(*args, **kwargs), retrying on 429 with server-suggested or exponential backoff."""
    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            err_mod = getattr(type(e), "__module__", "") or ""
            if "genai" not in err_mod and "google.genai" not in str(type(e)):
                raise
            # Check for 429 (rate limit / quota)
            is_429 = False
            if hasattr(e, "response") and e.response is not None:
                is_429 = getattr(e.response, "status_code", None) == 429
            if not is_429 and "429" not in str(e):
                raise
            delay = _parse_retry_delay_seconds(e)
            if delay is None:
                delay = (2 ** attempt) * 2  # 2, 4, 8, 16, 32 seconds
            delay = min(max(delay, 1), 120)  # clamp 1s to 2 min
            if attempt < max_retries:
                tqdm.write(f"Rate limited (429). Waiting {delay:.0f}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(delay)
            else:
                raise
    raise last_err  # type: ignore


def _format_gemini_error(e: Exception) -> str:
    """Format Gemini API errors with full details for debugging quota/billing."""
    lines = [
        "",
        "=" * 60,
        "Gemini API Error (verbose)",
        "=" * 60,
        f"Type: {type(e).__name__}",
        f"Message: {e}",
    ]
    if hasattr(e, "message") and e.message:
        lines.append(f"API message: {e.message}")
    if hasattr(e, "status"):
        lines.append(f"Status: {getattr(e.status, 'value', e.status)}")
    if hasattr(e, "response") and e.response is not None:
        r = e.response
        lines.append("")
        lines.append("HTTP Response:")
        if hasattr(r, "status_code"):
            lines.append(f"  status_code: {r.status_code}")
        if hasattr(r, "headers") and r.headers:
            for k, v in list(r.headers.items())[:10]:
                lines.append(f"  {k}: {v}")
        body = None
        if hasattr(r, "text") and r.text:
            body = r.text
        elif hasattr(r, "content") and r.content:
            try:
                body = r.content.decode("utf-8", errors="replace")
            except Exception:
                body = f"(binary, {len(r.content)} bytes)"
        if body:
            lines.append("  body (full):")
            for line in body.strip().split("\n")[:30]:
                lines.append(f"    {line}")
    # Extract error details from args if present (ClientError embeds JSON)
    if e.args:
        arg0 = str(e.args[0])
        if "429" in arg0 or "RESOURCE_EXHAUSTED" in arg0:
            lines.append("")
            lines.append("Common causes for 429 RESOURCE_EXHAUSTED:")
            lines.append("  - Rate limit: too many requests per minute")
            lines.append("  - Quota: daily/monthly token limit reached")
            lines.append("  - Billing: free tier exhausted or payment required")
            lines.append("")
            lines.append("Check: https://ai.google.dev/gemini-api/docs/rate-limits")
            lines.append("Usage: https://aistudio.google.com/ (API key usage)")
    lines.append("=" * 60)
    return "\n".join(lines)


def save_image_bytes(image_bytes: bytes, out_path: Path) -> None:
    out_path.write_bytes(image_bytes)

    try:
        with Image.open(out_path) as im:
            im.verify()
    except Exception as e:
        raise RuntimeError(f"Saved image failed verification: {out_path}") from e


def run(
    out: str = "dataset_out",
    digits: str = "0-9",
    per_digit: int = 50,
    provider: str = "openai",
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    image_model: Optional[str] = None,
    size: str = "1024x1024",
    image_size: Optional[str] = None,
    quality: str = "auto",
    output_format: str = "png",
    sleep: float = 0.2,
    seed: Optional[int] = None,
    gemini_model: Optional[str] = None,
    gemini_imagen_model: Optional[str] = None,
    gemini_llm_model: Optional[str] = None,
    use_batch: bool = False,
    batch_poll_seconds: float = 10.0,
    skip_confirm: bool = False,
    max_retries: int = 5,
    thinking_budget: Optional[int] = None,
    thinking_level: Optional[str] = None,
    temperature: Optional[float] = None,
) -> None:
    """Generate pareidolia dataset."""
    llm_provider = llm_provider or provider
    if llm_provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing OPENAI_API_KEY (required for OpenAI prompt generation).")
    if llm_provider in ("gemini", "gemini-imagen") or provider in ("gemini", "gemini-imagen"):
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            raise SystemExit(
                "Gemini provider requires GEMINI_API_KEY or GOOGLE_API_KEY."
            )

    llm_model = llm_model or os.getenv("LLM_MODEL")
    image_model = image_model or os.getenv("IMAGE_MODEL")
    gemini_llm_model = gemini_llm_model or os.getenv("GEMINI_LLM_MODEL", "gemini-3.1-flash-lite-preview")
    gemini_model = gemini_model or os.getenv(
        "GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image-preview"
    )
    gemini_imagen_model = gemini_imagen_model or os.getenv(
        "GEMINI_IMAGEN_MODEL", "imagen-4.0-generate-001"
    )
    if llm_provider == "openai":
        llm_model = llm_model or "gpt-5.2"
    else:
        llm_model = llm_model or gemini_llm_model
    if provider == "openai":
        image_model = image_model or "gpt-image-1"

    if provider == "openai" and size not in OPENAI_SIZE_OPTIONS:
        raise SystemExit(
            f"Invalid --size for OpenAI. Options: {', '.join(OPENAI_SIZE_OPTIONS)}"
        )
    if provider in ("gemini", "gemini-imagen") and image_size is not None:
        if image_size not in GEMINI_IMAGE_SIZE_OPTIONS:
            raise SystemExit(
                f"Invalid --image-size for Gemini. Options: {', '.join(GEMINI_IMAGE_SIZE_OPTIONS)}"
            )

    # For metadata: OpenAI uses size; Gemini uses image_size (default 1K)
    effective_size = size if provider == "openai" else (image_size or "1K")

    if seed is not None:
        random.seed(seed)

    out_dir = Path(out)
    ensure_dir(out_dir)
    images_dir = out_dir / "images"
    ensure_dir(images_dir)

    meta_path = out_dir / "metadata.jsonl"

    # parse digits arg
    digit_list: List[int]
    if digits.strip() == "0-9":
        digit_list = list(range(10))
    elif "-" in digits:
        a, b = digits.split("-", 1)
        digit_list = list(range(int(a), int(b) + 1))
    else:
        digit_list = [int(x.strip()) for x in digits.split(",") if x.strip()]

    for d in digit_list:
        if d < 0 or d > 9:
            raise SystemExit(f"Invalid digit: {d}")

    total = len(digit_list) * per_digit

    # Cost estimate and confirmation
    effective_image_model = (
        gemini_model if provider == "gemini" else gemini_imagen_model if provider == "gemini-imagen" else image_model
    )
    _, breakdown = _estimate_cost(
        total=total,
        llm_provider=llm_provider,
        llm_model=llm_model,
        provider=provider,
        image_model=effective_image_model,
        use_batch=use_batch,
        thinking_budget=thinking_budget,
        thinking_level=thinking_level,
        quality=quality,
        image_size=image_size,
    )
    if not _confirm_cost(breakdown, total, skip_confirm):
        raise SystemExit("Aborted by user.")

    openai_client = OpenAI() if llm_provider == "openai" or provider == "openai" else None

    pbar = tqdm(total=total, desc="Generating dataset", unit="img")

    try:
        # Pre-plan all samples so we can batch them.
        planned: List[Dict[str, Any]] = []
        for digit in digit_list:
            for _ in range(per_digit):
                planned.append({
                    "sample_id": uuid.uuid4().hex[:12],
                    "digit": digit,
                })

        if use_batch and (llm_provider == "gemini" or provider == "gemini"):
            if provider == "gemini-imagen":
                raise SystemExit(
                    "Batch mode with provider=gemini-imagen is not supported. "
                    "Use provider=gemini for Gemini native image generation (e.g. gemini-3.1-flash-image-preview)."
                )
            # ---- Gemini Batch: prompts + images ----
            from google.genai import types

            # Step 1: Gemini batch for prompts
            prompt_reqs: List[Dict[str, Any]] = []
            for row in planned:
                llm_prompt = _format_llm_prompt(row["digit"])
                row["prompt_to_llm"] = llm_prompt
                prompt_reqs.append({
                    "contents": [{"parts": [{"text": llm_prompt}], "role": "user"}],
                    "metadata": {"custom_id": row["sample_id"]},
                })
            _thinking_cfg: Optional[Any] = None
            if thinking_budget is not None or thinking_level is not None:
                tc_kw: Dict[str, Any] = {}
                if thinking_budget is not None:
                    tc_kw["thinking_budget"] = thinking_budget
                if thinking_level is not None:
                    tc_kw["thinking_level"] = thinking_level.upper()
                _thinking_cfg = types.ThinkingConfig(**tc_kw)
            prompt_responses = _gemini_run_batch(
                model=llm_model,
                requests=prompt_reqs,
                poll_seconds=batch_poll_seconds,
                thinking_config=_thinking_cfg,
                temperature=temperature,
            )
            # Map responses to planned rows (metadata may not be returned; use index as fallback)
            for i, inl in enumerate(prompt_responses):
                meta = getattr(inl, "metadata", None) or {}
                cid = meta.get("custom_id") if isinstance(meta, dict) else None
                if not cid and i < len(planned):
                    cid = planned[i]["sample_id"]
                row = next((r for r in planned if r["sample_id"] == cid), None)
                if row is None and i < len(planned):
                    row = planned[i]
                if row is None:
                    raise RuntimeError(f"Cannot match prompt response {i} to planned sample")
                err = getattr(inl, "error", None)
                if err:
                    raise RuntimeError(f"Prompt batch error for {row['sample_id']}: {err}")
                resp = getattr(inl, "response", None)
                if resp is None:
                    raise RuntimeError(f"Empty prompt response for {row['sample_id']}")
                text = _extract_gemini_text(resp)
                if not text:
                    raise RuntimeError(f"Empty prompt text for {row['sample_id']}")
                row["prompt_for_image"] = text

            # Step 2: Gemini batch for images
            image_reqs: List[Dict[str, Any]] = []
            for row in planned:
                prompt_sent = _build_image_prompt(row["prompt_for_image"], int(row["digit"]))
                image_reqs.append({
                    "contents": [{"parts": [{"text": prompt_sent}], "role": "user"}],
                    "metadata": {"custom_id": row["sample_id"]},
                })
            image_responses = _gemini_run_batch(
                model=gemini_model if provider == "gemini" else gemini_imagen_model,
                requests=image_reqs,
                poll_seconds=batch_poll_seconds,
                response_modalities=[types.Modality.IMAGE, types.Modality.TEXT],
                image_size=image_size,
            )
            # Map image responses to planned rows
            for i, inl in enumerate(image_responses):
                meta = getattr(inl, "metadata", None) or {}
                cid = meta.get("custom_id") if isinstance(meta, dict) else None
                if not cid and i < len(planned):
                    cid = planned[i]["sample_id"]
                row = next((r for r in planned if r["sample_id"] == cid), None)
                if row is None and i < len(planned):
                    row = planned[i]
                if row is None:
                    raise RuntimeError(f"Cannot match image response {i} to planned sample")
                err = getattr(inl, "error", None)
                if err:
                    raise RuntimeError(f"Image batch error for {row['sample_id']}: {err}")
                resp = getattr(inl, "response", None)
                if resp is None:
                    raise RuntimeError(f"Empty image response for {row['sample_id']}")
                image_bytes, ext = _extract_gemini_image(resp)

                sample_id = row["sample_id"]
                digit = int(row["digit"])
                created = utc_now_iso()
                digit_dir = images_dir / str(digit)
                ensure_dir(digit_dir)
                img_filename = f"{sample_id}.{ext}"
                img_path = digit_dir / img_filename
                save_image_bytes(image_bytes, img_path)

                effective_image_model = gemini_model if provider == "gemini" else gemini_imagen_model
                prompt_sent = _build_image_prompt(row["prompt_for_image"], digit)
                meta_obj = SampleMeta(
                    sample_id=sample_id,
                    digit=digit,
                    llm_model=llm_model,
                    image_model=effective_image_model,
                    provider=provider + "-batch",
                    created_utc=created,
                    prompt_for_image=prompt_sent,
                    image_path=str(img_path.relative_to(out_dir)),
                    size=effective_size,
                    quality=quality,
                    output_format=ext,
                    prompt_to_llm=row.get("prompt_to_llm"),
                )
                with meta_path.open("a", encoding="utf-8") as mf:
                    mf.write(json.dumps(asdict(meta_obj), ensure_ascii=False) + "\n")
                    mf.flush()
                pbar.update(1)

        elif provider == "openai" and use_batch:
            # ---- OpenAI Batch: prompts + images ----
            batch_dir = out_dir / "_batch"
            ensure_dir(batch_dir)

            prompt_reqs: List[Dict[str, Any]] = []
            for row in planned:
                cid = row["sample_id"]
                d = row["digit"]
                llm_prompt = _format_llm_prompt(d)
                row["prompt_to_llm"] = llm_prompt
                body: Dict[str, Any] = {
                    "model": llm_model,
                    "input": llm_prompt,
                }
                if temperature is not None:
                    body["temperature"] = temperature
                prompt_reqs.append({
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": body,
                })

            prompt_jsonl = batch_dir / "prompts.jsonl"
            _write_jsonl(prompt_jsonl, prompt_reqs)
            prompt_file_id = _openai_upload_batch_file(openai_client, prompt_jsonl)
            prompt_out_file_id, prompt_batch_id = _openai_run_batch(
                openai_client,
                input_file_id=prompt_file_id,
                endpoint="/v1/responses",
                poll_seconds=batch_poll_seconds,
            )

            prompt_out_bytes = _openai_download_file_bytes(openai_client, prompt_out_file_id)
            prompt_lines = _parse_openai_batch_output(prompt_out_bytes)

            # Attach prompts to planned rows.
            for row in planned:
                cid = row["sample_id"]
                line = prompt_lines.get(cid)
                if not line:
                    raise RuntimeError(f"Missing prompt output for custom_id={cid}")
                err = line.get("error")
                if err:
                    raise RuntimeError(f"Prompt batch error for {cid}: {err}")
                body = (((line.get("response") or {}).get("body")) or {})
                text = _extract_responses_text(body)
                if not text:
                    raise RuntimeError(f"Empty prompt text for sample_id={cid}")
                row["prompt_for_image"] = text

            # ---- Batch step 2: image generation via /v1/images/generations ----
            image_reqs: List[Dict[str, Any]] = []
            for row in planned:
                cid = row["sample_id"]
                prompt_sent = _build_image_prompt(row["prompt_for_image"], int(row["digit"]))
                image_reqs.append({
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/images/generations",
                    "body": {
                        "model": image_model,
                        "prompt": prompt_sent,
                        "size": size,
                        "quality": quality,
                        "output_format": output_format,
                        "n": 1,
                    },
                })

            images_jsonl = batch_dir / "images.jsonl"
            _write_jsonl(images_jsonl, image_reqs)
            images_file_id = _openai_upload_batch_file(openai_client, images_jsonl)
            images_out_file_id, images_batch_id = _openai_run_batch(
                openai_client,
                input_file_id=images_file_id,
                endpoint="/v1/images/generations",
                poll_seconds=batch_poll_seconds,
            )

            images_out_bytes = _openai_download_file_bytes(openai_client, images_out_file_id)
            images_lines = _parse_openai_batch_output(images_out_bytes)

            # Save images + metadata
            with meta_path.open("a", encoding="utf-8") as mf:
                for row in planned:
                    sample_id = row["sample_id"]
                    digit = int(row["digit"])
                    created = utc_now_iso()

                    digit_dir = images_dir / str(digit)
                    ensure_dir(digit_dir)

                    line = images_lines.get(sample_id)
                    if not line:
                        raise RuntimeError(f"Missing image output for custom_id={sample_id}")
                    err = line.get("error")
                    if err:
                        raise RuntimeError(f"Image batch error for {sample_id}: {err}")
                    body = (((line.get("response") or {}).get("body")) or {})
                    data = body.get("data")
                    if not (isinstance(data, list) and data and isinstance(data[0], dict)):
                        raise RuntimeError(f"Unexpected image body for {sample_id}: missing data")
                    b64_img = data[0].get("b64_json")
                    if not isinstance(b64_img, str):
                        raise RuntimeError(f"Unexpected image body for {sample_id}: missing b64_json")
                    image_bytes = base64.b64decode(b64_img)

                    img_filename = f"{sample_id}.{output_format}"
                    img_path = digit_dir / img_filename
                    save_image_bytes(image_bytes, img_path)

                    prompt_sent = _build_image_prompt(row["prompt_for_image"], digit)
                    meta = SampleMeta(
                        sample_id=sample_id,
                        digit=digit,
                        llm_model=llm_model,
                        image_model=image_model,
                        provider=provider + "-batch",
                        created_utc=created,
                        prompt_for_image=prompt_sent,
                        image_path=str(img_path.relative_to(out_dir)),
                        size=effective_size,
                        quality=quality,
                        output_format=output_format,
                        prompt_to_llm=row.get("prompt_to_llm"),
                    )
                    mf.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
                    mf.flush()

                    pbar.update(1)

        else:
            # ---- Synchronous (existing) path ----
            with meta_path.open("a", encoding="utf-8") as mf:
                for digit in digit_list:
                    digit_dir = images_dir / str(digit)
                    ensure_dir(digit_dir)

                    for _ in range(per_digit):
                        sample_id = uuid.uuid4().hex[:12]
                        created = utc_now_iso()

                        if llm_provider == "openai":
                            prompt_to_llm, prompt_for_image = generate_image_prompt_openai(
                                openai_client, llm_model, digit, temperature=temperature
                            )
                        else:
                            prompt_to_llm, prompt_for_image = generate_image_prompt_gemini(
                                llm_model=llm_model,
                                digit=digit,
                                max_retries=max_retries,
                                thinking_budget=thinking_budget,
                                thinking_level=thinking_level,
                                temperature=temperature,
                            )

                        prompt_for_image = _build_image_prompt(prompt_for_image, digit)

                        if provider == "openai":
                            image_bytes = openai_generate_image_bytes(
                                client=openai_client,
                                image_model=image_model,
                                prompt=prompt_for_image,
                                size=effective_size,
                                quality=quality,
                                output_format=output_format,
                            )
                            ext = output_format
                        elif provider == "gemini":
                            image_bytes, ext = gemini_generate_image_bytes(
                                prompt=prompt_for_image,
                                model=gemini_model,
                                aspect_ratio="1:1",
                                image_size=image_size,
                                max_retries=max_retries,
                            )
                        elif provider == "gemini-imagen":
                            image_bytes, ext = gemini_imagen_generate_image_bytes(
                                prompt=prompt_for_image,
                                model=gemini_imagen_model,
                                number_of_images=1,
                                aspect_ratio="1:1",
                                image_size=image_size,
                                max_retries=max_retries,
                            )
                        else:
                            raise SystemExit(f"Unsupported provider: {provider}")

                        img_filename = f"{sample_id}.{ext}"
                        img_path = digit_dir / img_filename
                        save_image_bytes(image_bytes, img_path)

                        effective_image_model = image_model
                        if provider == "gemini":
                            effective_image_model = gemini_model
                        elif provider == "gemini-imagen":
                            effective_image_model = gemini_imagen_model

                        meta = SampleMeta(
                            sample_id=sample_id,
                            digit=digit,
                            llm_model=llm_model,
                            image_model=effective_image_model,
                            provider=provider,
                            created_utc=created,
                            prompt_for_image=prompt_for_image,
                            image_path=str(img_path.relative_to(out_dir)),
                            size=effective_size,
                            quality=quality,
                            output_format=ext,
                            prompt_to_llm=prompt_to_llm,
                        )
                        mf.write(json.dumps(asdict(meta), ensure_ascii=False) + "\n")
                        mf.flush()

                        pbar.update(1)
                        if sleep > 0:
                            time.sleep(sleep)

    except Exception as e:
        err_mod = getattr(type(e), "__module__", "") or ""
        if "genai" in err_mod or "google.genai" in str(type(e)):
            print(_format_gemini_error(e), file=sys.stderr)
        raise
    finally:
        pbar.close()

    print(f"\nDone. Images: {images_dir}\nMetadata: {meta_path}")
