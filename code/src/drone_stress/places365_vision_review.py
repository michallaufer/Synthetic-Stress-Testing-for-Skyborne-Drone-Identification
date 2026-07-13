"""GPT-4o mini vision review for Places365 background candidates."""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from drone_stress.places365_clip_filter import CLIP_FILTERED_CATEGORIES
from drone_stress.places365_finalize import (
    MANIFEST_COLUMNS,
    _cell_str,
    normalize_decision,
    normalize_manifest_decisions,
    resolve_candidate_image_path,
    save_manifest_csv,
    summarize_manifest_decisions,
)

DEFAULT_MODEL = "gpt-4o-mini"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

API_KEY_PLACEHOLDERS = frozenset({"", "<your-key>", "your-key-here"})

VISION_EXTRA_COLUMNS = [
    "vision_decision",
    "vision_reason",
    "vision_error",
    "vision_model",
    "vision_confidence",
    "vision_usable_sky_region",
    "vision_corrected_category",
    "usable_sky_region",
    "corrected_category",
]

VALID_USABLE_SKY_REGIONS = frozenset({"large", "medium", "small", "none"})
VALID_VISION_TRIAGE_LABELS = frozenset({"accept_candidate", "review", "reject"})
VALID_VISION_DECISION_LABELS = VALID_VISION_TRIAGE_LABELS | {"error"}

VISION_TRIAGE_ALIASES = {
    "accept": "accept_candidate",
    "accept_candidate": "accept_candidate",
    "candidate": "accept_candidate",
    "keep": "accept_candidate",
    "approved": "accept_candidate",
    "review": "review",
    "maybe": "review",
    "uncertain": "review",
    "reject": "reject",
    "drop": "reject",
}

VISION_CORRECTED_CATEGORIES = frozenset(
    {
        *CLIP_FILTERED_CATEGORIES,
        "night_low_light",
        "unclassifiable",
    }
)

REVIEW_SYSTEM_PROMPT = """You are screening candidate background images for a synthetic drone-detection benchmark. Composited drones will be small (8-150 pixels) and placed somewhere within an open-air/sky region of the image. Your job is NOT to judge whether the image is "mostly sky." Your job is to judge whether the image contains a sky/open-air region large enough and plausible enough to place a small flying object in.

CRITICAL REASONING RULE:
Foreground content (buildings, trees, water, mountains, a runway, an aircraft wing, branches, silhouettes) is EXPECTED and NEVER a reason to reject by itself. An image can have buildings filling 70% of the frame and still be a great candidate if the remaining 30% is open sky. Category names like "cloudy_sky" or "sky_with_built_environment" describe the FOREGROUND CONTENT TYPE, not a requirement that sky be the majority of the pixels. A frame filled with clouds is the textbook example of "cloudy_sky" — never reject for that.

STEP 1 - Estimate usable sky/open-air area.
Look at the ENTIRE frame, including gaps between branches, between buildings, above a horizon line, around an aircraft wing, above a runway, etc. Estimate what fraction of the image is sky/cloud/open-air (not ground, water surface, foliage, structure, or close-up object).
  large  = roughly >50%
  medium = roughly 20-50%
  small  = roughly 5-20%
  none   = under 5%, no meaningful contiguous patch
A medium or small region is still USABLE if it is contiguous enough to place an 8-150px object without it obviously overlapping foreground clutter.

STEP 2 - Check hard-reject conditions ONLY.
Reject ONLY if one of these is true. If none are true, do not reject, regardless of how "busy" or foreground-heavy the image looks:
  - indoor scene
  - extreme close-up of food, flowers, an object, or a person/animal face
  - a person, animal, or vehicle is the clear subject filling most of the frame
  - painting, illustration, drawing, or heavily stylized/filtered image
  - a large airplane, bird, kite, or drone already dominates the sky region
  - usable_sky_region == none
  - heavy watermark, logo, or text overlay covering significant area
  - the image is abstract/textural with no recognizable scene (e.g. a macro shot where you cannot tell what real-world scene this is)

STEP 3 - Apply this decision logic exactly, in order:
  IF any hard-reject condition is true       -> reject
  ELIF usable_sky_region is large or medium   -> accept_candidate
  ELIF usable_sky_region is small             -> review
  (small is not rejected by default - only review)

STEP 4 - Lighting note.
Dusk, sunset, overcast, or hazy lighting is NOT a rejection reason and is not automatically "review" - it's a normal daytime condition. Only flag as corrected_category = "night_low_light" if the scene is genuinely dark (streetlights on, stars visible, near-black sky) such that a drone would be invisible if placed there.

STEP 5 - Category correction.
Assign corrected_category based on dominant foreground content, choosing the single best fit:
  clear_upper_sky          - sky/clouds dominate, minimal foreground
  cloudy_sky                - clouds are the main visual feature, any amount of horizon/land/water below is fine
  sky_with_natural_landscape - mountains, fields, coast, water, desert, valley, trees visible, with any usable sky
  sky_with_built_environment - buildings, roads, skyline, rooftops visible, with any usable sky
  runway_or_airport          - runway/airport visible, with any usable sky and no dominant large aircraft
  night_low_light            - genuinely dark/night scene
  unclassifiable              - none of the above fit and you are unsure

OUTPUT (JSON only, no other text):
{
  "vision_decision": "accept_candidate" | "review" | "reject",
  "decision": "accept" | "" | "reject",
  "usable_sky_region": "large" | "medium" | "small" | "none",
  "corrected_category": "<one of the categories above>",
  "vision_reason": "<under 15 words, must reference sky region size and hard-reject check, never just 'dominated by X'>",
  "confidence": "high" | "medium" | "low"
}
decision must equal "accept" only if vision_decision is accept_candidate.
decision must be "" if vision_decision is review.
decision must equal "reject" only if vision_decision is reject.

CALIBRATION EXAMPLES (apply this exact standard):
- A photo taken from an airplane window showing an aircraft wing in one corner and a huge expanse of sky/clouds/horizon filling the rest of the frame -> usable_sky_region = large -> accept_candidate. The wing is a small foreground element, not a disqualifier.
- A photo that is entirely white/grey clouds filling the frame, no visible blue sky -> usable_sky_region = large (clouds count as sky/air region) -> accept_candidate, corrected_category = cloudy_sky.
- A runway with mountains in the background and a large open sky above both -> usable_sky_region = large -> accept_candidate, corrected_category = runway_or_airport.
- A dark building silhouette occupying the bottom quarter of the frame with dramatic clouds filling the top three-quarters -> usable_sky_region = large -> accept_candidate, corrected_category = sky_with_built_environment.
- A tight close-up of tree foliage with only a tiny sliver of sky visible at one edge -> usable_sky_region = none or small -> review or reject depending on whether any contiguous patch exceeds ~5%.
"""


class VisionParseError(ValueError):
    def __init__(self, message: str, *, raw: str = "") -> None:
        super().__init__(message)
        self.raw = raw


@dataclass
class VisionReviewResult:
    vision_decision: str
    manifest_decision: str
    reason: str
    confidence: str
    usable_sky_region: str
    corrected_category: str
    model: str
    raw_response: str = ""


def _manifest_columns() -> list[str]:
    cols = list(MANIFEST_COLUMNS)
    for col in VISION_EXTRA_COLUMNS:
        if col not in cols:
            cols.append(col)
    return cols


def encode_image_for_vision(path: Path, *, max_size: int = 1024) -> tuple[str, str]:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return encoded, "image/jpeg"


def build_user_prompt(row: pd.Series) -> str:
    final_category = _cell_str(row, "final_category", "mapped_background_type")
    return (
        "Screen this Places365 candidate background.\n"
        f"candidate_id: {_cell_str(row, 'candidate_id')}\n"
        f"places365_category: {_cell_str(row, 'places365_category')}\n"
        f"mapped_background_type: {_cell_str(row, 'mapped_background_type')}\n"
        f"final_category: {final_category or 'unknown'}\n"
        f"upper_sky_ratio (heuristic): {row.get('upper_sky_ratio', '')}\n"
        f"clip_suitability_score: {row.get('clip_suitability_score', '')}\n"
        "Follow STEP 1-5 exactly. Foreground content alone is never a reject reason."
    )


def normalize_vision_triage_decision(value) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip().lower()
    if text in ("", "nan", "none"):
        return ""
    if text == "error":
        return "error"
    return VISION_TRIAGE_ALIASES.get(text, text if text in VALID_VISION_TRIAGE_LABELS else "")


def manifest_decision_from_vision(vision_decision: str) -> str:
    if vision_decision in {"accept_candidate", "review"}:
        return "accept"
    if vision_decision == "reject":
        return "reject"
    return ""


def _correct_over_rejection(vision_decision: str, usable_sky_region: str) -> str:
    """Re-apply STEP 3 when the model rejects despite adequate sky area."""
    if vision_decision != "reject":
        return vision_decision
    if usable_sky_region in {"large", "medium"}:
        return "accept_candidate"
    if usable_sky_region == "small":
        return "review"
    return vision_decision


def _sync_manifest_decision(vision_decision: str, gpt_decision: str) -> str:
    _ = gpt_decision  # GPT may echo decision; manifest mapping follows vision_decision.
    return manifest_decision_from_vision(vision_decision)


def _normalize_usable_sky_region(value) -> str:
    text = str(value or "").strip().lower()
    return text if text in VALID_USABLE_SKY_REGIONS else ""


def _normalize_corrected_category(value) -> str:
    text = str(value or "").strip()
    if text in {"", "none", "unchanged", "null"}:
        return ""
    if text in VISION_CORRECTED_CATEGORIES:
        return text
    return ""


def _apply_vision_result_to_row(
    manifest: pd.DataFrame,
    idx: int,
    result: VisionReviewResult,
) -> None:
    manifest.at[idx, "vision_decision"] = result.vision_decision
    manifest.at[idx, "decision"] = result.manifest_decision
    sky_note = f", sky={result.usable_sky_region}" if result.usable_sky_region else ""
    tag = result.vision_decision
    if result.vision_decision == "review":
        tag = "review->accept"
    manifest.at[idx, "notes"] = (
        f"gpt4o-mini [{tag}] ({result.confidence}{sky_note}): {result.reason}"
    )
    manifest.at[idx, "vision_reason"] = result.reason
    manifest.at[idx, "vision_error"] = ""
    manifest.at[idx, "vision_model"] = result.model
    manifest.at[idx, "vision_confidence"] = result.confidence
    manifest.at[idx, "vision_usable_sky_region"] = result.usable_sky_region
    manifest.at[idx, "vision_corrected_category"] = result.corrected_category
    manifest.at[idx, "usable_sky_region"] = result.usable_sky_region
    manifest.at[idx, "corrected_category"] = result.corrected_category
    if result.vision_decision in {"accept_candidate", "review"} and result.corrected_category in CLIP_FILTERED_CATEGORIES:
        manifest.at[idx, "final_category"] = result.corrected_category


def _apply_error_to_row(
    manifest: pd.DataFrame,
    idx: int,
    message: str,
    *,
    model: str = "",
    notes_prefix: str = "gpt_review_error",
) -> None:
    """Record API/parse/image errors without marking the row rejected."""
    manifest.at[idx, "decision"] = ""
    manifest.at[idx, "notes"] = f"{notes_prefix}: {message}"
    manifest.at[idx, "vision_decision"] = "error"
    manifest.at[idx, "vision_reason"] = ""
    manifest.at[idx, "vision_error"] = message
    manifest.at[idx, "vision_model"] = model
    manifest.at[idx, "vision_confidence"] = ""
    manifest.at[idx, "vision_usable_sky_region"] = ""
    manifest.at[idx, "vision_corrected_category"] = ""
    manifest.at[idx, "usable_sky_region"] = ""
    manifest.at[idx, "corrected_category"] = ""


def _apply_api_error_to_row(manifest: pd.DataFrame, idx: int, exc: Exception, *, model: str) -> None:
    _apply_error_to_row(manifest, idx, str(exc), model=model, notes_prefix="gpt_review_error")


def parse_vision_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Vision response is not a JSON object")
    return data


def review_image_with_gpt(
    *,
    client,
    image_path: Path,
    row: pd.Series,
    model: str = DEFAULT_MODEL,
    max_image_size: int = 1024,
) -> VisionReviewResult:
    encoded, mime = encode_image_for_vision(image_path, max_size=max_image_size)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_user_prompt(row)},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{encoded}",
                            "detail": "low",
                        },
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=400,
    )
    raw = response.choices[0].message.content or ""
    try:
        data = parse_vision_json(raw)
        usable_sky_region = _normalize_usable_sky_region(data.get("usable_sky_region"))
        vision_decision = normalize_vision_triage_decision(
            data.get("vision_decision", data.get("decision", "review"))
        )
        if vision_decision not in VALID_VISION_TRIAGE_LABELS:
            vision_decision = "review"
        vision_decision = _correct_over_rejection(vision_decision, usable_sky_region)
        manifest_decision = _sync_manifest_decision(
            vision_decision, data.get("decision", "")
        )
        reason = (
            str(data.get("vision_reason", data.get("reason", ""))).strip()
            or "no_reason_provided"
        )
        confidence = str(data.get("confidence", "medium")).strip().lower()
        corrected_category = _normalize_corrected_category(data.get("corrected_category"))
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise VisionParseError(str(exc), raw=raw or str(exc)) from exc

    return VisionReviewResult(
        vision_decision=vision_decision,
        manifest_decision=manifest_decision,
        reason=reason,
        confidence=confidence,
        usable_sky_region=usable_sky_region,
        corrected_category=corrected_category,
        model=model,
        raw_response=raw,
    )


def _read_key_from_env_file(env_path: Path, names: tuple[str, ...]) -> str:
    if not env_path.is_file():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() in names:
            value = value.strip().strip('"').strip("'")
            if value and value not in API_KEY_PLACEHOLDERS:
                return value
    return ""


def load_project_env(env_path: Path | None = None) -> None:
    """Load OPENAI_API_KEY from project .env, overriding shell placeholders."""
    path = env_path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return

    current = os.environ.get("OPENAI_API_KEY", "").strip()
    needs_override = current in API_KEY_PLACEHOLDERS

    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=needs_override)
    except ImportError:
        pass

    if needs_override or not os.environ.get("OPENAI_API_KEY", "").strip():
        file_key = _read_key_from_env_file(path, ("OPENAI_API_KEY", "OPEN_API_KEY"))
        if file_key:
            os.environ["OPENAI_API_KEY"] = file_key


def resolve_openai_api_key() -> str:
    load_project_env()
    for name in ("OPENAI_API_KEY", "OPEN_API_KEY"):
        api_key = os.environ.get(name, "").strip()
        if api_key and api_key not in API_KEY_PLACEHOLDERS:
            return api_key

    file_key = _read_key_from_env_file(PROJECT_ROOT / ".env", ("OPENAI_API_KEY", "OPEN_API_KEY"))
    if file_key:
        os.environ["OPENAI_API_KEY"] = file_key
        return file_key

    raise EnvironmentError(
        "OpenAI API key not found. Set OPENAI_API_KEY in the project .env file "
        "(not the README placeholder <your-key>) or export it in your shell."
    )


def create_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "GPT vision review requires the openai package. Install with: pip install openai"
        ) from exc
    return OpenAI(api_key=resolve_openai_api_key())


def _persist_manifest(
    manifest: pd.DataFrame,
    write_path: Path,
    *,
    manifest_path: Path,
    autosave_path: Path,
    warned_autosave: bool,
) -> tuple[Path, bool]:
    columns = _manifest_columns()
    if write_path != manifest_path:
        if write_path == autosave_path:
            manifest.to_csv(write_path, index=False, columns=columns)
            return write_path, warned_autosave
        save_manifest_csv(manifest, write_path, columns=columns)
        return write_path, warned_autosave
    try:
        save_manifest_csv(manifest, manifest_path, columns=columns)
        return manifest_path, warned_autosave
    except PermissionError:
        save_manifest_csv(manifest, autosave_path, columns=columns, atomic=False)
        if not warned_autosave:
            print(
                f"\nWarning: could not update locked manifest ({manifest_path.name}). "
                f"Saving GPT progress to {autosave_path.name}. "
                "Close Excel, then copy the autosave over the manifest when finished."
            )
        return autosave_path, True


def review_manifest_with_gpt(
    manifest_path: Path,
    *,
    candidate_root: Path,
    model: str = DEFAULT_MODEL,
    only_pending: bool = True,
    overwrite: bool = False,
    max_images: int | None = None,
    request_delay_s: float = 0.0,
    max_image_size: int = 1024,
) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    for col in MANIFEST_COLUMNS:
        if col not in manifest.columns:
            raise ValueError(f"Manifest missing required column: {col}")

    manifest = normalize_manifest_decisions(manifest)
    for col in VISION_EXTRA_COLUMNS:
        if col not in manifest.columns:
            manifest[col] = ""

    client = create_openai_client()

    indices = list(manifest.index)
    if overwrite:
        pass
    elif only_pending:
        indices = [
            idx
            for idx in indices
            if normalize_decision(manifest.at[idx, "decision"]) == ""
            or str(manifest.at[idx, "vision_decision"]).strip().lower() == "error"
        ]
    if max_images is not None:
        indices = indices[: max(0, max_images)]

    if not indices:
        print("No manifest rows selected for GPT vision review.")
        return manifest

    reviewed = 0
    accept_candidates = 0
    review_count = 0
    rejected = 0
    errors = 0
    autosave_path = manifest_path.parent / "places365_gpt_autosave.csv"
    write_path = manifest_path
    warned_autosave = False

    for idx in tqdm(indices, desc="GPT vision review", unit="image"):
        row = manifest.loc[idx]
        image_path = resolve_candidate_image_path(row, candidate_root)
        if image_path is None:
            _apply_error_to_row(
                manifest,
                idx,
                "image_not_found",
                model=model,
                notes_prefix="gpt_review",
            )
            errors += 1
            reviewed += 1
            write_path, warned_autosave = _persist_manifest(
                manifest,
                write_path,
                manifest_path=manifest_path,
                autosave_path=autosave_path,
                warned_autosave=warned_autosave,
            )
            continue

        try:
            result = review_image_with_gpt(
                client=client,
                image_path=image_path,
                row=row,
                model=model,
                max_image_size=max_image_size,
            )
            _apply_vision_result_to_row(manifest, idx, result)
            reviewed += 1
            if result.vision_decision == "accept_candidate":
                accept_candidates += 1
            elif result.vision_decision == "review":
                review_count += 1
            else:
                rejected += 1
        except VisionParseError as exc:
            _apply_error_to_row(
                manifest,
                idx,
                f"vision_parse_error: {exc}",
                model=model,
            )
            errors += 1
            reviewed += 1
        except Exception as exc:  # noqa: BLE001 - record per-image API failures
            _apply_api_error_to_row(manifest, idx, exc, model=model)
            errors += 1
            reviewed += 1

        write_path, warned_autosave = _persist_manifest(
            manifest,
            write_path,
            manifest_path=manifest_path,
            autosave_path=autosave_path,
            warned_autosave=warned_autosave,
        )
        if request_delay_s > 0:
            time.sleep(request_delay_s)

    summary = summarize_manifest_decisions(manifest)
    print(f"\nGPT vision reviewed: {reviewed}")
    print(f"  accept_candidate: {accept_candidates}")
    print(f"  review: {review_count}")
    print(f"  reject: {rejected}")
    if errors:
        print(f"  errors: {errors}")
    print(
        f"Manifest totals — pending: {summary['pending']}, "
        f"accept: {summary['accept']}, reject: {summary['reject']}"
    )
    if warned_autosave:
        print(f"GPT progress file: {autosave_path.resolve()}")
    else:
        print(f"Updated manifest: {manifest_path.resolve()}")
    return manifest


def reset_api_error_reviews(manifest_path: Path) -> int:
    """Clear GPT review fields for rows that failed due to API/auth errors."""
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    mask = (
        manifest.get("vision_decision", pd.Series(dtype=str))
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("error")
        | manifest.get("vision_reason", pd.Series(dtype=str))
        .astype(str)
        .str.contains("api_error", case=False, na=False)
        | manifest.get("vision_error", pd.Series(dtype=str))
        .astype(str)
        .str.len()
        .gt(0)
        | manifest.get("notes", pd.Series(dtype=str))
        .astype(str)
        .str.contains("gpt_review_error", case=False, na=False)
    )
    count = int(mask.sum())
    if count == 0:
        return 0
    for col in (
        "decision",
        "notes",
        "vision_decision",
        "vision_reason",
        "vision_error",
        "vision_model",
        "vision_confidence",
        "vision_usable_sky_region",
        "vision_corrected_category",
        "usable_sky_region",
        "corrected_category",
    ):
        if col in manifest.columns:
            manifest.loc[mask, col] = ""
    save_manifest_csv(manifest, manifest_path, columns=_manifest_columns())
    return count
