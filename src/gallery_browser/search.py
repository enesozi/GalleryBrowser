from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import torch

from .inference import DEFAULT_MODEL
from .utils import inference_autocast, load_rgb_image, load_video_frame, resolve_device, resolve_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass(slots=True)
class GalleryMatch:
    path: str
    kind: str
    score: float


@dataclass(slots=True)
class GallerySearchResult:
    scanned_files: int
    matches: list[GalleryMatch]
    skipped: list[str]


def search_gallery(
    *,
    gallery_dir: str,
    query: str,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 10,
    frame_index: int = 0,
    device: str = "auto",
) -> GallerySearchResult:
    gallery_path = resolve_path(gallery_dir)
    if not gallery_path.exists() or not gallery_path.is_dir():
        raise ValueError(f"Gallery folder does not exist or is not a directory: {gallery_path}")
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    media_files = _discover_media_files(gallery_path)
    if not media_files:
        return GallerySearchResult(scanned_files=0, matches=[], skipped=[])

    resolved_device = resolve_device(device)
    model = pe.CLIP.from_config(model_name, pretrained=True).to(resolved_device)
    model.eval()

    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    text_tensor = tokenizer([query]).to(resolved_device)

    matches: list[GalleryMatch] = []
    skipped: list[str] = []
    for path in media_files:
        kind = _detect_kind(path)
        try:
            if kind == "image":
                image, _ = load_rgb_image(str(path))
            else:
                image, _ = load_video_frame(str(path), frame_index)

            image_tensor = preprocess(image).unsqueeze(0).to(resolved_device)
            with torch.inference_mode(), inference_autocast(resolved_device):
                image_features, text_features, logit_scale = model(image_tensor, text_tensor)
                score = (logit_scale * image_features @ text_features.T)[0, 0]

            matches.append(
                GalleryMatch(path=str(path), kind=kind, score=float(score.detach().cpu().item()))
            )
        except Exception:
            skipped.append(str(path))

    matches.sort(key=lambda item: item.score, reverse=True)
    return GallerySearchResult(
        scanned_files=len(media_files),
        matches=matches[: max(top_k, 1)],
        skipped=skipped,
    )


def _discover_media_files(gallery_path: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(gallery_path.rglob("*")):
        if path.is_file() and _detect_kind(path) in {"image", "video"}:
            files.append(path)
    return files


def _detect_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "unknown"