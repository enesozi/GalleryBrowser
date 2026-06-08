from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
from decord import VideoReader
from PIL import Image
import torch
import torch.nn.functional as F

from .inference import DEFAULT_MODEL
from .utils import inference_autocast, load_rgb_image, resolve_device, resolve_path

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
    cache_hit: bool


@dataclass(slots=True)
class _GalleryEmbeddingEntry:
    path: str
    kind: str
    embeddings: torch.Tensor


@dataclass(slots=True)
class _GalleryEmbeddingCache:
    signature: tuple[str, ...]
    entries: list[_GalleryEmbeddingEntry]
    skipped: list[str]


_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any, Any]] = {}
_GALLERY_CACHE: dict[tuple[str, str, int], _GalleryEmbeddingCache] = {}


def clear_gallery_cache(*, clear_models: bool = False) -> None:
    _GALLERY_CACHE.clear()
    if clear_models:
        _MODEL_CACHE.clear()


def gallery_cache_info() -> dict[str, int]:
    return {
        "gallery_entries": len(_GALLERY_CACHE),
        "model_entries": len(_MODEL_CACHE),
    }


def search_gallery(
    *,
    gallery_dir: str,
    query: str,
    model_name: str = DEFAULT_MODEL,
    top_k: int = 10,
    video_frames: int = 4,
    device: str = "auto",
) -> GallerySearchResult:
    gallery_path = resolve_path(gallery_dir)
    if not gallery_path.exists() or not gallery_path.is_dir():
        raise ValueError(f"Gallery folder does not exist or is not a directory: {gallery_path}")
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    media_files = _discover_media_files(gallery_path)
    if not media_files:
        return GallerySearchResult(scanned_files=0, matches=[], skipped=[], cache_hit=False)

    resolved_device = resolve_device(device)
    model, preprocess, tokenizer = _get_model_bundle(model_name, resolved_device)
    text_embedding = _encode_text_embedding(model, tokenizer, query, resolved_device)

    effective_video_frames = max(int(video_frames), 1)
    cache_key = (str(gallery_path), model_name, effective_video_frames)
    signature = _gallery_signature(media_files)
    cached = _GALLERY_CACHE.get(cache_key)
    cache_hit = bool(cached and cached.signature == signature)
    if not cache_hit:
        cached = _build_gallery_embeddings(
            media_files=media_files,
            model=model,
            preprocess=preprocess,
            tokenizer=tokenizer,
            device=resolved_device,
            video_frames=effective_video_frames,
        )
        _GALLERY_CACHE[cache_key] = cached
    assert cached is not None

    matches: list[GalleryMatch] = []
    for entry in cached.entries:
        score = torch.matmul(entry.embeddings, text_embedding).max()
        matches.append(
            GalleryMatch(path=entry.path, kind=entry.kind, score=float(score.detach().cpu().item()))
        )

    matches.sort(key=lambda item: item.score, reverse=True)
    return GallerySearchResult(
        scanned_files=len(media_files),
        matches=matches[: max(top_k, 1)],
        skipped=cached.skipped,
        cache_hit=cache_hit,
    )


def _build_gallery_embeddings(
    *,
    media_files: list[Path],
    model,
    preprocess,
    tokenizer,
    device: torch.device,
    video_frames: int,
) -> _GalleryEmbeddingCache:
    entries: list[_GalleryEmbeddingEntry] = []
    skipped: list[str] = []
    for path in media_files:
        kind = _detect_kind(path)
        try:
            if kind == "image":
                embeddings = _embed_image(path, model, preprocess, tokenizer, device)
            elif kind == "video":
                embeddings = _embed_video(path, model, preprocess, tokenizer, device, video_frames)
            else:
                skipped.append(str(path))
                continue
            entries.append(_GalleryEmbeddingEntry(path=str(path), kind=kind, embeddings=embeddings))
        except Exception:
            skipped.append(str(path))

    return _GalleryEmbeddingCache(
        signature=_gallery_signature(media_files),
        entries=entries,
        skipped=skipped,
    )


def _embed_image(path: Path, model, preprocess, tokenizer, device: torch.device) -> torch.Tensor:
    image, _ = load_rgb_image(str(path))
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    return _encode_image_embeddings(model, tokenizer, image_tensor, device)


def _embed_video(
    path: Path,
    model,
    preprocess,
    tokenizer,
    device: torch.device,
    video_frames: int,
) -> torch.Tensor:
    frames = _load_video_frames(path, video_frames)
    video_tensor = torch.stack([preprocess(frame) for frame in frames], dim=0).to(device)
    return _encode_image_embeddings(model, tokenizer, video_tensor, device)


def _load_video_frames(path: Path, video_frames: int) -> list[Image.Image]:
    reader = VideoReader(str(path))
    frame_count = len(reader)
    if frame_count <= 0:
        raise ValueError(f"Video has no frames: {path}")

    frame_indices = _sample_frame_indices(frame_count, video_frames)
    frame_batch = reader.get_batch(frame_indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frame_batch]


def _sample_frame_indices(frame_count: int, video_frames: int) -> list[int]:
    frame_total = max(video_frames, 1)
    if frame_count == 1:
        return [0] * frame_total
    if frame_total == 1:
        return [frame_count // 2]
    return [int(index * (frame_count - 1) / (frame_total - 1)) for index in range(frame_total)]


def _encode_image_embeddings(
    model, tokenizer, image_tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
    with torch.inference_mode(), inference_autocast(device):
        if hasattr(model, "encode_image"):
            image_features = model.encode_image(image_tensor)
        else:
            anchor_text = tokenizer(["image"]).to(device)
            image_features, _, _ = model(image_tensor, anchor_text)
    return F.normalize(image_features, dim=-1).detach().cpu()


def _encode_text_embedding(model, tokenizer, query: str, device: torch.device) -> torch.Tensor:
    text_tensor = tokenizer([query]).to(device)
    with torch.inference_mode(), inference_autocast(device):
        if hasattr(model, "encode_text"):
            text_features = model.encode_text(text_tensor)
        else:
            dummy_image = torch.zeros((1, 3, model.image_size, model.image_size), device=device)
            _, text_features, _ = model(dummy_image, text_tensor)
    normalized = F.normalize(text_features, dim=-1).detach().cpu()
    return normalized[0]


def _get_model_bundle(model_name: str, device: torch.device):
    cache_key = (model_name, device.type)
    bundle = _MODEL_CACHE.get(cache_key)
    if bundle is not None:
        return bundle

    model = pe.CLIP.from_config(model_name, pretrained=True).to(device)
    model.eval()
    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    bundle = (model, preprocess, tokenizer)
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def _gallery_signature(media_files: list[Path]) -> tuple[str, ...]:
    return tuple(
        f"{path}:{path.stat().st_mtime_ns}:{path.stat().st_size}"
        for path in media_files
        if path.exists()
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
