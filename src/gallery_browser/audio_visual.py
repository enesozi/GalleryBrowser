from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .utils import (
    get_audio_visual_transform,
    inference_autocast,
    load_audio_visual_runtime,
    resolve_device,
    validate_parallel_lengths,
)

DEFAULT_AV_MODEL = "pe-av-small-16-frame"
AVAILABLE_AV_MODELS = [
    "pe-av-small-16-frame",
    "pe-av-base-16-frame",
    "pe-av-large-16-frame",
    "pe-av-small",
    "pe-av-base",
    "pe-av-large",
]


@dataclass(slots=True)
class AudioVisualEmbeddingResult:
    device: str
    model_name: str
    modalities: list[str]
    output_shapes: dict[str, tuple[int, ...]]
    dot_products: dict[str, list[float]]


def available_av_models() -> list[str]:
    return list(AVAILABLE_AV_MODELS)


def embed_audio_visual(
    *,
    video_paths: Sequence[str] | None = None,
    audio_paths: Sequence[str] | None = None,
    texts: Sequence[str] | None = None,
    model_name: str = DEFAULT_AV_MODEL,
    device: str = "auto",
) -> AudioVisualEmbeddingResult:
    if not any([video_paths, audio_paths, texts]):
        raise ValueError("Provide at least one of video_paths, audio_paths, or texts.")

    videos = list(video_paths) if video_paths else None
    audio = list(audio_paths) if audio_paths else None
    text = list(texts) if texts else None
    validate_parallel_lengths(videos=videos, audio=audio, text=text)

    PEAudioVisual, _ = load_audio_visual_runtime()
    resolved_device = resolve_device(device)
    model = PEAudioVisual.from_config(model_name, pretrained=True).to(resolved_device)
    model.eval()
    transform = get_audio_visual_transform(model_name, model)

    inputs = transform(videos=videos, audio=audio, text=text).to(resolved_device)
    with torch.inference_mode(), inference_autocast(resolved_device, dtype=torch.bfloat16):
        outputs = model(**inputs)

    output_shapes = _collect_output_shapes(outputs)
    dot_products = _collect_dot_products(outputs)
    modalities = [
        name for name, values in (("video", videos), ("audio", audio), ("text", text)) if values
    ]
    return AudioVisualEmbeddingResult(
        device=resolved_device.type,
        model_name=model_name,
        modalities=modalities,
        output_shapes=output_shapes,
        dot_products=dot_products,
    )


def _collect_output_shapes(outputs) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for name in (
        "visual_embeds",
        "visual_text_embeds",
        "audio_embeds",
        "audio_text_embeds",
        "audio_visual_embeds",
        "audio_visual_text_embeds",
        "audio_plus_text_embeds",
        "visual_plus_text_embeds",
    ):
        value = getattr(outputs, name)
        if value is not None:
            shapes[name] = tuple(value.shape)
    return shapes


def _collect_dot_products(outputs) -> dict[str, list[float]]:
    scores: dict[str, list[float]] = {}
    _maybe_add_dot_product(scores, "visual_text", outputs.visual_embeds, outputs.visual_text_embeds)
    _maybe_add_dot_product(
        scores, "audio_visual_text", outputs.audio_visual_embeds, outputs.audio_visual_text_embeds
    )
    _maybe_add_dot_product(scores, "audio_text", outputs.audio_embeds, outputs.audio_text_embeds)
    _maybe_add_dot_product(scores, "audio_visual", outputs.audio_embeds, outputs.visual_embeds)
    return scores


def _maybe_add_dot_product(
    scores: dict[str, list[float]],
    name: str,
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> None:
    if left is None or right is None:
        return
    scores[name] = torch.einsum("ij,ij->i", left, right).tolist()
