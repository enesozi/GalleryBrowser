from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .utils import (
    get_audio_frame_transform,
    load_audio_frame_runtime,
    resolve_device,
    validate_parallel_lengths,
)

DEFAULT_AUDIO_FRAME_MODEL = "pe-a-frame-small"
AVAILABLE_AUDIO_FRAME_MODELS = [
    "pe-a-frame-small",
    "pe-a-frame-base",
    "pe-a-frame-large",
]


@dataclass(slots=True)
class LocalizedEvent:
    description: str
    spans: list[tuple[float, float]]


@dataclass(slots=True)
class AudioLocalizationResult:
    device: str
    model_name: str
    audio_paths: list[str]
    events: list[LocalizedEvent]


def available_audio_frame_models() -> list[str]:
    return list(AVAILABLE_AUDIO_FRAME_MODELS)


def localize_audio_events(
    *,
    audio_paths: Sequence[str],
    descriptions: Sequence[str],
    model_name: str = DEFAULT_AUDIO_FRAME_MODEL,
    device: str = "auto",
    threshold: float = 0.3,
) -> AudioLocalizationResult:
    if not audio_paths:
        raise ValueError("Provide at least one audio path.")
    if not descriptions:
        raise ValueError("Provide at least one text description.")

    audio = list(audio_paths)
    text = list(descriptions)
    validate_parallel_lengths(audio=audio, text=text)

    PEAudioFrame, _ = load_audio_frame_runtime()
    resolved_device = resolve_device(device)
    model = PEAudioFrame.from_config(model_name, pretrained=True).to(resolved_device)
    model.eval()
    transform = get_audio_frame_transform(model_name, model)

    inputs = transform(audio=audio, text=text).to(resolved_device)
    with torch.inference_mode():
        outputs = model(**inputs, threshold=threshold)

    spans = outputs.spans or []
    events = [
        LocalizedEvent(
            description=description,
            spans=[(float(start), float(end)) for start, end in event_spans],
        )
        for description, event_spans in zip(text, spans)
    ]
    return AudioLocalizationResult(
        device=resolved_device.type,
        model_name=model_name,
        audio_paths=audio,
        events=events,
    )
