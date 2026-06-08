from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
from PIL import Image
import torch

from .utils import inference_autocast, load_rgb_image, load_video_frame, resolve_device

DEFAULT_MODEL = "PE-Core-B16-224"


@dataclass(slots=True)
class Prediction:
    label: str
    score: float


@dataclass(slots=True)
class PredictionResult:
    source: str
    source_kind: str
    device: str
    model_name: str
    image_feature_shape: tuple[int, ...]
    text_feature_shape: tuple[int, ...]
    logit_scale: float
    image_features: torch.Tensor
    text_features: torch.Tensor
    predictions: list[Prediction]


def available_clip_models() -> list[str]:
    return list(pe.CLIP.available_configs())


def classify_media(
    *,
    labels: Sequence[str],
    model_name: str = DEFAULT_MODEL,
    image_path: str | None = None,
    video_path: str | None = None,
    frame_index: int = 0,
    device: str = "auto",
) -> PredictionResult:
    if not labels:
        raise ValueError("At least one label is required.")
    if bool(image_path) == bool(video_path):
        raise ValueError("Provide exactly one of image_path or video_path.")

    resolved_device = resolve_device(device)
    model = pe.CLIP.from_config(model_name, pretrained=True).to(resolved_device)
    model.eval()

    preprocess = transforms.get_image_transform(model.image_size)
    tokenizer = transforms.get_text_tokenizer(model.context_length)
    image, source, source_kind = _load_source(
        image_path=image_path,
        video_path=video_path,
        frame_index=frame_index,
    )

    image_tensor = preprocess(image).unsqueeze(0).to(resolved_device)
    text_tensor = tokenizer(list(labels)).to(resolved_device)

    with torch.inference_mode(), inference_autocast(resolved_device):
        image_features, text_features, logit_scale = model(image_tensor, text_tensor)
        scores = (logit_scale * image_features @ text_features.T).softmax(dim=-1)[0]

    image_features = image_features.detach().cpu()
    text_features = text_features.detach().cpu()
    ranked = sorted(
        (
            Prediction(label=label, score=float(score))
            for label, score in zip(labels, scores.tolist())
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    return PredictionResult(
        source=source,
        source_kind=source_kind,
        device=resolved_device.type,
        model_name=model_name,
        image_feature_shape=tuple(image_features.shape),
        text_feature_shape=tuple(text_features.shape),
        logit_scale=float(logit_scale.detach().cpu().item()),
        image_features=image_features,
        text_features=text_features,
        predictions=ranked,
    )


def _load_source(
    *,
    image_path: str | None,
    video_path: str | None,
    frame_index: int,
) -> tuple[Image.Image, str, str]:
    if image_path:
        image, source = load_rgb_image(image_path)
        return image, source, "image"

    image, source = load_video_frame(video_path or "", frame_index)
    return image, source, "video"
