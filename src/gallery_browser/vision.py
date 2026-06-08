from __future__ import annotations

from dataclasses import dataclass

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms
import torch

from .utils import inference_autocast, load_rgb_image, resolve_device

DEFAULT_VISION_MODEL = "PE-Core-B16-224"


@dataclass(slots=True)
class VisionFeatureResult:
    source: str
    device: str
    model_name: str
    feature_shape: tuple[int, ...]
    features: torch.Tensor


def available_vision_models() -> list[str]:
    return list(pe.VisionTransformer.available_configs())


def extract_vision_features(
    *,
    image_path: str,
    model_name: str = DEFAULT_VISION_MODEL,
    device: str = "auto",
    layer_idx: int = -1,
    strip_cls_token: bool = True,
) -> VisionFeatureResult:
    resolved_device = resolve_device(device)
    model = pe.VisionTransformer.from_config(model_name, pretrained=True).to(resolved_device)
    model.eval()

    preprocess = transforms.get_image_transform(model.image_size)
    image, source = load_rgb_image(image_path)
    image_tensor = preprocess(image).unsqueeze(0).to(resolved_device)

    with torch.inference_mode(), inference_autocast(resolved_device):
        features = model.forward_features(
            image_tensor,
            layer_idx=layer_idx,
            strip_cls_token=strip_cls_token,
        )

    return VisionFeatureResult(
        source=source,
        device=resolved_device.type,
        model_name=model_name,
        feature_shape=tuple(features.shape),
        features=features,
    )
