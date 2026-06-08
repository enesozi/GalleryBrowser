from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from decord import AudioReader, VideoReader
from PIL import Image
import torch
import torchvision.transforms as T
from huggingface_hub import snapshot_download
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, BatchFeature

from core.vision_encoder.config import PE_VISION_CONFIG


@dataclass(slots=True)
class RuntimeStatus:
    ok: bool
    detail: str | None = None


def _pixel_to_float(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 255.0


class _FallbackAudioProcessor:
    def __init__(self, sampling_rate: int, hop_length: int):
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length

    def _reflect_pad(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.size(-1) % self.hop_length == 0:
            return wav
        padding = (0, self.hop_length - (wav.size(-1) % self.hop_length))
        return torch.nn.functional.pad(wav, padding, mode="reflect")

    def _load_audio(self, path: str) -> torch.Tensor:
        resolved = resolve_path(path)
        reader = AudioReader(str(resolved), sample_rate=self.sampling_rate, mono=True)
        audio = reader[:]
        return torch.from_numpy(audio.asnumpy())

    def __call__(self, raw_audio: str | list[str]) -> BatchFeature:
        if isinstance(raw_audio, str):
            raw_audio = [raw_audio]

        if isinstance(raw_audio, (list, tuple)) and raw_audio and isinstance(raw_audio[0], str):
            raw_audio = [self._load_audio(path) for path in raw_audio]

        processed = [self._reflect_pad(sample).T for sample in raw_audio]

        lengths = torch.tensor([sample.size(0) for sample in processed])
        input_values = pad_sequence(processed, batch_first=True).transpose(1, 2)
        padding_mask = torch.arange(lengths.max())[None] < lengths[:, None]
        return BatchFeature({"input_values": input_values, "padding_mask": padding_mask})


class _FallbackVideoProcessor:
    def __init__(self, fixed_len_video: bool, image_size: int):
        self.fixed_len_video = fixed_len_video
        self.frame_transform = T.Compose(
            [
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.Lambda(_pixel_to_float),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def _sample_indices(self, total_frames: int, num_frames: int = 16) -> list[int]:
        if total_frames <= 0:
            raise ValueError("Video contains no frames.")
        if total_frames == 1:
            return [0] * num_frames
        return [int(index * (total_frames - 1) / (num_frames - 1)) for index in range(num_frames)]

    def _load_video(self, path: str) -> torch.Tensor:
        resolved = resolve_path(path)
        reader = VideoReader(str(resolved))
        frame_count = len(reader)
        if frame_count == 0:
            raise ValueError(f"Video has no frames: {resolved}")
        if self.fixed_len_video:
            frames = reader.get_batch(self._sample_indices(frame_count)).asnumpy()
        else:
            frames = reader.get_batch(list(range(frame_count))).asnumpy()
        return torch.from_numpy(frames)

    def _transform_video(self, video: torch.Tensor) -> torch.Tensor:
        frames = []
        for frame in video:
            channels_first = frame.permute(2, 0, 1)
            frames.append(self.frame_transform(channels_first))
        return torch.stack(frames, dim=0)

    def __call__(self, raw_video: str | list[str]) -> BatchFeature:
        if isinstance(raw_video, str):
            raw_video = [raw_video]

        if isinstance(raw_video, (list, tuple)) and raw_video and isinstance(raw_video[0], str):
            raw_video = [self._load_video(path) for path in raw_video]

        videos = list(raw_video)
        transformed = [self._transform_video(video) for video in videos]
        lengths = torch.tensor([video.size(0) for video in transformed])
        padding_mask = torch.arange(lengths.max())[None] < lengths[:, None]
        pixel_values = pad_sequence(transformed, batch_first=True)
        return BatchFeature(
            {"pixel_values_videos": pixel_values, "padding_mask_videos": padding_mask}
        )


class FallbackPEAudioVisualTransform:
    def __init__(
        self,
        tokenizer,
        audio_processor: _FallbackAudioProcessor,
        video_processor: _FallbackVideoProcessor,
    ):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.video_processor = video_processor

    @classmethod
    def from_model_name_and_config(
        cls, model_name: str, config
    ) -> "FallbackPEAudioVisualTransform":
        checkpoint_dir = snapshot_download(
            repo_id=f"facebook/{model_name}", revision="perception_models"
        )
        vision_config = PE_VISION_CONFIG[config.audio_visual_model.visual_model.pe_encoder]
        return cls(
            tokenizer=AutoTokenizer.from_pretrained(checkpoint_dir),
            audio_processor=_FallbackAudioProcessor(
                sampling_rate=config.audio_visual_model.audio_model.dac_vae_encoder.sampling_rate,
                hop_length=config.audio_visual_model.audio_model.dac_vae_encoder.hop_length,
            ),
            video_processor=_FallbackVideoProcessor(
                fixed_len_video=config.audio_visual_model.visual_model.fixed_len_video,
                image_size=vision_config.image_size,
            ),
        )

    def __call__(
        self,
        *,
        text: list[str] | None = None,
        audio: list[str] | None = None,
        videos: list[str] | None = None,
    ) -> BatchFeature:
        batch = BatchFeature()
        if text is not None:
            batch.update(
                self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=512,
                )
            )
        if audio is not None:
            batch.update(self.audio_processor(audio))
        if videos is not None:
            batch.update(self.video_processor(videos))
        return batch


class FallbackPEAudioFrameTransform:
    def __init__(self, tokenizer, audio_processor: _FallbackAudioProcessor):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor

    @classmethod
    def from_model_name_and_config(cls, model_name: str, config) -> "FallbackPEAudioFrameTransform":
        checkpoint_dir = snapshot_download(
            repo_id=f"facebook/{model_name}", revision="perception_models"
        )
        return cls(
            tokenizer=AutoTokenizer.from_pretrained(checkpoint_dir),
            audio_processor=_FallbackAudioProcessor(
                sampling_rate=config.audio_model.dac_vae_encoder.sampling_rate,
                hop_length=config.audio_model.dac_vae_encoder.hop_length,
            ),
        )

    def __call__(
        self, *, text: list[str] | None = None, audio: list[str] | None = None
    ) -> BatchFeature:
        batch = BatchFeature()
        if text is not None:
            batch.update(
                self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=512,
                )
            )
        if audio is not None:
            batch.update(self.audio_processor(audio))
        return batch


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def inference_autocast(device: torch.device, dtype: torch.dtype | None = None) -> Any:
    if device.type == "cuda":
        kwargs = {"device_type": "cuda"}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return torch.autocast(**kwargs)
    return nullcontext()


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def load_rgb_image(path: str) -> tuple[Image.Image, str]:
    resolved = resolve_path(path)
    return Image.open(resolved).convert("RGB"), str(resolved)


def load_video_frame(path: str, frame_index: int) -> tuple[Image.Image, str]:
    resolved = resolve_path(path)
    reader = VideoReader(str(resolved))
    if len(reader) == 0:
        raise ValueError(f"Video has no frames: {resolved}")

    bounded_index = min(max(frame_index, 0), len(reader) - 1)
    frame = reader[bounded_index].asnumpy()
    return Image.fromarray(frame).convert("RGB"), str(resolved)


def validate_parallel_lengths(**items: list[str] | None) -> None:
    lengths = {name: len(values) for name, values in items.items() if values}
    distinct = {length for length in lengths.values() if length > 1}
    if len(distinct) > 1:
        pairs = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(
            f"Input lists must have matching lengths when batching modalities: {pairs}"
        )


def load_audio_visual_runtime():
    try:
        from core.audio_visual_encoder import PEAudioVisual, PEAudioVisualTransform
    except Exception as exc:  # pragma: no cover - depends on system runtime
        raise RuntimeError(_format_runtime_error("PE-AV", exc)) from exc
    return PEAudioVisual, PEAudioVisualTransform


def load_audio_frame_runtime():
    try:
        from core.audio_visual_encoder import PEAudioFrame, PEAudioFrameTransform
    except Exception as exc:  # pragma: no cover - depends on system runtime
        raise RuntimeError(_format_runtime_error("PE-A Frame", exc)) from exc
    return PEAudioFrame, PEAudioFrameTransform


def get_audio_visual_transform(model_name: str, model):
    _, transform_cls = load_audio_visual_runtime()
    if transform_cls is not None:
        return transform_cls.from_config(model_name)
    return FallbackPEAudioVisualTransform.from_model_name_and_config(model_name, model.config)


def get_audio_frame_transform(model_name: str, model):
    _, transform_cls = load_audio_frame_runtime()
    if transform_cls is not None:
        return transform_cls.from_config(model_name)
    return FallbackPEAudioFrameTransform.from_model_name_and_config(model_name, model.config)


def get_audio_runtime_status() -> RuntimeStatus:
    try:
        from core.audio_visual_encoder import PEAudioVisual
    except Exception as exc:
        return RuntimeStatus(ok=False, detail=str(exc))

    if PEAudioVisual is None:
        return RuntimeStatus(ok=False, detail="PE-AV model class could not be imported.")
    return RuntimeStatus(ok=True)


def _format_runtime_error(name: str, exc: Exception) -> str:
    return (
        f"{name} runtime unavailable: {exc}. "
        "This upstream path needs xformers plus a compatible torchcodec/FFmpeg runtime."
    )
