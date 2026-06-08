from __future__ import annotations

import argparse
from typing import Sequence

from gallery_browser.audio_events import (
    DEFAULT_AUDIO_FRAME_MODEL,
    available_audio_frame_models,
    localize_audio_events,
)
from gallery_browser.audio_visual import DEFAULT_AV_MODEL, available_av_models, embed_audio_visual
from gallery_browser.clipbench import clipbench_summary
from gallery_browser.inference import DEFAULT_MODEL, available_clip_models, classify_media
from gallery_browser.utils import get_audio_runtime_status
from gallery_browser.vision import DEFAULT_VISION_MODEL, available_vision_models, extract_vision_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gallery-browser",
        description="Run modular perception_models examples across image, video, audio, and vision tasks.",
    )
    subparsers = parser.add_subparsers(dest="command")

    clip_parser = subparsers.add_parser(
        "clip",
        help="PE core CLIP image or single-frame video feature extraction.",
    )
    clip_parser.add_argument("--image", help="Path to an input image.")
    clip_parser.add_argument(
        "--video",
        help="Path to an input video. The CLI uses a single extracted frame for prediction.",
    )
    clip_parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        default=[],
        help="Candidate label text. Repeat to provide multiple labels.",
    )
    clip_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Perception Encoder CLIP config to load. Default: {DEFAULT_MODEL}.",
    )
    clip_parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use, such as auto, cpu, or cuda.",
    )
    clip_parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to extract when --video is used.",
    )
    clip_parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="How many ranked predictions to print.",
    )
    clip_parser.set_defaults(func=_run_clip)

    vision_parser = subparsers.add_parser(
        "vision",
        help="Load PE core, PE lang, or PE spatial checkpoints and extract vision features.",
    )
    vision_parser.add_argument("--image", required=True, help="Path to an input image.")
    vision_parser.add_argument(
        "--model",
        default=DEFAULT_VISION_MODEL,
        help=f"VisionTransformer config to load. Default: {DEFAULT_VISION_MODEL}.",
    )
    vision_parser.add_argument("--device", default="auto", help="Torch device to use.")
    vision_parser.add_argument(
        "--layer-index",
        type=int,
        default=-1,
        help="Layer index passed to forward_features.",
    )
    vision_parser.add_argument(
        "--keep-cls-token",
        action="store_true",
        help="Keep the CLS token in the returned feature tensor.",
    )
    vision_parser.set_defaults(func=_run_vision)

    av_parser = subparsers.add_parser(
        "av",
        help="Run PE-AV embeddings for audio, video, and text inputs.",
    )
    av_parser.add_argument(
        "--video",
        dest="videos",
        action="append",
        default=[],
        help="Path to a video file. Repeat for batching.",
    )
    av_parser.add_argument(
        "--audio",
        dest="audio",
        action="append",
        default=[],
        help="Path to an audio or video file for audio decoding. Repeat for batching.",
    )
    av_parser.add_argument(
        "--text",
        dest="texts",
        action="append",
        default=[],
        help="Text description. Repeat for batching.",
    )
    av_parser.add_argument(
        "--model",
        default=DEFAULT_AV_MODEL,
        help=f"PE-AV config to load. Default: {DEFAULT_AV_MODEL}.",
    )
    av_parser.add_argument("--device", default="auto", help="Torch device to use.")
    av_parser.set_defaults(func=_run_av)

    audio_parser = subparsers.add_parser(
        "audio-localize",
        help="Run PE-A Frame audio event localization.",
    )
    audio_parser.add_argument(
        "--audio",
        dest="audio",
        action="append",
        default=[],
        help="Path to an audio or video file. Repeat for batching.",
    )
    audio_parser.add_argument(
        "--text",
        dest="texts",
        action="append",
        default=[],
        help="Text description to localize. Repeat for batching.",
    )
    audio_parser.add_argument(
        "--model",
        default=DEFAULT_AUDIO_FRAME_MODEL,
        help=f"PE-A Frame config to load. Default: {DEFAULT_AUDIO_FRAME_MODEL}.",
    )
    audio_parser.add_argument("--device", default="auto", help="Torch device to use.")
    audio_parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Probability threshold used to turn scores into spans.",
    )
    audio_parser.set_defaults(func=_run_audio_localize)

    models_parser = subparsers.add_parser(
        "models",
        help="List available CLIP, vision, PE-AV, and PE-A Frame checkpoint configs.",
    )
    models_parser.add_argument(
        "--family",
        choices=["clip", "vision", "av", "audio", "all"],
        default="all",
        help="Which config family to list.",
    )
    models_parser.set_defaults(func=_run_models)

    clipbench_parser = subparsers.add_parser(
        "clipbench",
        help="Print the upstream Clipbench evaluation tasks and reference documentation.",
    )
    clipbench_parser.set_defaults(func=_run_clipbench)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return

    try:
        args.func(args)
    except (RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


def _run_clip(args: argparse.Namespace) -> None:
    if bool(args.image) == bool(args.video):
        raise ValueError("Provide exactly one of --image or --video.")
    if not args.labels:
        raise ValueError("Provide at least one --label.")

    result = classify_media(
        labels=args.labels,
        model_name=args.model,
        image_path=args.image,
        video_path=args.video,
        frame_index=args.frame_index,
        device=args.device,
    )

    print(f"source: {result.source}")
    print(f"source_kind: {result.source_kind}")
    print(f"device: {result.device}")
    print(f"model: {result.model_name}")
    print(f"image_feature_shape: {result.image_feature_shape}")
    print(f"text_feature_shape: {result.text_feature_shape}")
    print(f"logit_scale: {result.logit_scale:.4f}")
    print("predictions:")
    for index, prediction in enumerate(result.predictions[: max(args.top_k, 1)], start=1):
        print(f"{index}. {prediction.label}: {prediction.score:.4f}")


def _run_vision(args: argparse.Namespace) -> None:
    result = extract_vision_features(
        image_path=args.image,
        model_name=args.model,
        device=args.device,
        layer_idx=args.layer_index,
        strip_cls_token=not args.keep_cls_token,
    )
    print(f"source: {result.source}")
    print(f"device: {result.device}")
    print(f"model: {result.model_name}")
    print(f"feature_shape: {result.feature_shape}")


def _run_av(args: argparse.Namespace) -> None:
    result = embed_audio_visual(
        video_paths=args.videos or None,
        audio_paths=args.audio or None,
        texts=args.texts or None,
        model_name=args.model,
        device=args.device,
    )
    print(f"device: {result.device}")
    print(f"model: {result.model_name}")
    print(f"modalities: {', '.join(result.modalities)}")
    print("output_shapes:")
    for name, shape in result.output_shapes.items():
        print(f"- {name}: {shape}")
    print("dot_products:")
    for name, values in result.dot_products.items():
        formatted = ", ".join(f"{value:.4f}" for value in values)
        print(f"- {name}: [{formatted}]")


def _run_audio_localize(args: argparse.Namespace) -> None:
    result = localize_audio_events(
        audio_paths=args.audio,
        descriptions=args.texts,
        model_name=args.model,
        device=args.device,
        threshold=args.threshold,
    )
    print(f"device: {result.device}")
    print(f"model: {result.model_name}")
    for event in result.events:
        span_str = ", ".join(f"({start:.2f}, {end:.2f})" for start, end in event.spans)
        print(f'"{event.description}": [{span_str}]')


def _run_models(args: argparse.Namespace) -> None:
    if args.family in {"clip", "all"}:
        print("clip:")
        for model_name in available_clip_models():
            print(f"- {model_name}")
    if args.family in {"vision", "all"}:
        print("vision:")
        for model_name in available_vision_models():
            print(f"- {model_name}")
    if args.family in {"av", "all"}:
        print("av:")
        for model_name in available_av_models():
            print(f"- {model_name}")
    if args.family in {"audio", "all"}:
        print("audio-localize:")
        for model_name in available_audio_frame_models():
            print(f"- {model_name}")


def _run_clipbench(_: argparse.Namespace) -> None:
    doc_url, tasks = clipbench_summary()
    print(f"doc: {doc_url}")
    print("tasks:")
    for task in tasks:
        print(f"- {task}")
    status = get_audio_runtime_status()
    if not status.ok:
        print("runtime_note:")
        print(f"- audio and video evaluation paths may need extra system runtime support: {status.detail}")


if __name__ == "__main__":
    main()