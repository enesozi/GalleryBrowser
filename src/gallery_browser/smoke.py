from __future__ import annotations

import ftfy
from decord import VideoReader

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

from .utils import get_audio_runtime_status


def main() -> None:
    fixed = ftfy.fix_text("caf\u00e9")
    print(f"ftfy_ok={fixed == 'café'}")
    print(f"decord_ok={VideoReader.__name__ == 'VideoReader'}")
    print(f"clip_config_count={len(pe.CLIP.available_configs())}")
    print(f"vision_config_count={len(pe.VisionTransformer.available_configs())}")
    print(f"tokenizer_available={hasattr(transforms, 'get_text_tokenizer')}")
    audio_status = get_audio_runtime_status()
    print(f"audio_runtime_ok={audio_status.ok}")
    if audio_status.detail:
        print(f"audio_runtime_detail={audio_status.detail}")


if __name__ == "__main__":
    main()