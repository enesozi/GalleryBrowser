# Gallery Browser

Minimal `uv`-based Python package for perception experiments with `perception_models`, `decord`, and `ftfy`.

## What It Includes

- A `uv` project with explicit Python 3.12 support.
- Git-backed `uv` integration with `perception_models` from the upstream repository.
- Modular commands for the official PE example families: CLIP, vision encoder checkpoints, PE-AV embeddings, and PE-A Frame audio localization.
- A smoke command that validates `ftfy`, `decord`, CLIP configs, vision configs, and the current audio runtime status.

## Setup

Install and sync dependencies:

```bash
uv sync
```

The project resolves `perception_models` directly from Git via `uv`:

```bash
uv add git+https://github.com/facebookresearch/perception_models.git
```

If you are bootstrapping from scratch, a complete dependency setup looks like:

```bash
uv sync
```

Run the smoke check:

```bash
uv run gallery-browser-smoke
```

List available CLIP and vision configs:

```bash
uv run gallery-browser models --family all
```

## Commands

Sample media is available under `assets/` for the commands below.

PE core CLIP image or single-frame video feature extraction:

```bash
uv run gallery-browser \
	clip \
	--image assets/cat.png \
	--label "a cat" \
	--label "a dog" \
	--label "a diagram"
```

Run the CLIP path on a video by classifying one extracted frame:

```bash
uv run gallery-browser \
	clip \
	--video assets/dog.mp4 \
	--frame-index 0 \
	--label "a person indoors" \
	--label "a dog jumping" \
	--label "an empty room"
```

Load PE core, PE lang, or PE spatial vision encoder checkpoints and print the feature shape:

```bash
uv run gallery-browser \
	vision \
	--image assets/cat.png \
	--model PE-Lang-L14-448
```

Run PE-AV embeddings for aligned video, audio, and text batches:

```bash
uv run gallery-browser \
	av \
	--video assets/train.mp4 \
	--video assets/office.mp4 \
	--audio assets/train.mp4 \
	--audio assets/office.mp4 \
	--text "A person talking with sirens and a train in the background" \
	--text "Two people talking in an office, with sounds of workers typing on a keyboard"
```

Run PE-A Frame audio event localization:

```bash
uv run gallery-browser \
	audio-localize \
	--audio assets/office.mp4 \
	--text "a person talking"
```

Print the upstream Clipbench evaluation task list and doc reference:

```bash
uv run gallery-browser clipbench
```

## Notes

- The first inference run downloads model weights from Hugging Face.
- `uv.lock` pins the current `perception_models` Git commit for reproducibility.
- The default model is `PE-Core-B16-224` to keep the initial example moderate.
- The CLIP video path uses `decord` to extract a frame, then applies the official image-text inference pattern.
- The PE-AV and PE-A Frame paths use an internal decord-based fallback when the upstream TorchCodec transform stack is unavailable.
- The first PE-AV or PE-A Frame run may spend significant time downloading model weights from Hugging Face.
- `xformers` is installed, but its compiled extensions do not match the local PyTorch build, so the package runs with the upstream warning still visible for audio/video paths.
