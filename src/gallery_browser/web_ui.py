from __future__ import annotations

import gradio as gr

from .inference import DEFAULT_MODEL, available_clip_models
from .search import search_gallery


def launch_web_ui(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    gallery_dir: str | None = None,
    device: str = "auto",
) -> None:
    default_gallery = gallery_dir or ""
    models = available_clip_models()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else DEFAULT_MODEL)

    def _run_search(
        folder: str,
        query: str,
        model_name: str,
        top_k: int,
        frame_index: int,
    ) -> tuple[list[list[str | float]], str]:
        try:
            result = search_gallery(
                gallery_dir=folder,
                query=query,
                model_name=model_name,
                top_k=top_k,
                frame_index=frame_index,
                device=device,
            )
        except ValueError as exc:
            return [], f"Error: {exc}"

        rows = [[match.score, match.kind, match.path] for match in result.matches]
        summary = (
            f"Scanned {result.scanned_files} media files. "
            f"Returned {len(result.matches)} results. "
            f"Skipped {len(result.skipped)} files."
        )
        return rows, summary

    with gr.Blocks(title="Gallery Browser") as demo:
        gr.Markdown("# Gallery Browser")
        gr.Markdown("Choose a gallery folder and search images/videos with a text description.")

        folder = gr.Textbox(label="Gallery folder", value=default_gallery, placeholder="/path/to/gallery")
        query = gr.Textbox(label="Description query", placeholder="a dog jumping over a fence")
        model_name = gr.Dropdown(label="Model", choices=models, value=default_model)
        top_k = gr.Slider(label="Top results", minimum=1, maximum=50, value=10, step=1)
        frame_index = gr.Number(label="Video frame index", value=0, precision=0)

        search_button = gr.Button("Search")
        results = gr.Dataframe(
            headers=["score", "kind", "path"],
            datatype=["number", "str", "str"],
            label="Matches",
            interactive=False,
            row_count=10,
            column_count=(3, "fixed"),
        )
        status = gr.Markdown()

        search_button.click(
            _run_search,
            inputs=[folder, query, model_name, top_k, frame_index],
            outputs=[results, status],
        )

    demo.launch(server_name=host, server_port=port)