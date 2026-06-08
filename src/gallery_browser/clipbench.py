from __future__ import annotations

EVALUATION_DOC_URL = "https://github.com/facebookresearch/perception_models/blob/main/apps/pe/docs/evaluation.md"

CLIPBENCH_TASKS = [
    "zero-shot image classification",
    "zero-shot image retrieval",
    "zero-shot video classification",
    "zero-shot video retrieval",
]


def clipbench_summary() -> tuple[str, list[str]]:
    return EVALUATION_DOC_URL, list(CLIPBENCH_TASKS)