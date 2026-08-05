import os

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.postprocessor.cohere_rerank import CohereRerank

COHERE_RERANK_MODEL = "rerank-multilingual-v3.0"


def build_postprocessors(top_n: int = 5) -> list[BaseNodePostprocessor]:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is not set")

    rerank = CohereRerank(
        api_key=api_key,
        model=COHERE_RERANK_MODEL,
        top_n=top_n,
    )
    return [rerank]
