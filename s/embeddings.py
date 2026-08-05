import math
import os

from llama_index.embeddings.cohere import CohereEmbedding

COHERE_EMBED_MODEL = "embed-multilingual-v3.0"


def build_embed_model(input_type: str = "search_document") -> CohereEmbedding:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. Create a .env file in rag-project/ "
            "with COHERE_API_KEY=<your key>."
        )

    return CohereEmbedding(
        api_key=api_key,
        model_name=COHERE_EMBED_MODEL,
        input_type=input_type,
    )


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)
