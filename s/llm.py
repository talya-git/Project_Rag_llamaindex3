import os

from llama_index.llms.cohere import Cohere

COHERE_LLM_MODEL = "command-r-plus-08-2024"


def build_llm() -> Cohere:
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("COHERE_API_KEY is not set")

    return Cohere(
        api_key=api_key,
        model=COHERE_LLM_MODEL,
        temperature=0.1,
    )
