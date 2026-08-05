from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

DEFAULT_TOP_K = 5


def build_retriever(index: VectorStoreIndex, top_k: int = DEFAULT_TOP_K) -> BaseRetriever:
    return index.as_retriever(similarity_top_k=top_k)


def retrieve(
    index: VectorStoreIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    postprocessors: list[BaseNodePostprocessor] | None = None,
) -> list[NodeWithScore]:
    retriever = build_retriever(index, top_k=top_k)
    nodes = retriever.retrieve(query)

    if postprocessors:
        query_bundle = QueryBundle(query_str=query)
        for pp in postprocessors:
            nodes = pp.postprocess_nodes(nodes, query_bundle=query_bundle)

    return nodes


def format_results(results: list[NodeWithScore]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        meta = r.node.metadata
        snippet = r.node.text[:120].replace("\n", " ").strip()
        lines.append(
            f"  #{i} score={r.score:.4f}  [{meta['tool']}/{meta['file_name']}]\n"
            f"      {snippet}..."
        )
    return "\n".join(lines)
