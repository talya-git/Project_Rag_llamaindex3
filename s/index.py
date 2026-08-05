from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode

from .embeddings import build_embed_model
from .vector_store import get_vector_store


def build_persistent_index(nodes: list[BaseNode]) -> VectorStoreIndex:
    embed_model = build_embed_model(input_type="search_document")
    vector_store, vector_count = get_vector_store()

    if vector_count > 0:
        print(f"  Vector store already has {vector_count} vectors - "
              f"skipping upload, connecting to existing collection.")
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )

    print(f"  Vector store is empty - embedding and uploading {len(nodes)} nodes...")
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
