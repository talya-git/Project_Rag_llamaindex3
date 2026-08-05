import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

from .config import PROJECT_ROOT

CHROMA_DB_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "rag-agentic-docs"


def get_vector_store() -> tuple[ChromaVectorStore, int]:
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    vector_count = collection.count()
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return vector_store, vector_count
