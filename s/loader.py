from pathlib import Path

from llama_index.core import Document
from llama_index.core import SimpleDirectoryReader

from .config import AGENTIC_TOOLS


def _make_metadata_fn(tool_name: str):
    def file_metadata(file_path: str) -> dict:
        path = Path(file_path)
        return {
            "tool": tool_name,
            "file_name": path.name,
            "file_path": str(path),
        }

    return file_metadata


def load_tool_documents(tool_name: str, tool_dir: Path) -> list[Document]:
    if not tool_dir.exists():
        raise FileNotFoundError(f"Tool directory not found: {tool_dir}")

    reader = SimpleDirectoryReader(
        input_dir=str(tool_dir),
        required_exts=[".md"],
        recursive=True,
        file_metadata=_make_metadata_fn(tool_name),
    )
    return reader.load_data()


def load_all_documents() -> list[Document]:
    all_docs: list[Document] = []
    for tool_name, tool_dir in AGENTIC_TOOLS.items():
        docs = load_tool_documents(tool_name, tool_dir)
        all_docs.extend(docs)
    return all_docs
