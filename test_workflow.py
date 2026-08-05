import asyncio
import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

from src.chunker import chunk_documents
from src.index import build_persistent_index
from src.llm import build_llm
from src.loader import load_all_documents
from src.postprocessor import build_postprocessors
from src.store import ItemStore
from src.workflow import RAGWorkflow

sys.stdout.reconfigure(encoding="utf-8")


async def main():
    print("Setup...")
    docs = load_all_documents()
    nodes = chunk_documents(docs)
    index = build_persistent_index(nodes)
    postprocessors = build_postprocessors(top_n=5)
    llm = build_llm()
    store = ItemStore()
    workflow = RAGWorkflow(index, llm, postprocessors, store=store, top_k=10, timeout=120)
    print(f"Ready. Store has: {store.counts()}\n")

    queries = [
        ("semantic (specific)", "מה הצבע העיקרי של הממשק?"),
        ("structured (list all)", "תן לי רשימה של כל ההחלטות הטכניות"),
        ("structured (filter)", "אילו אזהרות יש בפרויקט?"),
        ("structured (by scope)", "אילו כללים יש לגבי styling?"),
    ]

    for label, q in queries:
        print("=" * 70)
        print(f"[{label}] Q: {q}")
        print("=" * 70)
        result = await workflow.run(query=q)
        print(f"\nAnswer:\n{result['answer'][:500]}...\n"
              if len(result['answer']) > 500
              else f"\nAnswer:\n{result['answer']}\n")
        print("Trace:")
        for step_msg in result["trace"]:
            print(f"  - {step_msg}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
