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
from src.synthesizer import answer_question

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    print("Setting up...")
    docs = load_all_documents()
    nodes = chunk_documents(docs)
    index = build_persistent_index(nodes)
    postprocessors = build_postprocessors(top_n=5)
    llm = build_llm()
    print("Ready.\n")

    print("=" * 60)
    print("PART 2 STEP 3: FULL QA PIPELINE (retrieve -> rerank -> LLM)")
    print("=" * 60)

    queries = [
        "איך מתקינים את הפרויקט?",
        "מה הצבע העיקרי של הממשק?",
        "What did the docs say about routing?",
        "האם נעשה שימוש ב-Redux או בState אחר לניהול הסל?",
        "מה המתכון של עוגיות שוקולד צ'יפס?",
    ]

    for q in queries:
        print(f"\n{'=' * 60}")
        print(f"Q: {q}")
        print(f"{'=' * 60}")

        response = answer_question(index, q, llm, postprocessors=postprocessors)

        print(f"\nA: {response.response}")

        print(f"\nSources used ({len(response.source_nodes)} chunks):")
        seen = set()
        for s in response.source_nodes:
            m = s.node.metadata
            key = f"{m['tool']}/{m['file_name']}"
            if key not in seen:
                seen.add(key)
                print(f"  - [{key}]  (score={s.score:.4f})")


if __name__ == "__main__":
    main()
