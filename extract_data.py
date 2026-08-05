import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

from src.extractor import extract_all, save_extracted
from src.llm import build_llm
from src.loader import load_all_documents

sys.stdout.reconfigure(encoding="utf-8")

OUTPUT_PATH = "extracted_items.json"


def main():
    print("Loading documents...")
    docs = load_all_documents()
    print(f"  -> {len(docs)} documents\n")

    print("Building LLM...")
    llm = build_llm()
    print("  -> ok\n")

    print(f"Extracting items from {len(docs)} documents (this takes a few minutes)...")
    data = extract_all(llm, docs)

    print(f"\nSaving to {OUTPUT_PATH}...")
    save_extracted(data, OUTPUT_PATH)

    items = data["items"]
    print(f"\nDone. Totals:")
    print(f"  decisions: {len(items['decisions'])}")
    print(f"  rules:     {len(items['rules'])}")
    print(f"  warnings:  {len(items['warnings'])}")


if __name__ == "__main__":
    main()
