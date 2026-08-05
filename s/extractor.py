import json
import re

from llama_index.core import Document
from llama_index.core.llms import LLM, ChatMessage, MessageRole

EXTRACTION_PROMPT = """You are extracting structured items from documentation about a web project.
The documentation may be in Hebrew, English, or mixed.

Extract three kinds of items, ONLY if explicitly stated:
- DECISIONS: concrete technical choices made (e.g., "chose React Context for cart state").
- RULES: guidelines/conventions developers must follow (e.g., "all Hebrew text must use RTL").
- WARNINGS: things to avoid, sensitive code, risky operations.

Return ONLY a JSON object - no markdown fences, no commentary. Use this EXACT structure:

{{
  "decisions": [
    {{"title": "short title", "summary": "1-2 sentences", "tags": ["tag1", "tag2"]}}
  ],
  "rules": [
    {{"rule": "the rule in one sentence", "scope": "ui|api|styling|general|...", "notes": "optional caveats"}}
  ],
  "warnings": [
    {{"area": "auth|cart|routing|...", "message": "the warning", "severity": "low|medium|high"}}
  ]
}}

If a category has no items in this file, return an empty list [] for it.
Preserve the original language (Hebrew stays Hebrew, English stays English).

Documentation file: {source}
---
{text}
---

JSON output:"""


def _parse_json_response(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response")
    return json.loads(match.group(0))


def extract_from_document(llm: LLM, doc: Document) -> dict:
    source = f"{doc.metadata.get('tool')}/{doc.metadata.get('file_name')}"
    prompt = EXTRACTION_PROMPT.format(source=source, text=doc.text)

    messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
    response = llm.chat(messages)
    raw = str(response.message.content)
    data = _parse_json_response(raw)

    return {
        "source": {
            "tool": doc.metadata.get("tool"),
            "file": doc.metadata.get("file_name"),
            "file_path": doc.metadata.get("file_path"),
        },
        "decisions": data.get("decisions", []),
        "rules": data.get("rules", []),
        "warnings": data.get("warnings", []),
    }


def extract_all(llm: LLM, docs: list[Document]) -> dict:
    all_decisions: list[dict] = []
    all_rules: list[dict] = []
    all_warnings: list[dict] = []

    for i, doc in enumerate(docs, 1):
        source_str = f"{doc.metadata.get('tool')}/{doc.metadata.get('file_name')}"
        print(f"  [{i}/{len(docs)}] {source_str}...", end=" ", flush=True)

        try:
            result = extract_from_document(llm, doc)
            source = result["source"]

            for d in result["decisions"]:
                if not isinstance(d, dict):
                    continue
                d["id"] = f"dec-{len(all_decisions)+1:03d}"
                d["source"] = source
                all_decisions.append(d)

            for r in result["rules"]:
                if not isinstance(r, dict):
                    continue
                r["id"] = f"rule-{len(all_rules)+1:03d}"
                r["source"] = source
                all_rules.append(r)

            for w in result["warnings"]:
                if not isinstance(w, dict):
                    continue
                w["id"] = f"warn-{len(all_warnings)+1:03d}"
                w["source"] = source
                all_warnings.append(w)

            print(f"D={len(result['decisions'])} R={len(result['rules'])} W={len(result['warnings'])}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {str(e)[:80]}")

    return {
        "schema_version": "1.0",
        "items": {
            "decisions": all_decisions,
            "rules": all_rules,
            "warnings": all_warnings,
        },
    }


def save_extracted(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
