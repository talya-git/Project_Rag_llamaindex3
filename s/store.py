import json
from pathlib import Path

from .config import PROJECT_ROOT

DEFAULT_PATH = PROJECT_ROOT / "extracted_items.json"


class ItemStore:
    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = path
        with open(path, encoding="utf-8") as f:
            self.data = json.load(f)
        self.items = self.data["items"]

    def get_all(self, item_type: str) -> list[dict]:
        return self.items.get(item_type, [])

    def filter_by_tool(self, item_type: str, tool: str) -> list[dict]:
        return [
            item for item in self.get_all(item_type)
            if item.get("source", {}).get("tool") == tool
        ]

    def filter_by_severity(self, severity: str) -> list[dict]:
        return [w for w in self.get_all("warnings") if w.get("severity") == severity]

    def filter_by_tag(self, tag: str) -> list[dict]:
        tag_lower = tag.lower()
        return [
            d for d in self.get_all("decisions")
            if any(t.lower() == tag_lower for t in d.get("tags", []))
        ]

    def filter_by_scope(self, scope: str) -> list[dict]:
        return [r for r in self.get_all("rules") if r.get("scope") == scope]

    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.items.items()}
