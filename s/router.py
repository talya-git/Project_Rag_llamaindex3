import json
import re

from llama_index.core.llms import LLM, ChatMessage, MessageRole

ROUTER_PROMPT = """You are a query router for a documentation system about a web project.
The system has TWO ways to answer questions:

1. SEMANTIC SEARCH - good for specific factual questions, "how" / "what is" / "where" / "why".
   Example: "What color was chosen for primary buttons?", "How do I install the project?"

2. STRUCTURED LOOKUP - good for list / counting / filtering questions.
   Available structured types:
   - "decisions": all technical decisions made in the project
   - "rules": all rules/guidelines/conventions
   - "warnings": all warnings / sensitive areas / things to avoid
   Example: "List all decisions about routing", "What rules apply to the UI?", "What warnings are there?"

Decide which mode is best for this query. Return ONLY valid JSON, no fences or commentary:

{{"mode": "semantic"}}

OR

{{"mode": "structured", "item_type": "decisions|rules|warnings", "filter_field": "tool|severity|tag|scope|null", "filter_value": "<value or null>"}}

Examples:
Query: "מה הצבע של הכפתורים?"
{{"mode": "semantic"}}

Query: "תן לי רשימה של כל ההחלטות הטכניות"
{{"mode": "structured", "item_type": "decisions", "filter_field": null, "filter_value": null}}

Query: "אילו אזהרות חמורות יש בפרויקט?"
{{"mode": "structured", "item_type": "warnings", "filter_field": "severity", "filter_value": "high"}}

Query: "כל הכללים שקשורים ל-styling"
{{"mode": "structured", "item_type": "rules", "filter_field": "scope", "filter_value": "styling"}}

Query: "מה ההחלטות של claude?"
{{"mode": "structured", "item_type": "decisions", "filter_field": "tool", "filter_value": "claude"}}

Query: "{query}"
JSON:"""


def _parse_json(content: str) -> dict:
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError("No JSON object in router response")
    return json.loads(match.group(0))


def route_query(llm: LLM, query: str) -> dict:
    prompt = ROUTER_PROMPT.format(query=query)
    messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
    response = llm.chat(messages)
    raw = str(response.message.content)

    try:
        decision = _parse_json(raw)
        if decision.get("mode") not in ("semantic", "structured"):
            return {"mode": "semantic"}
        return decision
    except Exception:
        return {"mode": "semantic"}
