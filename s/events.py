from llama_index.core.schema import NodeWithScore
from llama_index.core.workflow import Event


class QueryValidatedEvent(Event):
    query: str


class NodesRetrievedEvent(Event):
    query: str
    nodes: list[NodeWithScore]


class NodesRerankedEvent(Event):
    query: str
    nodes: list[NodeWithScore]
    top_score: float


class RetryEvent(Event):
    query: str
    reason: str
    attempt: int
