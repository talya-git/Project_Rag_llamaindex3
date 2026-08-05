import json

from llama_index.core import VectorStoreIndex
from llama_index.core.llms import LLM, ChatMessage, MessageRole
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import QueryBundle
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step

from .events import (
    NodesRerankedEvent,
    NodesRetrievedEvent,
    QueryValidatedEvent,
    RetryEvent,
)
from .retriever import build_retriever
from .router import route_query
from .store import ItemStore
from .synthesizer import build_synthesizer

MIN_CONFIDENCE_SCORE = 0.001
MAX_RETRIES = 2
MIN_QUERY_LENGTH = 3


def _stop(answer: str, sources=None, trace=None) -> StopEvent:
    return StopEvent(result={
        "answer": answer,
        "sources": sources or [],
        "trace": trace or [],
    })


# A small event type for the structured path
from llama_index.core.workflow import Event


class StructuredQueryEvent(Event):
    query: str
    item_type: str
    filter_field: str | None
    filter_value: str | None


class RAGWorkflow(Workflow):
    def __init__(
        self,
        index: VectorStoreIndex,
        llm: LLM,
        postprocessors: list[BaseNodePostprocessor],
        store: ItemStore | None = None,
        top_k: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.index = index
        self.llm = llm
        self.postprocessors = postprocessors
        self.store = store
        self.top_k = top_k

    @step
    async def validate_query(
        self, ctx: Context, ev: StartEvent
    ) -> QueryValidatedEvent | StopEvent:
        query = (ev.query or "").strip()
        await ctx.store.set("original_query", query)
        await ctx.store.set("retry_attempts", 0)
        await ctx.store.set("trace", [])

        if not query:
            return _stop("אנא הזן שאלה.", trace=["validate: empty query"])
        if len(query) < MIN_QUERY_LENGTH:
            return _stop(
                "השאלה קצרה מדי. נסחי שאלה ברורה יותר.",
                trace=["validate: query too short"],
            )

        trace = await ctx.store.get("trace")
        trace.append(f"validate: ok ({len(query)} chars)")
        await ctx.store.set("trace", trace)

        return QueryValidatedEvent(query=query)

    @step
    async def route_query_step(
        self, ctx: Context, ev: QueryValidatedEvent
    ) -> NodesRetrievedEvent | StructuredQueryEvent | StopEvent:
        trace = await ctx.store.get("trace")

        # Skip routing on retries - retries always use semantic search
        attempts = await ctx.store.get("retry_attempts")
        if attempts > 0 or self.store is None:
            trace.append(f"route: skip routing (attempt={attempts}, store={'on' if self.store else 'off'}) -> semantic")
            await ctx.store.set("trace", trace)
            retriever = build_retriever(self.index, top_k=self.top_k)
            nodes = retriever.retrieve(ev.query)
            trace.append(f"retrieve: got {len(nodes)} candidates")
            await ctx.store.set("trace", trace)
            if not nodes:
                return _stop("לא מצאתי שום תוצאות.", trace=trace)
            return NodesRetrievedEvent(query=ev.query, nodes=nodes)

        decision = route_query(self.llm, ev.query)
        trace.append(f"route: decision={decision}")
        await ctx.store.set("trace", trace)

        if decision["mode"] == "structured":
            return StructuredQueryEvent(
                query=ev.query,
                item_type=decision.get("item_type", "decisions"),
                filter_field=decision.get("filter_field"),
                filter_value=decision.get("filter_value"),
            )

        retriever = build_retriever(self.index, top_k=self.top_k)
        nodes = retriever.retrieve(ev.query)
        trace.append(f"retrieve: got {len(nodes)} candidates")
        await ctx.store.set("trace", trace)
        if not nodes:
            return _stop("לא מצאתי שום תוצאות.", trace=trace)
        return NodesRetrievedEvent(query=ev.query, nodes=nodes)

    @step
    async def answer_structured(
        self, ctx: Context, ev: StructuredQueryEvent
    ) -> StopEvent:
        trace = await ctx.store.get("trace")
        store = self.store

        item_type = ev.item_type
        if ev.filter_field == "tool" and ev.filter_value:
            items = store.filter_by_tool(item_type, ev.filter_value)
        elif ev.filter_field == "severity" and ev.filter_value and item_type == "warnings":
            items = store.filter_by_severity(ev.filter_value)
        elif ev.filter_field == "tag" and ev.filter_value and item_type == "decisions":
            items = store.filter_by_tag(ev.filter_value)
        elif ev.filter_field == "scope" and ev.filter_value and item_type == "rules":
            items = store.filter_by_scope(ev.filter_value)
        else:
            items = store.get_all(item_type)

        trace.append(
            f"structured: type={item_type} filter={ev.filter_field}={ev.filter_value} "
            f"-> {len(items)} items"
        )

        if not items:
            await ctx.store.set("trace", trace)
            return _stop(
                f"לא מצאתי פריטים מהסוג '{item_type}' שתואמים את השאלה.",
                trace=trace,
            )

        items_json = json.dumps(items[:30], ensure_ascii=False, indent=2)
        prompt = (
            f"השאלה של המשתמש: {ev.query}\n\n"
            f"להלן נתונים מובנים מתוך בסיס הידע (סוג: {item_type}, "
            f"מספר פריטים: {len(items)}):\n"
            f"{items_json}\n\n"
            f"נסח תשובה מסודרת בשפה של המשתמש (עברית אם השאלה בעברית). "
            f"אם זאת רשימה - השתמש ב-bullet points. "
            f"לכל פריט ציין את המקור (tool/file). היה ספציפי, אל תמציא."
        )
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        response = self.llm.chat(messages)
        answer = str(response.message.content)

        trace.append(f"synthesize_structured: generated {len(answer)} char answer")
        await ctx.store.set("trace", trace)

        return _stop(answer=answer, sources=[], trace=trace)

    @step
    async def rerank_nodes(
        self, ctx: Context, ev: NodesRetrievedEvent
    ) -> NodesRerankedEvent | RetryEvent | StopEvent:
        query_bundle = QueryBundle(query_str=ev.query)
        nodes = ev.nodes
        for pp in self.postprocessors:
            nodes = pp.postprocess_nodes(nodes, query_bundle=query_bundle)

        trace = await ctx.store.get("trace")

        if not nodes:
            trace.append("rerank: 0 nodes after postprocess")
            await ctx.store.set("trace", trace)
            return _stop("התוצאות לא רלוונטיות מספיק.", trace=trace)

        top_score = nodes[0].score or 0.0
        attempts = await ctx.store.get("retry_attempts")

        if top_score < MIN_CONFIDENCE_SCORE:
            if attempts < MAX_RETRIES:
                attempts += 1
                await ctx.store.set("retry_attempts", attempts)
                trace.append(
                    f"rerank: low confidence ({top_score:.4f}) - retry #{attempts}"
                )
                await ctx.store.set("trace", trace)
                return RetryEvent(
                    query=ev.query,
                    reason=f"top_score={top_score:.4f}",
                    attempt=attempts,
                )
            else:
                trace.append(
                    f"rerank: low confidence ({top_score:.4f}) - max retries reached"
                )
                await ctx.store.set("trace", trace)
                original = await ctx.store.get("original_query")
                return _stop(
                    f"לא מצאתי במסמכים תשובה מספקת לשאלה: \"{original}\".",
                    sources=nodes[:3],
                    trace=trace,
                )

        trace.append(f"rerank: top score {top_score:.4f} - ok")
        await ctx.store.set("trace", trace)
        return NodesRerankedEvent(query=ev.query, nodes=nodes, top_score=top_score)

    @step
    async def reformulate_query(
        self, ctx: Context, ev: RetryEvent
    ) -> QueryValidatedEvent:
        prompt = (
            "אתה עוזר ניסוח שאלות לחיפוש בתיעוד טכני. "
            "המשתמש שאל שאלה שלא הניבה תוצאות טובות. "
            "נסח אותה מחדש כך שתהיה ספציפית יותר ותכלול מילות מפתח טכניות. "
            "אם השאלה בעברית - שמור על עברית. תן רק את השאלה המנוסחת מחדש, בלי הסברים.\n"
            f"\nשאלה מקורית: {ev.query}\n"
            "שאלה משופרת:"
        )
        messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
        response = await self.llm.achat(messages)
        new_query = str(response.message.content).strip().strip('"').strip("'")

        trace = await ctx.store.get("trace")
        trace.append(f"reformulate: {ev.query!r} -> {new_query!r}")
        await ctx.store.set("trace", trace)

        return QueryValidatedEvent(query=new_query)

    @step
    async def synthesize(
        self, ctx: Context, ev: NodesRerankedEvent
    ) -> StopEvent:
        synthesizer = build_synthesizer(self.llm)
        response = await synthesizer.asynthesize(query=ev.query, nodes=ev.nodes)

        trace = await ctx.store.get("trace")
        trace.append(f"synthesize: generated {len(str(response.response))} char answer")

        return _stop(
            answer=str(response.response),
            sources=response.source_nodes,
            trace=trace,
        )
