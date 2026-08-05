import sys

import truststore
truststore.inject_into_ssl()

from dotenv import load_dotenv

load_dotenv()

import gradio as gr

from src.chunker import chunk_documents
from src.index import build_persistent_index
from src.llm import build_llm
from src.loader import load_all_documents
from src.postprocessor import build_postprocessors
from src.store import ItemStore
from src.workflow import RAGWorkflow

sys.stdout.reconfigure(encoding="utf-8")

print("Initializing RAG pipeline (one-time)...")
DOCS = load_all_documents()
NODES = chunk_documents(DOCS)
INDEX = build_persistent_index(NODES)
POSTPROCESSORS = build_postprocessors(top_n=5)
LLM = build_llm()
STORE = ItemStore()
WORKFLOW = RAGWorkflow(
    index=INDEX,
    llm=LLM,
    postprocessors=POSTPROCESSORS,
    store=STORE,
    top_k=10,
    timeout=120,
)
print(f"Ready. {len(NODES)} chunks + {STORE.counts()} structured items.\n")


async def answer(question: str) -> tuple[str, str, str]:
    if not question:
        return "אנא הזן שאלה.", "", ""

    result = await WORKFLOW.run(query=question)

    answer_md = result["answer"]

    seen = set()
    source_lines = []
    for s in result.get("sources", []):
        m = s.node.metadata
        key = f"{m['tool']}/{m['file_name']}"
        if key in seen:
            continue
        seen.add(key)
        source_lines.append(
            f"- **{m['tool']}** / `{m['file_name']}` &nbsp; "
            f"*(relevance: {s.score:.4f})*"
        )
    sources_md = "\n".join(source_lines) if source_lines else "_(no sources)_"

    trace_lines = result.get("trace", [])
    trace_md = "\n".join(f"{i+1}. {t}" for i, t in enumerate(trace_lines))

    return answer_md, sources_md, trace_md


with gr.Blocks(title="RAG על תיעוד AI Coding Tools") as demo:
    gr.Markdown("# 🤖 שאל את התיעוד")
    gr.Markdown(
        f"מערכת RAG **Event-Driven** על **{len(NODES)} chunks** מתוך "
        f"**{len(DOCS)} קבצי md** של Claude Code ו-GitHub Copilot."
    )

    with gr.Row():
        with gr.Column(scale=4):
            question_box = gr.Textbox(
                label="שאלה",
                placeholder="לדוגמה: מה הצבע העיקרי של הממשק?",
                lines=2,
            )
        with gr.Column(scale=1):
            submit_btn = gr.Button("שאל", variant="primary", size="lg")

    gr.Examples(
        examples=[
            "איך מתקינים את הפרויקט?",
            "מה הצבע העיקרי של הממשק?",
            "האם נעשה שימוש ב-Redux או בState אחר לניהול הסל?",
            "מה ההחלטה לגבי routing?",
            "אילו קומפוננטות יצרנו עם Tailwind?",
            "מה המתכון של עוגיות?",
        ],
        inputs=question_box,
        label="שאלות לדוגמה",
    )

    gr.Markdown("### 💬 תשובה")
    answer_box = gr.Markdown()

    gr.Markdown("### 📚 מקורות")
    sources_box = gr.Markdown()

    with gr.Accordion("🔍 Workflow trace (התקדמות הזרימה)", open=False):
        trace_box = gr.Markdown()

    submit_btn.click(
        fn=answer, inputs=question_box,
        outputs=[answer_box, sources_box, trace_box],
    )
    question_box.submit(
        fn=answer, inputs=question_box,
        outputs=[answer_box, sources_box, trace_box],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)
