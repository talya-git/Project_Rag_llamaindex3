from llama_index.core import PromptTemplate, VectorStoreIndex, get_response_synthesizer
from llama_index.core.base.response.schema import Response
from llama_index.core.llms import LLM
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.response_synthesizers import BaseSynthesizer

from .retriever import retrieve

QA_PROMPT = PromptTemplate(
    "You are answering questions about documentation written by AI coding tools "
    "(Claude Code, GitHub Copilot). The documentation can be in Hebrew or English.\n"
    "\n"
    "Context information from the documentation is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "\n"
    "Rules:\n"
    "1. Answer based ONLY on the context above. Do not use prior knowledge.\n"
    "2. Respond in the SAME LANGUAGE as the question.\n"
    "3. If the answer is not in the context, say so clearly (in the question's language).\n"
    "4. Be concise but specific - include file/component names when relevant.\n"
    "\n"
    "Question: {query_str}\n"
    "Answer: "
)


def build_synthesizer(llm: LLM) -> BaseSynthesizer:
    return get_response_synthesizer(
        llm=llm,
        response_mode="compact",
        text_qa_template=QA_PROMPT,
    )


def answer_question(
    index: VectorStoreIndex,
    query: str,
    llm: LLM,
    postprocessors: list[BaseNodePostprocessor] | None = None,
    top_k: int = 10,
) -> Response:
    nodes = retrieve(index, query, top_k=top_k, postprocessors=postprocessors)
    synthesizer = build_synthesizer(llm)
    return synthesizer.synthesize(query=query, nodes=nodes)
