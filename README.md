# Smart Event-Driven RAG Core

Personal retrieval-augmented generation engine powered by LlamaIndex workflows and an interactive Gradio interface.

---

## Overview

An advanced RAG architecture driven by asynchronous event flows. At its center, a dynamic smart router intelligently splits processing pathways between a structured channel (for precise local data matching) and a semantic vector channel (for context-driven retrieval). This hybrid design ensures maximum accuracy, eliminates hallucinations, and anchors every generated response strictly in verified data.

---

## Project Core Objectives

The system processes free-text queries regarding system specs and complex tasks, automatically detecting intent and routing requests down the optimal pipeline:

| Query Type | Description | Processing Channel |
|---|---|---|
| Structured Queries | Formal rules, strict constraints, closed lists | Exact local data lookup |
| Semantic Content | General explanations, broad context | Vector database semantic search |

---

## Tech Stack

| Layer | Tool / Technology | Purpose |
|---|---|---|
| Framework | LlamaIndex (+ Workflows) | Native RAG support with event-driven execution |
| Embeddings | cohere embed-multilingual-v3.0 | 1024-dimension cross-lingual vectors with native Hebrew support |
| Vector Store | ChromaDB (Local Persistence) | Lightweight, zero-rate-limit local storage |
| Reranker | cohere rerank-multilingual-v3.0 | Drastically boosts top-K result precision |
| LLM | cohere command-r-plus-08-2024 | Superior Hebrew handling and robust RAG optimization |
| UI | Gradio | Rapid interactive chat interface under 100 lines |

---

## Workflow Architecture

The system orchestrates steps through asynchronous event triggers following LlamaIndex's event-driven paradigm:

User Query --> Input Validation --> Smart Router 
    --> Structured Retrieval (JSON)
    --> Semantic Vector Search (ChromaDB)
          │
          ▼
Answer Synthesis --> UI Output

---
Developed by Talya Toledano 💛
