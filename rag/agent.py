"""Single-shot RAG: retrieve once, then answer using only what was retrieved."""

from rag.llm import call_llm
from rag.retriever import Retriever


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks):
        lines.append(f"[{i+1}] ({c.get('title', 'untitled')}) {c['text']}")
    return "\n".join(lines)





ANSWER_PROMPT = """Answer the question using ONLY the numbered context passages below. \
If the passages don't contain enough information, say "insufficient information".
Respond with just the short answer, no explanation.

Context:
{context}

Question: {question}

Answer:"""


def answer_single_shot(question: str, retriever: Retriever, top_k: int) -> dict:
    chunks = retriever.query(question, top_k)
    context = _format_context(chunks)
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    answer = call_llm(prompt)
    return {
        "answer": answer,
        "retrieved_source_ids": [c["source_id"] for c in chunks],
    }