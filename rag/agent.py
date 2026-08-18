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



import json
import re

NEED_MORE_PROMPT = """You are trying to answer a multi-hop question. Here is what you've \
retrieved so far and the question. Decide if you have enough information to answer, or if \
you need more information on a specific missing entity or fact.

Do NOT call any tools or functions. Do NOT search the web. Only respond with the JSON object \
described below, as plain text.

Context so far:
{context}

Question: {question}

Respond with ONLY this JSON object, no other text:
{{"has_enough_info": true or false, "next_search_query": "a short query describing what specific information you need, else empty string"}}"""

def _parse_decision(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"has_enough_info": True, "next_search_query": ""}
    try:
        parsed = json.loads(match.group(0))
        return {
            "has_enough_info": bool(parsed.get("has_enough_info", True)),
            "next_search_query": str(parsed.get("next_search_query", "")),
        }
    except (json.JSONDecodeError, TypeError):
        return {"has_enough_info": True, "next_search_query": ""}



def answer_iterative(question: str, retriever: Retriever, top_k: int, max_hops: int) -> dict:
    all_chunks = []
    current_query = question
    hops_used = 0

    for hop in range(max_hops):
        hops_used = hop + 1

        new_chunks = retriever.query(current_query, top_k)
        seen = {c["chunk_id"] for c in all_chunks}
        all_chunks.extend(c for c in new_chunks if c["chunk_id"] not in seen)

        context = _format_context(all_chunks)
        decision_raw = call_llm(NEED_MORE_PROMPT.format(context=context, question=question))
        decision = _parse_decision(decision_raw)

        if decision["has_enough_info"] or not decision["next_search_query"]:
            break
        current_query = decision["next_search_query"]

    context = _format_context(all_chunks)
    final_prompt = ANSWER_PROMPT.format(context=context, question=question)
    answer = call_llm(final_prompt)

    return {
        "answer": answer,
        "retrieved_source_ids": [c["source_id"] for c in all_chunks],
        "hops": hops_used,
    }