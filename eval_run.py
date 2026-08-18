import json
import time

from datasets import load_dataset

from rag.chunking import chunk_paragraphs
from rag.retriever import Retriever
from rag.agent import answer_single_shot, answer_iterative
from rag.metrics import exact_match, f1_score, precision_recall_at_k

N_QUESTIONS = 10       # start small to prove it works, raise later
TOP_K = 5
MAX_HOPS = 3
CHUNK_SIZE = 300
OVERLAP = 50

ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{N_QUESTIONS}]")

rows = []

for i, example in enumerate(ds):
    print(f"[{i+1}/{N_QUESTIONS}] {example['question'][:60]}...")

    paragraphs = []
    for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
        paragraphs.append({"id": title, "title": title, "text": " ".join(sentences)})

    chunks = chunk_paragraphs(paragraphs, CHUNK_SIZE, OVERLAP)
    retriever = Retriever(collection_name=f"q_{i}")
    retriever.index(chunks)

    gold_ids = list(set(example["supporting_facts"]["title"]))

    t0 = time.time()
    single = answer_single_shot(example["question"], retriever, TOP_K)
    single_time = time.time() - t0

    t0 = time.time()
    iterative = answer_iterative(example["question"], retriever, TOP_K, MAX_HOPS)
    iterative_time = time.time() - t0

    def score(result, elapsed):
        rp = precision_recall_at_k(result["retrieved_source_ids"], gold_ids, TOP_K)
        return {
            "answer": result["answer"],
            "em": exact_match(result["answer"], example["answer"]),
            "f1": f1_score(result["answer"], example["answer"]),
            "retrieval_hit": rp["hit"],
            "retrieval_recall": rp["recall"],
            "hops": result.get("hops", 1),
            "time_sec": elapsed,
        }

    rows.append({
        "id": example["id"],
        "question": example["question"],
        "gold_answer": example["answer"],
        "single_shot": score(single, single_time),
        "iterative": score(iterative, iterative_time),
    })

with open("results/eval_results.json", "w") as f:
    json.dump(rows, f, indent=2)

def agg(strategy):
    n = len(rows)
    return {
        "em": sum(r[strategy]["em"] for r in rows) / n,
        "f1": sum(r[strategy]["f1"] for r in rows) / n,
        "retrieval_hit_rate": sum(r[strategy]["retrieval_hit"] for r in rows) / n,
        "retrieval_recall": sum(r[strategy]["retrieval_recall"] for r in rows) / n,
        "avg_hops": sum(r[strategy]["hops"] for r in rows) / n,
        "avg_time_sec": sum(r[strategy]["time_sec"] for r in rows) / n,
    }

print("\n=== Results (single-shot vs iterative) ===")
s, it = agg("single_shot"), agg("iterative")
print(f"{'metric':<20}{'single-shot':<15}{'iterative':<15}")
for key in s:
    print(f"{key:<20}{s[key]:<15.3f}{it[key]:<15.3f}")