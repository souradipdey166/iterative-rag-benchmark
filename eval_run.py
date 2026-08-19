import json
import time

from datasets import load_dataset

from rag.chunking import chunk_paragraphs
from rag.retriever import Retriever
from rag.agent import answer_single_shot, answer_iterative
from rag.metrics import exact_match, f1_score, precision_recall_at_k

N_QUESTIONS = 50
TOP_K = 5
MAX_HOPS = 3
CHUNK_SIZE = 300
OVERLAP = 50
RESULTS_PATH = "results/eval_results.jsonl"   # .jsonl now, one row per line

ds = load_dataset("hotpot_qa", "distractor", split=f"validation[:{N_QUESTIONS}]")

def call_with_retry(fn, *args, max_retries=5, **kwargs):
    """Retries on Groq rate-limit errors with backoff, instead of crashing the run."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 15 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exceeded max retries on rate limit")


rows = []
with open(RESULTS_PATH, "w") as out_f:
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
        single = call_with_retry(answer_single_shot, example["question"], retriever, TOP_K)
        single_time = time.time() - t0

        t0 = time.time()
        iterative = call_with_retry(answer_iterative, example["question"], retriever, TOP_K, MAX_HOPS)
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

        row = {
            "id": example["id"],
            "question": example["question"],
            "gold_answer": example["answer"],
            "single_shot": score(single, single_time),
            "iterative": score(iterative, iterative_time),
        }
        rows.append(row)
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()   # force it to disk immediately, don't wait for buffer

        time.sleep(2)   # a bit more breathing room given the TPM limit

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