import json
from rag.metrics import exact_match, contains_match

rows = [json.loads(l) for l in open("results/eval_results.jsonl")]

for strategy in ["single_shot", "iterative"]:
    n = len(rows)
    strict_em = sum(r[strategy]["em"] for r in rows) / n
    loose_em = sum(
        contains_match(r[strategy]["answer"], r["gold_answer"]) for r in rows
    ) / n
    print(f"{strategy:15} strict EM: {strict_em:.3f}   loose (contains) match: {loose_em:.3f}")