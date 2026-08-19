import json

rows = [json.loads(l) for l in open("results/eval_results.jsonl")]

wrong_iterative = [r for r in rows if r["iterative"]["em"] == 0]
print(f"{len(wrong_iterative)} / {len(rows)} wrong with iterative strategy\n")

for r in wrong_iterative[10:24]:
    print(f"Q: {r['question']}")
    print(f"Gold: {r['gold_answer']}")
    print(f"Model: {r['iterative']['answer']}")
    print(f"Hops: {r['iterative']['hops']}, retrieval_recall: {r['iterative']['retrieval_recall']}")
    print("-" * 60)