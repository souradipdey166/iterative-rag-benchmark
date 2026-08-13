Multi-hop RAG Agent

A retrieval-augmented QA agent built to answer multi-hop questions -- ones that can't be answered from a single lookup and require finding and connecting information across multiple documents. Evaluated on a subset of HotpotQA, a public multi-hop QA benchmark, so the results below are checkable by anyone, not self-graded.

Results

Fill in once the eval run (Day 7) is complete. Don't round up -- the whole point of this project is that the numbers are real and reproducible.

Metric	Single-shot retrieval	Iterative retrieval
Exact Match	TBD	TBD
F1	TBD	TBD
Retrieval hit-rate@k	TBD	TBD
Avg. hops	TBD	TBD
Avg. time/question	TBD	TBD

Ablation finding: one sentence once you have the numbers -- e.g. "iterative retrieval improved EM by X points at a cost of Y% more time/tokens."

Failure analysis and raw per-question results are in results/.

The question this project investigates

Standard RAG retrieves once and answers. On multi-hop questions, one retrieval pass often doesn't surface everything needed -- so this project builds and compares two strategies on the exact same questions:

Single-shot -- retrieve once, answer from those chunks.
Iterative -- retrieve, ask the model if it has enough information or needs to search again, loop up to a small hop limit, then answer.

The goal is a real, measured answer to: does the extra retrieval hop actually help, and what does it cost?

How it works
Corpus: for each HotpotQA question, the ~10 paragraphs it provides (2 that actually answer the question + distractors) are split into overlapping text chunks, keeping track of which chunk came from which source paragraph.
Retrieval: chunks are embedded and indexed in ChromaDB; a question retrieves its top-k most relevant chunks.
Answering: the LLM answers using only retrieved chunks -- single-shot retrieves once, iterative can retrieve again if it decides it doesn't have enough information yet.
Evaluation: answers are scored with exact match / F1 against HotpotQA's gold answers; retrieval quality is scored with precision/recall/ hit-rate@k against the question's actual supporting-fact paragraphs.
Why the "distractor setting" (stated honestly, not hidden)

This evaluates retrieval among the ~10 paragraphs HotpotQA provides per question, not open-domain retrieval over all of Wikipedia. That's the standard, legitimate "distractor setting" HotpotQA itself defines -- but it's an easier problem than full open-domain retrieval. A natural extension is a single shared index across all questions' paragraphs (see Limitations).

Setup
bash
git clone <this repo>
cd multihop-rag-agent
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
cp .env.example .env         # add your API key(s)

Verify everything is wired up correctly before building further:

bash
python check_setup.py
Run
bash
python -m eval.run_eval --n 75

This runs both strategies (single-shot and iterative) on the same set of questions and writes:

results/eval_results.jsonl -- per-question results, both strategies
Aggregate comparison table printed to console
Repo layout
rag/
  chunking.py     # splits paragraphs into overlapping chunks, tracks source
  metrics.py      # exact match, F1, precision/recall@k
  retriever.py    # ChromaDB-backed retrieval
  agent.py        # single-shot and iterative answering strategies
  llm.py          # LLM call wrapper -- swap providers by editing only this file
eval/
  build_corpus.py # loads HotpotQA, builds per-question corpus
  run_eval.py      # runs the ablation, computes metrics
results/
  eval_results.jsonl   # generated -- raw per-question results
docs/
  PLAN.md          # day-by-day build plan
Design choices worth explaining (things I'd expect to be asked about)
No LangChain/LangGraph for the core pipeline. The retrieval and agent-loop logic is plain Python so every mechanic (chunking, retrieval, the "do I need another hop" decision) is fully understood and explainable, not hidden behind a framework abstraction.
LLM provider is swappable. rag/llm.py isolates the model call behind one function (call_llm) -- everything else in the project is provider- agnostic. Currently using Groq's free tier for development; can swap to Claude by editing one file.
Embeddings run locally via sentence-transformers, not through a paid API -- keeps iteration cheap while developing.
Limitations (stated plainly, not hidden)
Retrieval is scoped to each question's own distractor set, not a shared open-domain index -- an easier retrieval problem than production RAG.
The "iterative" strategy's hop-decision is a single LLM call per hop, which can be inconsistent -- a more robust approach (e.g. a fine-tuned classifier for the stop/continue decision) is a natural extension.
Evaluated on a subset (see Results table for N), not the full HotpotQA validation set, due to time/cost constraints.
What I'd do with more time
Shared vector index across all questions (real open-domain retrieval)
A second ablation on retrieval breadth (top-k=3 vs top-k=10)
Swap the free dev model for Claude and report whether/how much answer quality changes