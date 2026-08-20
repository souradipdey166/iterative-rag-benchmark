# Multi-hop RAG Agent

A retrieval-augmented QA agent built to answer multi-hop questions -- ones
that can't be answered from a single lookup and require finding and
connecting information across multiple documents. Evaluated on a subset of
[HotpotQA](https://hotpotqa.github.io/), a public multi-hop QA benchmark, so
the results below are checkable by anyone, not self-graded.

## Results

| Metric | Single-shot | Iterative |
|---|---|---|
| Strict Exact Match | 0.380 | 0.520 |
| Loose Match (contains) | 0.520 | 0.680 |
| F1 | 0.488 | 0.658 |
| Retrieval hit-rate@5 | 0.980 | 0.980 |
| Retrieval recall@5 | 0.740 | 0.740 |
| Avg. hops | 1.00 | 1.48 |
| Avg. time/question | 2.48s | 14.69s |

N=50 questions, HotpotQA distractor validation split.

**Ablation finding**: iterative retrieval improves accuracy substantially
(+14 points strict EM, +16 points loose match) at roughly 6x the latency
cost per question. The improvement is NOT driven by better retrieval --
recall@5 is identical (0.740) between strategies -- suggesting the gain
comes from the model getting more reasoning attempts/context across hops
rather than finding better sources. Full failure taxonomy and the metric
investigation behind these numbers are in `results/manual_notes.md`.

**Note on evaluation strictness**: manual review of 24 "failed" answers found
42% were false negatives -- substantively correct answers penalized by
strict exact-match string comparison (e.g. "3,677" marked wrong against gold
"3,677 seated"). A looser contains-based match (`contains_match` in
`rag/metrics.py`) closes much of this gap; both metrics are reported above
for transparency. The single-shot vs. iterative gap holds across both
metrics, so the core finding isn't an artifact of scoring strictness.

## The question this project investigates

Standard RAG retrieves once and answers. On multi-hop questions, one
retrieval pass often doesn't surface everything needed -- so this project
builds and compares two strategies on the exact same questions:

1. **Single-shot** -- retrieve once, answer from those chunks.
2. **Iterative** -- retrieve, ask the model if it has enough information or
   needs to search again, loop up to a small hop limit, then answer.

The goal is a real, measured answer to: *does the extra retrieval hop
actually help, and what does it cost?*

## How it works

1. **Corpus**: for each HotpotQA question, the ~10 paragraphs it provides
   (2 that actually answer the question + distractors) are split into
   overlapping text chunks (`rag/chunking.py`), keeping track of which chunk
   came from which source paragraph.
2. **Retrieval**: chunks are embedded and indexed in ChromaDB
   (`rag/retriever.py`); a question retrieves its top-k most relevant
   chunks.
3. **Answering**: the LLM answers using only retrieved chunks
   (`rag/agent.py`) -- single-shot retrieves once; iterative can retrieve
   again if it decides it doesn't have enough information yet, up to a hop
   limit.
4. **Evaluation**: answers are scored with exact match / loose match / F1
   against HotpotQA's gold answers; retrieval quality is scored separately
   with precision/recall/hit-rate@k against the question's actual
   supporting-fact paragraphs (`rag/metrics.py`).

### Why the "distractor setting" (stated honestly, not hidden)

This evaluates retrieval among the ~10 paragraphs HotpotQA provides per
question, not open-domain retrieval over all of Wikipedia. That's the
standard, legitimate "distractor setting" HotpotQA itself defines -- but
it's an easier problem than full open-domain retrieval. A natural extension
is a single shared index across all questions' paragraphs (see
Limitations).

## Setup

```bash
git clone <this repo>
cd multihop-rag-agent
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
cp .env.example .env         # add your GROQ_API_KEY
```

## Run

Run the full ablation (both strategies, N questions -- currently 50 in
`eval_run.py`):

```bash
python eval_run.py
```

Writes `results/eval_results.jsonl` (per-question results, both strategies,
saved incrementally so a crash mid-run doesn't lose progress) and prints an
aggregate comparison table.

Re-score existing results with the looser `contains_match` metric, no API
calls needed:

```bash
python rescale_metrics.py
```

Inspect specific wrong answers by hand:

```bash
python analyze_failures.py
```

## Repo layout

```
rag/
  chunking.py          # splits paragraphs into overlapping chunks, tracks source
  metrics.py            # exact match, F1, contains_match, precision/recall@k
  retriever.py           # ChromaDB-backed retrieval
  agent.py                # single-shot and iterative answering strategies
  llm.py                   # LLM call wrapper (currently Groq) -- swap providers here
eval_run.py                  # main script: runs both strategies on N questions
analyze_failures.py           # inspects wrong answers by hand
rescale_metrics.py             # re-scores saved results with contains_match
results/
  eval_results.jsonl            # generated -- raw per-question results
  manual_notes.md                # full failure taxonomy and metric investigation
```

## Design choices worth explaining (things I'd expect to be asked about)

- **No LangChain/LangGraph for the core pipeline.** The retrieval and
  agent-loop logic is plain Python so every mechanic (chunking, retrieval,
  the "do I need another hop" decision) is fully understood and
  explainable, not hidden behind a framework abstraction.
- **LLM provider is swappable.** `rag/llm.py` isolates the model call
  behind one function (`call_llm`) -- everything else in the project is
  provider-agnostic. Currently using Groq's free tier for development; can
  swap to Claude by editing one file.
- **Embeddings via ChromaDB's built-in default embedding function**, not a
  separately-downloaded model -- runs locally, no extra dependency, no
  ongoing API cost for retrieval.
- **Two answer-quality metrics, not one.** Strict exact match understates
  real accuracy (see Results note above); `contains_match` was added after
  manual failure review revealed why, and both are reported for
  transparency rather than picking whichever number looks better.
- **Incremental result saving with retry-on-rate-limit.** `eval_run.py`
  writes each question's result to disk immediately and retries with
  backoff on API rate limits, so a long run surviving a transient failure
  doesn't lose completed work.

## Limitations (stated plainly, not hidden)

- Retrieval is scoped to each question's own distractor set, not a shared
  open-domain index -- an easier retrieval problem than production RAG.
- The iterative strategy's hop-decision is a single LLM call per hop, which
  can be inconsistent -- e.g. it failed to recognize a wrong-entity
  retrieval as insufficient in one documented case (see
  `results/manual_notes.md`, Example 2). A more robust approach (e.g. a
  fine-tuned classifier for the stop/continue decision) is a natural
  extension.
- Evaluated on N=50 questions, not the full HotpotQA validation set, due to
  free-tier API rate limits.
- Using Groq's free-tier `openai/gpt-oss-20b` for development rather than a
  larger/paid model -- answer quality may differ with a stronger model.

## What I'd do with more time

- Shared vector index across all questions (real open-domain retrieval)
- A second ablation on retrieval breadth (top-k=3 vs top-k=10)
- A more robust hop-decision step that can detect wrong-entity retrieval,
  not just missing-entity retrieval
- Swap the free dev model for Claude and report whether/how much answer
  quality changes
- Scale the eval run past N=50 once on a paid tier.