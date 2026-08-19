## Manual failure log

### Example 1
Q: Were Scott Derrickson and Ed Wood of the same nationality?
Gold: yes
Model: insufficient information
Sources retrieved: Ed Wood, Woodson Arkansas, Ed Wood (film), Conrad Brooks, Ed Wood (film)
Diagnosis: Retrieval miss -- all 5 retrieved chunks relate to Ed Wood; zero
chunks about Scott Derrickson were retrieved, despite the question requiring
both. Single-shot retrieval with top_k=5 wasn't sufficient here. This is
exactly the failure mode iterative retrieval (Day 5) is meant to address.

### Example 2
Q: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Gold: Chief of Protocol
Supporting facts (per HotpotQA): Kiss and Tell (1945 film), Shirley Temple
Model: insufficient information
Sources retrieved: Kiss and Tell (1945 film), A Kiss for Corliss, Janet Waldo,
Meet Corliss Archer (TV series), A Kiss for Corliss
Diagnosis: Entity disambiguation failure. Two different actresses played
Corliss Archer in different media (Janet Waldo on radio, Shirley Temple in
the 1945 film specifically). Retrieval surfaced the radio actress instead
of the film actress the question actually asks about; "Shirley Temple" was
never retrieved at all despite being one of the 10 provided paragraphs.

### Example 1 follow-up: iterative retrieval recovers the miss
Same question as Example 1 (Scott Derrickson / Ed Wood).
Single-shot: insufficient information (missed Scott Derrickson entirely)
Iterative (hops=2): Yes -- correct. Hop 2 retrieved 'Scott Derrickson' after
the model recognized the gap and issued a follow-up search.
This is a direct example of iterative retrieval fixing a single-shot
retrieval miss -- the core hypothesis behind this project's ablation.

### Example 2 follow-up: iterative retrieval does NOT recover this failure
Same question as Example 2 (Corliss Archer / government position).
Single-shot: insufficient information (wrong entity retrieved -- Janet Waldo
instead of Shirley Temple)
Iterative (hops=1): insufficient information -- IDENTICAL failure, loop
stopped after just 1 hop.
Diagnosis: the hop-decision step failed to recognize a problem. Having
retrieved *an* answer to "who played Corliss Archer," the model treated
that as sufficient, without realizing the retrieved entity was wrong for
the specific film version the question asks about. This suggests iterative
retrieval fixes MISSING-entity failures (Example 1) but not WRONG-entity /
disambiguation failures (Example 2) -- the hop-decision step isn't
sophisticated enough to detect that the entity it found doesn't actually
match the question's constraints.

### Metric investigation: hit_rate vs recall
hit_rate stayed at 1.000 for both strategies across 10 questions --
misleadingly perfect. Adding retrieval_recall revealed the real picture:
0.700 for both. hit_rate only checks "found at least one gold source,"
which top_k=5 on a 10-paragraph corpus makes too easy to fail. recall
(fraction of ALL gold sources found) is the more honest retrieval metric
going forward.

Also noted: retrieval_recall is identical between single-shot and
iterative (0.700 both), yet EM differs substantially (0.4 vs 0.7 in this
run). This suggests iterative's improvement isn't coming from better
retrieval recall -- it's coming from something else, possibly giving the
model more reasoning attempts/context across hops even when the same
sources are ultimately retrieved. Worth investigating further if time
allows.

## Failure taxonomy (first 10 of 24 iterative failures, N=50 run)

Important finding: 5/10 "failures" were false negatives from strict exact-match
scoring -- the model's answer was substantively correct but phrased
differently from HotpotQA's gold string (e.g. "3,677" vs "3,677 seated",
"1969-1974" vs "1969 until 1974"). This suggests the true EM of 0.520 likely
understates real answer accuracy; F1 (0.658) partially captures this via
partial credit, but a normalized/fuzzy match would give a more accurate
picture. Noting this as a known limitation of the evaluation, not the agent.

Real failures broke down into three categories:
- Retrieval miss (info never found, even across hops) -- e.g. Guns N' Roses
  question, retrieval_recall=0.0
- Wrong entity retrieved entirely -- e.g. Corliss Archer question, retrieval
  surfaced Janet Waldo (the radio actress) instead of Shirley Temple (the
  film actress the question specifically asks about); Shirley Temple was
  never retrieved
- Reasoning error despite perfect retrieval (recall=1.0) -- e.g. Random House
  Tower yes/no question, most concerning category since more retrieval
  wouldn't fix this

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
rather than finding better sources.

**Note on evaluation strictness**: manual review of 24 "failed" answers
found 42% were false negatives -- substantively correct answers penalized
by strict exact-match string comparison (e.g. "3,677" marked wrong against
gold "3,677 seated"). A looser contains-based match closes much of this
gap; both metrics are reported for transparency. The single-shot vs.
iterative gap is consistent across both metrics, suggesting the core
finding is not an artifact of scoring strictness.