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