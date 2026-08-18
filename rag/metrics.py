"""
Standard evaluation metrics -- these follow the same normalization used by
SQuAD/HotpotQA's own official eval scripts, so your numbers are comparable
to published results, not an invented scoring scheme.
"""

import re
import string


def normalize_answer(s: str) -> str:
    """Lowercase, strip punctuation/articles/extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def exact_match(prediction: str, gold: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(gold))


from collections import Counter


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def precision_recall_at_k(retrieved_ids: list[str], gold_ids: list[str], k: int) -> dict:
    """
    retrieved_ids: source_ids returned by the retriever, in ranked order.
    gold_ids: the source_ids HotpotQA says actually contain the answer
              (from supporting_facts).
    """
    top_k = set(retrieved_ids[:k])
    gold_set = set(gold_ids)
    if not gold_set:
        return {"precision": 0.0, "recall": 0.0, "hit": 0}

    hits = top_k & gold_set
    precision = len(hits) / k if k > 0 else 0.0
    recall = len(hits) / len(gold_set)
    hit = int(len(hits) > 0)
    return {"precision": precision, "recall": recall, "hit": hit}