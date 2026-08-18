from rag.metrics import exact_match, normalize_answer

print(exact_match("1889.", "1889"))       # expect 1
print(exact_match("Yes", "yes"))          # expect 1
print(exact_match("The Chief of Protocol", "Chief of Protocol"))  # expect 1 (article stripped)
print(exact_match("insufficient information", "yes"))  # expect 0

from rag.metrics import f1_score

print(f1_score("Chief", "Chief of Protocol"))
print(f1_score("Chief of Protocol", "Chief of Protocol"))  # expect 1.0, perfect match
print(f1_score("banana", "Chief of Protocol"))  # expect 0.0, no overlap

from rag.metrics import precision_recall_at_k

print(precision_recall_at_k(["a", "b", "c"], ["b", "x"], k=3))