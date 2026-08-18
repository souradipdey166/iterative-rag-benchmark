from datasets import load_dataset
from rag.chunking import chunk_paragraphs
from rag.retriever import Retriever
from rag.agent import answer_single_shot, answer_iterative

ds = load_dataset("hotpot_qa", "distractor", split="validation[1:2]")
example = ds[0]

print("Question:", example["question"])
print("Gold answer:", example["answer"])

print()

paragraphs = []
for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
    paragraphs.append({"id": title, "title": title, "text": " ".join(sentences)})

chunks = chunk_paragraphs(paragraphs, chunk_size=300, overlap=50)
retriever = Retriever()
retriever.index(chunks)

print("=== Single-shot ===")
single = answer_single_shot(example["question"], retriever, top_k=5)
print("Answer:", single["answer"])
print("Sources:", single["retrieved_source_ids"])
print()

print("=== Iterative ===")
iterative = answer_iterative(example["question"], retriever, top_k=5, max_hops=3)
print("Answer:", iterative["answer"])
print("Sources:", iterative["retrieved_source_ids"])
print("Hops used:", iterative["hops"])
