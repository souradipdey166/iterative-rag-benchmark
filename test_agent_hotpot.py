from datasets import load_dataset
from rag.chunking import chunk_paragraphs
from rag.retriever import Retriever
from rag.agent import answer_single_shot

ds = load_dataset("hotpot_qa", "distractor", split="validation[1:2]")
example = ds[0]

print("Question:", example["question"])
print("Gold answer:", example["answer"])
print()

print("Supporting fact titles:", example["supporting_facts"]["title"])
print("Supporting fact sent_ids:", example["supporting_facts"]["sent_id"])
print()
print("All context paragraph titles:", example["context"]["title"])

paragraphs = []
for title, sentences in zip(example["context"]["title"], example["context"]["sentences"]):
    paragraphs.append({"id": title, "title": title, "text": " ".join(sentences)})

chunks = chunk_paragraphs(paragraphs, chunk_size=300, overlap=50)
retriever = Retriever()
retriever.index(chunks)

for c in chunks:
    if c["source_id"] == "Janet Waldo":
        print(f"--- chunk {c['chunk_id']} ---")
        print(c["text"])
        print()

result = answer_single_shot(example["question"], retriever, top_k=5)
print("Model answer:", result["answer"])
print("Sources used:", result["retrieved_source_ids"])