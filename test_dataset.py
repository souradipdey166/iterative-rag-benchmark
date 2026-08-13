from datasets import load_dataset

# Load just 3 examples first -- fast, and enough to sanity-check the format
ds = load_dataset("hotpot_qa", "distractor", split="validation[:3]")

print("Number of examples loaded:", len(ds))
print()

example = ds[0]
print("Question:", example["question"])
print("Answer:", example["answer"])
print("Number of context paragraphs:", len(example["context"]["title"]))
print("Context titles:", example["context"]["title"])