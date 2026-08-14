from rag.chunking import chunk_text

text = "a" * 500
chunks = chunk_text(text, chunk_size=200, overlap=40)

print("Number of chunks:", len(chunks))
for i, c in enumerate(chunks):
    print(f"chunk {i}: length={len(c)}")


from rag.chunking import chunk_paragraphs

paragraphs = [{"id": "p1", "title": "Test", "text": "short text"}]
result = chunk_paragraphs(paragraphs, chunk_size=200, overlap=40)

print(result)