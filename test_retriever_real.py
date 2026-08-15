from rag.chunking import chunk_paragraphs
from rag.retriever import Retriever

paragraphs = [
    {"id": "p1", "title": "Eiffel Tower", "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was completed in 1889."},
    {"id": "p2", "title": "Tokyo", "text": "Tokyo is the capital of Japan. It is one of the most populous cities in the world."},
    {"id": "p3", "title": "Bananas", "text": "Bananas are a good source of potassium and are grown in tropical climates."},
]

chunks = chunk_paragraphs(paragraphs, chunk_size=200, overlap=40)

retriever = Retriever()
retriever.index(chunks)

results = retriever.query("What year was the Eiffel Tower completed?", top_k=2)
for r in results:
    print(f"source={r['source_id']} distance={r['distance']:.3f}")
    print(f"  {r['text']}")