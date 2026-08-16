from rag.chunking import chunk_paragraphs
from rag.retriever import Retriever
from rag.agent import answer_single_shot

paragraphs = [
    {"id": "p1", "title": "Eiffel Tower", "text": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was completed in 1889."},
    {"id": "p2", "title": "Tokyo", "text": "Tokyo is the capital of Japan. It is one of the most populous cities in the world."},
]

chunks = chunk_paragraphs(paragraphs, chunk_size=200, overlap=40)
retriever = Retriever()
retriever.index(chunks)

result = answer_single_shot("What year was the Eiffel Tower completed?", retriever, top_k=2)
print("Answer:", result["answer"])
print("Sources used:", result["retrieved_source_ids"])