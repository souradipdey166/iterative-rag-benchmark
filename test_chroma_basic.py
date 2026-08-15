import chromadb
from rag.embeddings import TfidfEmbedder

embedder = TfidfEmbedder()

class _Wrapped:
    def __call__(self, input):
        return embedder(list(input))

client = chromadb.EphemeralClient()
collection = client.create_collection("test", embedding_function=_Wrapped())

collection.add(
    ids=["a", "b", "c"],
    documents=[
        "The Eiffel Tower is in Paris.",
        "The capital of Japan is Tokyo.",
        "Bananas are a good source of potassium.",
    ],
)

results = collection.query(query_texts=["Where is the Eiffel Tower?"], n_results=2)
print(results)