"""
Vector retriever backed by ChromaDB, using its default local embedding
model (downloads once, cached after that -- no ongoing internet needed).
"""

import chromadb


class Retriever:
    def __init__(self, collection_name: str = "corpus"):
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(collection_name)

    def index(self, chunks: list[dict]):
        """chunks: list of {"chunk_id", "source_id", "title", "text"}"""
        if not chunks:
            return
        self.collection.add(
            ids=[c["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[{"source_id": c["source_id"], "title": c.get("title", "")} for c in chunks],
        )

    def query(self, question: str, top_k: int) -> list[dict]:
        """Returns ranked list of {"chunk_id", "source_id", "title", "text", "distance"}."""
        result = self.collection.query(query_texts=[question], n_results=top_k)
        out = []
        for cid, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            out.append({
                "chunk_id": cid,
                "source_id": meta["source_id"],
                "title": meta["title"],
                "text": doc,
                "distance": dist,
            })
        return out