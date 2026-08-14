"""Simple character-based chunking with overlap."""

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if not text:
        return []

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks

def chunk_paragraphs(paragraphs: list[dict], chunk_size: int, overlap: int) -> list[dict]:
    """
    paragraphs: list of {"id": ..., "title": ..., "text": ...}
    Returns chunks that remember which paragraph they came from.
    """
    out = []
    for para in paragraphs:
        pieces = chunk_text(para["text"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            out.append({
                "chunk_id": f"{para['id']}::{i}",
                "source_id": para["id"],
                "title": para.get("title", ""),
                "text": piece,
            })
    return out