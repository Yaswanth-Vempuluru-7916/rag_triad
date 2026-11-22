from ragscope.schemas import Document, Chunk

def chunk(doc : Document, size : int, overlap : int) -> list[Chunk] :
    if size <= 0:
        raise ValueError("Chunk size must be greater than 0")
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be between 0 and chunk size")

    text = doc.text.strip()
    step = size - overlap
    chunks = []

    for i, start in enumerate(range(0,len(text), step)):
        end = min(start + size, len(text))
        chunk_text = text[start:end]
        chunks.append(Chunk(doc_id=doc.id, chunk_idx=i, text=chunk_text))
        if end >= len(text):
            break
    return chunks