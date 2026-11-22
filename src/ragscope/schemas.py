from dataclasses import dataclass

@dataclass(frozen=True)
class Document:
    id : str
    title : str
    text : str

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    chunk_idx: int
    text: str