from ragscope.schemas import Document
from ragscope.chunking import chunk

doc = Document("1", "Niklaus", "I am Klaus Mikaelson")
result = chunk(doc, size=5, overlap=2)
print([c.text for c in result])
