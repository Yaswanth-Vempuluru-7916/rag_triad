from ragscope.llm import LLM
from ragscope.schemas import RetrievedChunk

PROMPT = """Answer the question using ONLY the context below.
If the context does not contain the answer, reply exactly: I don't know.

Context:
{context}

Question: {question}
Answer:"""


def build_prompt(question: str, chunks: list[RetrievedChunk], template: str = PROMPT) -> str:
    context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(chunks, 1))
    return template.format(context=context, question=question)


def answer(llm: LLM, question: str, chunks: list[RetrievedChunk], template: str = PROMPT) -> str:
    if not question.strip():
        raise ValueError("question must not be empty")
    if not chunks:
        return "I don't know."  # nothing retrieved, so no point paying for an LLM call
    return llm.complete(build_prompt(question, chunks, template)).strip()
