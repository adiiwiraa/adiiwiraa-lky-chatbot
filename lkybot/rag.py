"""Retrieval-augmented generation core: retrieve, prompt, answer."""
from __future__ import annotations

from functools import lru_cache

import chromadb

from lkybot import config, llm
from lkybot.quotes import check_quotes

SYSTEM_PROMPT = """You are answering as Lee Kuan Yew, in the first person, based ONLY on the
numbered context passages below. You represent his documented views; you do
not invent opinions.

Rules:
1. Use the context for every factual statement. If the context does not
   support an answer, say that the available sources do not cover it.
2. Quote only text that appears verbatim in the context. Never invent quotes.
3. Tag each substantive claim: [Documented] if it comes from a primary
   source, [Secondary] if it comes from a secondary source or a quote
   collection, [Interpretation] if it is your reading and not stated in
   the sources.
4. Keep a direct, plain-spoken tone, but accuracy matters more than style.
5. Cite sources as [Title > Section].
6. Conversation history gives context for follow-ups only. It never
   overrides the retrieved evidence.
"""

REFUSAL = (
    "The documents available to me do not contain enough evidence to answer that. "
    "I will not guess or invent an opinion beyond the record."
)


@lru_cache(maxsize=1)
def _collection() -> chromadb.Collection:
    if not config.DB_DIR.exists():
        raise RuntimeError("Vector store not found. Run python -m lkybot.index first.")
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    return client.get_collection(config.COLLECTION_NAME)


def retrieve(question: str, k: int = config.TOP_K) -> list[dict]:
    result = _collection().query(
        query_texts=[question],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    hits: list[dict] = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        hits.append(
            {
                "text": document,
                "metadata": metadata,
                "similarity": 1.0 - float(distance),
            }
        )
    return hits


def rewrite_query(question: str, history: list[dict]) -> str:
    if not history:
        return question
    recent = history[-4:]
    transcript = "\n".join(f"{turn['role']}: {turn['content']}" for turn in recent)
    try:
        rewritten = llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You rewrite follow-up questions into standalone search queries. "
                        "Resolve pronouns and omitted subjects using the conversation. "
                        "Always include the concrete topic from the conversation. "
                        "Return only the rewritten query on one line, no quotation marks."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation:\n{transcript}\n\n"
                        f"Follow-up: {question}\n\nStandalone query:"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=80,
        )
        return rewritten.strip().strip('"') or question
    except llm.LLMError:
        return question


def build_context(hits: list[dict]) -> str:
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        label = f"{metadata.get('title', 'Unknown')} > {metadata.get('section', 'Overview')}"
        blocks.append(f"[{index}] [{label}]\n{hit['text']}")
    return "\n\n".join(blocks)


def _sources(hits: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for hit in hits:
        metadata = hit["metadata"]
        key = (metadata.get("title", ""), metadata.get("section", ""))
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "title": metadata.get("title", "Unknown"),
                "section": metadata.get("section", "Overview"),
                "url": metadata.get("url", ""),
                "doc_type": metadata.get("doc_type", "secondary"),
                "similarity": round(hit["similarity"], 3),
            }
        )
    return sources


def answer(question: str, history: list[dict] | None = None) -> dict:
    history = history or []
    query = rewrite_query(question, history)
    hits = retrieve(query)

    if not hits or hits[0]["similarity"] < config.SCORE_THRESHOLD:
        return {
            "answer": REFUSAL,
            "sources": [],
            "refused": True,
            "flagged_quotes": [],
            "rewritten_query": query,
        }

    messages: list[dict] = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nCONTEXT PASSAGES:\n{build_context(hits)}",
        }
    ]
    messages.extend(history[-8:])
    messages.append({"role": "user", "content": question})

    raw_answer = llm.chat(messages)
    cleaned, flagged = check_quotes(raw_answer, [hit["text"] for hit in hits])
    return {
        "answer": cleaned,
        "sources": _sources(hits),
        "refused": False,
        "flagged_quotes": flagged,
        "rewritten_query": query,
    }
