"""Chunk documents and build the ChromaDB index.

Run from the project root: python -m lkybot.index
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb

from lkybot import config

HEADING_RE = re.compile(r"^\s*==+\s*(.+?)\s*==+\s*$", re.MULTILINE)
BATCH_SIZE = 64


def split_sections(text: str) -> list[tuple[str, str]]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("Overview", text.strip())] if text.strip() else []
    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("Overview", preamble))
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            sections.append((match.group(1).strip(), body))
    return sections


def _split_long_paragraph(paragraph: str, max_words: int) -> list[str]:
    words = paragraph.split()
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), max_words)]


def chunk_section(section: str, body: str) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append(" ".join(current))
            current = []
            current_words = 0

    for paragraph in paragraphs:
        words = len(paragraph.split())
        if words > config.CHUNK_MAX_WORDS:
            flush()
            chunks.extend(_split_long_paragraph(paragraph, config.CHUNK_MAX_WORDS))
            continue
        if current_words and current_words + words > config.CHUNK_MAX_WORDS:
            flush()
        current.append(paragraph)
        current_words += words
        if current_words >= config.CHUNK_MIN_WORDS:
            flush()
    flush()
    return chunks


def load_documents() -> list[tuple[dict, str]]:
    documents: list[tuple[dict, str]] = []
    for text_path in sorted(config.RAW_DIR.glob("*.txt")):
        meta_path = text_path.with_suffix(".json")
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        text = text_path.read_text(encoding="utf-8")
        documents.append((meta, text))
    return documents


def build_chunks() -> tuple[list[str], list[dict], list[str]]:
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    for meta, text in load_documents():
        for section, body in split_sections(text):
            for piece in chunk_section(section, body):
                chunk_text = f"{meta['title']} > {section}\n{piece}"
                ids.append(f"{meta['id']}:{len(documents)}")
                documents.append(chunk_text)
                metadatas.append(
                    {
                        "doc_id": meta["id"],
                        "title": meta["title"],
                        "section": section,
                        "doc_type": meta.get("doc_type", "secondary"),
                        "url": meta.get("url", ""),
                    }
                )
    return ids, documents, metadatas


def build_index() -> chromadb.Collection:
    config.DB_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids, documents, metadatas = build_chunks()
    for start in range(0, len(documents), BATCH_SIZE):
        collection.add(
            ids=ids[start : start + BATCH_SIZE],
            documents=documents[start : start + BATCH_SIZE],
            metadatas=metadatas[start : start + BATCH_SIZE],
        )
    return collection


def preview(collection: chromadb.Collection, queries: list[str]) -> None:
    for query in queries:
        result = collection.query(query_texts=[query], n_results=3)
        print(f"\nQuery: {query}")
        docs = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for doc, meta, distance in zip(docs, metadatas, distances):
            similarity = 1.0 - float(distance)
            first_line = doc.splitlines()[0]
            print(f"  {similarity:.3f}  {first_line}")


def main() -> None:
    ids, documents, _ = build_chunks()
    if not ids:
        raise SystemExit("No chunks found. Run python -m lkybot.scrape first.")
    collection = build_index()
    avg_words = sum(len(doc.split()) for doc in documents) / len(documents)
    print(f"Documents: {len(load_documents())}")
    print(f"Chunks: {len(ids)}")
    print(f"Average chunk length: {avg_words:.1f} words")
    preview(
        collection,
        [
            "Why did Singapore separate from Malaysia?",
            "What did Lee Kuan Yew say about meritocracy?",
            "How does Singapore defend itself?",
        ],
    )


if __name__ == "__main__":
    main()
