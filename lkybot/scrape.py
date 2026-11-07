"""Collect public documents from Wikipedia and Wikiquote via the MediaWiki API.

Run from the project root: python -m lkybot.scrape
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests
from bs4 import BeautifulSoup

from lkybot import config

USER_AGENT = "lkybot/0.1 (educational RAG take-home; contact: adiiwiraa@users.noreply.github.com)"
HEADERS = {"User-Agent": USER_AGENT}
TIMEOUT = 25
DELAY_SECONDS = 1.0
RETRIES = 3

WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIQUOTE_API = "https://en.wikiquote.org/w/api.php"

TOPICS: list[tuple[str, str]] = [
    ("wikipedia", "Lee Kuan Yew"),
    ("wikipedia", "History of Singapore"),
    ("wikipedia", "Separation of Singapore from Malaysia"),
    ("wikipedia", "People's Action Party"),
    ("wikipedia", "Economy of Singapore"),
    ("wikipedia", "Singapore–China relations"),
    ("wikipedia", "Singapore–United States relations"),
    ("wikipedia", "Housing and Development Board"),
    ("wikipedia", "National service in Singapore"),
    ("wikipedia", "Bilingualism policy in Singapore"),
    ("wikipedia", "Goh Chok Tong"),
    ("wikiquote", "Lee Kuan Yew"),
]


def log(message: str) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {message}"
    with config.SCRAPE_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _get(api: str, params: dict) -> dict | None:
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(api, params=params, headers=HEADERS, timeout=TIMEOUT)
            if response.status_code == 403:
                log(f"BLOCKED 403 {api} params={params.get('srsearch') or params.get('titles')}")
                return None
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            log(f"RETRY {attempt}/{RETRIES} {api} error={exc}")
            time.sleep(attempt * 2)
    return None


def search_title(api: str, topic: str) -> str | None:
    data = _get(
        api,
        {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srlimit": 1,
            "format": "json",
            "formatversion": 2,
        },
    )
    if not data:
        return None
    results = data.get("query", {}).get("search", [])
    return results[0]["title"] if results else None


def fetch_extract(api: str, title: str) -> str:
    data = _get(
        api,
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
            "formatversion": 2,
        },
    )
    if not data:
        return ""
    pages = data.get("query", {}).get("pages", [])
    if not pages or pages[0].get("missing"):
        return ""
    return (pages[0].get("extract") or "").strip()


def fetch_parsed_text(api: str, title: str) -> str:
    data = _get(
        api,
        {
            "action": "parse",
            "page": title,
            "prop": "text",
            "format": "json",
            "formatversion": 2,
            "redirects": 1,
        },
    )
    if not data:
        return ""
    html = data.get("parse", {}).get("text", "")
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one(".mw-parser-output") or soup
    parts: list[str] = []
    for node in body.find_all(["h2", "h3", "p", "li"]):
        text = node.get_text(" ", strip=True)
        if not text or text.startswith("["):
            continue
        if node.name == "h2":
            parts.append(f"\n== {text} ==")
        elif node.name == "h3":
            parts.append(f"\n=== {text} ===")
        else:
            parts.append(text)
    return "\n".join(parts).strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:90] or "document"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def existing_hashes() -> set[str]:
    hashes: set[str] = set()
    for meta_path in config.RAW_DIR.glob("*.json"):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("content_hash"):
                hashes.add(meta["content_hash"])
        except (OSError, ValueError):
            continue
    return hashes


def save_document(doc_id: str, text: str, meta: dict) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    (config.RAW_DIR / f"{doc_id}.txt").write_text(text, encoding="utf-8")
    (config.RAW_DIR / f"{doc_id}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def scrape_topic(source: str, topic: str, seen_hashes: set[str]) -> bool:
    api = WIKIPEDIA_API if source == "wikipedia" else WIKIQUOTE_API
    doc_type = "secondary" if source == "wikipedia" else "quotes"
    title = search_title(api, topic)
    if not title:
        log(f"UNRESOLVED source={source} topic={topic!r}")
        return False

    doc_id = slugify(f"{source}-{title}")
    if (config.RAW_DIR / f"{doc_id}.txt").exists():
        log(f"SKIP existing doc_id={doc_id}")
        return False

    text = fetch_extract(api, title)
    if not text:
        text = fetch_parsed_text(api, title)
    if not text:
        log(f"EMPTY source={source} title={title!r}")
        return False

    digest = content_hash(text)
    if digest in seen_hashes:
        log(f"DUPLICATE source={source} title={title!r} hash={digest[:12]}")
        return False

    page_url = (
        f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        if source == "wikipedia"
        else f"https://en.wikiquote.org/wiki/{title.replace(' ', '_')}"
    )
    meta = {
        "id": doc_id,
        "title": title,
        "topic": topic,
        "source": "Wikipedia" if source == "wikipedia" else "Wikiquote",
        "doc_type": doc_type,
        "url": page_url,
        "license": "CC BY-SA 4.0",
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "content_hash": digest,
        "status": "ok",
    }
    save_document(doc_id, text, meta)
    seen_hashes.add(digest)
    log(f"OK doc_id={doc_id} chars={len(text)}")
    return True


def scrape_pdfs() -> int:
    if not config.PDF_DIR.exists():
        return 0
    try:
        import pymupdf
    except ImportError:
        log("PDF skipped: pymupdf not installed")
        return 0

    saved = 0
    for pdf_path in sorted(config.PDF_DIR.glob("*.pdf")):
        doc_id = slugify(f"pdf-{pdf_path.stem}")
        if (config.RAW_DIR / f"{doc_id}.txt").exists():
            log(f"SKIP existing doc_id={doc_id}")
            continue
        try:
            with pymupdf.open(pdf_path) as document:
                text = "\n".join(page.get_text() for page in document).strip()
        except Exception as exc:  # noqa: BLE001 - log and continue per runbook
            log(f"PDF ERROR file={pdf_path.name} error={exc}")
            continue
        if not text:
            log(f"PDF EMPTY file={pdf_path.name}")
            continue
        meta = {
            "id": doc_id,
            "title": pdf_path.stem,
            "topic": pdf_path.stem,
            "source": "Local PDF",
            "doc_type": "primary",
            "url": pdf_path.as_posix(),
            "license": "unknown",
            "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "content_hash": content_hash(text),
            "status": "ok",
        }
        save_document(doc_id, text, meta)
        saved += 1
        log(f"OK PDF doc_id={doc_id} chars={len(text)}")
    return saved


def main() -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    seen_hashes = existing_hashes()
    saved = 0
    for source, topic in TOPICS:
        if scrape_topic(source, topic, seen_hashes):
            saved += 1
        time.sleep(DELAY_SECONDS)
    saved += scrape_pdfs()
    documents = sorted(config.RAW_DIR.glob("*.txt"))
    print(f"Saved {saved} new document(s). Total documents: {len(documents)}")
    print(f"Log: {config.SCRAPE_LOG}")


if __name__ == "__main__":
    main()
