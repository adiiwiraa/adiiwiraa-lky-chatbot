"""Evaluation: retrieval quality, refusal behaviour, and LLM-judge scores.

Run from the project root: python -m lkybot.evaluate
All numbers are produced by this script; none are hardcoded.
"""
from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone

from lkybot import config, llm, rag

JUDGE_MODEL = os.getenv("JUDGE_MODEL", config.LLM_MODEL)


def load_eval_set() -> list[dict]:
    with config.EVAL_SET.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def retrieval_rank(hits: list[dict], gold_doc: str, gold_keyword: str) -> int | None:
    needle = gold_keyword.lower()
    for rank, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        if metadata.get("doc_id") != gold_doc:
            continue
        if needle in hit["text"].lower():
            return rank
    return None


def _parse_json(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except ValueError:
        return None


def judge(question: str, answer: str, context: str) -> dict | None:
    prompt = f"""You are a strict evaluator for a retrieval-augmented chatbot.
Score the answer from 1 to 5 on each dimension:
- groundedness: every factual claim is supported by the context
- relevance: the answer addresses the question
- faithfulness: no invented quotations or opinions beyond the context
The answer is intentionally written in the first person as Lee Kuan Yew.
Do NOT penalize the persona voice or first-person framing; score only
unsupported factual claims and invented quotations.
Return ONLY valid JSON:
{{"groundedness": 1, "relevance": 1, "faithfulness": 1, "reason": "short reason"}}

Question: {question}

Context:
{context}

Answer:
{answer}
"""
    try:
        raw = llm.chat(
            [
                {"role": "system", "content": "Return only valid JSON. No markdown fences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=300,
        )
    except llm.LLMError:
        return None
    parsed = _parse_json(raw)
    if not parsed:
        return None
    return {
        "groundedness": parsed.get("groundedness"),
        "relevance": parsed.get("relevance"),
        "faithfulness": parsed.get("faithfulness"),
        "reason": parsed.get("reason", ""),
    }


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def main() -> None:
    rows = load_eval_set()
    answerable = [row for row in rows if row["answerable"].lower() == "true"]
    unanswerable = [row for row in rows if row["answerable"].lower() != "true"]

    recall_hits = 0
    reciprocal_ranks: list[float] = []
    correct_refusals = 0
    false_refusals = 0
    generated_answers = 0
    flagged_count = 0
    groundedness: list[float] = []
    relevance: list[float] = []
    faithfulness: list[float] = []
    details: list[dict] = []

    for row in rows:
        question = row["question"]
        is_answerable = row["answerable"].lower() == "true"
        hits = rag.retrieve(question)
        rank = None
        if is_answerable:
            rank = retrieval_rank(hits, row["gold_doc"], row["gold_keyword"])
            if rank is not None:
                recall_hits += 1
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        result = rag.answer(question)
        if result["refused"]:
            if is_answerable:
                false_refusals += 1
            else:
                correct_refusals += 1
        else:
            generated_answers += 1
            if result["flagged_quotes"]:
                flagged_count += 1

        scores = None
        if not result["refused"]:
            context = rag.build_context(hits)
            scores = judge(question, result["answer"], context)
            if scores:
                for key, bucket in (
                    ("groundedness", groundedness),
                    ("relevance", relevance),
                    ("faithfulness", faithfulness),
                ):
                    value = scores.get(key)
                    if isinstance(value, (int, float)):
                        bucket.append(float(value))

        details.append(
            {
                "question": question,
                "answerable": is_answerable,
                "gold_doc": row["gold_doc"],
                "retrieval_rank": rank,
                "refused": result["refused"],
                "flagged_quotes": result["flagged_quotes"],
                "answer": result["answer"],
                "sources": result["sources"],
                "judge": scores,
            }
        )
        status = "REFUSED" if result["refused"] else f"rank={rank}"
        print(f"[{len(details):02d}/{len(rows)}] {status}  {question}")

    metrics = {
        "n_questions": len(rows),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "recall_at_5": round(recall_hits / len(answerable), 3) if answerable else None,
        "mrr": _mean(reciprocal_ranks),
        "correct_refusal_rate": round(correct_refusals / len(unanswerable), 3) if unanswerable else None,
        "false_refusal_rate": round(false_refusals / len(answerable), 3) if answerable else None,
        "flagged_quote_share": round(flagged_count / generated_answers, 3) if generated_answers else 0.0,
        "judge_groundedness": _mean(groundedness),
        "judge_relevance": _mean(relevance),
        "judge_faithfulness": _mean(faithfulness),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.LLM_MODEL,
        "judge_model": JUDGE_MODEL,
        "top_k": config.TOP_K,
        "score_threshold": config.SCORE_THRESHOLD,
        "documents": len(list(config.RAW_DIR.glob("*.txt"))),
        "chunks": rag._collection().count(),
        "metrics": metrics,
        "questions": details,
    }
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.RESULTS_DIR / "eval_results.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nMetrics")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
