"""Question-independent, bounded routing catalogs over existing MM-Mem records.

No model/GPU imports: compilation can be tested independently of inference.
Only summaries are compressed; full records and every routing edge are retained.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

from m3bench import as_seconds, l3_search_text, visible_span

SCHEMA_VERSION = 1


def token_count(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False))


def preview(tokenizer, text, limit):
    """Explicitly marked preview; the source record is never truncated."""
    text = " ".join(str(text).split())
    if token_count(tokenizer, text) <= limit:
        return text
    # Decode/re-encode can change token boundaries, so measure the final string.
    ids = tokenizer.encode(text, add_special_tokens=False)
    for n in range(min(limit, len(ids)), -1, -1):
        result = tokenizer.decode(ids[:n], skip_special_tokens=True).strip() + " …"
        if token_count(tokenizer, result) <= limit:
            return result
    raise ValueError("Summary token budget is too small")


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_records(memory_dir, video_id, until_sec=None):
    """Filter BEFORE compiling summaries, including L3's linked L2 interval.

    All visible L1 records remain addressable, even if L2 discarded them.
    L3 content is attached to its visible L2 record rather than duplicated.
    """
    root = Path(memory_dir) / video_id
    if not root.is_dir():
        raise FileNotFoundError(f"Memory directory does not exist: {root}")
    l1 = read_jsonl(root / "l1_memory.jsonl")
    l2 = read_jsonl(root / "l2_memory.jsonl")
    if not l1 and not l2:
        raise FileNotFoundError(f"No L1/L2 memory records in {root}")
    l3 = {row["l2_index"]: row for row in read_jsonl(root / "l3_memory.jsonl")}
    records = {}
    for layer, rows, key in (("L1", l1, "l1_index"), ("L2", l2, "l2_index")):
        for row in rows:
            start, end = as_seconds(row.get("start_sec")), as_seconds(row.get("end_sec"))
            if not visible_span(start, end, until_sec):
                continue
            if start is not None and not math.isfinite(start):
                raise ValueError("Non-finite memory timestamp")
            if end is not None and (not math.isfinite(end) or (start is not None and end < start)):
                raise ValueError("Invalid memory interval")
            rid = f"{layer}:{int(row[key])}"
            if rid in records:
                raise ValueError(f"Duplicate record ID: {rid}")
            caption = row.get("caption") or ""
            if not isinstance(caption, str):
                caption = json.dumps(caption, ensure_ascii=False)
            parts = [caption]
            if layer == "L2":
                parts.extend([str(row.get("transcript") or ""), l3_search_text(l3.get(row[key], {}))])
            records[rid] = {
                "id": rid, "layer": layer, "index": int(row[key]),
                "start_sec": start, "end_sec": end,
                "text": "\n".join(part for part in parts if part),
                "frame_paths": row.get("frame_paths") or [],
                "frame_timestamps": row.get("frame_timestamps") or [],
                "l1_ids": [f"L1:{int(i)}" for i in row.get("constituent_l1_indices", [])],
            }
    # Never expose a pointer to a filtered-out record.
    for record in records.values():
        record["l1_ids"] = [rid for rid in record["l1_ids"] if rid in records]
    return records


def render_entries(title, entries):
    lines = [title, "Routing summaries only; read records for evidence."]
    for entry in entries:
        start, end = entry["start_sec"], entry["end_sec"]
        span = f"{start:g}-{end:g}s" if start is not None and end is not None else "untimed"
        lines.append(f"[{entry['id']}] {span} | {entry['summary']}")
    return "\n".join(lines)


def _bounded_entries(title, entries, tokenizer, budget):
    """Retain ALL pointers; shorten previews, never slice the catalog text."""
    entries = [dict(entry) for entry in entries]
    while token_count(tokenizer, render_entries(title, entries)) > budget:
        if not entries:
            raise ValueError("Catalog budget cannot fit its header")
        longest = max(entries, key=lambda e: token_count(tokenizer, e["summary"]))
        size = token_count(tokenizer, longest["summary"])
        if size <= 8:
            raise ValueError("Catalog budget cannot fit pointers and minimal summaries; increase it")
        longest["summary"] = preview(tokenizer, longest["summary"], max(8, size // 2))
    return entries


def compile_catalog(records, tokenizer, budget=2000, page_budget=2000, fanout=8,
                    summary_tokens=96, summarizer=None):
    """Build a bounded tree. External pages grow; the visible root stays bounded.

    Chronological grouping is a prototype policy, not a learned architecture.
    summarizer(source, max_tokens) optionally supplies model-written parent summaries.
    """
    if min(budget, page_budget) < 64 or fanout < 2 or summary_tokens < 8:
        raise ValueError("Budgets must be >=64, fanout >=2, summary_tokens >=8")
    entries = []
    for record in sorted(records.values(), key=lambda r: (
            r["start_sec"] if r["start_sec"] is not None else float("inf"), r["id"])):
        entries.append({"id": record["id"], "start_sec": record["start_sec"],
                        "end_sec": record["end_sec"],
                        "summary": preview(tokenizer, record["text"] or "No text; visual evidence available.", summary_tokens)})
    pages = {}
    level = 0
    while len(entries) > fanout:
        parents = []
        for offset in range(0, len(entries), fanout):
            children = entries[offset:offset + fanout]
            pid = f"P{level}:{offset // fanout}"
            children = _bounded_entries(f"Page {pid}", children, tokenizer, page_budget)
            pages[pid] = children
            source = "\n".join(entry["summary"] for entry in children)
            # Extractive previews are an explicit low-cost fallback, not model summaries.
            summary = summarizer(source, summary_tokens) if summarizer else source
            if not isinstance(summary, str) or not summary.strip():
                raise ValueError(f"Empty or invalid summary for {pid}")
            starts = [e["start_sec"] for e in children if e["start_sec"] is not None]
            ends = [e["end_sec"] for e in children if e["end_sec"] is not None]
            parents.append({"id": pid, "start_sec": min(starts) if starts else None,
                            "end_sec": max(ends) if ends else None,
                            "summary": preview(tokenizer, summary, summary_tokens)})
        entries = parents
        level += 1
    root = _bounded_entries("Memory catalog", entries, tokenizer, budget)
    catalog = {"schema_version": SCHEMA_VERSION, "root": root, "pages": pages,
               "records": records, "summary_mode": "model" if summarizer else "extractive",
               "budget": budget, "page_budget": page_budget}
    validate_catalog(catalog, tokenizer)
    return catalog


def validate_catalog(catalog, tokenizer):
    if catalog.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported catalog schema")
    visited_pages, reachable = set(), set()

    def walk(entries, ancestors):
        for entry in entries:
            rid = entry["id"]
            if rid in catalog["records"]:
                reachable.add(rid)
            elif rid in catalog["pages"]:
                if rid in ancestors:
                    raise ValueError("Catalog contains a cycle")
                visited_pages.add(rid)
                walk(catalog["pages"][rid], ancestors | {rid})
            else:
                raise ValueError(f"Dangling catalog pointer: {rid}")

    walk(catalog["root"], set())
    if reachable != set(catalog["records"]) or visited_pages != set(catalog["pages"]):
        raise ValueError("Catalog has unreachable records or pages")
    if token_count(tokenizer, render_entries("Memory catalog", catalog["root"])) > catalog["budget"]:
        raise ValueError("Catalog exceeds token budget")
    for pid, entries in catalog["pages"].items():
        if token_count(tokenizer, render_entries(f"Page {pid}", entries)) > catalog["page_budget"]:
            raise ValueError(f"Page exceeds token budget: {pid}")


def cached_catalog(memory_dir, video_id, tokenizer, *, until_sec=None, budget=2000,
                   page_budget=2000, fanout=8, summary_tokens=96,
                   summarizer=None, model_key="", rebuild=False):
    """Content-addressed snapshots; a later cutoff never populates an earlier one."""
    if until_sec is not None and (not math.isfinite(until_sec) or until_sec < 0):
        raise ValueError("until_sec must be finite and nonnegative")
    records = load_records(memory_dir, video_id, until_sec)
    identity = {"version": SCHEMA_VERSION, "records": records, "cutoff": until_sec,
                "budget": budget, "page_budget": page_budget, "fanout": fanout,
                "summary_tokens": summary_tokens, "model": model_key,
                "summary_mode": "model" if summarizer else "extractive",
                "tokenizer": str(getattr(tokenizer, "name_or_path", type(tokenizer).__name__)),
                "tokenizer_init": getattr(tokenizer, "init_kwargs", {}),
                "vocab_hash": hashlib.sha256(json.dumps(tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest()
                if hasattr(tokenizer, "get_vocab") else None}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
    folder = Path(memory_dir) / video_id / "catalogs"
    path = folder / f"{digest}.json"
    if path.exists() and not rebuild:
        catalog = json.loads(path.read_text())
        validate_catalog(catalog, tokenizer)
        if catalog["records"] != records:
            raise ValueError("Cached catalog records do not match source")
        return catalog, str(path), True
    catalog = compile_catalog(records, tokenizer, budget, page_budget, fanout, summary_tokens, summarizer)
    catalog.update({"fingerprint": digest, "until_sec": until_sec, "video_id": video_id})
    folder.mkdir(parents=True, exist_ok=True)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", dir=folder, delete=False, suffix=".tmp") as handle:
        tmp = handle.name
        json.dump(catalog, handle, ensure_ascii=False)
    os.replace(tmp, path)
    return catalog, str(path), False
