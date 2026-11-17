"""Open-ended answers from MM-Mem L1/L2/L3 memory.

Multiple-choice Video-MME prompts stay in memory_retrieve.py. This path asks
for a free-text sentence and drops any L1, L2, or linked L3 segment whose end
is after the question cutoff.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m3bench import cutoff_seconds, l3_search_text, open_answer_prompt, resolve_span, visible_span
from transformers_backend import TransformersLLM

ROOT = Path("/home/hzhao/long/MM-Mem/output")
RESULTS = Path("/home/hzhao/long/m3-results/MM-Mem")
EMB = "/home/hzhao/long/weights/bge-large-en-v1.5"
VLM = "/home/hzhao/long/weights/Qwen3-VL-8B-Instruct"
TOP_K = 5
VIDEOS = [
    ("study_17", "/home/hzhao/long/m3-agent/data/annotations/robot.json"),
    ("MaStjHoN8RE", "/home/hzhao/long/m3-agent/data/annotations/web.json"),
]


def _memory_file(video_id: str, name: str) -> Path | None:
    nested = ROOT / video_id / video_id / name
    if nested.is_file():
        return nested
    flat = ROOT / video_id / name
    if flat.is_file():
        return flat
    return None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_captions(video_id: str) -> list[dict]:
    l2_times: dict = {}
    l2_path = _memory_file(video_id, "l2_memory.jsonl")
    if l2_path is not None:
        for item in _read_jsonl(l2_path):
            idx = item.get("l2_index")
            if idx is None:
                continue
            l2_times[idx] = resolve_span("l2_memory.jsonl", item, {})

    rows = []
    for name in ("l1_memory.jsonl", "l2_memory.jsonl", "l3_memory.jsonl"):
        path = _memory_file(video_id, name)
        if path is None:
            continue
        for item in _read_jsonl(path):
            if name == "l3_memory.jsonl":
                text = l3_search_text(item)
            else:
                text = item.get("caption") or item.get("summary") or ""
                if isinstance(text, dict):
                    text = json.dumps(text, ensure_ascii=False)
                text = str(text).strip()
                if name == "l2_memory.jsonl":
                    transcript = str(item.get("transcript") or "").strip()
                    if transcript:
                        text = f"{text}\n{transcript}".strip()
            if not text:
                continue
            start, end = resolve_span(name, item, l2_times)
            rows.append({
                "file": name,
                "id": item.get("l1_index", item.get("l2_index", item.get("l3_index"))),
                "text": text[:1500],
                "start_sec": start,
                "end_sec": end,
            })
    return rows


def select_retrieved(captions: list[dict], matrix: np.ndarray, query: np.ndarray, cutoff: float | None) -> list[dict]:
    """Top-5 by cosine similarity inside this question's own candidate pool.

    Embeddings may be computed once. The pool is rebuilt from `cutoff` every
    call, so a later question's cutoff never becomes the shared candidate set.
    A node that starts before the cutoff and ends after it is dropped whole.
    """
    scores = np.asarray(matrix @ query, dtype=np.float64).copy()
    if cutoff is not None:
        allowed = np.array(
            [visible_span(row.get("start_sec"), row.get("end_sec"), cutoff) for row in captions],
            dtype=bool,
        )
        if allowed.shape[0] != scores.shape[0]:
            raise RuntimeError("embedding rows do not match memory nodes")
        scores = np.where(allowed, scores, -np.inf)
    retrieved = []
    for index in np.argsort(-scores):
        index = int(index)
        if not np.isfinite(scores[index]):
            break
        row = captions[index]
        if cutoff is not None and not visible_span(row.get("start_sec"), row.get("end_sec"), cutoff):
            raise RuntimeError(
                f"cutoff filter leaked {row['file']} id={row['id']} "
                f"start={row.get('start_sec')} end={row.get('end_sec')} cutoff={cutoff}"
            )
        retrieved.append({
            "id": row["id"],
            "file": row["file"],
            "score": float(scores[index]),
            "text": row["text"],
        })
        if len(retrieved) >= TOP_K:
            break
    return retrieved


def _force_ids() -> set[str]:
    return {part.strip() for part in os.environ.get("MM_MEM_FORCE_IDS", "").split(",") if part.strip()}


def _load_existing(out: Path) -> tuple[dict, dict, list[str]]:
    done = {}
    original_lines = {}
    ordered_ids = []
    if not out.is_file():
        return done, original_lines, ordered_ids
    for line in out.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        qid = rec["question_id"]
        done[qid] = rec
        original_lines[qid] = line
        if qid not in ordered_ids:
            ordered_ids.append(qid)
    return done, original_lines, ordered_ids


def _persist(out: Path, ordered_ids: list[str], done: dict, original_lines: dict, recomputed: set[str]) -> None:
    parts = []
    for qid in ordered_ids:
        if qid not in done:
            continue
        if qid in recomputed:
            parts.append(json.dumps(done[qid], ensure_ascii=False))
        else:
            parts.append(original_lines[qid])
    text = "\n".join(parts) + "\n"
    tmp = out.parent / (out.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, out)


def main() -> None:
    encoder = SentenceTransformer(EMB)
    llm = None
    only = os.environ.get("MM_MEM_VIDEO", "").strip()
    force_ids = _force_ids()
    print(f"video filter={only or 'ALL'} force={sorted(force_ids)}", flush=True)
    for video_id, ann_path in VIDEOS:
        if only and video_id != only:
            continue
        captions = load_captions(video_id)
        print(f"{video_id} captions {len(captions)}", flush=True)
        if not captions:
            print(f"skip {video_id}: memory not built", flush=True)
            continue
        # One embedding pass. Each question slices its own candidate pool below.
        matrix = encoder.encode([row["text"] for row in captions], normalize_embeddings=True)
        ann = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        out = RESULTS / f"{video_id}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done, original_lines, ordered_ids = _load_existing(out)
        recomputed: set[str] = set()
        for qa in ann[video_id]["qa_list"]:
            qid = qa["question_id"]
            if qid in done and qid not in force_ids and not str(done[qid].get("final_answer", "")).startswith("Error"):
                continue
            if qid not in ordered_ids:
                ordered_ids.append(qid)
            cutoff = cutoff_seconds(qa.get("before_clip"))
            query = encoder.encode([qa["question"]], normalize_embeddings=True)[0]
            retrieved = select_retrieved(captions, matrix, query, cutoff)
            context = "\n".join(f"[{item['file']} {item['id']}] {item['text']}" for item in retrieved)
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": open_answer_prompt(qa["question"], context)}],
            }]
            if llm is None:
                llm = TransformersLLM(VLM)
            answer = llm.generate_messages(messages, max_new_tokens=256, temperature=0.0).strip()
            record = {
                "question_id": qid,
                "question": qa["question"],
                "gold_answer": qa.get("answer"),
                "before_clip": qa.get("before_clip"),
                "cutoff_sec": cutoff,
                "timestamp": qa.get("timestamp"),
                "rounds": [{"retrieval_query": qa["question"], "retrieved": retrieved, "reasoning_text": answer}],
                "final_answer": answer,
            }
            done[qid] = record
            recomputed.add(qid)
            _persist(out, ordered_ids, done, original_lines, recomputed)
            brief = " ".join(f"{item['file']}:{item['id']}" for item in retrieved)
            print(f"WROTE {qid} cutoff_sec={cutoff} n={len(retrieved)} {brief}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
