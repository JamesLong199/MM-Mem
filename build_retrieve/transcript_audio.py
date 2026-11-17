"""WorldMM audio transcripts for MM-Mem L2 episodes.

Uses the same faster-whisper model as WorldMM (`distil-large-v3.5`, or the
local CTranslate2 snapshot when `model.bin` is present) and calls WorldMM's
transcribe script. An existing non-empty `.srt` is reused and not rewritten.
"""
from __future__ import annotations

import logging
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

WORLDM_TRANSCRIBE = Path("/home/hzhao/long/WorldMM/data/Video-MME/utils/transcribe.py")
WORLDM_PYTHON = Path("/home/hzhao/long/WorldMM/.venv/bin/python")
WHISPER_SNAPSHOT = Path("/home/hzhao/long/weights/distil-large-v3.5-ct2")

_TIMESTAMP = re.compile(
    r"(?P<h>\d+):(?P<m>\d+):(?P<s>\d+)[,.](?P<ms>\d+)"
)


def whisper_model() -> str:
    """Same choice as WorldMM `run_m3bench.whisper_model`."""
    if (WHISPER_SNAPSHOT / "model.bin").is_file():
        return str(WHISPER_SNAPSHOT)
    return "distil-large-v3.5"


def resolve_transcript_file(transcript_path: str, video_id: str) -> Path:
    path = Path(transcript_path)
    if path.is_dir():
        return path / f"{video_id}.srt"
    return path


def ensure_transcript(video_path: str, transcript_path: str, video_id: str) -> Path:
    """Return an `.srt` path, reusing it when it already has content.

    A missing file is created by WorldMM's transcribe script. This does not
    choose a different ASR model.
    """
    path = resolve_transcript_file(transcript_path, video_id)
    if path.is_file() and path.stat().st_size > 0:
        logger.info("Reusing existing transcript %s", path)
        return path
    if path.is_file() and path.stat().st_size == 0:
        path.unlink()
    if not WORLDM_TRANSCRIBE.is_file():
        raise FileNotFoundError(f"WorldMM transcribe script not found: {WORLDM_TRANSCRIBE}")
    path.parent.mkdir(parents=True, exist_ok=True)
    python = str(WORLDM_PYTHON) if WORLDM_PYTHON.is_file() else sys.executable
    model = whisper_model()
    command = [
        python,
        str(WORLDM_TRANSCRIBE),
        "--input-path",
        video_path,
        "--output-path",
        str(path),
        "--model",
        model,
        "--batch-size",
        "1",
    ]
    logger.info("Transcribing %s with faster-whisper %s", video_path, model)
    subprocess.run(command, check=True)
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Transcript was not written: {path}")
    return path


def _timestamp_to_seconds(text: str) -> float:
    match = _TIMESTAMP.search(text.strip())
    if match is None:
        raise ValueError(f"Bad SRT timestamp: {text}")
    millis = match.group("ms")
    millis = (millis + "000")[:3]
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(millis) / 1000.0
    )


def parse_srt(path: str | Path) -> list[dict]:
    """Parse an `.srt` file into start/end seconds plus the original timestamps."""
    raw = Path(path).read_text(encoding="utf-8", errors="replace")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    entries = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        time_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if time_index is None:
            continue
        start_stamp, end_stamp = [part.strip() for part in lines[time_index].split("-->", 1)]
        text = " ".join(line.strip() for line in lines[time_index + 1:] if line.strip())
        if not text:
            continue
        entries.append({
            "start": _timestamp_to_seconds(start_stamp),
            "end": _timestamp_to_seconds(end_stamp),
            "start_timestamp": start_stamp,
            "end_timestamp": end_stamp,
            "text": text,
        })
    return entries


def overlapping_transcript_text(entries: list[dict], start_sec: float, end_sec: float) -> str:
    """Transcript lines that overlap `[start_sec, end_sec)`, WorldMM's overlap test."""
    lines = []
    for entry in entries:
        if entry["end"] > start_sec and entry["start"] < end_sec:
            lines.append(
                f"- [{entry['start_timestamp']} --> {entry['end_timestamp']}] {entry['text']}"
            )
    return "\n".join(lines)
