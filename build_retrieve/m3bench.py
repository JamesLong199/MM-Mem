"""Shared M3-Bench cutoff and open-question prompt.

Clip length is 30 seconds. A question with `before_clip` may use memory whose
end is at or before `(before_clip + 1) * 30` seconds. A segment that starts
before that cutoff and ends after it is not searchable: the whole segment is
dropped, and its text is not trimmed to the boundary.
"""
from __future__ import annotations

CLIP_SECONDS = 30


def as_seconds(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


def cutoff_seconds(before_clip):
    """Inclusive end of clip `before_clip`. Clips are 30 seconds."""
    if before_clip is None:
        return None
    if isinstance(before_clip, str) and not before_clip.strip():
        return None
    return (int(before_clip) + 1) * CLIP_SECONDS


def resolve_span(name: str, item: dict, l2_times: dict) -> tuple:
    """Span used for the cutoff check.

    L1 and L2 use their own start_sec/end_sec. An L3 row that has no end_sec
    inherits the linked L2 span. Untimed text stays untimed.
    """
    start = as_seconds(item.get("start_sec"))
    end = as_seconds(item.get("end_sec"))
    if name == "l3_memory.jsonl" and end is None:
        linked = l2_times.get(item.get("l2_index"))
        if linked is not None and linked[1] is not None:
            end = linked[1]
            if start is None:
                start = linked[0]
    return start, end


def visible_at(end, cutoff) -> bool:
    """True when this segment may be retrieved.

    With no cutoff, every segment is visible. With a cutoff, a missing end is
    not searchable, and an end after the cutoff drops the segment whole.
    """
    if cutoff is None:
        return True
    if end is None:
        return False
    return float(end) <= float(cutoff)


def visible_span(start, end, cutoff) -> bool:
    """Same rule as `visible_at`, and a start after the cutoff is also dropped."""
    if not visible_at(end, cutoff):
        return False
    if cutoff is not None and start is not None and float(start) > float(cutoff):
        return False
    return True


def l3_search_text(item: dict) -> str:
    """Text searched for an L3 record. L3 rows store entities, not a caption."""
    parts = []
    entity_bits = []
    for entity in item.get("entities") or []:
        if isinstance(entity, dict):
            entity_bits.append(
                f"{entity.get('entity name', '')}: {entity.get('description', '')}".strip(": ")
            )
        else:
            entity_bits.append(str(entity))
    if any(entity_bits):
        parts.append("Entities: " + "; ".join(bit for bit in entity_bits if bit))

    action_bits = []
    for action in item.get("actions") or []:
        if isinstance(action, dict):
            action_bits.append(
                f"{action.get('entity name', '')}: {action.get('action description', '')}".strip(": ")
            )
        else:
            action_bits.append(str(action))
    if any(action_bits):
        parts.append("Actions: " + "; ".join(bit for bit in action_bits if bit))

    scene_bits = []
    for scene in item.get("scenes") or []:
        if isinstance(scene, dict):
            scene_bits.append(str(scene.get("location", "")))
        else:
            scene_bits.append(str(scene))
    if any(scene_bits):
        parts.append("Scenes: " + "; ".join(bit for bit in scene_bits if bit))
    return " | ".join(parts)


def open_answer_prompt(question: str, context_text: str = "") -> str:
    """Free-text prompt for open M3-Bench questions.

    The gold answer is not an argument and is never written into the prompt.
    """
    question = (question or "").strip()
    context = (context_text or "").strip()
    if context:
        return (
            "You are given an open-ended question about a video, along with relevant "
            "retrieved information from the video's memory.\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Using the retrieved memory, answer the question in free text. "
            "Reply with a short sentence, not a single letter.\n"
            f"Question: {question}\n"
            "Answer:"
        )
    return (
        "Answer the following open-ended question about the video in free text. "
        "Reply with a short sentence, not a single letter.\n"
        f"Question: {question}\n"
        "Answer:"
    )
