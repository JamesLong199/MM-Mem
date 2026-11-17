"""
Memory building module - bottom-up construction of L1/L2/L3 three-layer multimodal memory
"""

import os
import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

from config import MemoryConfig
from models import ModelManager, build_messages, generate_text
from video_utils import (
    get_l1_segments,
    extract_frames_at_fps,
    save_frames,
)
from knowledge_graph import build_knowledge_graph, save_graph

logger = logging.getLogger(__name__)


# ============================================================
# Data structures
# ============================================================

@dataclass
class L1Node:
    l1_index: int
    video_id: str
    start_sec: float
    end_sec: float
    caption: str = ""
    frame_paths: List[str] = field(default_factory=list)
    frame_timestamps: List[float] = field(default_factory=list)


@dataclass
class L2Node:
    l2_index: int
    video_id: str
    start_sec: float
    end_sec: float
    caption: str = ""
    constituent_l1_indices: List[int] = field(default_factory=list)
    frame_paths: List[str] = field(default_factory=list)
    frame_timestamps: List[float] = field(default_factory=list)
    transcript: str = ""


@dataclass
class L3NodeData:
    l2_index: int
    entities: List[Dict] = field(default_factory=list)
    actions: List[Dict] = field(default_factory=list)
    scenes: List[Dict] = field(default_factory=list)
    frame_paths: List[str] = field(default_factory=list)
    frame_timestamps: List[float] = field(default_factory=list)


# ============================================================
# Prompt templates
# ============================================================

L1_CAPTION_PROMPT = """Analyze the provided video frames and describe what you see in detail.
Focus on:
- Objects and people present in the scene
- Actions being performed
- Spatial relationships and positions
- Visual attributes (colors, sizes, expressions)
- Any text or symbols visible

Output your description as a JSON object:
```json
{
    "description": "A detailed description of the scene content",
    "objects": ["list", "of", "key", "objects"],
    "actions": ["list", "of", "actions", "observed"],
    "spatial_info": "Description of spatial layout and relationships"
}
```"""

L2_DECISION_PROMPT = """You are a video memory organizer. You must decide how to handle a new scene segment relative to the current memory episode.

Current memory episode:
- Caption: {current_l2_caption}
- Time range: {current_l2_start:.1f}s - {current_l2_end:.1f}s

New scene segment:
- Caption: {new_l1_caption}
- Time range: {new_l1_start:.1f}s - {new_l1_end:.1f}s

Based on the visual content of both the current episode frames and the new segment frames, decide one of:
- ADD_NEW: The new segment is significantly different from the current episode. Finalize the current episode and start a new one.
- MERGE: The new segment is a continuation of the current episode. Add it to the current episode.
- DISCARD: The new segment contains noise, transitions, or redundant content. Skip it.

Respond with exactly one word: ADD_NEW, MERGE, or DISCARD."""

L2_CAPTION_PROMPT = """Analyze the provided video frames which span a continuous episode in a video.
Focus on the narrative flow and progression:
- What events occur in sequence?
- How does the scene develop over time?
- What is the overall narrative arc of this episode?
- Key transitions or changes within the scene

Output your description as a JSON object:
```json
{
    "episode_summary": "A narrative summary of the entire episode",
    "event_sequence": ["event1", "event2", "event3"],
    "key_transitions": "Description of major changes within the episode"
}
```"""


def l2_caption_prompt(transcript_text: str) -> str:
    """L2 caption prompt plus transcript lines that overlap the episode."""
    block = (transcript_text or "").strip() or "- No transcript lines overlap this episode."
    return (
        L2_CAPTION_PROMPT
        + "\n\nTranscript lines whose times overlap this episode:\n"
        + block
        + "\nInclude relevant speech from the transcript lines in the episode summary. "
        + "Output only the JSON object."
    )


L3_ENTITY_PROMPT = """Please analyze the given video frames and extract key information in a structured JSON format in English. Identify and describe:

Entities: List all distinct objects, people, animals, or other significant elements visible.
Actions: Describe the actions and interactions of the entities.
Scenes: Identify the locations, environments, or contexts.

If the video is filmed from a first-person point of view, describe the subject as "me" and include their actions.

Output strictly in this JSON format:
```json
{
    "entities": [{"entity name": "", "description": ""}],
    "actions": [{"entity name": "", "action description": ""}],
    "scenes": [{"location": ""}]
}
```
Each section should be detailed but concise. Avoid text outside the JSON."""


# ============================================================
# JSON parsing utilities
# ============================================================

def _salvage_json(response: str) -> Optional[Dict]:
    """Recover a dict when the model hits max_new_tokens mid-JSON."""
    start = response.find("{")
    if start < 0:
        return None
    blob = response[start:]
    fence = blob.find("```")
    if fence > 0:
        blob = blob[:fence]
    try:
        obj, _ = json.JSONDecoder().raw_decode(blob)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    last = max(blob.rfind("}"), blob.rfind("]"))
    if last < 0:
        return None
    chunk = blob[: last + 1]
    stack = []
    in_str = False
    escape = False
    for ch in chunk:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
    if in_str:
        return None
    closers = {"{": "}", "[": "]"}
    try:
        obj = json.loads(chunk + "".join(closers[item] for item in reversed(stack)))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_json_from_response(response: str) -> Optional[Dict]:
    """
    Parse JSON from model response, supporting ```json...``` format.
    A truncated object is closed so a max-token cutoff still yields entities.
    """
    # Try to match ```json ... ```
    match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to match { ... }
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # Try to parse directly
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return _salvage_json(response)


def parse_decision(response: str) -> str:
    """Parse L2 decision result: ADD_NEW / MERGE / DISCARD"""
    response_upper = response.strip().upper()
    for decision in ["ADD_NEW", "MERGE", "DISCARD"]:
        if decision in response_upper:
            return decision
    # Default to MERGE (conservative strategy)
    logger.warning(f"Cannot parse L2 decision result: {response}, defaulting to MERGE")
    return "MERGE"


# ============================================================
# JSONL read/write
# ============================================================

def save_jsonl(data_list: List[Dict], filepath: str):
    """Save as JSONL format"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# ============================================================
# L1 memory building
# ============================================================

def build_l1_memory(
    video_path: str,
    output_dir: str,
    video_id: str,
    config: MemoryConfig,
    model_manager: ModelManager,
) -> List[L1Node]:
    """
    Build L1 layer detail memory.
    1. PySceneDetect segmentation + split segments >10s
    2. Extract frames at 5fps per segment
    3. Generate detail captions with base model
    """
    logger.info("===== Starting L1 memory building =====")
    llm, processor = model_manager.get_base_model()

    # Step 1: Segmentation
    segments = get_l1_segments(
        video_path,
        pyscenedetect_threshold=config.pyscenedetect_threshold,
        max_segment_duration=config.l1_max_segment_duration,
    )

    l1_nodes = []
    for seg in segments:
        idx = seg["l1_index"]
        start = seg["start_sec"]
        end = seg["end_sec"]

        logger.info(f"  L1[{idx}]: {start:.1f}s - {end:.1f}s")

        # Step 2: Extract frames at 5fps
        frames, timestamps = extract_frames_at_fps(
            video_path, start, end, fps=config.l1_fps
        )

        if not frames:
            logger.warning(f"  L1[{idx}]: Unable to extract frames, skipping")
            continue

        # Save frames
        frame_dir = os.path.join(output_dir, video_id, "L1_frames", f"l1_{idx:04d}")
        frame_paths = save_frames(frames, timestamps, frame_dir)

        # Step 3: Generate caption with base model (pass file paths to avoid vLLM cache issues)
        full_paths = [os.path.join(frame_dir, p) for p in frame_paths]
        messages = build_messages(L1_CAPTION_PROMPT, images=full_paths)
        response = generate_text(
            llm, processor, messages,
            max_new_tokens=config.caption_max_new_tokens,
        )

        caption_data = parse_json_from_response(response)
        caption = json.dumps(caption_data, ensure_ascii=False) if caption_data else response

        node = L1Node(
            l1_index=idx,
            video_id=video_id,
            start_sec=start,
            end_sec=end,
            caption=caption,
            frame_paths=frame_paths,
            frame_timestamps=timestamps,
        )
        l1_nodes.append(node)
        logger.info(f"  L1[{idx}]: caption complete, {len(frames)} frames")

    # Save L1 memory
    l1_path = os.path.join(output_dir, video_id, "l1_memory.jsonl")
    save_jsonl([asdict(n) for n in l1_nodes], l1_path)
    logger.info(f"L1 memory building complete: {len(l1_nodes)} nodes, saved to {l1_path}")

    return l1_nodes


# ============================================================
# L2 memory building
# ============================================================


# 360 images at 1024x576 is the largest caption that fit in this GPU (~100GB).
# Longer episodes are sampled down to that count; fit_images_to_context still
# scales resolution so the language-model context stays under 262144 tokens.
MAX_CONTEXT_IMAGES = 360



def _capped_fps(start_sec: float, end_sec: float, fps: float, limit: int = MAX_CONTEXT_IMAGES) -> float:
    """Keep extracted frame count at or below the count that fits in GPU memory."""
    duration = max(0.0, end_sec - start_sec)
    count = duration * fps
    if count <= limit:
        return fps
    capped = limit / max(duration, 1e-3)
    logger.info(
        "Capping frame extraction from %.2f fps to %.3f fps (about %d frames -> %d)",
        fps,
        capped,
        int(count),
        limit,
    )
    return capped


def _limit_image_paths(paths, limit: int = MAX_CONTEXT_IMAGES):
    if len(paths) <= limit:
        return paths
    import numpy as np
    indices = np.linspace(0, len(paths) - 1, limit, dtype=int)
    chosen = []
    seen = set()
    for index in indices.tolist():
        index = int(index)
        if index in seen:
            continue
        seen.add(index)
        chosen.append(paths[index])
    logger.info(
        "Sampling %d frames down to %d for the context window",
        len(paths),
        len(chosen),
    )
    return chosen


def _finalize_l2_node(
    current_l2: L2Node,
    video_path: str,
    output_dir: str,
    config: MemoryConfig,
    llm,
    processor,
    transcript_entries: Optional[list] = None,
) -> L2Node:
    """
    Finalize an L2 node:
    1. Extract 2fps frames from the original video as L2 visual memory
    2. Generate narrative caption with base model (vLLM)

    When `transcript_entries` is provided, overlapping lines are stored on the
    node and inserted into the caption prompt. None leaves the visual prompt unchanged.
    """
    if transcript_entries is not None:
        from transcript_audio import overlapping_transcript_text
        current_l2.transcript = overlapping_transcript_text(
            transcript_entries, current_l2.start_sec, current_l2.end_sec
        )
    # Extract 2fps frames
    frames, timestamps = extract_frames_at_fps(
        video_path,
        current_l2.start_sec,
        current_l2.end_sec,
        fps=_capped_fps(current_l2.start_sec, current_l2.end_sec, config.l2_fps),
    )

    if frames:
        frame_dir = os.path.join(
            output_dir, current_l2.video_id, "L2_frames", f"l2_{current_l2.l2_index:04d}"
        )
        frame_paths = save_frames(frames, timestamps, frame_dir)
        current_l2.frame_paths = frame_paths
        current_l2.frame_timestamps = timestamps

        # Generate narrative caption with base model (pass file paths)
        full_paths = _limit_image_paths([os.path.join(frame_dir, p) for p in frame_paths])
        caption_prompt = (
            l2_caption_prompt(current_l2.transcript)
            if transcript_entries is not None
            else L2_CAPTION_PROMPT
        )
        messages = build_messages(caption_prompt, images=full_paths)
        response = generate_text(
            llm, processor, messages,
            max_new_tokens=config.l2_caption_max_new_tokens,
        )
        caption_data = parse_json_from_response(response)
        current_l2.caption = json.dumps(caption_data, ensure_ascii=False) if caption_data else response

    return current_l2


def build_l2_memory(
    video_path: str,
    l1_nodes: List[L1Node],
    output_dir: str,
    video_id: str,
    config: MemoryConfig,
    model_manager: ModelManager,
    transcript_path: str = "",
) -> List[L2Node]:
    """
    Build L2 layer narrative memory.
    Process L1 nodes sequentially, using the fine-tuned model to decide ADD_NEW/MERGE/DISCARD.

    `transcript_path` points at an `.srt` file or a directory containing `{video_id}.srt`.
    An existing non-empty transcript is reused. A missing one is produced by WorldMM's
    faster-whisper script and then overlapped onto each episode caption.
    """
    logger.info("===== Starting L2 memory building =====")
    transcript_entries = None
    if transcript_path:
        from transcript_audio import ensure_transcript, parse_srt
        srt_path = ensure_transcript(video_path, transcript_path, video_id)
        transcript_entries = parse_srt(srt_path)
        logger.info("Loaded %d transcript lines from %s", len(transcript_entries), srt_path)

    if not l1_nodes:
        logger.warning("No L1 nodes available, skipping L2 building")
        return []

    base_llm, base_processor = model_manager.get_base_model()
    ft_llm, ft_processor = model_manager.get_finetuned_model()  # Returns base model when no fine-tuned model

    l2_nodes = []
    current_l2: Optional[L2Node] = None
    l2_index = 0

    # Cache of current L2 frame paths for L2 decision
    current_l2_frame_paths: List[str] = []
    ckpt_path = os.path.join(output_dir, video_id, "l2_checkpoint.json")
    start_i = 0
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            payload = json.load(f)
        start_i = int(payload["next_i"])
        l2_index = int(payload["l2_index"])
        l2_nodes = [L2Node(**node) for node in payload["l2_nodes"]]
        current_l2_frame_paths = payload.get("current_l2_frame_paths", [])
        if payload.get("current"):
            current_l2 = L2Node(**payload["current"])
        logger.info(
            "Resuming L2 at L1 position %d with %d completed episodes",
            start_i,
            len(l2_nodes),
        )

    def save_ckpt(next_i: int) -> None:
        current = None
        if current_l2 is not None:
            current = asdict(current_l2)
        payload = {
            "next_i": next_i,
            "l2_index": l2_index,
            "l2_nodes": [asdict(node) for node in l2_nodes],
            "current": current,
            "current_l2_frame_paths": current_l2_frame_paths,
        }
        tmp_path = ckpt_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, ckpt_path)

    for i, l1_node in enumerate(l1_nodes):
        if i < start_i:
            continue
        # Current L1 frame file paths
        l1_frame_dir = os.path.join(output_dir, video_id, "L1_frames", f"l1_{l1_node.l1_index:04d}")
        l1_frame_full_paths = [os.path.join(l1_frame_dir, p) for p in l1_node.frame_paths]

        if not l1_frame_full_paths or not os.path.exists(l1_frame_full_paths[0]):
            logger.warning(f"  L1[{l1_node.l1_index}]: No frames available, skipping")
            continue

        if current_l2 is None:
            # First L1 node, default ADD_NEW
            logger.info(f"  L1[{l1_node.l1_index}]: Default ADD_NEW (first L1)")
            current_l2 = L2Node(
                l2_index=l2_index,
                video_id=video_id,
                start_sec=l1_node.start_sec,
                end_sec=l1_node.end_sec,
                constituent_l1_indices=[l1_node.l1_index],
            )
            # Use L1 frame paths as visual reference for current L2
            current_l2_frame_paths = list(l1_frame_full_paths)
            continue

        # Subsequent L1 nodes: use fine-tuned model to decide
        decision_text = L2_DECISION_PROMPT.format(
            current_l2_caption=current_l2.caption or "(caption not yet generated)",
            current_l2_start=current_l2.start_sec,
            current_l2_end=current_l2.end_sec,
            new_l1_caption=l1_node.caption,
            new_l1_start=l1_node.start_sec,
            new_l1_end=l1_node.end_sec,
        )

        # Combine L2 frame paths and L1 frame paths as visual input
        combined_paths = current_l2_frame_paths + l1_frame_full_paths
        if len(combined_paths) > config.l2_max_input_frames:
            import numpy as np
            indices = np.linspace(0, len(combined_paths) - 1, config.l2_max_input_frames, dtype=int)
            combined_paths = [combined_paths[i] for i in indices]

        messages = build_messages(decision_text, images=combined_paths)
        response = generate_text(
            ft_llm, ft_processor, messages,
            max_new_tokens=config.l2_decision_max_new_tokens,
        )

        decision = parse_decision(response)
        logger.info(f"  L1[{l1_node.l1_index}]: Decision result={decision}")

        if decision == "ADD_NEW":
            # Finalize the current L2 node
            current_l2 = _finalize_l2_node(
                current_l2, video_path, output_dir, config, base_llm, base_processor,
                transcript_entries=transcript_entries,
            )
            l2_nodes.append(current_l2)
            logger.info(f"  L2[{current_l2.l2_index}]: Completed ({current_l2.start_sec:.1f}s - {current_l2.end_sec:.1f}s)")

            # Start a new L2
            l2_index += 1
            current_l2 = L2Node(
                l2_index=l2_index,
                video_id=video_id,
                start_sec=l1_node.start_sec,
                end_sec=l1_node.end_sec,
                constituent_l1_indices=[l1_node.l1_index],
            )
            current_l2_frame_paths = list(l1_frame_full_paths)

        elif decision == "MERGE":
            # Merge L1 into current L2
            current_l2.constituent_l1_indices.append(l1_node.l1_index)
            current_l2.end_sec = max(current_l2.end_sec, l1_node.end_sec)
            current_l2_frame_paths.extend(l1_frame_full_paths)

        elif decision == "DISCARD":
            # Skip this L1 node
            logger.info(f"  L1[{l1_node.l1_index}]: DISCARD, skipping")

        save_ckpt(i + 1)

    # Finalize the last L2 node
    if current_l2 is not None:
        current_l2 = _finalize_l2_node(
            current_l2, video_path, output_dir, config, base_llm, base_processor,
            transcript_entries=transcript_entries,
        )
        l2_nodes.append(current_l2)
        logger.info(f"  L2[{current_l2.l2_index}]: Completed ({current_l2.start_sec:.1f}s - {current_l2.end_sec:.1f}s)")

    # Save L2 memory
    l2_path = os.path.join(output_dir, video_id, "l2_memory.jsonl")
    save_jsonl([asdict(n) for n in l2_nodes], l2_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    logger.info(f"L2 memory building complete: {len(l2_nodes)} nodes, saved to {l2_path}")

    return l2_nodes


# ============================================================
# L3 memory building (knowledge graph)
# ============================================================

def build_l3_memory(
    video_path: str,
    l2_nodes: List[L2Node],
    output_dir: str,
    video_id: str,
    config: MemoryConfig,
    model_manager: ModelManager,
) -> Tuple[Any, Any, List[L3NodeData]]:
    """
    Build L3 layer knowledge graph.
    1. Extract 1fps frames for each L2 node
    2. Extract entities/actions/scenes with base model
    3. Build knowledge graph
    """
    logger.info("===== Starting L3 memory building =====")

    if not l2_nodes:
        logger.warning("No L2 nodes available, skipping L3 building")
        import networkx as nx
        return nx.DiGraph(), {}, []

    base_llm, base_processor = model_manager.get_base_model()
    embedding_model = model_manager.get_embedding_model()

    l3_data_list = []
    ckpt_path = os.path.join(output_dir, video_id, "l3_checkpoint.json")
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            payload = json.load(f)
        l3_data_list = [L3NodeData(**node) for node in payload.get("nodes", [])]
        logger.info("Resuming L3 with %d completed episodes", len(l3_data_list))

    done_ids = {node.l2_index for node in l3_data_list}

    def save_l3_ckpt() -> None:
        tmp_path = ckpt_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump({"nodes": [asdict(node) for node in l3_data_list]}, f)
        os.replace(tmp_path, ckpt_path)

    for l2_node in l2_nodes:
        idx = l2_node.l2_index
        if idx in done_ids:
            continue
        logger.info(f"  L2[{idx}]: Extracting entities ({l2_node.start_sec:.1f}s - {l2_node.end_sec:.1f}s)")

        # Extract 1fps frames
        frames, timestamps = extract_frames_at_fps(
            video_path,
            l2_node.start_sec,
            l2_node.end_sec,
            fps=_capped_fps(l2_node.start_sec, l2_node.end_sec, config.l3_fps),
        )

        if not frames:
            logger.warning(f"  L2[{idx}]: Unable to extract 1fps frames")
            l3_data_list.append(L3NodeData(l2_index=idx))
            continue

        # Save L3 frames
        frame_dir = os.path.join(output_dir, video_id, "L3_frames", f"l2_{idx:04d}")
        frame_paths = save_frames(frames, timestamps, frame_dir)

        # Extract entities/actions/scenes with base model (with retries)
        entity_data = None
        full_paths = _limit_image_paths([os.path.join(frame_dir, p) for p in frame_paths])
        for attempt in range(config.l3_entity_max_retries):
            messages = build_messages(L3_ENTITY_PROMPT, images=full_paths)
            response = generate_text(
                base_llm, base_processor, messages,
                max_new_tokens=config.l3_entity_max_new_tokens,
            )
            entity_data = parse_json_from_response(response)
            if entity_data and "entities" in entity_data:
                break
            logger.warning(
                "  L2[%s]: JSON parsing failed (attempt %d), response=%r",
                idx,
                attempt + 1,
                response[:1200],
            )

        if entity_data is None:
            entity_data = {"entities": [], "actions": [], "scenes": []}

        l3_node = L3NodeData(
            l2_index=idx,
            entities=entity_data.get("entities", []),
            actions=entity_data.get("actions", []),
            scenes=entity_data.get("scenes", []),
            frame_paths=frame_paths,
            frame_timestamps=timestamps,
        )
        l3_data_list.append(l3_node)
        done_ids.add(idx)
        save_l3_ckpt()
        logger.info(
            f"  L2[{idx}]: Extracted {len(l3_node.entities)} entities, "
            f"{len(l3_node.actions)} actions, {len(l3_node.scenes)} scenes"
        )

    # Build knowledge graph
    kg_input = [
        {
            "l2_index": d.l2_index,
            "entities": d.entities,
            "actions": d.actions,
            "scenes": d.scenes,
        }
        for d in l3_data_list
    ]

    video_graph, entity_graph = build_knowledge_graph(
        kg_input, embedding_model, dedup_threshold=config.l3_entity_dedup_threshold
    )

    # Save
    graph_path = os.path.join(output_dir, video_id, "knowledge_graph.pkl")
    save_graph(video_graph, entity_graph, graph_path)

    l3_path = os.path.join(output_dir, video_id, "l3_memory.jsonl")
    save_jsonl([asdict(d) for d in l3_data_list], l3_path)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    logger.info(f"L3 memory building complete: {len(l3_data_list)} nodes, saved to {l3_path}")
    return video_graph, entity_graph, l3_data_list


# ============================================================
# Complete memory building pipeline
# ============================================================

def build_all_memory(
    video_path: str,
    output_dir: str,
    config: MemoryConfig,
    model_manager: ModelManager,
    transcript_path: str = "",
) -> Dict[str, Any]:
    """
    Complete bottom-up memory building: L1 -> L2 -> L3.

    `transcript_path` is forwarded to L2 captioning. Empty keeps the original
    visual-only episode prompt.
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    logger.info(f"Starting memory building for video: {video_id}")
    if not transcript_path:
        transcript_path = getattr(config, "transcript_path", "") or ""

    # L1
    l1_nodes = build_l1_memory(video_path, output_dir, video_id, config, model_manager)

    # L2
    l2_nodes = build_l2_memory(
        video_path, l1_nodes, output_dir, video_id, config, model_manager,
        transcript_path=transcript_path,
    )

    # L3
    video_graph, entity_graph, l3_data = build_l3_memory(
        video_path, l2_nodes, output_dir, video_id, config, model_manager
    )

    result = {
        "video_id": video_id,
        "num_l1_nodes": len(l1_nodes),
        "num_l2_nodes": len(l2_nodes),
        "num_l3_entities": len(entity_graph) if entity_graph else 0,
        "num_graph_edges": video_graph.number_of_edges() if video_graph else 0,
        "output_dir": os.path.join(output_dir, video_id),
    }

    logger.info(f"Memory building complete: {json.dumps(result, ensure_ascii=False, indent=2)}")
    return result
