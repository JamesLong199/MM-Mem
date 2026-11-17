"""
Video processing utility module
- PySceneDetect scene detection
- Frame extraction (at specified FPS)
- Frame saving/loading
"""

import os
import math
import logging
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def detect_scenes(
    video_path: str,
    threshold: float = 27.0,
) -> List[Tuple[float, float]]:
    """
    Detect video scene boundaries using PySceneDetect.
    Returns a list of [(start_sec, end_sec), ...].
    """
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        # No scene transitions detected, treat the entire video as one scene
        vr = open_video_reader(video_path)
        duration = len(vr) / vr.get_avg_fps()
        logger.info(f"No scene transitions detected, treating entire video as a single scene: 0.0 - {duration:.2f}s")
        return [(0.0, duration)]

    scenes = []
    for start_tc, end_tc in scene_list:
        scenes.append((start_tc.get_seconds(), end_tc.get_seconds()))

    logger.info(f"Detected {len(scenes)} scenes")
    return scenes


def split_segments(
    start_sec: float,
    end_sec: float,
    max_duration: float = 10.0,
) -> List[Tuple[float, float]]:
    """
    If segment duration exceeds max_duration, split into multiple sub-segments.
    Each sub-segment is at most max_duration seconds; the last one may be shorter.

    Example: (0.0, 25.0), max_duration=10.0 -> [(0.0, 10.0), (10.0, 20.0), (20.0, 25.0)]
    """
    duration = end_sec - start_sec
    if duration <= max_duration:
        return [(start_sec, end_sec)]

    segments = []
    current = start_sec
    while current < end_sec:
        seg_end = min(current + max_duration, end_sec)
        segments.append((current, seg_end))
        current = seg_end

    return segments


def get_l1_segments(
    video_path: str,
    pyscenedetect_threshold: float = 27.0,
    max_segment_duration: float = 10.0,
) -> List[Dict]:
    """
    Complete L1 segmentation pipeline: scene detection + splitting segments exceeding duration limit.
    Returns:
    [
        {
            "l1_index": int,
            "start_sec": float,
            "end_sec": float,
            "source_scene_index": int,
        },
        ...
    ]
    """
    scenes = detect_scenes(video_path, threshold=pyscenedetect_threshold)

    l1_segments = []
    l1_index = 0

    for scene_idx, (start, end) in enumerate(scenes):
        sub_segments = split_segments(start, end, max_duration=max_segment_duration)
        for sub_start, sub_end in sub_segments:
            l1_segments.append({
                "l1_index": l1_index,
                "start_sec": sub_start,
                "end_sec": sub_end,
                "source_scene_index": scene_idx,
            })
            l1_index += 1

    logger.info(f"L1 segmentation complete: {len(scenes)} scenes -> {len(l1_segments)} L1 nodes")
    return l1_segments


def open_video_reader(video_path: str):
    """Open a Decord reader on one thread.

    The default thread pool aborts on some VP9/AV1 files with
    ``threaded_decoder.cc: Check failed: run_.load()``.
    """
    import decord
    decord.bridge.set_bridge("native")
    return decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)


def _read_indices_decord(video_path: str, frame_indices: List[int]) -> np.ndarray:
    last = None
    for attempt in range(2):
        try:
            vr = open_video_reader(video_path)
            return vr.get_batch(frame_indices).asnumpy()
        except Exception as exc:
            last = exc
            logger.warning("decord batch read failed (%s), attempt %s", exc, attempt + 1)
    frames = []
    for idx in frame_indices:
        vr = open_video_reader(video_path)
        frames.append(vr[int(idx)].asnumpy())
    if not frames:
        raise RuntimeError(f"decord could not read {video_path}") from last
    return np.stack(frames)


def _read_indices_pyav(video_path: str, frame_indices: List[int], video_fps: float) -> np.ndarray:
    """Seek each timestamp with PyAV when Decord cannot decode the file."""
    import av
    frames = []
    for idx in frame_indices:
        target = float(idx) / video_fps
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            seek = int(target / stream.time_base) if stream.time_base else 0
            container.seek(seek, stream=stream, any_frame=False, backward=True)
            chosen = None
            for frame in container.decode(stream):
                if frame.time is None or frame.time + 1e-3 >= target:
                    chosen = frame.to_ndarray(format="rgb24")
                    break
            if chosen is None:
                raise RuntimeError(f"PyAV found no frame at {target:.3f}s in {video_path}")
            frames.append(chosen)
    return np.stack(frames)


def extract_frames_at_fps(
    video_path: str,
    start_sec: float,
    end_sec: float,
    fps: float = 5.0,
) -> Tuple[List[Image.Image], List[float]]:
    """
    Extract frames uniformly at specified FPS from the original video.
    Returns (frame_list, timestamp_list).
    """
    vr = open_video_reader(video_path)
    video_fps = vr.get_avg_fps()
    total_frames = len(vr)

    start_frame = int(start_sec * video_fps)
    end_frame = min(int(end_sec * video_fps), total_frames)

    if start_frame >= end_frame:
        logger.warning(f"Invalid frame range: start={start_frame}, end={end_frame}")
        return [], []

    duration = end_sec - start_sec
    num_frames = max(1, int(duration * fps))

    # Calculate uniformly distributed frame indices
    if num_frames == 1:
        frame_indices = [start_frame]
    else:
        frame_indices = np.linspace(start_frame, end_frame - 1, num_frames, dtype=int).tolist()

    # Deduplicate and ensure within valid range
    frame_indices = sorted(set(
        max(0, min(idx, total_frames - 1)) for idx in frame_indices
    ))

    if not frame_indices:
        return [], []

    try:
        frames_data = _read_indices_decord(video_path, frame_indices)
    except Exception as exc:
        logger.warning("decord frame read failed (%s); using PyAV", exc)
        frames_data = _read_indices_pyav(video_path, frame_indices, video_fps)

    frames = []
    timestamps = []
    for i, idx in enumerate(frame_indices):
        img = Image.fromarray(frames_data[i])
        frames.append(img)
        timestamps.append(idx / video_fps)

    return frames, timestamps


def save_frames(
    frames: List[Image.Image],
    timestamps: List[float],
    output_dir: str,
) -> List[str]:
    """
    Save frames as JPEG files.
    Returns a list of relative paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for frame, ts in zip(frames, timestamps):
        filename = f"frame_{ts:.3f}.jpg"
        filepath = os.path.join(output_dir, filename)
        frame.save(filepath, "JPEG", quality=95)
        paths.append(filename)
    return paths


def load_frames(
    frame_paths: List[str],
    base_dir: str,
) -> List[Image.Image]:
    """Load previously saved frames from disk"""
    frames = []
    for path in frame_paths:
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            frames.append(Image.open(full_path).convert("RGB"))
        else:
            logger.warning(f"Frame file does not exist: {full_path}")
    return frames


def subsample_frames(
    frames: List[Image.Image],
    timestamps: List[float],
    max_frames: int,
) -> Tuple[List[Image.Image], List[float]]:
    """Uniformly subsample when frame count exceeds the limit"""
    if len(frames) <= max_frames:
        return frames, timestamps

    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    sampled_frames = [frames[i] for i in indices]
    sampled_timestamps = [timestamps[i] for i in indices]
    return sampled_frames, sampled_timestamps
