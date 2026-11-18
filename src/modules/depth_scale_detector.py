"""
depth_scale_detector.py — Automatically detect depth scale factor.

Problem:  depth maps may be in mm, cm, m, or raw sensor units.
         If DataLoader always uses scale=1.0, the 3-D geometry
         is completely wrong and TSDF produces garbage.

Solution: Compare median depth against the expected scene scale
          derived from pose translations, or use heuristics...
"""

import logging
import numpy as np
from typing import List

log = logging.getLogger("depth_scale_detector")

def detect_depth_scale(
    frames: List[dict],
    pose_scale_hint: float = 1.0,
    expected_scene_radius_m: float = 5.0,
    min_frames: int = 10,
) -> float:
    n = min(len(frames), min_frames)

    # Helper: works for both dict and object
    def _get(f, key):
        if isinstance(f, dict):
            return f[key]
        return getattr(f, key)

    medians = []
    for f in frames[:n]:
        d = _get(f, "depth")
        valid = d[d > 0]
        if len(valid) > 0:
            medians.append(float(np.median(valid)))
    if not medians:
        log.warning("  No valid depth values — defaulting scale=1.0")
        return 1.0

    median_raw = float(np.median(medians))
    log.info(f"  Median raw depth = {median_raw:.4f}")

    scale_heuristic = _heuristic_scale(median_raw)
    log.info(f"  Heuristic scale  = {scale_heuristic}")

    scale_pose = _pose_based_scale(frames, median_raw, n)
    if scale_pose is not None:
        log.info(f"  Pose-based scale = {scale_pose}")
        ratio = max(scale_pose, scale_heuristic) / max(min(scale_pose, scale_heuristic), 1e-9)
        if ratio < 3.0:
            final = scale_pose
        else:
            log.warning("  Pose and heuristic disagree — using heuristic")
            final = scale_heuristic
    else:
        final = scale_heuristic

    metric_depth = median_raw * final
    if metric_depth > 100 or metric_depth < 0.05:
        log.warning(f"  Metric depth = {metric_depth:.2f} m looks wrong. Auto-correcting.")
        final = _correct_scale(metric_depth, median_raw)

    log.info(f"  ★ Detected depth_scale = {final}")
    return final

def _heuristic_scale(median_raw: float) -> float:
    """Common depth encodings."""
    if median_raw > 5000:     # uint16 millimetres (very common for RealSense/Azure)
        return 0.001
    if median_raw > 500:      # millimetres (float)
        return 0.001
    if median_raw > 50:       # centimetres
        return 0.01
    if median_raw > 0.3:      # metres
        return 1.0
    # Very small → might be normalised [0,1] or decimetres
    if median_raw > 0.05:
        return 1.0            # assume metres, just a close scene
    return 1.0                # fallback


def _pose_based_scale(frames, median_raw, n):
    try:
        def _get(f, key):
            if isinstance(f, dict):
                return f[key]
            return getattr(f, key)

        translations = np.array([_get(f, "pose")[:3, 3] for f in frames[:n]])
        diffs = np.diff(translations, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        dists = dists[dists > 0]
        if len(dists) == 0:
            return None
        median_step = float(np.median(dists))
        if median_step < 1e-8:
            return None
        expected_metric = median_step * 10
        scale = expected_metric / median_raw
        return scale
    except Exception:
        return None


def _correct_scale(metric_depth, median_raw) -> float:
    """Try to find a scale that gives a metric depth in [0.3, 20] m."""
    for candidate in [1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001, 0.0001]:
        d = median_raw * candidate
        if 0.3 <= d <= 20.0:
            log.info(f"  Corrected: raw×{candidate} = {d:.2f} m")
            return candidate
    return 0.001  # most common default