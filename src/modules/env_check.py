"""
modules/env_check.py   (Phase 1.1, new)

Fast environment self-test, run at the start of the pipeline (about 1-2 s).

Why it exists
-------------
Open3D 0.20.0 (PyPI wheel) has a regression: ScalableTSDFVolume.integrate()
succeeds but extract_point_cloud() / extract_triangle_mesh() return NOTHING.
A synthetic wall gives 8,374 points on Open3D 0.19.0 and 0 on 0.20.0.
The pipeline then failed far downstream with a confusing empty-mesh error.
This check reproduces that scenario in miniature so the failure is reported
immediately, with the fix.
"""
import logging
from typing import Dict

import numpy as np

log = logging.getLogger("env_check")

KNOWN_GOOD_OPEN3D = "0.19.0"


def tsdf_self_test() -> Dict:
    """
    Fuse a synthetic flat wall into a ScalableTSDFVolume and extract it.
    Returns dict(ok, n_points, open3d_version, error).
    """
    import open3d as o3d
    res = dict(ok=False, n_points=0, open3d_version=o3d.__version__, error=None)
    try:
        depth = np.full((120, 160), 2.0, np.float32)
        intr = o3d.camera.PinholeCameraIntrinsic(160, 120, 150., 150., 80., 60.)
        vol = o3d.pipelines.integration.ScalableTSDFVolume(
            0.02, 0.08, o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.full((120, 160, 3), 128, np.uint8)),
            o3d.geometry.Image(depth),
            depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
        for _ in range(3):
            vol.integrate(rgbd, intr, np.eye(4))
        res["n_points"] = len(vol.extract_point_cloud().points)
        res["ok"] = res["n_points"] > 1000
    except Exception as e:                                   # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def check_environment(strict: bool = True) -> Dict:
    """
    Raises RuntimeError (when strict) if the TSDF self-test fails.
    Only warns about a CPU-only torch build, since it is not fatal.
    """
    report: Dict = {}
    t = tsdf_self_test()
    report["tsdf"] = t
    if t["ok"]:
        log.info(f"  env check: Open3D {t['open3d_version']} TSDF OK "
                 f"({t['n_points']:,} points on the synthetic wall)")
    else:
        msg = (f"Open3D {t['open3d_version']}: ScalableTSDFVolume returned "
               f"{t['n_points']} points on a synthetic wall"
               f"{' (' + t['error'] + ')' if t['error'] else ''}. "
               f"This build cannot reconstruct. Fix:  "
               f"python -m pip install open3d=={KNOWN_GOOD_OPEN3D}")
        log.error("ENVIRONMENT CHECK FAILED: " + msg)
        if strict:
            raise RuntimeError(msg)

    try:
        import torch
        report["torch"] = dict(version=torch.__version__,
                               cuda=bool(torch.cuda.is_available()))
        if not torch.cuda.is_available():
            log.warning(
                f"  env check: torch {torch.__version__} has no CUDA support "
                f"(CPU-only build). GPU texture baking and the GPU proximity "
                f"crop will fall back to the CPU. Install a CUDA build of "
                f"torch (RTX 50-series needs CUDA 12.8 or newer).")
    except Exception:                                        # noqa: BLE001
        report["torch"] = None
        log.warning("  env check: torch is not importable (CPU fallbacks only)")
    return report
