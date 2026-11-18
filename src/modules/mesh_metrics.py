"""
modules/mesh_metrics.py   (Phase 2, new)

Measurable mesh-quality proxies. No ground truth is used or claimed.


1. depth_fidelity()   Ray-cast the mesh into sampled camera views and compare
                      the hit distance with the depth the sensor measured.
                        coverage       fraction of high-confidence sensor pixels
                                       where the mesh is hit at all
                        median/p90     |mesh_depth - sensor_depth| (m) on hits
                        within_tol     fraction of hits within `tol_m`
                        front_floaters fraction of hits that are more than
                                       `gross_m` IN FRONT of the sensor depth
                                       (geometry floating before the real
                                       surface; typical "bridging" artefact)
                      LIMITS: this measures agreement with the *input* depth.
                      A systematic sensor error (e.g. glass) is invisible to
                      it, and the sampled views come from the same data the
                      mesh was built from, so it is a consistency check and not
                      an independent accuracy measure.

                      
2. topology()        faces, vertices, connected components, boundary-edge
                      count and length (open borders = holes / unobserved
                      regions), non-manifold edges, degenerate triangles.
"""


import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)
_FLIP = np.diag([1., -1., -1., 1.])


def _read_depth(path: Path) -> Optional[np.ndarray]:
    path = Path(path)
    try:
        b = path.with_suffix(".bin")
        if b.is_file():
            raw = np.fromfile(str(b), dtype=np.float32)
            for h, w in ((192, 256), (480, 640), (360, 480)):
                if raw.size == h * w:
                    return raw.reshape(h, w)
        from src.modules.exr_reader import read_exr_depth
        return read_exr_depth(path).astype(np.float32)
    except Exception:                                        # noqa: BLE001
        return None


def depth_fidelity(mesh, frames: list, n_views: int = 24, tol_m: float = 0.03,
                   gross_m: float = 0.10, stride: int = 2,
                   min_depth: float = 0.3, max_depth: float = 9.0,
                   min_conf_eval: int = 2) -> Dict:
    """`mesh` is an open3d TriangleMesh (legacy)."""
    import open3d as o3d
    if len(mesh.triangles) == 0:
        return dict(coverage=0.0, n_samples=0, error="empty mesh")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    idx = np.unique(np.linspace(0, len(frames) - 1, n_views).astype(int))
    errs: List[np.ndarray] = []
    n_px = n_hit = n_front = 0
    for k in idx:
        fr = frames[int(k)]
        d = _read_depth(fr.depth_path)
        if d is None:
            continue
        h, w = d.shape
        valid = (d > min_depth) & (d < max_depth) & np.isfinite(d)
        cp = getattr(fr, "conf_path", None)
        if min_conf_eval > 0 and cp is not None and Path(cp).is_file():
            c = np.fromfile(str(cp), dtype=np.uint8)
            if c.size == d.size:
                valid &= (c.reshape(d.shape) >= min_conf_eval)
        K = fr.intrinsics.K_for(w, h)
        v, u = np.mgrid[0:h:stride, 0:w:stride]
        dirs_cam = np.stack([(u + 0.5 - K[0, 2]) / K[0, 0],
                             (v + 0.5 - K[1, 2]) / K[1, 1],
                             np.ones_like(u, dtype=np.float64)], -1).reshape(-1, 3)
        T = fr.c2w @ _FLIP
        R, o = T[:3, :3], T[:3, 3]
        dirs_w = dirs_cam @ R.T
        norm = np.linalg.norm(dirs_w, axis=1, keepdims=True)
        dirs_w = dirs_w / norm
        rays = np.concatenate([np.tile(o, (len(dirs_w), 1)), dirs_w], axis=1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays))
        t_hit = ans["t_hit"].numpy()
        # t_hit is along the unit world ray; optical-axis depth = t * (dir_cam.z / |dir_cam|)
        z_mesh = t_hit * (dirs_cam[:, 2] / np.linalg.norm(dirs_cam, axis=1))
        zs = d[v, u].reshape(-1)
        ok = valid[v, u].reshape(-1)
        hit = np.isfinite(t_hit) & ok
        n_px += int(ok.sum()); n_hit += int(hit.sum())
        diff = z_mesh[hit] - zs[hit]
        n_front += int((diff < -gross_m).sum())
        errs.append(np.abs(diff))
    if not errs or n_px == 0:
        return dict(coverage=0.0, n_samples=0, error="no data")
    e = np.concatenate(errs)
    return dict(
        coverage=float(n_hit / n_px),
        n_samples=int(e.size),
        median_abs_err_m=float(np.median(e)) if e.size else None,
        p90_abs_err_m=float(np.percentile(e, 90)) if e.size else None,
        within_tol=float((e < tol_m).mean()) if e.size else None,
        tol_m=tol_m,
        front_floaters=float(n_front / max(n_hit, 1)),
        n_views=len(idx),
    )


def topology(mesh) -> Dict:
    V = np.asarray(mesh.vertices)
    F = np.asarray(mesh.triangles)
    if len(F) == 0:
        return dict(n_faces=0, n_vertices=len(V))
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    key = e[:, 0].astype(np.int64) * (int(V.shape[0]) + 1) + e[:, 1]
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    boundary = uniq[cnt == 1]
    b0 = (boundary // (V.shape[0] + 1)).astype(int)
    b1 = (boundary % (V.shape[0] + 1)).astype(int)
    blen = np.linalg.norm(V[b0] - V[b1], axis=1).sum() if len(boundary) else 0.0
    tri = V[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    labels, counts, _ = mesh.cluster_connected_triangles()
    counts = np.asarray(counts)
    return dict(
        n_vertices=int(len(V)), n_faces=int(len(F)),
        n_components=int(len(counts)),
        largest_component_face_fraction=float(counts.max() / len(F)),
        components_under_1pct=int((counts < 0.01 * len(F)).sum()),
        boundary_edges=int(len(boundary)),
        boundary_length_m=float(blen),
        non_manifold_edges=int((cnt > 2).sum()),
        degenerate_faces=int((area < 1e-10).sum()),
        total_area_m2=float(area.sum()),
    )


def evaluate_mesh(mesh_path, frames: list, **kw) -> Dict:
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(str(mesh_path))
    return dict(topology=topology(m), depth_fidelity=depth_fidelity(m, frames, **kw))
