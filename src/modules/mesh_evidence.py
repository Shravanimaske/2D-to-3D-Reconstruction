"""
modules/mesh_evidence.py   (Phase 3, new)

Per-face evidence from the captured data.  Every triangle of a mesh is checked
against the cameras and the sensor depth with exact ray casting:

  pixel pass      one ray per depth pixel from each sampled camera.
                  seen      the face is the first surface hit by >= 1 ray
                  support   ... and the hit distance agrees with the measured
                            depth (|mesh - sensor| <= tol): the face lies on a
                            surface the sensor actually measured
                  front     the face is in front of the measured surface
                            (floating / bridging geometry)
                  behind    the face is behind the measured surface
  visibility pass a ray from each camera to each face centroid.  A camera
                  "sees" the face if the first hit is that face (inside the
                  frustum and facing the camera).  Gives n_visible and the best
                  truly-visible camera per face (the same score the texture
                  baker uses, but with the occlusion test it lacks).

Faces that no camera sees, and faces that the sensor never supports, cannot be
textured correctly by anyone: they are hidden slivers, back-sides or geometry
extrapolated into unobserved space.  filter_by_evidence() removes them instead
of fabricating colour for them, and reports exactly what was removed.

LIMITS: evidence comes from the same captured data used to build the mesh, so a
surface the scan never observed is reported as unobserved, not "verified".
Faces smaller than a depth pixel can be missed by pixel rays; the visibility
pass (centroid rays) is used to protect them.
"""
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)
_FLIP = np.diag([1., -1., -1., 1.])


def _read_depth(path: Path) -> Optional[np.ndarray]:
    try:
        b = Path(path).with_suffix(".bin")
        if b.is_file():
            raw = np.fromfile(str(b), dtype=np.float32)
            for h, w in ((192, 256), (480, 640), (360, 480)):
                if raw.size == h * w:
                    return raw.reshape(h, w)
        from src.modules.exr_reader import read_exr_depth
        return read_exr_depth(Path(path)).astype(np.float32)
    except Exception:                                        # noqa: BLE001
        return None


def compute_evidence(mesh, frames: list, step: int = 2, pixel_stride: int = 1,
                     tol_m: float = 0.05, min_conf: int = 1,
                     min_facing: float = 0.15, centroid_tol_m: float = 0.03,
                     min_depth: float = 0.3, max_depth: float = 9.0) -> Dict:
    """
    mesh: legacy open3d TriangleMesh.  Uses every `step`-th frame.
    Returns arrays of length n_faces: seen, support, front, behind, n_visible,
    best_cam (index into `frames`, -1 = none), plus the face areas/normals.
    """
    import open3d as o3d
    V = np.asarray(mesh.vertices)
    Fc = np.asarray(mesh.triangles)
    nF = len(Fc)
    v0, v1, v2 = V[Fc[:, 0]], V[Fc[:, 1]], V[Fc[:, 2]]
    cen = ((v0 + v1 + v2) / 3.0).astype(np.float32)
    nrm = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm = (nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)).astype(np.float32)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    ev = {k: np.zeros(nF, np.int32) for k in ("seen", "support", "front", "behind", "n_visible")}
    best_cam = np.full(nF, -1, np.int32)
    best_sc = np.full(nF, -np.inf, np.float32)

    for fi in range(0, len(frames), max(1, step)):
        fr = frames[fi]
        T = (np.asarray(fr.c2w, float) @ _FLIP)
        R, o = T[:3, :3], T[:3, 3]

        # ---- pixel pass -------------------------------------------------------
        d = _read_depth(fr.depth_path)
        if d is not None:
            h, w = d.shape
            valid = (d > min_depth) & (d < max_depth) & np.isfinite(d)
            cp = getattr(fr, "conf_path", None)
            if min_conf > 0 and cp is not None and Path(cp).is_file():
                c = np.fromfile(str(cp), dtype=np.uint8)
                if c.size == d.size:
                    valid &= (c.reshape(d.shape) >= min_conf)
            K = fr.intrinsics.K_for(w, h)
            vv, uu = np.mgrid[0:h:pixel_stride, 0:w:pixel_stride]
            dc = np.stack([(uu + .5 - K[0, 2]) / K[0, 0], (vv + .5 - K[1, 2]) / K[1, 1],
                           np.ones_like(uu, float)], -1).reshape(-1, 3)
            dw = dc @ R.T
            dw /= np.linalg.norm(dw, axis=1, keepdims=True)
            rays = np.concatenate([np.tile(o, (len(dw), 1)), dw], 1).astype(np.float32)
            ans = scene.cast_rays(o3d.core.Tensor(rays))
            t = ans["t_hit"].numpy()
            pid = ans["primitive_ids"].numpy().astype(np.int64)
            hit = np.isfinite(t)
            zm = t * (dc[:, 2] / np.linalg.norm(dc, axis=1))
            zs = d[vv, uu].reshape(-1)
            ok = valid[vv, uu].reshape(-1)
            dd = zm - zs
            ev["seen"] += np.bincount(pid[hit], minlength=nF).astype(np.int32)
            ev["support"] += np.bincount(pid[hit & ok & (np.abs(dd) <= tol_m)], minlength=nF).astype(np.int32)
            ev["front"] += np.bincount(pid[hit & ok & (dd < -tol_m)], minlength=nF).astype(np.int32)
            ev["behind"] += np.bincount(pid[hit & ok & (dd > tol_m)], minlength=nF).astype(np.int32)

        # ---- visibility pass (centroid rays) ----------------------------------
        K = fr.intrinsics.K()
        w2c = np.linalg.inv(T)
        cam = (np.hstack([cen, np.ones((nF, 1), np.float32)]) @ w2c.T)[:, :3]
        Z = cam[:, 2]
        iw, ih = fr.intrinsics.width, fr.intrinsics.height
        with np.errstate(all="ignore"):
            u = K[0, 0] * cam[:, 0] / Z + K[0, 2]
            v = K[1, 1] * cam[:, 1] / Z + K[1, 2]
        to = o - cen
        dist = np.linalg.norm(to, axis=1) + 1e-9
        facing = np.einsum("ij,ij->i", nrm, to / dist[:, None])
        cand = np.where((Z > 0.05) & (u >= 0) & (u < iw) & (v >= 0) & (v < ih)
                        & (facing > min_facing))[0]
        if len(cand):
            dirs = -(to[cand] / dist[cand, None])
            r2 = np.concatenate([np.tile(o, (len(cand), 1)), dirs], 1).astype(np.float32)
            a2 = scene.cast_rays(o3d.core.Tensor(r2))
            t2 = a2["t_hit"].numpy()
            p2 = a2["primitive_ids"].numpy().astype(np.int64)
            sees = (p2 == cand) | (np.abs(t2 - dist[cand]) < centroid_tol_m)
            c2 = cand[sees]
            ev["n_visible"][c2] += 1
            sc = (facing[c2] ** 2) / (dist[c2] * 0.1 + 1.0)      # same score as the baker
            better = sc > best_sc[c2]
            best_sc[c2[better]] = sc[better]
            best_cam[c2[better]] = fi

    ev["best_cam"] = best_cam
    ev["area"] = area.astype(np.float32)
    ev["normal"] = nrm
    return ev


def evidence_summary(ev: Dict) -> Dict:
    nF = len(ev["seen"])
    A = ev["area"].sum() + 1e-12
    vis = ev["n_visible"] > 0
    seen = ev["seen"] > 0
    sup = ev["support"] > 0
    observed = vis | seen
    return dict(
        n_faces=int(nF),
        no_camera_sees_face_pct=float((~observed).mean() * 100),
        no_camera_sees_face_area_pct=float(ev["area"][~observed].sum() / A * 100),
        never_sensor_consistent_pct=float((~sup).mean() * 100),
        sensor_supported_pct=float(sup.mean() * 100),
        sensor_supported_area_pct=float(ev["area"][sup].sum() / A * 100),
    )


def filter_by_evidence(mesh, ev: Dict, require_visible: bool = True,
                       require_support: bool = False, min_support: int = 1,
                       conflict_ratio: Optional[float] = None, min_conflict_votes: int = 3,
                       min_component_fraction: float = 0.005):
    """
    Remove faces the captured data does not back.  A face is kept if
      (it is visible to a camera or hit by a pixel ray)   [require_visible]   DEFAULT ON
      and it has >= min_support sensor-consistent votes    [require_support]   default off
      and it is not mostly contradicted by the sensor      [conflict_ratio]    default off
    Tiny connected components left behind are dropped.

    Measured on Full_Refuge_data (see Phase 3 report): visibility alone removes
    25 % of the faces / 6 % of the area with NO change in coverage, median error
    or floaters.  Requiring sensor support (+-5 cm) or applying the contradiction
    rule removed 2/3 of the mesh and cost 6-10 points of coverage, because normal
    sensor noise at long range exceeds the tolerance - so both are opt-in.
    Returns (new_mesh, stats).
    """
    import open3d as o3d
    nF = len(ev["seen"])
    keep = np.ones(nF, bool)
    if require_visible:
        keep &= (ev["n_visible"] > 0) | (ev["seen"] > 0)
    if require_support:
        keep &= ev["support"] >= min_support
    if conflict_ratio is not None:
        votes = ev["support"] + ev["front"] + ev["behind"]
        against = ev["front"] + ev["behind"]
        keep &= ~((votes >= min_conflict_votes) & (against > conflict_ratio * np.maximum(votes, 1)))

    m = o3d.geometry.TriangleMesh(mesh)
    m.remove_triangles_by_mask(~keep)
    m.remove_unreferenced_vertices()
    n_after_filter = len(m.triangles)
    if n_after_filter and min_component_fraction > 0:
        lab, cnt, _ = m.cluster_connected_triangles()
        lab = np.asarray(lab); cnt = np.asarray(cnt)
        small = cnt < max(50, min_component_fraction * len(lab))
        m.remove_triangles_by_mask(small[lab])
        m.remove_unreferenced_vertices()
    stats = dict(faces_before=int(nF), removed_by_evidence=int((~keep).sum()),
                 removed_small_components=int(n_after_filter - len(m.triangles)),
                 faces_after=int(len(m.triangles)),
                 area_removed_pct=float(ev["area"][~keep].sum() / (ev["area"].sum() + 1e-12) * 100))
    log.info(f"  evidence filter: {stats['faces_before']:,} -> {stats['faces_after']:,} faces "
             f"({stats['removed_by_evidence']:,} unsupported, "
             f"{stats['removed_small_components']:,} in tiny components; "
             f"{stats['area_removed_pct']:.1f}% of the surface area)")
    return m, stats
