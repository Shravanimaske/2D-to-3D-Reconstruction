"""
modules/pose_diagnostics.py   (Phase 1, new module)

Measures how well camera poses + intrinsics + depth agree with each other,
WITHOUT any ground truth.

Method (multiview depth agreement)
----------------------------------
For a frame pair (i, j): back-project the depth of frame i into the world with
pose i, project those 3-D points into frame j with pose j, and compare the
projected depth z with the depth *measured* in frame j at that pixel.
If poses, intrinsics, axis convention and depth scale are all consistent,
|measured - projected| is small (sensor noise); wrong intrinsics, a wrong
axis convention, pose drift or a depth/pose time offset make it large.

What it IS:   a proxy for geometric self-consistency of the input data.
What it is NOT: absolute accuracy.  It cannot detect an error that is
              consistent across all frames (e.g. a global scale error that
              affects depth and pose translation equally), and
              pairs with little overlap are skipped, so large-baseline
              numbers describe only the overlapping region.

Only pixels with the highest confidence value (when .conf files exist) are
used, so sensor-flagged garbage does not pollute the statistic.

Typical use
-----------
    diag = PoseDiagnostics(frames, depth_scale=1.0)
    before = diag.run(label="input")
    ... modify poses ...
    after  = diag.run(label="aligned")
    keep   = PoseDiagnostics.alignment_is_not_worse(before, after)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)

# camera axis flip: OpenGL-style camera (y up, -z forward) -> OpenCV (y down, +z)
_FLIP = np.diag([1., -1., -1., 1.])


def _read_depth(path: Path) -> Optional[np.ndarray]:
    path = Path(path)
    try:
        if path.suffix.lower() == ".npy":
            return np.load(str(path)).astype(np.float32)
        if path.suffix.lower() == ".bin":
            raw = np.fromfile(str(path), dtype=np.float32)
            for h, w in ((192, 256), (480, 640), (360, 480), (720, 960)):
                if raw.size == h * w:
                    return raw.reshape(h, w)
            return None
        from src.modules.exr_reader import read_exr_depth
        return read_exr_depth(path).astype(np.float32)
    except Exception as e:                                   # noqa: BLE001
        log.debug(f"  depth read failed {path.name}: {e}")
        return None


class PoseDiagnostics:

    def __init__(
        self,
        frames: list,
        depth_scale: float = 1.0,
        separations: Sequence[int] = (12, 60, 150),
        pairs_per_separation: int = 16,
        tol_m: float = 0.05,
        min_depth: float = 0.1,
        pixel_stride: int = 2,
        use_flip: bool = True,
        output_dir: Optional[Path] = None,
    ):
        self.frames       = frames
        self.depth_scale  = depth_scale
        self.separations  = tuple(int(s) for s in separations)
        self.n_pairs      = int(pairs_per_separation)
        self.tol          = float(tol_m)
        self.min_depth    = float(min_depth)
        self.stride       = max(1, int(pixel_stride))
        self.flip         = _FLIP if use_flip else np.eye(4)
        self.output_dir   = Path(output_dir) if output_dir else None
        self._cache: Dict[int, tuple] = {}

    # -- data access ---------------------------------------------------------

    def _depth_and_conf(self, k: int):
        """(depth_m, high_conf_mask) for frame index k of self.frames."""
        if k in self._cache:
            return self._cache[k]
        fr = self.frames[k]
        d = _read_depth(fr.depth_path)
        out = (None, None)
        if d is not None:
            # per-frame scale from the pose file wins; otherwise the global one
            fs = float(getattr(fr, "depth_scale", 1.0) or 1.0)
            d = d * (fs if fs != 1.0 else float(self.depth_scale))
            d[~np.isfinite(d)] = 0.0
            mask = d > self.min_depth
            cp = getattr(fr, "conf_path", None)
            if cp is not None and Path(cp).is_file():
                c = np.fromfile(str(cp), dtype=np.uint8)
                if c.size == d.size:
                    c = c.reshape(d.shape)
                    mask &= (c == c.max())
            out = (d, mask)
        self._cache[k] = out
        return out

    def _K(self, k: int, shape):
        h, w = shape
        K = self.frames[k].intrinsics.K_for(w, h)
        return K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # -- trajectory ------------------------------------------------------------

    def trajectory_stats(self) -> Dict:
        c2w = np.array([f.c2w for f in self.frames])
        pos = c2w[:, :3, 3]
        steps = np.linalg.norm(np.diff(pos, axis=0), axis=1) if len(pos) > 1 else np.zeros(1)
        fwd = -c2w[:, :3, 2]
        sweep = np.degrees(np.arccos(np.clip(fwd @ fwd[0], -1, 1))).max()
        # angular step between consecutive frames
        Rrel = np.einsum("nji,njk->nik", c2w[:-1, :3, :3], c2w[1:, :3, :3])
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        med = float(np.median(steps)) if len(steps) else 0.0
        return dict(
            n_frames=len(self.frames),
            span_m=float(np.linalg.norm(pos.max(0) - pos.min(0))),
            path_length_m=float(steps.sum()),
            median_step_m=med,
            max_step_m=float(steps.max()),
            n_translation_jumps=int((steps > max(10 * med, 0.05)).sum()),
            max_view_direction_change_deg=float(sweep),
            max_angular_step_deg=float(ang.max()) if len(ang) else 0.0,
            n_rotation_jumps=int((ang > 10.0).sum()) if len(ang) else 0,
        )

    # -- multiview agreement ------------------------------------------------------

    def _pair_residuals(self, i: int, j: int):
        di, mi = self._depth_and_conf(i)
        dj, mj = self._depth_and_conf(j)
        if di is None or dj is None:
            return None
        H, W = di.shape
        if dj.shape != di.shape:
            return None
        s = self.stride
        v, u = np.mgrid[0:H:s, 0:W:s]
        z = di[v, u]
        ok0 = mi[v, u]
        fx, fy, cx, cy = self._K(i, di.shape)
        x = (u + 0.5 - cx) / fx * z
        y = (v + 0.5 - cy) / fy * z
        Pc = np.stack([x, y, z, np.ones_like(z)], -1)[ok0].reshape(-1, 4)
        Ti = self.frames[i].c2w @ self.flip
        Tj = self.frames[j].c2w @ self.flip
        Pj = (Pc @ (np.linalg.inv(Tj) @ Ti).T)[:, :3]
        zj = Pj[:, 2]
        fx, fy, cx, cy = self._K(j, dj.shape)
        ok = zj > self.min_depth
        zs = np.where(ok, zj, 1.0)
        uj = fx * Pj[:, 0] / zs + cx - 0.5
        vj = fy * Pj[:, 1] / zs + cy - 0.5
        ui = np.round(uj).astype(int)
        vi = np.round(vj).astype(int)
        ok &= (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if ok.sum() < 300:
            return None
        meas = dj[vi[ok], ui[ok]]
        valid = mj[vi[ok], ui[ok]]
        dz = (meas - zj[ok])[valid]
        # gross outliers (occlusions / disocclusions) are not pose errors
        dz = dz[np.abs(dz) < 0.5]
        if dz.size < 300:
            return None
        return np.abs(dz), float(ok.mean())

    def multiview_agreement(self) -> Dict:
        n = len(self.frames)
        result: Dict[str, Dict] = {}
        for sep in self.separations:
            if sep >= n - 1:
                continue
            starts = np.unique(np.linspace(0, n - 1 - sep, self.n_pairs).astype(int))
            res, ov = [], []
            for i in starts:
                r = self._pair_residuals(int(i), int(i) + sep)
                if r is not None:
                    res.append(r[0]); ov.append(r[1])
            if not res:
                result[str(sep)] = dict(n_pairs=0)
                continue
            a = np.concatenate(res)
            result[str(sep)] = dict(
                n_pairs=len(res),
                n_samples=int(a.size),
                median_abs_dz_m=float(np.median(a)),
                p90_abs_dz_m=float(np.percentile(a, 90)),
                frac_within_tol=float((a < self.tol).mean()),
                mean_overlap=float(np.mean(ov)),
            )
        return result

    # -- public --------------------------------------------------------------------

    def run(self, label: str = "poses", save: bool = True) -> Dict:
        rep = dict(
            label=label,
            tolerance_m=self.tol,
            trajectory=self.trajectory_stats(),
            multiview_agreement=self.multiview_agreement(),
            notes=("Self-consistency proxy, not absolute accuracy. "
                   "Pixels limited to highest-confidence depth when .conf exists."),
        )
        vals = [v["frac_within_tol"] for v in rep["multiview_agreement"].values()
                if v.get("n_pairs", 0) > 0]
        rep["mean_frac_within_tol"] = float(np.mean(vals)) if vals else None
        self._log(rep)
        if save and self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            p = self.output_dir / f"pose_report_{label}.json"
            p.write_text(json.dumps(rep, indent=2))
            log.info(f"  pose report -> {p}")
        return rep

    @staticmethod
    def alignment_is_not_worse(before: Dict, after: Dict, tolerance: float = 0.005) -> bool:
        """
        True if the mean fraction of depth samples agreeing within tolerance did
        not drop by more than `tolerance` (absolute) after modifying the poses.
        If a metric is unavailable we do NOT claim an improvement -> False.
        """
        b, a = before.get("mean_frac_within_tol"), after.get("mean_frac_within_tol")
        if b is None or a is None:
            return False
        return a >= b - tolerance

    def _log(self, rep: Dict):
        t = rep["trajectory"]
        log.info(f"  [{rep['label']}] trajectory: span={t['span_m']:.2f}m  "
                 f"path={t['path_length_m']:.2f}m  view sweep="
                 f"{t['max_view_direction_change_deg']:.0f} deg  "
                 f"jumps(t/r)={t['n_translation_jumps']}/{t['n_rotation_jumps']}")
        for sep, m in rep["multiview_agreement"].items():
            if m.get("n_pairs", 0) == 0:
                log.info(f"  [{rep['label']}] sep={sep:>3}: not enough overlap")
                continue
            log.info(f"  [{rep['label']}] sep={sep:>3}: median|dz|="
                     f"{m['median_abs_dz_m']*100:.1f}cm  p90={m['p90_abs_dz_m']*100:.1f}cm  "
                     f"within {self.tol*100:.0f}cm={m['frac_within_tol']*100:.1f}%  "
                     f"(pairs={m['n_pairs']})")
        warn = []
        if t["n_translation_jumps"] or t["n_rotation_jumps"]:
            warn.append("trajectory contains jumps (possible tracking loss)")
        m12 = rep["multiview_agreement"].get(str(self.separations[0]), {})
        if m12.get("n_pairs", 0) and m12["frac_within_tol"] < 0.6:
            warn.append("poor short-baseline agreement: check intrinsics / axis "
                        "convention / depth scale")
        rep["warnings"] = warn
        for w in warn:
            log.warning(f"  [{rep['label']}] {w}")
