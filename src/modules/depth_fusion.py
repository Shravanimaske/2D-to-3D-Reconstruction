"""
modules/depth_fusion.py  ── v8  "coordinate-consistent"  (+ Phase 2 changes)

[P2-1] inpainting no longer undoes the border / sky / flying-pixel masks: a
       masked pixel is only refilled if the RAW sensor had no depth there.
[P2-2] optional depth-confidence masking (min_confidence; uses frame.conf_path).
[P2-3] sdf_trunc_m (absolute metres) overrides voxel*multiplier; the multiplier
       clamp is widened from 6 to 16 (a 6.7 request used to be silently cut to 6,
       i.e. a 45 cm truncation band at a 7.5 cm voxel).
[P2-4] the fallback `cx = cy = a, b` tuple bug is fixed.
[P2-5] _fill_mesh_holes no longer calls the non-existent legacy
       TriangleMesh.fill_holes (a silent no-op); smoothing iterations configurable.
[P2-6] last_stats records frames fused / raw points for the quality gates.

CRITICAL FIX from v7:
  Confirmed that np.linalg.inv(fr.c2w @ _FLIP) is CORRECT for TSDF
  integration when c2w is in OpenGL convention (Record3D/ARKit).

Additional fixes:
  - Removed overly aggressive flying pixel suppression (was removing valid edges)
  - Made inpainting optional and conservative (was creating false geometry)
  - Simplified dynamic weighting (image-space comparison doesn't work for moving cameras)
  - Added depth/RGB resolution mismatch handling for EXR vs BIN depth
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

_FLIP = np.diag([1., -1., -1., 1.])


def _load_exr(path: Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(str(path)).astype(np.float32)
    try:
        import OpenEXR, Imath
        f   = OpenEXR.InputFile(str(path))
        dw  = f.header()["dataWindow"]
        W, H = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1
        ch  = next(iter(f.header()["channels"].keys()))
        raw = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
        return np.frombuffer(raw, dtype=np.float32).reshape(H, W)
    except (ImportError, OSError):
        pass
    try:
        import cv2
        d = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if d is not None:
            return d.astype(np.float32)
    except Exception:
        pass
    from src.modules.exr_reader import read_exr_depth
    return read_exr_depth(path).astype(np.float32)


def _suppress_flying_pixels(depth: np.ndarray,
                             jump_thresh_m: float = 0.15,
                             min_bad_neighbours: int = 6) -> np.ndarray:
    """
    CONSERVATIVE flying pixel removal (v8: raised min_bad_neighbours from 5→6).
    Only removes pixels that differ from MOST neighbours, not just some.
    This preserves valid thin structures and edges .
    """
    if depth is None or depth.size == 0:
        return depth

    H, W = depth.shape
    out  = depth.copy()
    valid = depth > 0.05

    bad_count = np.zeros((H, W), dtype=np.int32)

    for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]:
        shifted = np.zeros_like(depth)
        sy = slice(max(0, dy), H + min(0, dy))
        sx = slice(max(0, dx), W + min(0, dx))
        ty = slice(max(0, -dy), H + min(0, -dy))
        tx = slice(max(0, -dx), W + min(0, -dx))
        shifted[ty, tx] = depth[sy, sx]

        neighbour_valid = shifted > 0.05
        both_valid = valid & neighbour_valid
        big_jump = both_valid & (np.abs(depth - shifted) > jump_thresh_m)
        bad_count += big_jump.astype(np.int32)

    # ★ RAISED: require 6/8 bad neighbours (was 5) — more conservative
    flying = valid & (bad_count >= min_bad_neighbours)
    out[flying] = 0.0

    n_removed = int(flying.sum())
    if n_removed > 0:
        log.debug(f"  Flying pixels removed: {n_removed}")

    return out


def _suppress_sky_depth(depth: np.ndarray,
                         rgb: Optional[np.ndarray],
                         sky_bright_thresh: int = 230,
                         sky_depth_min_m: float = 2.0) -> np.ndarray:
    if rgb is None or rgb.shape[:2] != depth.shape:
        return depth

    overexposed = (rgb[:, :, 0].astype(np.int32) > sky_bright_thresh) & \
                  (rgb[:, :, 1].astype(np.int32) > sky_bright_thresh) & \
                  (rgb[:, :, 2].astype(np.int32) > sky_bright_thresh)

    sky_mask = overexposed & (depth > sky_depth_min_m)

    out = depth.copy()
    out[sky_mask] = 0.0
    return out


def _mask_depth_borders(depth: np.ndarray, border_frac: float = 0.02) -> np.ndarray:
    H, W = depth.shape
    bh = max(2, int(H * border_frac))
    bw = max(2, int(W * border_frac))
    out = depth.copy()
    out[:bh,  :] = 0.0
    out[-bh:, :] = 0.0
    out[:,  :bw] = 0.0
    out[:, -bw:] = 0.0
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Depth inpainting — CONSERVATIVE in v8
# ══════════════════════════════════════════════════════════════════════════════

def _inpaint_depth(depth: np.ndarray, max_hole_px: int = 15) -> np.ndarray:
    """
    Fill zero-depth holes with INPAINTING.

    v8 CHANGE: Reduced radius from 20→15 and added safety check.
    Inpainting can create false geometry if holes are too large.
    Only inpaint if the hole fraction is reasonable (<40% of image).
    """
    hole_mask = (depth <= 0).astype(np.uint8)
    if hole_mask.sum() == 0:
        return depth

    # Safety: don't inpaint if too much is missing (would create fantasy geometry)
    hole_fraction = float(hole_mask.sum()) / hole_mask.size
    if hole_fraction > 0.40:
        log.debug(f"  Inpaint skipped: hole fraction {hole_fraction:.1%} too large")
        return depth

    try:
        import cv2
        valid = depth > 0
        if not valid.any():
            return depth
        d_max  = float(depth[valid].max()) + 1e-6
        d_u16  = (np.clip(depth / d_max, 0, 1) * 65535).astype(np.uint16)
        filled = cv2.inpaint(d_u16, hole_mask, inpaintRadius=max_hole_px,
                             flags=cv2.INPAINT_NS)
        result = filled.astype(np.float32) / 65535.0 * d_max
        out = depth.copy()
        out[hole_mask.astype(bool)] = result[hole_mask.astype(bool)]
        return out
    except ImportError:
        pass

    # scipy fallback: nearest-neighbour
    try:
        from scipy.ndimage import distance_transform_edt
        out   = depth.copy()
        valid = depth > 0
        _, indices = distance_transform_edt(~valid, return_indices=True)
        out[~valid] = depth[tuple(indices[:, ~valid])]
        return out
    except ImportError:
        pass

    return depth


# ══════════════════════════════════════════════════════════════════════════════
# Depth smoothing with tighter sigma
# ══════════════════════════════════════════════════════════════════════════════

def _smooth_depth_bilateral(depth: np.ndarray,
                              d: int = 7,
                              sigma_color: float = 0.02,
                              sigma_space: float = 3.0) -> np.ndarray:
    try:
        import cv2
        return cv2.bilateralFilter(depth.astype(np.float32), d,
                                   sigma_color, sigma_space)
    except ImportError:
        try:
            from scipy.ndimage import gaussian_filter
            return gaussian_filter(depth, sigma=0.8).astype(np.float32)
        except ImportError:
            return depth


# ══════════════════════════════════════════════════════════════════════════════
# Depth range validation — hard clamp with outlier rejection
# ══════════════════════════════════════════════════════════════════════════════

def _validate_depth_range(depth: np.ndarray,
                           min_d: float,
                           max_d: float) -> np.ndarray:
    """
    Beyond simple min/max clamp: also detect implausible local spikes.
    A pixel that is 3× deeper than its local neighbourhood median is
    likely a LiDAR multipath reflection — zero it.
    """
    out = np.where((depth >= min_d) & (depth <= max_d), depth, 0.0).astype(np.float32)

    try:
        import cv2

        valid = out > 0
        if valid.sum() < 100:
            return out

        d_max = float(out[valid].max()) + 1e-6
        out_u8 = np.clip(out / d_max, 0, 1)
        out_u8 = (out_u8 * 255).astype(np.uint8)

        median_u8 = cv2.medianBlur(out_u8, 15)

        median = median_u8.astype(np.float32) / 255.0 * d_max

        spike = valid & (median > 0) & (out > median * 2.5)
        out[spike] = 0.0
    except ImportError:
        pass

    return out


# ══════════════════════════════════════════════════════════════════════════════
# Planar snap (unchanged from v7, proven to work)
# ══════════════════════════════════════════════════════════════════════════════

def _snap_to_planes(pcd,
                    plane_distance_thresh: float = 0.015,
                    n_planes: int = 6,
                    snap_dist: float = 0.05,
                    remove_dist: float = 0.15,
                    min_inlier_fraction: float = 0.02):
    try:
        import open3d as o3d
    except ImportError:
        return pcd

    pts     = np.asarray(pcd.points).copy()
    n_total = len(pts)
    if n_total < 500:
        return pcd

    remaining_idx = np.arange(n_total)
    planes        = []

    for _ in range(n_planes):
        if len(remaining_idx) < 200:
            break
        sub = pcd.select_by_index(remaining_idx)
        try:
            plane_model, inliers = sub.segment_plane(
                distance_threshold=plane_distance_thresh,
                ransac_n=3, num_iterations=1000)
        except Exception:
            break
        if len(inliers) < n_total * min_inlier_fraction:
            break
        planes.append(plane_model)
        mask = np.ones(len(remaining_idx), dtype=bool)
        mask[inliers] = False
        remaining_idx = remaining_idx[mask]

    if not planes:
        return pcd

    normals    = np.array([[p[0], p[1], p[2]] for p in planes], dtype=np.float32)
    ds         = np.array([p[3] for p in planes], dtype=np.float32)
    norm_len   = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    unit_norms = normals / norm_len

    signed_dists = (pts @ normals.T + ds[None]) / (norm_len.T + 1e-9)
    abs_dists    = np.abs(signed_dists)
    nearest_p    = abs_dists.argmin(axis=1)
    min_dist     = abs_dists.min(axis=1)

    snap_mask = min_dist < snap_dist
    for i in range(len(planes)):
        pm = snap_mask & (nearest_p == i)
        if pm.any():
            pts[pm] -= signed_dists[pm, i][:, None] * unit_norms[i]

    ghost_mask = np.zeros(n_total, dtype=bool)
    for i in range(len(planes)):
        dists_i = signed_dists[:, i]
        on_plane = abs_dists[:, i] < snap_dist
        if on_plane.sum() == 0:
            continue
        pos_side = np.sum(dists_i[on_plane] > 0)
        neg_side = np.sum(dists_i[on_plane] < 0)
        if pos_side > neg_side:
            ghost_candidates = (dists_i < -snap_dist) & (dists_i > -1.0)
        else:
            ghost_candidates = (dists_i > snap_dist) & (dists_i < 1.0)
        ghost_mask |= ghost_candidates

    far_mask = min_dist > remove_dist

    # ── NEW: protect non-planar (curved/thin) structures like columns ──────
    # Estimate local curvature via normal-consistency in a small neighborhood.
    # High curvature = not a wall/floor/ceiling = likely a column/pillar/pipe.
    # Never remove far_mask points there — only planar-snap logic should touch flats.
    pcd_tmp = o3d.geometry.PointCloud()
    pcd_tmp.points = o3d.utility.Vector3dVector(pts)
    pcd_tmp.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=snap_dist * 2, max_nn=30))
    normals_pc = np.asarray(pcd_tmp.normals)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        _, nbr_idx = tree.query(pts, k=8, workers=-1)
        nbr_normals = normals_pc[nbr_idx]                      # (N,8,3)
        mean_n = nbr_normals.mean(axis=1)
        mean_n /= (np.linalg.norm(mean_n, axis=1, keepdims=True) + 1e-9)
        cos_sim = np.einsum('ij,ij->i', normals_pc, mean_n)
        is_curved = cos_sim < 0.85   # normals disagree with neighbors -> curved surface
    except ImportError:
        is_curved = np.zeros(n_total, dtype=bool)

    far_mask = far_mask & ~is_curved   # ★ don't strip curved geometry just for being "far from a plane"

    try:
        labels = np.array(pcd.cluster_dbscan(eps=0.06, min_points=25,
                                              print_progress=False))
        valid_l = labels[labels >= 0]
        main_c  = int(np.bincount(valid_l).argmax()) if len(valid_l) > 0 else -1
        in_main = labels == main_c
    except Exception:
        in_main = np.ones(n_total, dtype=bool)

    remove = (far_mask & ~in_main) | ghost_mask
    keep   = ~remove

    import open3d as o3d
    pcd_out = pcd.select_by_index(np.where(keep)[0])
    pcd_out.points = o3d.utility.Vector3dVector(pts[keep])
    if pcd.has_colors():
        pcd_out.colors = o3d.utility.Vector3dVector(
            np.asarray(pcd.colors)[keep])
    log.info(f"  Planar snap: removed {int(remove.sum()):,} pts "
             f"(protected {int(is_curved.sum()):,} curved pts)")
    return pcd_out


# ══════════════════════════════════════════════════════════════════════════════
# Mesh hole filling
# ══════════════════════════════════════════════════════════════════════════════

def _merge_inpaint(depth: np.ndarray, filled: np.ndarray, raw: np.ndarray,
                   keep_masks: bool = True) -> np.ndarray:
    """
    [P2-1] Combine the inpainted depth with the masked depth.
    keep_masks=True : only pixels that had NO valid depth in the raw sensor data
                      are taken from `filled`; pixels masked on purpose stay 0.
    keep_masks=False: legacy behaviour (everything that is zero is refilled).
    """
    if not keep_masks:
        return filled
    sensor_hole = ~(np.isfinite(raw) & (raw > 0.05))
    return np.where(sensor_hole & (depth <= 0.05), filled, depth).astype(np.float32)


def _fill_mesh_holes(mesh, max_hole_size: int = 50, smooth_iterations: int = 2):
    """
    [P2-5] The legacy open3d TriangleMesh has no fill_holes(); the old call was a
    silent no-op.  We only call it if it exists, then smooth lightly.
    Holes are deliberately NOT invented here: unobserved regions are reported
    by the quality stage rather than capped with fabricated geometry.
    """
    try:
        import open3d as o3d  # noqa: F401
    except ImportError:
        return mesh

    if hasattr(mesh, "fill_holes"):
        n_before = len(mesh.triangles)
        try:
            filled = mesh.fill_holes(hole_size=max_hole_size)
            if hasattr(filled, "triangles") and len(filled.triangles) > n_before:
                mesh = filled
                log.info(f"  Hole fill: {n_before:,} -> {len(mesh.triangles):,} faces")
        except Exception as e:
            log.debug(f"  fill_holes skipped: {e}")

    if smooth_iterations > 0:
        try:
            mesh = mesh.filter_smooth_laplacian(
                number_of_iterations=int(smooth_iterations), lambda_filter=0.5)
            mesh.compute_vertex_normals()
        except Exception:
            pass
    return mesh


# ══════════════════════════════════════════════════════════════════════════════
# Confetti removal (two-pass)
# ══════════════════════════════════════════════════════════════════════════════

def _remove_confetti(pcd, voxel_size: float):
    if len(pcd.points) < 50:
        return pcd
    n0 = len(pcd.points)
    # Pass 1: radius filter — loosened nb_points 10→6 so grazing-angle/
    # sparse-but-real column surfaces survive; SOR pass 2 still catches true noise
    pcd, _ = pcd.remove_radius_outlier(nb_points=6, radius=voxel_size * 4.5)
    log.info(f"  Confetti P1 (radius): {n0:,} → {len(pcd.points):,}")
    if len(pcd.points) < 50:
        return pcd
    n1 = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    log.info(f"  Confetti P2 (SOR):    {n1:,} → {len(pcd.points):,}")
    return pcd


# ══════════════════════════════════════════════════════════════════════════════
# Main class
# ══════════════════════════════════════════════════════════════════════════════

class DepthFusion:

    def __init__(
        self,
        output_dir,
        depth_scale: float = 1.0,
        min_depth: float = 0.15,
        max_depth: float = 3.5,
        voxel_size: float = 0.025,
        frame_skip: int = 1,
        stop_flag: Optional[Callable[[], bool]] = None,
        scene_mode: bool = True,
        # Smoothing
        use_bilateral: bool = True,
        # Glass weight
        use_glass_weight: bool = False,
        glass_bright_threshold: int = 220,
        glass_min_weight: float = 0.20,
        # Border
        use_border_mask: bool = True,
        border_frac: float = 0.020,
        # Depth inpainting
        use_inpaint: bool = True,
        inpaint_radius_px: int = 15,
        # Soft multi-view consistency
        use_soft_consistency: bool = False,
        consistency_window: int = 3,
        consistency_threshold_m: float = 0.08,
        consistency_min_weight: float = 0.25,
        # Dynamic object weighting
        use_dynamic_weight: bool = True,
        dynamic_motion_thresh_m: float = 0.08,
        dynamic_min_weight: float = 0.0,
        # Planar snap
        use_planar_snap: bool = True,
        planar_ransac_thresh: float = 0.015,
        n_planes: int = 6,
        planar_snap_dist: float = 0.05,
        planar_remove_dist: float = 0.15,
        # Mesh hole filling
        use_hole_fill: bool = True,
        max_hole_size: int = 50,
        # TSDF
        sdf_trunc_multiplier: float = 3.0,
        # v8 new parameters
        use_flying_pixel_suppress: bool = True,
        flying_pixel_jump_thresh_m: float = 0.15,
        use_sky_suppress: bool = True,
        sky_bright_thresh: int = 230,
        sky_depth_min_m: float = 2.0,
        use_spike_filter: bool = True,
        # Legacy compatibility
        use_consistency: bool = False,
        consistency_threshold: float = 0.08,
        use_foreground_mask: bool = False,
        foreground_depth_band: float = 0.35,
        use_grabcut_refine: bool = False,
        grabcut_downscale: int = 2,
        use_glass_mask: bool = False,
        use_multiview_consistency: bool = False,
        multiview_window: int = 3,
        multiview_threshold_m: float = 0.08,
        use_planar_prior: bool = False,
        floating_dist_thresh: float = 0.12,
        min_view_count: int = 2,
        # Phase 2 additions (backward compatible defaults)
        min_confidence: int = 0,
        sdf_trunc_m: Optional[float] = None,
        mesh_smooth_iterations: int = 2,
        inpaint_keep_masks: bool = True,
    ):
        self.output_dir                  = Path(output_dir)
        self.depth_scale                 = depth_scale
        self.min_depth                   = min_depth
        self.max_depth                   = max_depth
        self.voxel_size                  = voxel_size
        self.frame_skip                  = frame_skip
        self.stop_flag                   = stop_flag or (lambda: False)
        self.scene_mode                  = scene_mode
        self.use_bilateral               = use_bilateral
        self.use_glass_weight            = use_glass_weight
        self.glass_bright_threshold      = glass_bright_threshold
        self.glass_min_weight            = glass_min_weight
        self.use_border_mask             = use_border_mask
        self.border_frac                 = border_frac
        self.use_inpaint                 = use_inpaint
        self.inpaint_radius_px           = inpaint_radius_px
        self.use_soft_consistency        = use_soft_consistency
        self.consistency_window          = consistency_window
        self.consistency_threshold_m     = consistency_threshold_m
        self.consistency_min_weight      = consistency_min_weight
        self.use_dynamic_weight          = use_dynamic_weight
        self.dynamic_motion_thresh_m     = dynamic_motion_thresh_m
        self.dynamic_min_weight          = dynamic_min_weight
        self.use_planar_snap             = use_planar_snap
        self.planar_ransac_thresh        = planar_ransac_thresh
        self.n_planes                    = n_planes
        self.planar_snap_dist            = planar_snap_dist
        self.planar_remove_dist          = planar_remove_dist
        self.use_hole_fill               = use_hole_fill
        self.max_hole_size               = max_hole_size
        self.sdf_trunc_multiplier        = float(np.clip(sdf_trunc_multiplier, 1.5, 16.0))  # [P2-3]
        self.sdf_trunc_m                 = float(sdf_trunc_m) if sdf_trunc_m else None
        self.min_confidence              = int(min_confidence)
        self.mesh_smooth_iterations      = int(mesh_smooth_iterations)
        self.inpaint_keep_masks          = bool(inpaint_keep_masks)   # [P2-1] False = legacy behaviour
        self.last_stats                  = {}
        self.use_flying_pixel_suppress   = use_flying_pixel_suppress
        self.flying_pixel_jump_thresh_m  = flying_pixel_jump_thresh_m
        self.use_sky_suppress            = use_sky_suppress
        self.sky_bright_thresh           = sky_bright_thresh
        self.sky_depth_min_m             = sky_depth_min_m
        self.use_spike_filter            = use_spike_filter
        self._frames_ref                 = None
        self.last_tsdf_mesh_path         = None

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────────

    def fuse(self, frames: list) -> Path:
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d is required: pip install open3d")

        self._frames_ref = frames
        out_path  = self.output_dir / "fused_pointcloud.ply"
        sdf_trunc = (self.sdf_trunc_m if self.sdf_trunc_m
                     else self.voxel_size * self.sdf_trunc_multiplier)   # [P2-3]

        log.info(f"  TSDF v8: voxel={self.voxel_size}m  "
                 f"sdf_trunc={sdf_trunc:.4f}m  "
                 f"depth=[{self.min_depth},{self.max_depth}]m")

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=self.voxel_size,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

        sel   = frames[::max(1, self.frame_skip)]
        n_sel = len(sel)
        log.info(f"  Integrating {n_sel}/{len(frames)} frames ...")

        # Pre-load depth cache for dynamic weight computation
        depth_cache: Dict[int, np.ndarray] = {}
        for i, fr in enumerate(sel):
            try:
                depth_cache[i] = (_load_exr(fr.depth_path).astype(np.float32)
                                  * self.depth_scale)
            except Exception as e:
                log.warning(f"  Frame {getattr(fr, 'idx', i)}: depth cache failed ({e})")
        log.info(f"  Depth cache: {len(depth_cache)} frames loaded")

        n_fused = 0
        for i, fr in enumerate(sel):
            if self.stop_flag():
                break
            if i not in depth_cache:
                continue

            depth = depth_cache[i].copy()

            # [P2-2] drop sensor-flagged low-confidence pixels (ARKit .conf: 0/1/2)
            if self.min_confidence > 0:
                depth = self._apply_confidence(depth, fr)

            # ── Step 1: Border strip ────────────────────────────────────────
            if self.use_border_mask:
                depth = _mask_depth_borders(depth, self.border_frac)

            # ── Load RGB ────────────────────────────────────────────────────
            rgb_np = None
            try:
                from PIL import Image as PILImage
                rgb_pil = PILImage.open(fr.rgb_path).convert("RGB")
                rgb_np  = np.asarray(rgb_pil, dtype=np.uint8)
                if rgb_np.shape[:2] != depth.shape:
                    rgb_np = np.asarray(
                        rgb_pil.resize((depth.shape[1], depth.shape[0]),
                                       PILImage.BILINEAR),
                        dtype=np.uint8)
            except Exception:
                pass

            # ── Step 2: Sky/window suppression ──────────────────────────────
            if self.use_sky_suppress:
                depth = _suppress_sky_depth(depth, rgb_np,
                                             self.sky_bright_thresh,
                                             self.sky_depth_min_m)

            # ── Step 3: Bilateral smoothing (tighter sigma) ─────────────────
            if self.use_bilateral:
                depth = _smooth_depth_bilateral(depth,
                                                sigma_color=0.02,
                                                sigma_space=3.0)

            # ── Step 4: Flying pixel suppression ────────────────────────────
            if self.use_flying_pixel_suppress:
                depth = _suppress_flying_pixels(depth,
                                                jump_thresh_m=self.flying_pixel_jump_thresh_m)

            # ── Step 5: Inpaint holes (conservative) ────────────────────────
            if self.use_inpaint:
                filled = _inpaint_depth(depth, max_hole_px=self.inpaint_radius_px)
                # [P2-1] pixels removed ON PURPOSE (border, sky, flying pixels,
                # low confidence) stay removed; only genuine sensor holes are filled
                depth = _merge_inpaint(depth, filled, depth_cache[i],
                                       self.inpaint_keep_masks)

            # ── Step 6: Depth range + spike filter ──────────────────────────
            if self.use_spike_filter:
                depth = _validate_depth_range(depth, self.min_depth, self.max_depth)
            else:
                depth = np.where(
                    (depth >= self.min_depth) & (depth <= self.max_depth),
                    depth, 0.0).astype(np.float32)

            # ── Step 7: Dynamic object down-weighting (SIMPLIFIED in v8) ────
            # v8 FIX: The previous image-space dynamic mask was broken for
            # moving cameras — same world point projects to different pixels
            # in different frames → high std everywhere → massive over-masking.
            # Now we use a simple motion-based confidence that only down-weights
            # pixels whose depth changed significantly from the previous frame.
            # This is less aggressive but doesn't destroy static geometry.
            conf = np.ones(depth.shape, dtype=np.float32)
            if self.use_dynamic_weight and i > 0 and (i - 1) in depth_cache:
                depth_prev = depth_cache[i - 1]
                if depth_prev is not None and depth_prev.shape == depth.shape:
                    both = (depth > 0.05) & (depth_prev > 0.05)
                    if both.sum() > 100:
                        diff = np.zeros_like(depth)
                        diff[both] = np.abs(depth[both] - depth_prev[both])
                        # Only down-weight large changes
                        dyn_ratio = np.clip(diff / self.dynamic_motion_thresh_m,
                                            0.0, 1.0)
                        # ★ v8 FIX: Only apply down-weighting where diff is
                        # significantly above threshold (was too aggressive)
                        strong_motion = diff > self.dynamic_motion_thresh_m * 2.0
                        conf[strong_motion] = np.clip(
                            1.0 - dyn_ratio[strong_motion] * 0.5, 0.3, 1.0)
                        conf[depth <= 0.05] = 0.0

            # Apply confidence: blend toward smooth estimate for low-conf pixels
            if conf.min() < 0.99:
                smooth  = _smooth_depth_bilateral(depth, d=9,
                                                  sigma_color=0.05,
                                                  sigma_space=4.0)
                valid   = depth > 0.05
                depth   = np.where(valid,
                                   conf * depth + (1.0 - conf) * smooth,
                                   depth).astype(np.float32)

            # ── Prepare RGBD for TSDF ───────────────────────────────────────
            # Resize rgb to match depth if needed
            rgb_tsdf = (rgb_np if rgb_np is not None
                        else np.zeros((*depth.shape, 3), np.uint8))
            if rgb_tsdf.shape[:2] != depth.shape:
                try:
                    from PIL import Image as PILImage2
                    rgb_tsdf = np.asarray(
                        PILImage2.fromarray(rgb_tsdf).resize(
                            (depth.shape[1], depth.shape[0]),
                            PILImage2.BILINEAR), dtype=np.uint8)
                except Exception:
                    rgb_tsdf = np.zeros((*depth.shape, 3), np.uint8)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(rgb_tsdf.astype(np.uint8)),
                o3d.geometry.Image(depth),
                depth_scale=1.0,
                depth_trunc=self.max_depth,
                convert_rgb_to_intensity=False,
            )

            # ── Intrinsics: scale to match depth resolution ─────────────────
            intr = fr.intrinsics
            depth_h, depth_w = depth.shape

            # ★ v8 FIX: Handle case where intr.width/height might not exist
            # (some frame types store intrinsics differently)
            if hasattr(intr, 'width') and hasattr(intr, 'height'):
                rgb_w = intr.width
                rgb_h = intr.height
            elif hasattr(intr, 'K'):
                # Estimate from K matrix
                K_arr = np.array(intr.K) if not isinstance(intr.K, np.ndarray) else intr.K
                rgb_w = int(K_arr[0, 2] * 2) if K_arr.shape == (3, 3) else depth_w
                rgb_h = int(K_arr[1, 2] * 2) if K_arr.shape == (3, 3) else depth_h
            else:
                rgb_w = depth_w
                rgb_h = depth_h

            scale_x = depth_w / max(rgb_w, 1)
            scale_y = depth_h / max(rgb_h, 1)

            # Get focal lengths and principal point from intrinsics object
            if hasattr(intr, 'fx') and hasattr(intr, 'fy'):
                fx, fy = intr.fx, intr.fy
                cx = getattr(intr, 'cx', rgb_w / 2.0)
                cy = getattr(intr, 'cy', rgb_h / 2.0)
            elif hasattr(intr, 'K'):
                K_arr = np.array(intr.K) if not isinstance(intr.K, np.ndarray) else intr.K
                if K_arr.shape == (3, 3):
                    fx, fy = K_arr[0, 0], K_arr[1, 1]
                    cx, cy = K_arr[0, 2], K_arr[1, 2]
                else:
                    fx = fy = depth_w * 0.5 / np.tan(np.radians(30))
                    cx, cy = depth_w / 2.0, depth_h / 2.0
            else:
                fx = fy = depth_w * 0.5 / np.tan(np.radians(30))
                cx, cy = depth_w / 2.0, depth_h / 2.0   # [P2-4] was a tuple assignment
                log.warning(f"  Frame {i}: estimating intrinsics from depth size")

            o3d_intr = o3d.camera.PinholeCameraIntrinsic(
                depth_w, depth_h,
                fx * scale_x, fy * scale_y,
                cx * scale_x, cy * scale_y,
            )

            # ════════════════════════════════════════════════════════════════
            # ★★★ CRITICAL: TSDF integration extrinsic ★★★
            #
            # Open3D TSDF expects the EXTRINSIC matrix = w2c in OpenCV convention
            # (Y down, Z forward).
            #
            # Our c2w is in OpenGL convention (Y up, Z backward) from Record3D.
            # Conversion:  c2w_cv = c2w_gl @ _FLIP
            # Then:        w2c_cv = inv(c2w_cv) = inv(c2w_gl @ _FLIP)
            #
            # This is CONFIRMED CORRECT. The _FLIP handles the Y/Z axis swap.
            # ════════════════════════════════════════════════════════════════
            c2w_gl = fr.c2w
            w2c_cv = np.linalg.inv(c2w_gl @ _FLIP)

            volume.integrate(rgbd, o3d_intr, w2c_cv)
            n_fused += 1
            if n_fused % 50 == 0:
                log.info(f"  {n_fused}/{n_sel} frames integrated...")

        log.info(f"  {n_fused} frames fused")

        # ── Extract point cloud ──────────────────────────────────────────────
        log.info("Extracting point cloud...")
        pcd = volume.extract_point_cloud()
        log.info(f"  Raw: {len(pcd.points):,} points")
        self.last_stats = dict(n_frames_selected=n_sel, n_fused=n_fused,
                               raw_points=len(pcd.points),
                               voxel_size=self.voxel_size, sdf_trunc_m=float(sdf_trunc))   # [P2-6]

        if len(pcd.points) == 0:
            log.error("TSDF produced 0 points — check depth range and poses.")
            o3d.io.write_point_cloud(str(out_path), pcd)
            return out_path

        pcd = self._cleanup_scene(pcd)

        # ── Extract TSDF mesh and fill holes ─────────────────────────────────
        try:
            tsdf_mesh = volume.extract_triangle_mesh()
            if len(tsdf_mesh.triangles) > 0:
                tsdf_mesh.compute_vertex_normals()
                if self.use_hole_fill:
                    tsdf_mesh = _fill_mesh_holes(tsdf_mesh, self.max_hole_size,
                                                 self.mesh_smooth_iterations)
                self.last_tsdf_mesh_path = self.output_dir / "fused_tsdf_mesh.ply"
                o3d.io.write_triangle_mesh(str(self.last_tsdf_mesh_path), tsdf_mesh)
                log.info(f"Saved TSDF mesh -> {self.last_tsdf_mesh_path}")
        except Exception as e:
            log.warning(f"TSDF mesh extraction failed: {e}")

        o3d.io.write_point_cloud(str(out_path), pcd)
        log.info(f"Saved -> {out_path}  ({len(pcd.points):,} points)")
        return out_path

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_confidence(self, depth: np.ndarray, fr) -> np.ndarray:
        """[P2-2] zero depth where the per-pixel confidence is below min_confidence."""
        cp = getattr(fr, "conf_path", None)
        if cp is None or not Path(cp).is_file():
            return depth
        try:
            c = np.fromfile(str(cp), dtype=np.uint8)
            if c.size != depth.size:
                return depth
            c = c.reshape(depth.shape)
            out = depth.copy()
            out[c < self.min_confidence] = 0.0
            return out
        except Exception:
            return depth

    def _cleanup_scene(self, pcd):
        import open3d as o3d

        pts = np.asarray(pcd.points)
        ok  = np.isfinite(pts).all(axis=1)
        if (~ok).sum() > 0:
            pcd = pcd.select_by_index(np.where(ok)[0])

        pcd = pcd.voxel_down_sample(self.voxel_size)
        log.info(f"  After voxel downsample: {len(pcd.points):,}")

        pcd = self._camera_proximity_crop(pcd)

        # Two-pass confetti removal
        pcd = _remove_confetti(pcd, self.voxel_size)

        # Planar snap
        if self.use_planar_snap and len(pcd.points) > 500:
            pcd = _snap_to_planes(
                pcd,
                plane_distance_thresh=self.planar_ransac_thresh,
                n_planes=self.n_planes,
                snap_dist=self.planar_snap_dist,
                remove_dist=self.planar_remove_dist,
            )
            log.info(f"  After planar snap: {len(pcd.points):,}")

        # DBSCAN — keep clusters ≥ 0.5% of largest
        try:
            labels = np.array(pcd.cluster_dbscan(
                eps=float(self.voxel_size * 6.0),
                min_points=30,
                print_progress=False,
            ))
        except Exception as e:
            log.warning(f"  DBSCAN skipped: {e}")
            return pcd

        if labels.size == 0 or np.all(labels < 0):
            return pcd

        valid   = labels >= 0
        if not np.any(valid):
            return pcd
        counts  = np.bincount(labels[valid])
        largest = int(counts.max())

        min_pts   = max(100, int(largest * 0.005))
        keep_mask = (labels >= 0) & (counts[labels] >= min_pts)
        removed   = int((~keep_mask).sum())
        if removed > 0:
            log.info(f"  DBSCAN: removed {removed:,} pts, "
                     f"kept {int(keep_mask.sum()):,}")
            pcd = pcd.select_by_index(np.where(keep_mask)[0])

        return pcd

    def _camera_proximity_crop(self, pcd):
        if self._frames_ref is None:
            return pcd

        pts = np.asarray(pcd.points)
        cam_positions = np.array([
            (fr.c2w @ _FLIP)[:3, 3]
            for i, fr in enumerate(self._frames_ref)
            if i % 5 == 0
        ])
        threshold = self.max_depth * 1.15
        log.info(f"  Proximity crop: threshold={threshold:.2f}m ...")

        try:
            import torch
            if torch.cuda.is_available():
                device   = torch.device('cuda')
                pts_t    = torch.from_numpy(pts).float().to(device)
                cams_t   = torch.from_numpy(cam_positions).float().to(device)
                chunk    = 100_000
                min_dist = np.full(len(pts), np.inf, dtype=np.float32)
                for i in range(0, len(pts), chunk):
                    end = min(i + chunk, len(pts))
                    d   = torch.cdist(pts_t[i:end], cams_t)
                    min_dist[i:end] = d.min(dim=1).values.cpu().numpy()
                keep = min_dist <= threshold
                log.info(f"  Proximity crop (GPU): kept {keep.sum():,}")
                return pcd.select_by_index(np.where(keep)[0])
        except ImportError:
            pass

        chunk    = 50_000
        min_dist = np.full(len(pts), np.inf, dtype=np.float32)
        for start in range(0, len(pts), chunk):
            p = pts[start:start+chunk]
            d = np.linalg.norm(
                p[:, None, :] - cam_positions[None, :, :], axis=2
            ).min(axis=1)
            min_dist[start:start+chunk] = d
        keep = min_dist <= threshold
        log.info(f"  Proximity crop (CPU): kept {keep.sum():,}")
        return pcd.select_by_index(np.where(keep)[0])
