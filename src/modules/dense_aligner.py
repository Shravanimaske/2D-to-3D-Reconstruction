import logging, re, gc, copy, time
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple

log = logging.getLogger("dense_aligner")

# Phase 1 changes (see inline [P1-*] tags)
#  [P1-1] intrinsics are taken from frame.intrinsics (scaled to the depth
#         resolution) instead of silently falling back to a guessed 60-deg FOV
#  [P1-2] pose-graph nodes are camera-to-world poses (they were world-to-camera
#         while the edges assumed camera-to-world: noise-free input poses were
#         moved by ~1.4 m / up to 177 deg in a synthetic test)
#  [P1-3] an ICP correction is accepted only if it measurably lowers the
#         registration error (evaluate_registration before/after)
#  [P1-4] loop-closure edges now carry the ICP measurement (they used the
#         current poses, i.e. no information, and had the wrong orientation)
#  [P1-5] the pose graph is only optimised when loop closures exist

_FLIP = np.diag([1., -1., -1., 1.])

# ─────────────────────────────────────────────────────────────────────────────
#  Optional deps
# ─────────────────────────────────────────────────────────────────────────────
try:
    import imageio; HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import cv2; HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _g(frame, key, default=None):
    try:
        return getattr(frame, key)
    except AttributeError:
        pass
    try:
        return frame[key]
    except (KeyError, TypeError, IndexError):
        pass
    return default


# ─────────────────────────────────────────────────────────────────────────────
#  Depth file I/O  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

DEPTH_EXTS = {'.bin', '.exr', '.png', '.npy', '.npz', '.tiff', '.tif', '.bmp', '.jpg', '.jpeg'}


def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower()
            for t in re.split(r'(\d+)', str(s))]

def _discover_files(directory: Path, exts: set) -> List[Path]:
    directory = Path(directory)
    if not directory.is_dir():
        return []
    files = [f for f in directory.rglob("*")
             if f.is_file() and f.suffix.lower() in exts]
    files.sort(key=lambda p: _natural_key(p.stem))
    return files


def _load_depth_bin(path: Path) -> Optional[np.ndarray]:
    try:
        raw = np.fromfile(str(path), dtype=np.float32)
    except Exception as e:
        log.debug(f"  .bin read failed for {path.name}: {e}")
        return None
    if raw.size == 0:
        return None
    for ext in ('.png', '.jpg', '.jpeg'):
        sibling = path.with_suffix(ext)
        if sibling.is_file() and HAS_CV2:
            try:
                img = cv2.imread(str(sibling))
                if img is not None:
                    h, w = img.shape[:2]
                    if h * w == raw.size:
                        return raw.reshape(h, w).astype(np.float64)
                    if (h // 2) * (w // 2) == raw.size:
                        return raw.reshape(h // 2, w // 2).astype(np.float64)
            except Exception:
                pass
    candidates = [
        (192, 256), (256, 192), (480, 640), (640, 480),
        (720, 960), (960, 720), (1080, 1440), (1440, 1080),
        (1440, 1920), (1920, 1440), (576, 768), (768, 576),
        (360, 480), (480, 360),
    ]
    for h, w in candidates:
        if h * w == raw.size:
            return raw.reshape(h, w).astype(np.float64)
    n = raw.size
    for w in range(32, 4096):
        if n % w == 0:
            h = n // w
            if 1.0 <= max(h, w) / min(h, w) <= 2.5:
                return raw.reshape(h, w).astype(np.float64)
    return None


def _is_valid_exr(path: Path) -> bool:
    try:
        with open(str(path), 'rb') as f:
            return f.read(4) == b'\x76\x2f\x31\x01'
    except Exception:
        return False


def _load_depth(path) -> Optional[np.ndarray]:
    path = Path(path)
    if not path.is_file():
        return None
    suf = path.suffix.lower()
    if suf == '.bin':
        return _load_depth_bin(path)
    if suf == '.npy':
        try:
            d = np.load(str(path))
            return d.astype(np.float64) if d.ndim == 2 else None
        except Exception:
            return None
    if suf == '.npz':
        try:
            data = np.load(str(path))
            d = data[list(data.keys())[0]]
            return d.astype(np.float64) if d.ndim == 2 else None
        except Exception:
            return None
    if suf == '.exr' and not _is_valid_exr(path):
        log.debug(f"  .exr has wrong magic bytes: {path.name}")
        return None
    if HAS_IMAGEIO:
        try:
            d = imageio.imread(str(path))
            if d.ndim == 2:
                return d.astype(np.float64)
            if d.ndim == 3:
                return d[:, :, 0].astype(np.float64)
        except Exception:
            pass
    if HAS_CV2:
        try:
            d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if d is not None:
                if d.ndim == 3:
                    d = d[:, :, 0]
                return d.astype(np.float64)
        except Exception:
            pass
    if suf == '.exr':
        try:
            import OpenEXR, Imath
            exr = OpenEXR.InputFile(str(path))
            hdr = exr.header()
            dw = hdr['dataWindow']
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            ch = 'Y' if 'Y' in hdr['channels'] else 'R'
            raw = exr.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
            return np.frombuffer(raw, dtype=np.float32).reshape(h, w).astype(np.float64)
        except Exception:
            pass
    return None


def _load_rgb_size(path) -> Optional[Tuple[int, int]]:
    if HAS_CV2:
        try:
            img = cv2.imread(str(path))
            if img is not None:
                return img.shape[:2]
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Depth-scale detection  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def detect_depth_scale(frames, depth_dir=None, max_samples=50):
    depth_medians = []
    for f in frames[:max_samples]:
        for attr in ('depth', 'depth_data', 'depth_image', 'depth_map',
                     'depth_array', 'depth_img'):
            d = _g(f, attr)
            if isinstance(d, np.ndarray) and d.ndim == 2:
                d = d[d > 0].astype(np.float64)
                if len(d) > 100:
                    depth_medians.append(np.median(d))
                break
    if not depth_medians and depth_dir is not None:
        files = _discover_files(Path(depth_dir), DEPTH_EXTS)
        for fp in files[:max_samples]:
            d = _load_depth(fp)
            if d is not None:
                d = d[d > 0]
                if len(d) > 100:
                    depth_medians.append(np.median(d))
    if not depth_medians:
        log.warning("Cannot detect depth scale — using 1.0")
        return 1.0
    median_depth = np.median(depth_medians)
    pose_dists = []
    for i in range(1, min(len(frames), max_samples)):
        p0 = _get_pose(frames[i - 1])
        p1 = _get_pose(frames[i])
        if p0 is not None and p1 is not None:
            dist = np.linalg.norm(p1[:3, 3] - p0[:3, 3])
            if 0.001 < dist < 50:
                pose_dists.append(dist)
    median_pose_dist = np.median(pose_dists) if pose_dists else 0.1
    if median_depth > 500:
        scale, label = 0.001, "millimeters"
    elif median_depth > 50:
        scale, label = 0.01, "centimeters"
    elif median_depth > 10:
        ratio = median_depth / max(median_pose_dist, 0.001)
        if 8 < median_depth < 30 and ratio > 50:
            scale, label = 0.1, "decimeters"
        else:
            scale, label = 1.0, "meters"
    else:
        scale, label = 1.0, "meters"
    log.info(f"Depth scale auto-detect: median_depth={median_depth:.2f}, "
             f"median_pose_spacing={median_pose_dist:.3f}m → scale={scale} ({label})")
    return scale


# ─────────────────────────────────────────────────────────────────────────────
#  Pose helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _get_pose(frame) -> Optional[np.ndarray]:
    for attr in ('pose', 'extrinsic', 'cam_to_world', 'c2w', 'transform'):
        p = _g(frame, attr)
        if p is not None:
            p = np.asarray(p, dtype=np.float64)
            if p.shape == (4, 4):
                return p
    return None


def _set_pose(frame, pose: np.ndarray):
    for attr in ('pose', 'extrinsic', 'cam_to_world', 'c2w', 'transform'):
        if hasattr(frame, attr):
            try:
                setattr(frame, attr, pose)
                return
            except Exception:
                pass
    for attr in ('pose', 'extrinsic', 'cam_to_world', 'c2w', 'transform'):
        try:
            frame[attr] = pose
            return
        except (TypeError, KeyError):
            pass
    log.warning("Cannot write pose back to frame")


def _get_K(frame) -> Optional[np.ndarray]:
    for attr in ('intrinsic', 'intrinsics', 'K', 'camera_matrix', 'cam_intrinsic'):
        val = _g(frame, attr)
        if val is not None:
            val = np.asarray(val)
            if val.shape == (3, 3):
                return val.astype(np.float64)
    fx = _g(frame, 'fx')
    fy = _g(frame, 'fy')
    if fx is not None and fy is not None:
        w  = _g(frame, 'width')  or _g(frame, 'w')  or 640
        h  = _g(frame, 'height') or _g(frame, 'h')  or 480
        cx = _g(frame, 'cx') or float(w) / 2
        cy = _g(frame, 'cy') or float(h) / 2
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    fl = _g(frame, 'focal_length') or _g(frame, 'focal')
    if fl is not None:
        fx = fy = float(fl) if np.isscalar(fl) else float(np.asarray(fl).ravel()[0])
        w  = _g(frame, 'width') or 640
        h  = _g(frame, 'height') or 480
        return np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]], dtype=np.float64)
    return None


def _find_K_file(data_dir: Path) -> Optional[np.ndarray]:
    for search in [data_dir, data_dir.parent]:
        if not search.is_dir():
            continue
        for name in ['intrinsics.txt', 'K.txt', 'camera_matrix.txt',
                     'calibration.txt', 'intrinsic.txt', 'intrinsics.npy']:
            p = search / name
            if p.is_file():
                try:
                    K = np.load(str(p)) if p.suffix == '.npy' else np.loadtxt(str(p))
                    if K.shape == (3, 3):
                        log.info(f"  Found intrinsics file: {p}")
                        return K
                except Exception:
                    pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Transform helpers  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _safe_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]; t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti


def _is_valid_transform(T: np.ndarray, max_trans: float = 200.0) -> bool:
    if T is None or T.shape != (4, 4):
        return False
    if not np.isfinite(T).all():
        return False
    det = np.linalg.det(T[:3, :3])
    if abs(det - 1.0) > 0.15:
        return False
    if np.linalg.norm(T[:3, 3]) > max_trans:
        return False
    return True


def _extract_frame_index(path: Path) -> int:
    m = re.match(r'(\d+)', path.stem)
    return int(m.group(1)) if m else -1


# ─────────────────────────────────────────────────────────────────────────────
#  Depth-file index map  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def _build_depth_index_map(depth_files: List[Path]) -> Dict[int, Path]:
    from collections import defaultdict, Counter
    groups: Dict[int, List[Path]] = defaultdict(list)
    for fp in depth_files:
        idx = _extract_frame_index(fp)
        if idx >= 0:
            groups[idx].append(fp)

    def _is_confidence_file(p: Path) -> bool:
        s = p.stem.lower()
        return p.suffix.lower() == '.conf' or 'conf' in s or 'confidence' in s or s.endswith('_c')

    EXT_PRIORITY = {'.bin': 0, '.npy': 1, '.npz': 1, '.exr': 2,
                    '.tiff': 3, '.tif': 3, '.png': 4, '.bmp': 5, '.jpg': 6, '.jpeg': 6}
    index_map: Dict[int, Path] = {}
    for idx, candidates in groups.items():
        depth_c = [p for p in candidates if not _is_confidence_file(p)] or candidates
        depth_c.sort(key=lambda p: (EXT_PRIORITY.get(p.suffix.lower(), 99), p.name))
        index_map[idx] = depth_c[0]
    return index_map


# ─────────────────────────────────────────────────────────────────────────────
#  Point-cloud utilities   [SPEED] optimized
# ─────────────────────────────────────────────────────────────────────────────

def _make_pcd_from_depth(
    depth_m: np.ndarray,
    intrinsic: o3d.camera.PinholeCameraIntrinsic,
    max_depth: float,
    voxel_ds: float,
    compute_normals: bool = False,       # [SPEED] lazy normals
) -> Optional[o3d.geometry.PointCloud]:
    dm = depth_m.astype(np.float32)
    dm[dm <= 0.0] = 0.0
    dm[dm > max_depth] = 0.0
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(dm),
        intrinsic,
        extrinsic=np.eye(4),
        depth_scale=1.0,
        depth_trunc=max_depth,
    )
    if len(pcd.points) < 100:
        return None
    pcd = pcd.voxel_down_sample(voxel_ds)
    if len(pcd.points) < 50:
        return None
    if compute_normals:                   # [SPEED] only when needed
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_ds * 4, max_nn=30))
        pcd.orient_normals_towards_camera_location(np.zeros(3))
    return pcd


def _ensure_normals(pcd: o3d.geometry.PointCloud, voxel_ds: float) -> o3d.geometry.PointCloud:
    """Add normals to a point cloud if missing (lazy, in-place)."""
    if not pcd.has_normals():
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_ds * 4, max_nn=30))
    return pcd


# [SPEED] vectorized overlap — uses C++ compute_point_cloud_distance
def _overlap_fraction(src: o3d.geometry.PointCloud,
                      tgt: o3d.geometry.PointCloud,
                      radius: float) -> float:
    if len(src.points) == 0 or len(tgt.points) == 0:
        return 0.0
    pts = np.asarray(src.points)
    step = max(1, len(pts) // 500)
    indices = list(range(0, len(pts), step))
    sampled = src.select_by_index(indices)
    dists = np.asarray(sampled.compute_point_cloud_distance(tgt))
    return float(np.mean(dists < radius))


# [SPEED] fast pose-based pre-check before expensive cloud creation
def _quick_pose_overlap(c2w_a: np.ndarray, c2w_b: np.ndarray,
                        max_dist: float) -> bool:
    """True if cameras are close enough to possibly see the same region."""
    return np.linalg.norm(c2w_a[:3, 3] - c2w_b[:3, 3]) < max_dist


# ─────────────────────────────────────────────────────────────────────────────
#  ICP helpers   [SPEED] completely rewritten — pure C++ Open3D ICP
# ─────────────────────────────────────────────────────────────────────────────

def _run_icp(src_world: o3d.geometry.PointCloud,
             tgt_world: o3d.geometry.PointCloud,
             icp_dist: float,
             voxel_ds: float,
             min_fitness: float,
             init: np.ndarray = None
             ) -> Tuple[np.ndarray, float, float]:
    """
    Multi-stage ICP using Open3D C++ backend.
    Returns (T_correction_world, fitness, rmse).
    """
    if init is None:
        init = np.eye(4, dtype=np.float64)

    # ── Stage 1: Coarse point-to-point (heavily downsampled, C++) ──────
    coarse_vox = voxel_ds * 4
    src_c = src_world.voxel_down_sample(coarse_vox)
    tgt_c = tgt_world.voxel_down_sample(coarse_vox)

    if len(src_c.points) < 10 or len(tgt_c.points) < 10:
        return np.eye(4, dtype=np.float64), 0.0, float('inf')

    try:
        res1 = o3d.pipelines.registration.registration_icp(
            src_c, tgt_c, icp_dist * 2.5, init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
        T_coarse = res1.transformation
    except Exception:
        T_coarse = init

    # ── Stage 2: Fine point-to-plane (moderately downsampled, C++) ─────
    fine_vox = voxel_ds * 2
    src_f = src_world.voxel_down_sample(fine_vox)
    tgt_f = tgt_world.voxel_down_sample(fine_vox)

    _ensure_normals(tgt_f, fine_vox)          # target needs normals
    _ensure_normals(src_f, fine_vox)          # source too for P2Plane

    best_T = T_coarse
    best_fit = 0.0
    best_rmse = float('inf')

    try:
        res2 = o3d.pipelines.registration.registration_icp(
            src_f, tgt_f, icp_dist, T_coarse,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50))
        best_T = res2.transformation
        best_fit = float(res2.fitness)
        best_rmse = float(res2.inlier_rmse)
    except Exception:
        pass

    # ── Stage 3: Point-to-point fallback if P2Plane failed ─────────────
    if best_fit < min_fitness:
        try:
            res3 = o3d.pipelines.registration.registration_icp(
                src_f, tgt_f, icp_dist * 2.0, T_coarse,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60))
            if float(res3.fitness) > best_fit:
                best_T = res3.transformation
                best_fit = float(res3.fitness)
                best_rmse = float(res3.inlier_rmse)
        except Exception:
            pass

    log.debug(f"    Fast-ICP: fit={best_fit:.4f}, rmse={best_rmse:.6f}")
    return best_T, best_fit, best_rmse


def _registration_improves(src: o3d.geometry.PointCloud,
                           tgt: o3d.geometry.PointCloud,
                           T: np.ndarray, dist: float, voxel: float,
                           min_rel_gain: float = 0.02) -> bool:
    """
    [P1-3] True only if applying T lowers the inlier RMSE by at least
    `min_rel_gain` (relative) without losing inlier fitness, compared with
    leaving the poses untouched (identity).  Prevents ICP from "correcting"
    already consistent ARKit poses into worse ones.
    """
    try:
        s = src.voxel_down_sample(voxel * 2)
        t = tgt.voxel_down_sample(voxel * 2)
        e0 = o3d.pipelines.registration.evaluate_registration(
            s, t, dist, np.eye(4))
        e1 = o3d.pipelines.registration.evaluate_registration(s, t, dist, T)
        if e0.fitness <= 0.0:
            return e1.fitness > 0.0
        return (e1.inlier_rmse < e0.inlier_rmse * (1.0 - min_rel_gain)
                and e1.fitness >= e0.fitness - 0.01)
    except Exception:                                           # noqa: BLE001
        return False


def _run_icp_consecutive(src_world: o3d.geometry.PointCloud,
                         tgt_world: o3d.geometry.PointCloud,
                         icp_dist: float,
                         voxel_ds: float,
                         min_fitness: float,
                         init: np.ndarray = None
                         ) -> Tuple[np.ndarray, float, float]:
    """
    [SPEED] Lightweight single-stage ICP for consecutive frames.
    Assumes initial alignment is already close — skips coarse stage.
    """
    if init is None:
        init = np.eye(4, dtype=np.float64)

    vox = voxel_ds * 3
    src_d = src_world.voxel_down_sample(vox)
    tgt_d = tgt_world.voxel_down_sample(vox)

    if len(src_d.points) < 10 or len(tgt_d.points) < 10:
        return np.eye(4, dtype=np.float64), 0.0, float('inf')

    # P2Point first (no normals needed)
    try:
        res = o3d.pipelines.registration.registration_icp(
            src_d, tgt_d, icp_dist, init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40))
        best_T = res.transformation
        best_fit = float(res.fitness)
        best_rmse = float(res.inlier_rmse)
    except Exception:
        return np.eye(4, dtype=np.float64), 0.0, float('inf')

    # Quick P2Plane polish if P2Point was decent
    if best_fit > min_fitness * 0.5:
        _ensure_normals(src_d, vox)
        _ensure_normals(tgt_d, vox)
        try:
            res2 = o3d.pipelines.registration.registration_icp(
                src_d, tgt_d, icp_dist * 0.8, best_T,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25))
            if float(res2.fitness) > best_fit:
                best_T = res2.transformation
                best_fit = float(res2.fitness)
                best_rmse = float(res2.inlier_rmse)
        except Exception:
            pass

    return best_T, best_fit, best_rmse


# ─────────────────────────────────────────────────────────────────────────────
#  DenseAligner v10  (speed-optimized)
# ─────────────────────────────────────────────────────────────────────────────

class DenseAligner:
    """
    Keyframe-anchored + consecutive-frame dense ICP pose refinement.
    v10: Speed-optimized rewrite of v9.

    Speed changes vs v9
    -------------------
    1. Replaced Python SVD-ICP with Open3D C++ ICP  (~50-100× faster ICP)
    2. Streaming point clouds — only keyframes kept in memory
    3. Lazy normal estimation — only computed when P2Plane ICP needs them
    4. Vectorized overlap fraction via compute_point_cloud_distance
    5. Fast pose-based pre-check skips impossible overlaps before cloud creation
    6. Lightweight ICP for consecutive frames (single-stage, no coarse)
    7. Removed global RANSAC fallback (expensive, rarely helps)
    8. Removed pose-quality pre-check when skip_if_poses_good=False
    """

    def __init__(
        self,
        output_dir: Path,
        depth_scale: float               = 1.0,
        max_depth: float                 = 10.0,
        voxel_downsample_size: float     = 0.008,
        icp_max_correspondence: float    = 0.10,
        min_icp_fitness: float           = 0.25,
        keyframe_interval: int           = 4,
        min_overlap_fraction: float      = 0.20,
        loop_closure: bool               = False,
        loop_closure_spatial_dist: float = 0.5,
        loop_closure_min_frame_gap: int  = 20,
        stop_flag: Optional[Callable[[], bool]] = None,
        depth_dir: Optional[Path]        = None,
        rgb_dir: Optional[Path]          = None,
        skip_if_poses_good: bool         = False,
        pose_quality_threshold: float    = 0.15,
    ):
        self.output_dir            = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.depth_scale           = depth_scale
        self.max_depth             = max_depth
        self.voxel_ds              = voxel_downsample_size
        self.icp_dist              = icp_max_correspondence
        self.min_fitness           = min_icp_fitness
        self.keyframe_interval     = keyframe_interval
        self.min_overlap           = min_overlap_fraction
        self.loop_closure          = loop_closure
        self.lc_spatial_dist       = loop_closure_spatial_dist
        self.lc_min_gap            = loop_closure_min_frame_gap
        self.stop_flag             = stop_flag
        self.depth_dir             = Path(depth_dir) if depth_dir else None
        self.rgb_dir               = Path(rgb_dir)   if rgb_dir   else None
        self.skip_if_poses_good    = skip_if_poses_good
        self.pose_quality_threshold = pose_quality_threshold

        self._depth_files: List[Path]       = []
        self._rgb_files:   List[Path]       = []
        self._depth_index_map: Dict[int, Path] = {}
        self._intrinsic: Optional[o3d.camera.PinholeCameraIntrinsic] = None
        self._img_h = self._img_w = 0
        self._adaptive_icp_dist: Optional[float] = None

    # ── internal helpers ──────────────────────────────────────────────────

    @property
    def _eff_icp_dist(self) -> float:
        return self._adaptive_icp_dist if self._adaptive_icp_dist is not None else self.icp_dist

    def _ensure_files(self):
        if not self._depth_files and self.depth_dir:
            self._depth_files = _discover_files(self.depth_dir, DEPTH_EXTS)
            log.info(f"  Depth files: {len(self._depth_files)} in {self.depth_dir}")
            self._depth_index_map = _build_depth_index_map(self._depth_files)
            log.info(f"  Depth index map: {len(self._depth_index_map)} entries")
            from collections import Counter
            ext_counts = Counter(fp.suffix.lower() for fp in self._depth_files)
            log.info(f"  Depth file types: {dict(sorted(ext_counts.items()))}")
        if not self._rgb_files and self.rgb_dir:
            self._rgb_files = _discover_files(
                self.rgb_dir, {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'})

    def _get_image_size(self, frames) -> Tuple[int, int]:
        if self._img_h and self._img_w:
            return self._img_h, self._img_w
        if frames:
            dres = _g(frames[0], 'depth_res')          # [P1-1] (W, H) of depth maps
            if dres is not None and len(dres) == 2 and dres[0] and dres[1]:
                self._img_w, self._img_h = int(dres[0]), int(dres[1])
                return self._img_h, self._img_w
            w = _g(frames[0], 'width')  or _g(frames[0], 'w')
            h = _g(frames[0], 'height') or _g(frames[0], 'h')
            if h and w:
                self._img_h, self._img_w = int(h), int(w)
                return self._img_h, self._img_w
            for attr in ('depth', 'depth_data', 'depth_image', 'rgb', 'color', 'image'):
                val = _g(frames[0], attr)
                if isinstance(val, np.ndarray) and val.ndim >= 2:
                    self._img_h, self._img_w = val.shape[0], val.shape[1]
                    return self._img_h, self._img_w
        self._ensure_files()
        if self._depth_files:
            d = _load_depth(self._depth_files[0])
            if d is not None:
                self._img_h, self._img_w = d.shape[:2]
                return self._img_h, self._img_w
        if self._rgb_files:
            sz = _load_rgb_size(self._rgb_files[0])
            if sz:
                self._img_h, self._img_w = sz
                return self._img_h, self._img_w
        return 0, 0

    def _resolve_intrinsics(self, frames) -> o3d.camera.PinholeCameraIntrinsic:
        self._ensure_files()
        h, w = self._get_image_size(frames)
        if frames and h > 0 and w > 0:
            # [P1-1] real camera from the pose file, scaled to the depth size.
            # Median over frames (values vary by <0.5 % between ARKit frames).
            try:
                ks = [fr.intrinsics.K_for(w, h) for fr in frames
                      if getattr(fr, 'intrinsics', None) is not None
                      and hasattr(fr.intrinsics, 'K_for')]
                if ks:
                    ks = np.array(ks)
                    fx, fy = float(np.median(ks[:, 0, 0])), float(np.median(ks[:, 1, 1]))
                    cx, cy = float(np.median(ks[:, 0, 2])), float(np.median(ks[:, 1, 2]))
                    log.info(f"  Intrinsics from frame.intrinsics (scaled to depth "
                             f"{w}x{h}): fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")
                    return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)
            except Exception as e:                              # noqa: BLE001
                log.warning(f"  could not use frame.intrinsics ({e}) - falling back")
        if frames:
            K = _get_K(frames[0])
            if K is not None and h > 0 and w > 0:
                log.info(f"  Intrinsics from frame: fx={K[0,0]:.1f} {w}×{h}")
                return o3d.camera.PinholeCameraIntrinsic(
                    w, h, float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]))
        for d in [self.depth_dir, self.rgb_dir]:
            if d:
                K = _find_K_file(d)
                if K is not None and h > 0 and w > 0:
                    log.info(f"  Intrinsics from file: fx={K[0,0]:.1f} {w}×{h}")
                    return o3d.camera.PinholeCameraIntrinsic(
                        w, h, float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]))
        if h > 0 and w > 0:
            fx = fy = w * 0.5 / np.tan(np.radians(30))
            log.warning(f"  ★ Estimated intrinsics: fx={fx:.0f} from {w}×{h} (60° H-FOV). "
                        f"Add intrinsics.txt for better results.")
            return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, w/2.0, h/2.0)
        raise RuntimeError("Cannot determine camera intrinsics or image size.")

    def _load_depth_for_frame(self, frame_idx: int, frame=None) -> Optional[np.ndarray]:
        raw = None
        if frame is not None:
            for attr in ('depth', 'depth_data', 'depth_image', 'depth_map',
                         'depth_array', 'depth_img'):
                d = _g(frame, attr)
                if isinstance(d, np.ndarray):
                    if d.ndim == 2:
                        raw = d.astype(np.float64); break
                    if d.ndim == 3 and d.shape[2] == 1:
                        raw = d[:, :, 0].astype(np.float64); break
                    if d.ndim == 3 and d.shape[0] == 1:
                        raw = d[0].astype(np.float64); break
            if raw is None:
                for attr in ('depth_path', 'depth_file', 'depth_filename', 'depth_fn'):
                    p = _g(frame, attr)
                    if p is not None:
                        raw = _load_depth(p)
                        if raw is not None:
                            break
        if raw is None:
            self._ensure_files()
            if frame_idx in self._depth_index_map:
                raw = _load_depth(self._depth_index_map[frame_idx])
            elif self._depth_files and frame_idx < len(self._depth_files):
                raw = _load_depth(self._depth_files[frame_idx])
        return None if raw is None else raw * self.depth_scale

    def _make_world_pcd(self, frame_idx, c2w_orig, frame=None,
                        compute_normals=False):                # [SPEED] lazy normals
        depth_m = self._load_depth_for_frame(frame_idx, frame)
        if depth_m is None:
            return None
        pcd_cam = _make_pcd_from_depth(
            depth_m, self._intrinsic, self.max_depth, self.voxel_ds,
            compute_normals=compute_normals)
        if pcd_cam is None:
            return None
        return pcd_cam.transform(c2w_orig @ _FLIP)

    # ── main refine ───────────────────────────────────────────────────────

    def refine(self, frames: list) -> list:
        t_total = time.time()
        n = len(frames)
        log.info(f"═══ DenseAligner v10 (fast) : {n} frames ═══")

        if n < 3:
            log.warning("Too few frames — skipping dense alignment")
            return frames

        log.info(f"  depth_dir  : {self.depth_dir}")
        log.info(f"  depth_scale: {self.depth_scale}")
        log.info(f"  keyframe_interval: {self.keyframe_interval}")

        # ── Intrinsics ────────────────────────────────────────────────────
        self._intrinsic = self._resolve_intrinsics(frames)
        fx, fy = self._intrinsic.get_focal_length()
        log.info(f"  Camera: {self._intrinsic.width}×{self._intrinsic.height} "
                 f"fx={fx:.1f} fy={fy:.1f}")

        # ── Collect original poses ────────────────────────────────────────
        orig: Dict[int, np.ndarray] = {}
        for i in range(n):
            p = _get_pose(frames[i])
            orig[i] = p.astype(np.float64) if p is not None else None
        if orig[0] is None:
            log.error("No pose on frame 0 — cannot align"); return frames

        # ── Measure frame spacing ─────────────────────────────────────────
        spacings = []
        for i in range(1, min(n, 500)):
            if orig[i] is not None and orig[i-1] is not None:
                d = np.linalg.norm(orig[i][:3,3] - orig[i-1][:3,3])
                if d > 1e-6:
                    spacings.append(d)
        median_spacing = float(np.median(spacings)) if spacings else 0.01
        p95_spacing    = float(np.percentile(spacings, 95)) if spacings else median_spacing * 5

        # ── Adaptive ICP distance ─────────────────────────────────────────
        adaptive_icp = max(
            self.voxel_ds * 3,
            min(median_spacing * 10.0, self.icp_dist))
        self._adaptive_icp_dist = adaptive_icp
        max_correction = median_spacing * 3.0

        overlap_radius = max(self.voxel_ds * 4, median_spacing * 3.0)
        # [SPEED] max camera distance for quick pre-check (generous)
        quick_max_cam_dist = max_depth_factor = min(
            self.max_depth * 1.5, median_spacing * 50)

        log.info(f"  Median spacing  : {median_spacing:.5f}m  p95={p95_spacing:.5f}m")
        log.info(f"  Adaptive ICP    : {adaptive_icp:.5f}m")
        log.info(f"  Max correction  : {max_correction:.5f}m")

        # ── Keyframes ─────────────────────────────────────────────────────
        ki = self.keyframe_interval
        keyframe_set = set(range(0, n, ki))
        log.info(f"  Keyframes: {len(keyframe_set)} (every {ki} frames)")

        # [SPEED] Build keyframe clouds upfront (with normals for P2Plane)
        # Non-KF clouds are created on-the-fly and discarded after use.
        log.info("  Step 1/5: Building keyframe point clouds...")
        t0 = time.time()
        kf_clouds: Dict[int, o3d.geometry.PointCloud] = {}
        kf_fail = 0
        for i in sorted(keyframe_set):
            if self.stop_flag and self.stop_flag():
                return frames
            c2w = orig[i] if orig[i] is not None else np.eye(4)
            # [SPEED] keyframe clouds get normals (used as ICP targets)
            pcd = self._make_world_pcd(i, c2w, frames[i] if i < len(frames) else None,
                                       compute_normals=True)
            if pcd is not None:
                kf_clouds[i] = pcd
            else:
                kf_fail += 1
                if kf_fail < 5:
                    log.warning(f"  KF {i}: could not build world pcd")

        log.info(f"  {len(kf_clouds)} KF clouds built ({kf_fail} failed) "
                 f"in {time.time()-t0:.1f}s")

        # [SPEED] Skip pose quality check entirely when not needed
        if self.skip_if_poses_good and len(kf_clouds) > 3:
            # Quick quality check using only keyframe clouds
            kf_list = sorted(kf_clouds.keys())
            ovlp_errs = []
            for k in range(1, min(len(kf_list), 20)):
                i0, i1 = kf_list[k-1], kf_list[k]
                ovlp = _overlap_fraction(kf_clouds[i0], kf_clouds[i1], overlap_radius)
                ovlp_errs.append(1.0 - ovlp)
            if ovlp_errs:
                mean_err = float(np.mean(ovlp_errs))
                log.info(f"  Pose quality (KF sample): err={mean_err:.4f}")
                if mean_err < self.pose_quality_threshold:
                    log.info(f"  ★ Poses already excellent — skipping ICP.")
                    self._save_poses(frames, n)
                    log.info("═══ DenseAligner v10: SKIPPED ═══")
                    return frames

        # ── Step 2: Streaming KF-anchored + consecutive ICP ──────────────
        log.info("  Step 2/5: Streaming dual-target ICP (v10 fast)...")
        t0 = time.time()

        pose_graph = o3d.pipelines.registration.PoseGraph()
        refined_c2w: Dict[int, np.ndarray] = {}
        valid: List[int] = []

        current_kf_idx:   Optional[int]                    = None
        current_kf_cloud: Optional[o3d.geometry.PointCloud] = None
        current_kf_c2w:   Optional[np.ndarray]              = None

        prev_refined_idx:   Optional[int]                    = None
        prev_refined_cloud: Optional[o3d.geometry.PointCloud] = None
        prev_refined_c2w:   Optional[np.ndarray]              = None

        good_icp    = 0
        rejected_no_gain = 0          # [P1-3] ICP result did not improve registration
        skipped_overlap = 0
        skipped_pose = 0  # [SPEED] pre-check skips
        node_k      = 0
        node_map: Dict[int, int] = {}
        chain_prev_node: Optional[int]        = None   # [P1-5] odometry chain
        chain_prev_c2w:  Optional[np.ndarray] = None

        for i in range(n):
            if self.stop_flag and self.stop_flag():
                return frames

            c2w_orig = orig[i]
            if c2w_orig is None:
                continue

            is_kf = (i in keyframe_set)

            # ── Keyframe: lock pose, store cloud ──────────────────────────
            if is_kf:
                if i in kf_clouds:
                    c2w_new = c2w_orig.copy()
                    refined_c2w[i] = c2w_new
                    valid.append(i)
                    current_kf_idx   = i
                    current_kf_cloud = kf_clouds[i]
                    current_kf_c2w   = c2w_new
                    prev_refined_idx   = i
                    prev_refined_cloud = kf_clouds[i]
                    prev_refined_c2w   = c2w_new

                    # [P1-2] node pose = camera-to-world (matches the edge convention)
                    pose_graph.nodes.append(
                        o3d.pipelines.registration.PoseGraphNode(c2w_new.copy()))
                    node_map[i] = node_k
                    if chain_prev_node is not None:
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                                node_k, chain_prev_node,
                                _safe_inv(chain_prev_c2w) @ c2w_new,
                                np.eye(6), uncertain=False))
                    chain_prev_node, chain_prev_c2w = node_k, c2w_new.copy()
                    node_k += 1
                continue

            # ── Non-keyframe: create cloud on-the-fly ─────────────────────
            # [SPEED] Quick pre-check: is this frame close enough to KF?
            if current_kf_c2w is not None:
                if not _quick_pose_overlap(c2w_orig, current_kf_c2w, quick_max_cam_dist):
                    # Also check against previous frame
                    if prev_refined_c2w is None or not _quick_pose_overlap(
                            c2w_orig, prev_refined_c2w, quick_max_cam_dist):
                        skipped_pose += 1
                        continue

            # [SPEED] No normals for source cloud (P2Point doesn't need them;
            # P2Plane will add them to downsampled copies inside _run_icp)
            src_w = self._make_world_pcd(i, c2w_orig,
                                         frames[i] if i < len(frames) else None,
                                         compute_normals=False)
            if src_w is None:
                continue

            valid.append(i)
            c2w_init = c2w_orig.copy()
            best_c2w_new = c2w_init.copy()
            best_fitness  = -1.0

            # --- Attempt A: align to nearest keyframe ---
            if current_kf_cloud is not None and current_kf_c2w is not None:
                ovlp_kf = _overlap_fraction(src_w, current_kf_cloud, overlap_radius)
                if ovlp_kf >= self.min_overlap:
                    T_kf, fit_kf, rmse_kf = _run_icp(
                        src_w, current_kf_cloud,
                        self._eff_icp_dist, self.voxel_ds, self.min_fitness)
                    c2w_proposed_kf = T_kf @ c2w_init
                    delta_kf = np.linalg.norm(c2w_proposed_kf[:3,3] - c2w_init[:3,3])
                    if (fit_kf >= self.min_fitness
                            and _is_valid_transform(c2w_proposed_kf)
                            and delta_kf <= max_correction):
                        if not _registration_improves(
                                src_w, current_kf_cloud, T_kf,
                                self._eff_icp_dist, self.voxel_ds):
                            rejected_no_gain += 1          # [P1-3]
                        elif fit_kf > best_fitness:
                            best_fitness  = fit_kf
                            best_c2w_new  = c2w_proposed_kf
                            good_icp += 1
                else:
                    skipped_overlap += 1

            # --- Attempt B: align to previous refined frame (lightweight) ---
            if (prev_refined_cloud is not None
                    and prev_refined_c2w is not None
                    and prev_refined_idx != current_kf_idx):
                ovlp_prev = _overlap_fraction(src_w, prev_refined_cloud, overlap_radius)
                if ovlp_prev >= self.min_overlap:
                    # [SPEED] Use lightweight ICP for consecutive frames
                    T_prev, fit_prev, rmse_prev = _run_icp_consecutive(
                        src_w, prev_refined_cloud,
                        self._eff_icp_dist, self.voxel_ds, self.min_fitness)
                    c2w_proposed_prev = T_prev @ c2w_init
                    delta_prev = np.linalg.norm(c2w_proposed_prev[:3,3] - c2w_init[:3,3])
                    if (fit_prev >= self.min_fitness
                            and _is_valid_transform(c2w_proposed_prev)
                            and delta_prev <= max_correction
                            and fit_prev > best_fitness):
                        if _registration_improves(
                                src_w, prev_refined_cloud, T_prev,
                                self._eff_icp_dist, self.voxel_ds):
                            best_fitness  = fit_prev
                            best_c2w_new  = c2w_proposed_prev
                        else:
                            rejected_no_gain += 1          # [P1-3]

            c2w_new = best_c2w_new
            refined_c2w[i] = c2w_new

            # Update previous refined frame tracker
            # [SPEED] Don't keep prev cloud reference if it's not a keyframe
            # to allow GC. We only need prev_cloud for the next 1-2 frames.
            prev_refined_idx   = i
            prev_refined_cloud = src_w       # keep alive for next frame
            prev_refined_c2w   = c2w_new

            # Pose graph node  [P1-2] camera-to-world
            pose_graph.nodes.append(
                o3d.pipelines.registration.PoseGraphNode(c2w_new.copy()))
            node_map[i] = node_k
            if chain_prev_node is not None:
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        node_k, chain_prev_node,
                        _safe_inv(chain_prev_c2w) @ c2w_new,
                        np.eye(6) * 0.5, uncertain=False))
            chain_prev_node, chain_prev_c2w = node_k, c2w_new.copy()

            # Pose graph edge to keyframe
            if current_kf_idx is not None and current_kf_idx in node_map:
                kf_node_k = node_map[current_kf_idx]
                T_kf_to_i = _safe_inv(current_kf_c2w) @ c2w_new

                info = np.eye(6, dtype=np.float64)
                if best_fitness >= self.min_fitness and current_kf_cloud is not None:
                    try:
                        T_corr = c2w_new @ _safe_inv(c2w_init)
                        info = o3d.pipelines.registration \
                            .get_information_matrix_from_point_clouds(
                                src_w, current_kf_cloud,
                                self._eff_icp_dist * 2, T_corr)
                    except Exception:
                        info = np.eye(6) * min(best_fitness, 1.0)
                else:
                    info *= 1e-6

                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(
                        node_k, kf_node_k, T_kf_to_i, info,
                        uncertain=(best_fitness < self.min_fitness)))

            node_k += 1

            # [SPEED] Explicitly release non-KF source cloud after a few frames
            # to keep memory bounded (only keep prev for consecutive ICP)
            if (prev_refined_idx is not None
                    and prev_refined_idx not in kf_clouds
                    and prev_refined_idx != i
                    and i - prev_refined_idx > 2):
                # The old prev_refined_cloud is no longer needed
                # (it was only kept for the frame immediately after it)
                pass  # Python GC will handle it when ref is overwritten

            if node_k % 200 == 0:
                log.info(f"    node {node_k}  good_icp={good_icp}  "
                         f"skip_overlap={skipped_overlap}  "
                         f"skip_pose={skipped_pose}  "
                         f"{time.time()-t0:.1f}s")

        log.info(f"  ICP results: good={good_icp}  "
                 f"skipped_overlap={skipped_overlap}  "
                 f"skipped_pose={skipped_pose}  "
                 f"in {time.time()-t0:.1f}s")

        log.info(f"  ICP corrections rejected (no measurable gain): {rejected_no_gain}")

        # ── Step 3: Loop closure ──────────────────────────────────────────
        n_lc = 0
        if self.loop_closure and len(kf_clouds) > self.lc_min_gap:
            log.info("  Step 3/5: Loop closure (keyframe pairs only)...")
            t0 = time.time()
            kf_list = sorted(kf_clouds.keys())
            positions = {vi: refined_c2w.get(vi, orig[vi])[:3, 3]
                         for vi in kf_list if vi in refined_c2w}
            for ki_idx in range(len(kf_list)):
                fi = kf_list[ki_idx]
                if fi not in refined_c2w or fi not in node_map:
                    continue
                for kj_idx in range(ki_idx + self.lc_min_gap, len(kf_list)):
                    fj = kf_list[kj_idx]
                    if fj not in refined_c2w or fj not in node_map:
                        continue
                    if np.linalg.norm(positions[fi] - positions[fj]) > self.lc_spatial_dist:
                        continue
                    ovlp_lc = _overlap_fraction(kf_clouds[fi], kf_clouds[fj], overlap_radius)
                    if ovlp_lc < 0.25:
                        continue
                    T_lc, fit_lc, _ = _run_icp(
                        kf_clouds[fi], kf_clouds[fj],
                        self._eff_icp_dist, self.voxel_ds, 0.25)
                    if (fit_lc > 0.30 and _is_valid_transform(T_lc)
                            and _registration_improves(
                                kf_clouds[fi], kf_clouds[fj], T_lc,
                                self._eff_icp_dist, self.voxel_ds)):
                        ni = node_map[fi]
                        nj = node_map[fj]
                        # [P1-4] measured relative pose: ICP moves world points
                        # of cloud i by T_lc, so camera i becomes T_lc @ c2w_i.
                        # Edge (i -> j) stores  c2w_j^-1 @ (T_lc @ c2w_i).
                        T_ij = _safe_inv(refined_c2w[fj]) @ T_lc @ refined_c2w[fi]
                        try:
                            info_lc = o3d.pipelines.registration \
                                .get_information_matrix_from_point_clouds(
                                    kf_clouds[fi], kf_clouds[fj],
                                    self._eff_icp_dist * 4, T_lc)
                        except Exception:
                            info_lc = np.eye(6) * fit_lc
                        pose_graph.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(
                                ni, nj, T_ij, info_lc, uncertain=True))
                        n_lc += 1
            log.info(f"  {n_lc} loop-closure edges in {time.time()-t0:.1f}s")

        # Release non-KF clouds
        if prev_refined_cloud is not None and prev_refined_idx not in kf_clouds:
            prev_refined_cloud = None
        gc.collect()

        # ── Step 4: Pose-graph optimisation ──────────────────────────────
        log.info("  Step 4/5: Pose-graph optimisation...")
        t0 = time.time()
        optimised = False
        if n_lc > 0:
            try:
                o3d.pipelines.registration.global_optimization(
                    pose_graph,
                    o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                    o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                    o3d.pipelines.registration.GlobalOptimizationOption(
                        max_correspondence_distance=self._eff_icp_dist * 2,
                        edge_prune_threshold=0.25,
                        reference_node=0))
                optimised = True
            except Exception as e:
                log.warning(f"  Pose-graph optimisation failed: {e} — using sequential ICP poses")
        else:
            log.info("  No loop closures -> pose graph not needed "
                     "(refined poses used directly)")
        log.info(f"  Pose-graph done in {time.time()-t0:.1f}s")

        # ── Step 5: Apply refined poses ───────────────────────────────────
        log.info("  Step 5/5: Applying refined poses...")
        opt_c2w: Dict[int, np.ndarray] = {}
        for vi in valid:
            try:
                if optimised and vi in node_map:
                    # [P1-2] node poses are camera-to-world: no inversion
                    opt_c2w[vi] = np.asarray(
                        pose_graph.nodes[node_map[vi]].pose, dtype=np.float64)
                elif vi in refined_c2w:
                    opt_c2w[vi] = refined_c2w[vi]
                else:
                    opt_c2w[vi] = orig[vi]
            except Exception:
                opt_c2w[vi] = refined_c2w.get(vi, orig[vi])

        corrections = []
        for i in range(n):
            orig_pose = orig[i]
            if orig_pose is None:
                continue

            try:
                frames[i].pose_original = orig_pose.copy()
            except (AttributeError, TypeError):
                pass

            if i in opt_c2w:
                new_pose = opt_c2w[i]
            else:
                # Interpolate between neighbours
                prev_v = next_v = None
                for vi in valid:
                    if vi <= i: prev_v = vi
                    if vi >= i and next_v is None: next_v = vi
                if prev_v is not None and next_v is not None and prev_v != next_v:
                    t  = (i - prev_v) / max(1, next_v - prev_v)
                    tp = opt_c2w[prev_v][:3, 3]
                    tn = opt_c2w[next_v][:3, 3]
                    new_pose = np.eye(4, dtype=np.float64)
                    new_pose[:3, :3] = opt_c2w[prev_v][:3,:3] if t < 0.5 else opt_c2w[next_v][:3,:3]
                    new_pose[:3, 3]  = (1-t)*tp + t*tn
                elif prev_v is not None:
                    delta    = opt_c2w[prev_v] @ _safe_inv(orig[prev_v])
                    new_pose = delta @ orig_pose
                else:
                    new_pose = orig_pose

            # Final sanity check
            delta_t = np.linalg.norm(new_pose[:3,3] - orig_pose[:3,3])
            # loop-closure drift correction may legitimately exceed the
            # per-frame ICP limit; plain ICP refinement may not
            limit = max(max_correction * 2.0, 0.10) if optimised else max_correction * 2.0
            if not _is_valid_transform(new_pose) or delta_t > limit:
                log.debug(f"  Frame {i}: post-PG correction {delta_t:.5f}m too large — keeping orig")
                new_pose = orig_pose

            _set_pose(frames[i], new_pose.astype(np.float64))
            corrections.append(delta_t)

        self._save_poses(frames, n)

        if corrections:
            log.info(f"  Corrections: mean={np.mean(corrections):.5f}m  "
                     f"max={np.max(corrections):.5f}m  "
                     f"median={np.median(corrections):.5f}m")

        total_s = time.time() - t_total
        log.info(f"═══ DenseAligner v10 complete — {total_s:.1f}s total ═══")
        return frames

    def _save_poses(self, frames, n):
        p = self.output_dir / "refined_poses.txt"
        with open(p, "w") as f:
            for i in range(n):
                pose = _get_pose(frames[i])
                if pose is not None:
                    f.write(f"{i} " + " ".join(f"{v:.8f}" for v in pose.flatten()) + "\n")
        log.info(f"  Saved → {p.name}")