"""
src/modules/mast3r_sfm_aligner.py  — v1.0

Drop-in replacement for COLMAPDatabaseBuilder (Stage 3).
Uses MASt3R sparse global alignment to produce camera poses that are
then blended back onto the full frame list via SE(3) interpolation.

Requirements (install once):
    git clone --recursive https://github.com/naver/mast3r
    cd mast3r
    pip install -r requirements.txt
    pip install -r dust3r/requirements.txt
    # Optional but recommended — enables retrieval-based pairing (ASMK):
    pip install cython
    git clone https://github.com/jenicek/asmk && cd asmk/cython && cythonize *.pyx && cd .. && pip install .

Checkpoint download (run once):
    mkdir -p checkpoints
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/
    # Optional retrieval model (needed if use_retrieval=True):
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth -P checkpoints/
    wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl -P checkpoints/
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

log = logging.getLogger("mast3r_sfm")


# ─────────────────────────────────────────────────────────────────────────────
#  Lazy MASt3R import helper
# ─────────────────────────────────────────────────────────────────────────────

def _import_mast3r(mast3r_repo_path: Optional[str] = None):
    """
    Add the MASt3R repo to sys.path if needed, then import the
    key symbols we depend on.  Raises ImportError with a helpful message
    if the repo is not found.
    """
    if mast3r_repo_path:
        p = str(Path(mast3r_repo_path).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
        dust = str(Path(mast3r_repo_path) / "dust3r")
        if dust not in sys.path:
            sys.path.insert(0, dust)

    try:
        from mast3r.model import AsymmetricMASt3R               # noqa: F401
        from mast3r.fast_nn import fast_reciprocal_NNs          # noqa: F401
        from dust3r.inference import inference                   # noqa: F401
        from dust3r.utils.image import load_images              # noqa: F401
        from dust3r.cloud_opt.sparse_ga import sparse_global_alignment  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(
            "MASt3R not found. "
            "Clone https://github.com/naver/mast3r, install its requirements, "
            "then pass mast3r_repo_path=<path> to MASt3RSfMAligner."
        ) from exc

    from mast3r.model import AsymmetricMASt3R
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.cloud_opt.sparse_ga import sparse_global_alignment

    # make_pairs lives in different places across MASt3R versions
    make_pairs = None
    for mod_path in (
        "mast3r.utils.misc",
        "dust3r.utils.misc",
        "dust3r.cloud_opt.sparse_ga",
    ):
        try:
            import importlib
            m = importlib.import_module(mod_path)
            if hasattr(m, "make_pairs"):
                make_pairs = m.make_pairs
                break
        except ModuleNotFoundError:
            continue

    if make_pairs is None:
        # Fallback: sequential pairs
        def make_pairs(imgs, scene_graph="swin-5", prefilter=None, symmetrize=True):
            pairs = []
            n = len(imgs)
            win = 5
            for i in range(n):
                for j in range(i + 1, min(i + win + 1, n)):
                    pairs.append((imgs[i], imgs[j]))
                    if symmetrize:
                        pairs.append((imgs[j], imgs[i]))
            return pairs

    return AsymmetricMASt3R, inference, load_images, sparse_global_alignment, make_pairs


# ─────────────────────────────────────────────────────────────────────────────
#  SE(3) interpolation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rot_to_axis_angle(R: np.ndarray):
    """Rodrigues rotation → axis-angle vector."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.ravel()


def _axis_angle_to_rot(v: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(v.reshape(3, 1))
    return R


def _slerp_pose(P0: np.ndarray, P1: np.ndarray, t: float) -> np.ndarray:
    """
    Linearly interpolate between two 4×4 camera-to-world matrices.
    Translation is lerp'd; rotation uses axis-angle lerp (≈ slerp for small angles).
    """
    R0, t0 = P0[:3, :3], P0[:3, 3]
    R1, t1 = P1[:3, :3], P1[:3, 3]

    # relative rotation  R_rel = R1 @ R0.T
    R_rel = R1 @ R0.T
    aa = _rot_to_axis_angle(R_rel)
    R_interp = _axis_angle_to_rot(aa * t) @ R0

    t_interp = t0 + t * (t1 - t0)

    P = np.eye(4, dtype=np.float64)
    P[:3, :3] = R_interp
    P[:3, 3] = t_interp
    return P


def _propagate_poses_to_all_frames(
    frames: list,
    keyframe_indices: List[int],
    kf_poses_c2w: List[np.ndarray],
) -> list:
    """
    Given refined poses for keyframe_indices (a subset of frames),
    interpolate / extrapolate poses to every frame in the full list.

    Strategy:
      - Between two keyframes: slerp
      - Before the first keyframe: copy first keyframe pose
      - After the last keyframe: copy last keyframe pose
    """
    if len(keyframe_indices) != len(kf_poses_c2w):
        raise ValueError("keyframe_indices and kf_poses_c2w must have the same length")

    n = len(frames)
    kf_idx = sorted(range(len(keyframe_indices)), key=lambda i: keyframe_indices[i])
    sorted_kf_indices = [keyframe_indices[i] for i in kf_idx]
    sorted_kf_poses  = [kf_poses_c2w[i]     for i in kf_idx]

    # Build a lookup of {frame_idx: pose_c2w}
    pose_map: dict[int, np.ndarray] = {
        idx: p for idx, p in zip(sorted_kf_indices, sorted_kf_poses)
    }

    result_poses: List[np.ndarray] = [None] * n

    for fi in range(n):
        if fi in pose_map:
            result_poses[fi] = pose_map[fi]
            continue

        # Find bounding keyframes
        lo_idx, hi_idx = None, None
        for ki in sorted_kf_indices:
            if ki < fi:
                lo_idx = ki
            elif ki > fi and hi_idx is None:
                hi_idx = ki

        if lo_idx is None and hi_idx is None:
            result_poses[fi] = np.eye(4)
        elif lo_idx is None:
            result_poses[fi] = pose_map[hi_idx].copy()
        elif hi_idx is None:
            result_poses[fi] = pose_map[lo_idx].copy()
        else:
            span = hi_idx - lo_idx
            t = (fi - lo_idx) / span
            result_poses[fi] = _slerp_pose(pose_map[lo_idx], pose_map[hi_idx], t)

    # Apply refined poses back to frames
    updated_frames = []
    for fi, frame in enumerate(frames):
        f = frame  # shallow copy — frames are typically dicts or simple objects
        pose_c2w = result_poses[fi]
        if pose_c2w is not None:
            # Support both dict-like frames and object-like frames
            if isinstance(frame, dict):
                f = dict(frame)
                f["pose"] = pose_c2w
            else:
                try:
                    import copy
                    f = copy.copy(frame)
                    f.pose = pose_c2w
                except AttributeError:
                    pass  # immutable frame — skip
        updated_frames.append(f)

    return updated_frames


# ─────────────────────────────────────────────────────────────────────────────
#  Scale alignment helper
# ─────────────────────────────────────────────────────────────────────────────

def _align_scale_to_sensor(
    mast3r_poses_c2w: List[np.ndarray],
    sensor_poses_c2w: List[np.ndarray],
) -> tuple[List[np.ndarray], float]:
    """
    Align MASt3R poses (metric but may drift) to sensor trajectory scale
    using a simple Procrustes-style scale factor derived from trajectory length.

    Returns (scale-corrected poses, scale_factor).
    """
    if len(mast3r_poses_c2w) < 2:
        return mast3r_poses_c2w, 1.0

    def traj_length(poses):
        pts = np.array([p[:3, 3] for p in poses])
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    m_len = traj_length(mast3r_poses_c2w)
    s_len = traj_length(sensor_poses_c2w)

    if m_len < 1e-6 or s_len < 1e-6:
        log.warning("  Scale alignment: near-zero trajectory length, skipping")
        return mast3r_poses_c2w, 1.0

    scale = s_len / m_len
    log.info(f"  Scale alignment: MASt3R traj={m_len:.3f}m  "
             f"sensor traj={s_len:.3f}m  scale={scale:.4f}")

    corrected = []
    for P in mast3r_poses_c2w:
        Pc = P.copy()
        Pc[:3, 3] *= scale
        corrected.append(Pc)

    return corrected, scale


# ─────────────────────────────────────────────────────────────────────────────
#  Main class
# ─────────────────────────────────────────────────────────────────────────────

class MASt3RSfMAligner:
    """
    Runs MASt3R sparse global alignment on a keyframe subset and propagates
    refined camera-to-world poses back to all frames.

    Parameters
    ----------
    output_dir          : Path to write debug / intermediate files.
    mast3r_repo_path    : Path to the cloned naver/mast3r repo.
                          Set to None if it's already on sys.path.
    checkpoint_path     : Path to the .pth model checkpoint.
    retrieval_model_path: Path to *_retrieval_trainingfree.pth  (optional).
    retrieval_codebook  : Path to *_retrieval_codebook.pkl      (optional).
    keyframe_every      : Sample 1 frame every N for SfM (keep <100 frames).
    image_size          : Resolution fed to ViT (512 is the MASt3R default).
    device              : 'cuda' | 'cpu'.
    niter               : Optimisation iterations for sparse_global_alignment.
    schedule            : LR schedule — 'cosine' or 'linear'.
    lr                  : Learning rate for global alignment optimisation.
    scene_graph         : Pair-selection strategy ('swin-5' | 'complete' | 'retrieval').
    align_scale         : Whether to rescale MASt3R poses to match sensor scale.
    blend_alpha         : 0 = pure sensor, 1 = pure MASt3R.
                          Values in (0,1) do a weighted slerp blend.
    min_conf_thr        : Minimum confidence threshold for MASt3R outputs.
    stop_flag           : Optional callable; if it returns True, abort early.
    """

    def __init__(
        self,
        output_dir: Path,
        mast3r_repo_path: Optional[str]  = None,
        checkpoint_path: str             = r"checkpoints\MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        retrieval_model_path: Optional[str] = None,
        retrieval_codebook: Optional[str]   = None,
        keyframe_every: int              = 8,
        image_size: int                  = 512,
        device: str                      = "cuda",
        niter: int                       = 300,
        schedule: str                    = "cosine",
        lr: float                        = 0.01,
        scene_graph: str                 = "swin-5",
        align_scale: bool                = True,
        blend_alpha: float               = 1.0,
        min_conf_thr: float              = 1.5,
        stop_flag: Optional[Callable]    = None,
    ):
        self.output_dir           = Path(output_dir)
        self.mast3r_repo_path     = mast3r_repo_path
        self.checkpoint_path      = checkpoint_path
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_codebook   = retrieval_codebook
        self.keyframe_every       = keyframe_every
        self.image_size           = image_size
        self.device               = device
        self.niter                = niter
        self.schedule             = schedule
        self.lr                   = lr
        self.scene_graph          = scene_graph
        self.align_scale          = align_scale
        self.blend_alpha          = max(0.0, min(1.0, blend_alpha))
        self.min_conf_thr         = min_conf_thr
        self.stop_flag            = stop_flag or (lambda: False)

        # Loaded lazily on first call to align()
        self._model = None

    # ──────────────────────────────────────────────────────────────────────
    #  Model loading
    # ──────────────────────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return

        (
            AsymmetricMASt3R,
            self._inference,
            self._load_images,
            self._sparse_ga,
            self._make_pairs,
        ) = _import_mast3r(self.mast3r_repo_path)

        ckpt = Path(self.checkpoint_path)
        if not ckpt.is_file():
            raise FileNotFoundError(
                f"MASt3R checkpoint not found: {ckpt}\n"
                "Download with:\n"
                "  wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/"
                "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth -P checkpoints/"
            )

        log.info(f"  Loading MASt3R model from {ckpt.name} ...")
        t0 = time.time()
        self._model = AsymmetricMASt3R.from_pretrained(str(ckpt)).to(self.device).eval()
        log.info(f"  Model loaded in {time.time()-t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────
    #  Pose extraction from MASt3R scene object
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_poses_c2w(scene) -> List[np.ndarray]:
        """
        Extract camera-to-world 4×4 matrices from a MASt3R scene object.
        MASt3R stores world-to-camera internally; we invert.
        """
        poses = []
        try:
            # scene.get_im_poses() returns (N,4,4) cam2world tensors
            import torch
            raw = scene.get_im_poses()
            if isinstance(raw, torch.Tensor):
                raw = raw.detach().cpu().numpy()
            for P in raw:
                poses.append(P.astype(np.float64))
        except Exception as e:
            log.warning(f"  get_im_poses() failed ({e}); trying .im_poses attribute")
            try:
                import torch
                raw = scene.im_poses
                if isinstance(raw, torch.Tensor):
                    raw = raw.detach().cpu().numpy()
                for P in raw:
                    poses.append(np.linalg.inv(P.astype(np.float64)))
            except Exception as e2:
                log.error(f"  Cannot extract poses from MASt3R scene: {e2}")
        return poses

    # ──────────────────────────────────────────────────────────────────────
    #  Frame → tmp image file helper
    # ──────────────────────────────────────────────────────────────────────

    def _save_keyframe_images(
        self,
        frames: list,
        kf_indices: List[int],
        rgb_dir: Optional[Path],
    ) -> tuple[Path, List[Path]]:
        """
        Write RGB images for keyframes to a temp directory so MASt3R can read them.
        Returns (tmp_dir, list_of_image_paths).
        """
        tmp_dir = self.output_dir / "_mast3r_kf_images"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        img_paths: List[Path] = []
        for ki in kf_indices:
            frame = frames[ki]

            # ── Try 1: frame has a .rgb_path attribute ─────────────────
            rgb_path = None
            for attr in ("rgb_path", "color_path", "image_path", "path"):
                v = frame.get(attr) if isinstance(frame, dict) else getattr(frame, attr, None)
                if v and Path(v).is_file():
                    rgb_path = Path(v)
                    break

            # ── Try 2: reconstruct from rgb_dir and frame name/index ───
            if rgb_path is None and rgb_dir is not None:
                stem = None
                for attr in ("name", "stem", "frame_name", "frame_id"):
                    v = frame.get(attr) if isinstance(frame, dict) else getattr(frame, attr, None)
                    if v:
                        stem = v
                        break
                if stem is None:
                    stem = f"{ki:06d}"
                for ext in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
                    candidate = rgb_dir / (str(stem) + ext)
                    if candidate.is_file():
                        rgb_path = candidate
                        break

            if rgb_path is not None:
                img_paths.append(rgb_path)
            else:
                # ── Try 3: frame carries raw numpy RGB ─────────────────
                img_np = None
                for attr in ("rgb", "color", "image"):
                    v = frame.get(attr) if isinstance(frame, dict) else getattr(frame, attr, None)
                    if v is not None:
                        img_np = v
                        break

                if img_np is not None:
                    dst = tmp_dir / f"kf_{ki:06d}.jpg"
                    cv2.imwrite(str(dst), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                    img_paths.append(dst)
                else:
                    log.warning(f"  Keyframe {ki}: no image found, skipping")

        return tmp_dir, img_paths

    # ──────────────────────────────────────────────────────────────────────
    #  Sensor pose helper
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_sensor_pose_c2w(frame) -> np.ndarray:
        """Extract 4×4 camera-to-world from a frame (handles dict & object)."""
        for attr in ("pose", "c2w", "camera_to_world", "transform"):
            v = frame.get(attr) if isinstance(frame, dict) else getattr(frame, attr, None)
            if v is not None:
                return np.asarray(v, dtype=np.float64).reshape(4, 4)
        return np.eye(4, dtype=np.float64)

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    def align(
        self,
        frames: list,
        rgb_dir: Optional[Path] = None,
    ) -> list:
        """
        Run MASt3R-SfM on a keyframe subset and propagate refined poses
        to all frames via SE(3) interpolation.

        Parameters
        ----------
        frames  : Full list of frames as returned by DataLoader / DenseAligner.
        rgb_dir : Directory containing RGB images (needed if frames don't
                  carry .rgb_path attributes).

        Returns
        -------
        Updated frame list with refined poses.
        """
        if self.stop_flag():
            log.info("  MASt3RSfMAligner: stop flag set, skipping")
            return frames

        self._load_model()

        # ── Select keyframes ───────────────────────────────────────────
        kf_indices = list(range(0, len(frames), self.keyframe_every))
        # Always include last frame for good coverage
        if kf_indices[-1] != len(frames) - 1:
            kf_indices.append(len(frames) - 1)
        log.info(f"  MASt3R keyframes: {len(kf_indices)} / {len(frames)}")

        # Cap at ~100 to avoid OOM on typical GPUs
        MAX_KF = 100
        if len(kf_indices) > MAX_KF:
            step = len(kf_indices) // MAX_KF
            kf_indices = kf_indices[::step]
            log.warning(f"  Capped to {len(kf_indices)} keyframes to avoid GPU OOM")

        # ── Save keyframe images ───────────────────────────────────────
        _, img_paths = self._save_keyframe_images(frames, kf_indices, rgb_dir)
        if len(img_paths) < 2:
            log.error("  Not enough images for MASt3R — falling back to sensor poses")
            return frames

        # ── Load images into MASt3R format ────────────────────────────
        log.info(f"  Loading {len(img_paths)} images at size {self.image_size}...")
        imgs = self._load_images([str(p) for p in img_paths], size=self.image_size)

        # ── Build pairs ────────────────────────────────────────────────
        sg = self.scene_graph
        if sg == "retrieval" and self.retrieval_model_path is None:
            log.warning("  retrieval scene_graph requested but no retrieval model path given; "
                        "falling back to swin-5")
            sg = "swin-5"

        try:
            if sg == "retrieval":
                # Try retrieval-based pairing (needs asmk)
                try:
                    from mast3r.retrieval.processor import Retriever
                    retriever = Retriever(
                        self.retrieval_model_path,
                        codebook=self.retrieval_codebook,
                        device=self.device,
                    )
                    pairs = retriever.make_pairs(imgs, topk=10)
                    log.info(f"  Retrieval pairs: {len(pairs)}")
                except Exception as e:
                    log.warning(f"  Retrieval pairing failed ({e}); falling back to swin-5")
                    pairs = self._make_pairs(imgs, scene_graph="swin-5", symmetrize=True)
            else:
                pairs = self._make_pairs(imgs, scene_graph=sg, symmetrize=True)
        except Exception as e:
            log.warning(f"  make_pairs failed ({e}); using sequential pairing")
            # Minimal sequential fallback
            pairs = []
            for i in range(len(imgs) - 1):
                pairs.append((imgs[i], imgs[i + 1]))
                pairs.append((imgs[i + 1], imgs[i]))

        log.info(f"  Image pairs: {len(pairs)}")

        # ── MASt3R inference ───────────────────────────────────────────
        log.info("  Running MASt3R inference...")
        t0 = time.time()
        output = self._inference(pairs, self._model, self.device,
                                 batch_size=1, verbose=False)
        log.info(f"  Inference done in {time.time()-t0:.1f}s")

        if self.stop_flag():
            log.info("  Stop flag — skipping global alignment")
            return frames

        # ── Sparse global alignment ────────────────────────────────────
        log.info(f"  Sparse global alignment (niter={self.niter})...")
        t0 = time.time()
        try:
            scene = self._sparse_ga(
                output,
                niter=self.niter,
                schedule=self.schedule,
                lr=self.lr,
                ftol=1e-4,
                min_conf_thr=self.min_conf_thr,
                matching_conf_thr=2.0,
                verbose=False,
            )
        except TypeError:
            # Older API without some kwargs
            scene = self._sparse_ga(
                output,
                niter=self.niter,
                schedule=self.schedule,
                lr=self.lr,
                verbose=False,
            )
        log.info(f"  Global alignment done in {time.time()-t0:.1f}s")

        # ── Extract refined poses ──────────────────────────────────────
        mast3r_poses_c2w = self._extract_poses_c2w(scene)

        if len(mast3r_poses_c2w) != len(kf_indices):
            log.warning(
                f"  Pose count mismatch: MASt3R={len(mast3r_poses_c2w)} "
                f"kf={len(kf_indices)} — truncating to min"
            )
            n = min(len(mast3r_poses_c2w), len(kf_indices))
            mast3r_poses_c2w = mast3r_poses_c2w[:n]
            kf_indices = kf_indices[:n]

        if len(mast3r_poses_c2w) == 0:
            log.error("  No poses extracted from MASt3R — using sensor poses")
            return frames

        # ── Optional scale alignment ───────────────────────────────────
        if self.align_scale and len(mast3r_poses_c2w) >= 2:
            sensor_kf_poses = [self._get_sensor_pose_c2w(frames[ki]) for ki in kf_indices]
            mast3r_poses_c2w, scale = _align_scale_to_sensor(
                mast3r_poses_c2w, sensor_kf_poses
            )
        else:
            scale = 1.0

        # ── Optional blending with sensor poses ────────────────────────
        if self.blend_alpha < 1.0:
            blended = []
            for ki, mp in zip(kf_indices, mast3r_poses_c2w):
                sp = self._get_sensor_pose_c2w(frames[ki])
                bp = _slerp_pose(sp, mp, self.blend_alpha)
                blended.append(bp)
            mast3r_poses_c2w = blended
            log.info(f"  Blended MASt3R ↔ sensor poses (alpha={self.blend_alpha:.2f})")

        # ── Propagate to all frames ────────────────────────────────────
        log.info(f"  Propagating poses to all {len(frames)} frames...")
        updated_frames = _propagate_poses_to_all_frames(
            frames, kf_indices, mast3r_poses_c2w
        )

        # ── Save a debug ply of MASt3R sparse pointcloud ───────────────
        try:
            pts3d = scene.get_pts3d()
            conf  = scene.get_conf()
            self._save_debug_ply(pts3d, conf, scale)
        except Exception:
            pass  # non-critical

        log.info(f"  MASt3R-SfM alignment complete  (scale={scale:.4f})")
        return updated_frames

    def _save_debug_ply(self, pts3d, conf, scale: float):
        try:
            import torch
            import open3d as o3d

            pts_list, col_list = [], []
            for p in pts3d:
                if isinstance(p, torch.Tensor):
                    p = p.detach().cpu().numpy()
                pts_list.append(p.reshape(-1, 3))

            pts_all = np.concatenate(pts_list, axis=0) * scale
            if conf is not None:
                conf_list = []
                for c in conf:
                    if isinstance(c, torch.Tensor):
                        c = c.detach().cpu().numpy()
                    conf_list.append(c.ravel())
                conf_all = np.concatenate(conf_list)
                mask = conf_all > self.min_conf_thr
                pts_all = pts_all[mask]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_all)
            dst = self.output_dir / "mast3r_sparse_cloud.ply"
            o3d.io.write_point_cloud(str(dst), pcd)
            log.info(f"  Debug cloud → {dst.name}  ({len(pts_all):,} pts)")
        except Exception as e:
            log.debug(f"  Debug ply save failed: {e}")