"""
modules/data_loader.py   (Phase 1 - "intrinsics-correct")

Loads RGB images, depth maps, and pose files.
Returns a list of Frame dataclasses - the common currency of the pipeline.

Changes vs the previous version
-------------------------------
[FIX-1]  The pose-file parser used to skip every line that starts with "#".
         The per-frame lines

             # intrinsics fx=1348.37 fy=1348.37 cx=958.41 cy=721.95 w=1920 h=1440
             # depth_metadata depth_scale=1.0 depth_resolution=256x192

         start with "#", so they were never read and every frame silently fell
         back to fx = 0.8*width (1536 px instead of ~1348 px => 64.0 deg
         instead of 70.9 deg horizontal FOV). All downstream modules (TSDF,
         ICP, texture projection) inherited the wrong camera.
[FIX-2]  The parser now reads key=value pairs from those comment lines.
[FIX-3]  depth_res is filled in (from the pose file, or by probing one depth
         file) so intrinsics can be scaled to the depth resolution.
[FIX-4]  New validate() / DataLoader.report: rotation orthonormality, intrinsic
         plausibility, RGB/depth/pose count and stem agreement.  A missing
         intrinsics block now logs a WARNING and is recorded in the report.

Backward compatible: Frame / CameraIntrinsics keep all previous fields; new
members are optional with defaults.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# --- Frame dataclass --------------------------------------------------------

@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width:  int
    height: int
    # where the numbers came from: "pose_file" | "default"
    source: str = "pose_file"

    def K(self) -> np.ndarray:
        # 3x3 intrinsic matrix (pixel units of the RGB image).
        return np.array([
            [self.fx,    0, self.cx],
            [   0,    self.fy, self.cy],
            [   0,       0,      1  ],
        ], dtype=np.float64)

    def scaled(self, width: int, height: int) -> "CameraIntrinsics":
        # Intrinsics for the same camera at another image resolution."""
        sx = float(width) / max(self.width, 1)
        sy = float(height) / max(self.height, 1)
        return CameraIntrinsics(self.fx * sx, self.fy * sy,
                                self.cx * sx, self.cy * sy,
                                int(width), int(height), self.source)

    def K_for(self, width: int, height: int) -> np.ndarray:
        return self.scaled(width, height).K()

    def hfov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.width / (2.0 * self.fx))))


@dataclass
class Frame:
    idx:        int
    rgb_path:   Path
    depth_path: Path
    c2w:        np.ndarray          # 4x4 camera-to-world transform
    intrinsics: CameraIntrinsics
    depth_scale: float = 1.0       # metres per raw unit
    depth_res:  Optional[Tuple[int, int]] = None  # (W, H) of depth image
    conf_path:  Optional[Path] = None             # optional per-pixel confidence (.conf)


# --- Parser helpers ----------------------------------------------------------

_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def _parse_kv_line(line: str, intr: dict, depth_meta: dict) -> bool:
    
    # Parse 'key=value' pairs from one line (comment prefix already stripped).
    # Returns True if at least one pair was consumed.
    
    found = False
    for k, v in _KV.findall(line):
        found = True
        kl = k.lower()
        if kl in ("depth_resolution", "depth_res") and "x" in v.lower():
            try:
                w, h = v.lower().split("x")
                depth_meta["depth_w"] = int(w)
                depth_meta["depth_h"] = int(h)
            except ValueError:
                pass
            continue
        try:
            val = float(v)
        except ValueError:
            continue
        if kl in ("fx", "fy", "cx", "cy", "w", "h"):
            intr[kl] = val
        else:
            depth_meta[kl] = val
    return found


def _parse_pose_file(pose_path: Path) -> List[dict]:
    """
    Flexible pose file parser.

    Layout 1 (one block per frame):
        # Frame <N>
        r00 r01 r02 tx
        r10 r11 r12 ty
        r20 r21 r22 tz
        0   0   0   1
        # intrinsics fx=... fy=... cx=... cy=... w=... h=...
        # depth_metadata depth_scale=... depth_resolution=WxH      (optional)

    Layout 2: 16 numbers per line (row-major 4x4), no intrinsics.
    """
    text   = pose_path.read_text(encoding="utf-8-sig")
    blocks = re.split(r"#\s*Frame\s+\d+", text)
    if len(blocks) <= 1:
        return _parse_plain_matrices(text)

    frames_data = []
    for block in blocks[1:]:           # first element is text before "# Frame 0"
        fd = _parse_block(block)
        if fd:
            frames_data.append(fd)
    return frames_data


def _parse_block(block: str) -> Optional[dict]:
    lines = [l.strip() for l in block.strip().splitlines() if l.strip()]

    matrix_rows: List[List[float]] = []
    intr: dict       = {}
    depth_meta: dict = {}

    for line in lines:
        if line.startswith("#"):
            # [FIX-1/2] comment lines can carry intrinsics / depth metadata
            _parse_kv_line(line.lstrip("#"), intr, depth_meta)
            continue
        if "=" in line:
            _parse_kv_line(line, intr, depth_meta)
            continue
        nums = line.split()
        if len(nums) == 4:
            try:
                matrix_rows.append([float(x) for x in nums])
            except ValueError:
                pass

    if len(matrix_rows) < 4:
        return None

    c2w = np.array(matrix_rows[:4], dtype=np.float64)
    return dict(c2w=c2w, intr=intr, depth_meta=depth_meta)


def _parse_plain_matrices(text: str) -> List[dict]:
    # Fallback: each non-blank line = 16 space-separated floats (row-major 4x4).
    frames_data = []
    for line in text.splitlines():
        nums = line.strip().split()
        if len(nums) == 16:
            c2w = np.array(nums, dtype=np.float64).reshape(4, 4)
            frames_data.append(dict(c2w=c2w, intr={}, depth_meta={}))
    return frames_data


# --- DataLoader ----------------------------------------------------------------

class DataLoader:
    """
    Parameters
    ----------
    data_dir     : root directory
    rgb_subdir   : subfolder for RGB images  (relative to data_dir)
    depth_subdir : subfolder for depth maps  (relative to data_dir)
    pose_file    : pose filename             (relative to data_dir)
    default_intrinsics : used ONLY if the pose file has no intrinsics
    depth_scale  : metres per raw depth unit
    """

    RGB_EXTS   = {".jpg", ".jpeg", ".png"}
    DEPTH_EXTS = {".exr"}
    CONF_EXT   = ".conf"

    def __init__(
        self,
        data_dir:     Path,
        rgb_subdir:   str = "rgb",
        depth_subdir: str = "depth",
        pose_file:    str = "poses.txt",
        default_intrinsics: Optional[CameraIntrinsics] = None,
        depth_scale:  float = 1.0,
    ):
        self.data_dir    = Path(data_dir)
        self.rgb_dir     = self.data_dir / rgb_subdir
        self.depth_dir   = self.data_dir / depth_subdir
        self.pose_path   = self.data_dir / pose_file
        self.default_intr = default_intrinsics
        self.depth_scale  = depth_scale
        self.report: Dict = {}

    # -- public -----------------------------------------------------------------

    def load_all(self) -> List[Frame]:
        self._validate_dirs()
        rgb_paths   = self._sorted_images(self.rgb_dir,   self.RGB_EXTS)
        depth_paths = self._sorted_images(self.depth_dir, self.DEPTH_EXTS)
        pose_data   = _parse_pose_file(self.pose_path)

        n = min(len(rgb_paths), len(depth_paths), len(pose_data))
        if n == 0:
            raise RuntimeError("No matching frames found. Check directory structure.")
        count_mismatch = (len(rgb_paths) != len(depth_paths)
                          or len(rgb_paths) != len(pose_data))
        if count_mismatch:
            log.warning(
                f"Count mismatch: RGB={len(rgb_paths)} depth={len(depth_paths)} "
                f"poses={len(pose_data)}. Using first {n} frames."
            )

        # depth resolution: from pose file if given, otherwise probe one file
        probed_res = None
        if not any("depth_w" in pd.get("depth_meta", {}) for pd in pose_data[:n]):
            probed_res = self._probe_depth_res(depth_paths[0])

        frames: List[Frame] = []
        n_default_intr = 0
        for i in range(n):
            pd   = pose_data[i]
            intr, from_file = self._resolve_intrinsics(pd["intr"], rgb_paths[i])
            if not from_file:
                n_default_intr += 1
            dm   = pd.get("depth_meta", {})
            dep_res = None
            if "depth_w" in dm and "depth_h" in dm:
                dep_res = (int(dm["depth_w"]), int(dm["depth_h"]))
            elif probed_res is not None:
                dep_res = probed_res

            conf = depth_paths[i].with_suffix(self.CONF_EXT)
            frames.append(Frame(
                idx         = i,
                rgb_path    = rgb_paths[i],
                depth_path  = depth_paths[i],
                c2w         = pd["c2w"],
                intrinsics  = intr,
                depth_scale = float(dm.get("depth_scale", self.depth_scale)),
                depth_res   = dep_res,
                conf_path   = conf if conf.is_file() else None,
            ))

        if n_default_intr:
            log.warning(
                f"  {n_default_intr}/{n} frames have NO intrinsics in the pose "
                f"file - using a guessed camera (fx = 0.8*width). Geometry and "
                f"texture projection will be inaccurate. Add an '# intrinsics' "
                f"line per frame or pass default_intrinsics.")

        self.report = self.validate(frames, rgb_paths, depth_paths,
                                    count_mismatch, n_default_intr)
        log.info(f"DataLoader: {n} frames ready  "
                 f"(fx={frames[0].intrinsics.fx:.2f}, "
                 f"hfov={frames[0].intrinsics.hfov_deg():.1f} deg, "
                 f"depth_res={frames[0].depth_res})")
        return frames

    # -- validation ---------------------------------------------------------------

    @staticmethod
    def validate(frames: List[Frame], rgb_paths=None, depth_paths=None,
                 count_mismatch: bool = False, n_default_intr: int = 0) -> Dict:
        """
        Cheap structural checks.  Returns a dict with 'errors' (fatal for
        geometry) and 'warnings' (suspicious but usable).
        """
        errors: List[str] = []
        warnings: List[str] = []

        R = np.array([f.c2w[:3, :3] for f in frames])
        finite = bool(np.isfinite(np.array([f.c2w for f in frames])).all())
        if not finite:
            errors.append("non-finite values in poses")
            R = np.nan_to_num(R)
        orth_err = float(np.abs(np.einsum("nij,nkj->nik", R, R)
                                - np.eye(3)).max())
        dets = np.linalg.det(R)
        bottom = np.array([f.c2w[3] for f in frames])
        if orth_err > 1e-2:
            errors.append(f"rotations not orthonormal (max err {orth_err:.2e}) - "
                          f"scale or shear in pose matrices?")
        elif orth_err > 1e-4:
            warnings.append(f"rotations slightly non-orthonormal ({orth_err:.2e})")
        if (np.abs(dets - 1.0) > 1e-2).any():
            errors.append("rotation determinant != +1 (mirrored pose?)")
        if np.nan_to_num(np.abs(bottom - np.array([0, 0, 0, 1.0]))).max() > 1e-6:
            warnings.append("pose bottom row is not [0 0 0 1]")

        fxs = np.array([f.intrinsics.fx for f in frames])
        ws  = np.array([f.intrinsics.width for f in frames])
        ratio = fxs / ws
        if ((ratio < 0.4) | (ratio > 3.0)).any():
            errors.append("implausible focal length (fx/width outside 0.4..3.0)")
        if n_default_intr:
            warnings.append(f"{n_default_intr} frames use guessed intrinsics")
        if (fxs.max() - fxs.min()) / max(float(np.median(fxs)), 1e-9) > 0.05:
            warnings.append("focal length varies by >5% between frames")

        if count_mismatch:
            warnings.append("rgb / depth / pose counts differ")
        if rgb_paths is not None and depth_paths is not None:
            n = len(frames)
            bad = sum(1 for i in range(n)
                      if rgb_paths[i].stem != depth_paths[i].stem)
            if bad:
                warnings.append(
                    f"{bad}/{n} frames: RGB and depth file names differ "
                    f"(pairing is by sorted order)")

        pos = np.array([f.c2w[:3, 3] for f in frames])
        steps = (np.linalg.norm(np.diff(pos, axis=0), axis=1)
                 if len(pos) > 1 else np.zeros(1))
        rep = dict(
            n_frames=len(frames),
            fx_median=float(np.median(fxs)),
            hfov_deg=float(frames[0].intrinsics.hfov_deg()),
            rot_orthonormality_max_err=orth_err,
            trajectory_span_m=float(np.linalg.norm(pos.max(0) - pos.min(0))),
            path_length_m=float(steps.sum()),
            max_step_m=float(steps.max()),
            errors=errors, warnings=warnings,
        )
        for e in errors:
            log.error(f"  INPUT ERROR: {e}")
        for w in warnings:
            log.warning(f"  input warning: {w}")
        return rep

    # -- private ------------------------------------------------------------------

    def _validate_dirs(self):
        for d in (self.rgb_dir, self.depth_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"Directory not found: {d}")
        if not self.pose_path.is_file():
            raise FileNotFoundError(f"Pose file not found: {self.pose_path}")

    @staticmethod
    def _sorted_images(folder: Path, exts: set) -> List[Path]:
        return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)

    @staticmethod
    def _probe_depth_res(depth_path: Path) -> Optional[Tuple[int, int]]:
        """(W, H) of one depth file; None if it cannot be read."""
        try:
            if depth_path.suffix.lower() == ".npy":
                h, w = np.load(str(depth_path), mmap_mode="r").shape[:2]
                return int(w), int(h)
            from src.modules.exr_reader import read_exr_depth
            h, w = read_exr_depth(depth_path).shape[:2]
            return int(w), int(h)
        except Exception as e:                       # noqa: BLE001
            log.debug(f"  depth resolution probe failed: {e}")
            return None

    def _resolve_intrinsics(self, intr_dict: dict, rgb_path: Path):
        """
        Returns (CameraIntrinsics, from_file: bool).
        Values from the pose file always win; defaults are used only when the
        pose file really has no focal length.
        """
        w = int(intr_dict.get("w", 0))
        h = int(intr_dict.get("h", 0))

        if w == 0 or h == 0:
            try:
                from PIL import Image
                with Image.open(rgb_path) as im:
                    w, h = im.size
            except Exception:                        # noqa: BLE001
                w, h = 1920, 1440

        if "fx" in intr_dict:
            fx = float(intr_dict["fx"])
            fy = float(intr_dict.get("fy", fx))
            cx = float(intr_dict.get("cx", w / 2.0))
            cy = float(intr_dict.get("cy", h / 2.0))
            return CameraIntrinsics(fx, fy, cx, cy, w, h, "pose_file"), True

        if self.default_intr is not None:
            d = self.default_intr
            return CameraIntrinsics(d.fx, d.fy, d.cx, d.cy,
                                    d.width, d.height, "default"), False

        fx = w * 0.8
        return CameraIntrinsics(fx, fx, w / 2.0, h / 2.0, w, h, "default"), False
