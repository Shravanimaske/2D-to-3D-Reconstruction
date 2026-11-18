"""
da3_refiner.py — Depth Anything 3 depth refinement stage.

Replaces or blends sensor depth with DA3 metric depth predictions.
DA3 is cross-view consistent, so it dramatically improves ICP quality downstream.

Install the official DA3 package (REQUIRED):
    pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

Usage:
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained("depth-anything/da3-large")
    result = model.inference(images)
    result.depth        # [N, H, W] float32
    result.conf         # [N, H, W] float32
    result.extrinsics   # [N, 3, 4] float32  (w2c poses)
    result.intrinsics   # [N, 3, 3] float32
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, List
import cv2

log = logging.getLogger("da3_refiner")

# ── Frame attribute helpers ─────────────────────────────────────────────────

def _g(frame, key, default=None):
    try:
        return getattr(frame, key)
    except AttributeError:
        pass
    try:
        return frame[key]
    except (KeyError, TypeError):
        pass
    return default


def _set(frame, key, val):
    if hasattr(frame, key):
        try:
            setattr(frame, key, val)
            return
        except Exception:
            pass
    try:
        frame[key] = val
    except (TypeError, KeyError):
        pass


# ── Model name normalisation ────────────────────────────────────────────────
# The HuggingFace repos use UPPERCASE (DA3-LARGE) but the official
# from_pretrained() may expect lowercase.  We map both.

_MODEL_NAME_MAP = {
    "depth-anything/da3-large": "depth-anything/DA3-LARGE",
    "depth-anything/da3-small": "depth-anything/DA3-SMALL",
    "depth-anything/da3-base":  "depth-anything/DA3-BASE",
    "depth-anything/DA3-LARGE": "depth-anything/DA3-LARGE",
    "depth-anything/DA3-SMALL": "depth-anything/DA3-SMALL",
    "depth-anything/DA3-BASE":  "depth-anything/DA3-BASE",
}


class DA3Refiner:
    """
    Refines sensor depth using Depth Anything 3 metric depth predictions.

    Modes:
      'replace'  — use DA3 depth entirely (best when sensor depth is very noisy)
      'blend'    — weighted blend of sensor + DA3  (safest default)
      'fill'     — use DA3 only where sensor depth is missing/zero
      'conf_blend' — blend weighted by DA3 confidence (best quality)
    """

    def __init__(
        self,
        output_dir: Path,
        depth_scale: float = 1.0,
        mode: str = "blend",
        sensor_weight: float = 0.6,
        da3_weight: float = 0.4,
        model_name: str = "depth-anything/DA3-LARGE",
        batch_size: int = 4,
        max_frames: int = 9999,
        device: str = "auto",
        rgb_dir: Optional[Path] = None,
        half_precision: bool = True,
        use_confidence: bool = True,          # use DA3 confidence maps
        export_depth: bool = True,            # save refined depth as .npy
        export_poses: bool = True,            # save DA3 camera poses
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.depth_scale = depth_scale
        self.mode = mode
        self.sensor_w = sensor_weight
        self.da3_w = da3_weight
        self.model_name = _MODEL_NAME_MAP.get(model_name, model_name)
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.rgb_dir = Path(rgb_dir) if rgb_dir else None
        self.half = half_precision
        self.use_confidence = use_confidence
        self.export_depth = export_depth
        self.export_poses = export_poses

        if device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None

    # ── Model loading ────────────────────────────────────────────────────────

    @staticmethod
    def _mock_moviepy():
        """
        Create a lightweight stub for moviepy.editor ONLY.

        moviepy 2.x removed the 'editor' submodule, which blocks
        the import of depth_anything_3.api.  We provide a minimal
        stub so the import succeeds.

        IMPORTANT: Do NOT mock xformers, e3nn, pycolmap, etc.
        DA3 has its own try/except around those and must see the
        real ImportError so it falls back to vanilla PyTorch code.
        """
        import sys
        import types

        for name in ('moviepy', 'moviepy.editor'):
            if name not in sys.modules:
                sys.modules[name] = types.ModuleType(name)

        # Link parent → child so `import moviepy.editor` resolves
        sys.modules['moviepy'].editor = sys.modules['moviepy.editor']

    @staticmethod
    def _unmock_moviepy():
        """Remove the moviepy stubs so they don't poison other imports."""
        import sys
        for name in ('moviepy.editor', 'moviepy'):
            sys.modules.pop(name, None)

    def _load_model(self):
        if self._model is not None:
            return

        import torch
        log.info(f"  Loading DA3 model: {self.model_name}  device={self.device}")

        # ── Strategy 1: Official depth_anything_3 package ──────────────
        # pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git --no-deps
        # pip install torch torchvision transformers accelerate pillow opencv-python timm
        try:
            self._mock_moviepy()
            from depth_anything_3.api import DepthAnything3  # type: ignore

            self._model = DepthAnything3.from_pretrained(self.model_name)
            self._model = self._model.to(device=self.device)
            if self.half and self.device == "cuda":
                self._model = self._model.half()
            self._model.eval()
            log.info("  DA3 loaded via official depth_anything_3 package")
            return
        except ImportError as e:
            log.warning(f"  depth_anything_3 import failed: {e}")
        except Exception as e:
            log.warning(f"  Official package load failed ({e})")
        finally:
            # Always clean up — if DA3 succeeded the module is already
            # imported; if it failed we must not leave stubs that break
            # torchvision for the V2 fallback.
            self._unmock_moviepy()

        # ── Strategy 2: Depth Anything V2 via transformers ─────────────
        try:
            from transformers import (
                AutoModelForDepthEstimation,
                AutoImageProcessor,
            )

            v2_name = "depth-anything/Depth-Anything-V2-Large-hf"
            log.info(f"  DA3 unavailable — falling back to V2: {v2_name}")
            self._processor = AutoImageProcessor.from_pretrained(v2_name)
            dtype = (
                torch.float16
                if (self.half and self.device == "cuda")
                else torch.float32
            )
            self._model = AutoModelForDepthEstimation.from_pretrained(
                v2_name, torch_dtype=dtype,
            ).to(self.device)
            self._model.eval()
            self._is_v2 = True
            log.info("  Depth Anything V2 loaded via transformers")
            return
        except Exception as e:
            log.error(f"  V2 fallback also failed: {e}")

        raise RuntimeError(
            "Cannot load any depth model.\n\n"
            "SOLUTION — install the official DA3 package:\n"
            "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git --no-deps\n"
            "  pip install torch torchvision transformers accelerate pillow opencv-python timm\n\n"
            "Or for V2 fallback:\n"
            "  pip install transformers accelerate"
        )

    # ── RGB / depth helpers ──────────────────────────────────────────────────

    def _get_rgb(self, frame, frame_idx: int) -> Optional[np.ndarray]:
        """Returns HxWx3 uint8 RGB array."""
        for attr in ('rgb', 'color', 'image', 'color_image', 'rgb_image'):
            img = _g(frame, attr)
            if isinstance(img, np.ndarray) and img.ndim == 3:
                if img.shape[2] == 3:
                    return img.astype(np.uint8)
                if img.shape[2] == 4:
                    return img[:, :, :3].astype(np.uint8)

        for attr in ('rgb_path', 'color_path', 'image_path', 'rgb_file'):
            p = _g(frame, attr)
            if p is not None:
                try:
                    img = cv2.imread(str(p))
                    if img is not None:
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                except Exception:
                    pass

        if self.rgb_dir:
            try:
                from src.modules.dense_aligner import _natural_key
            except ImportError:
                import re as _re

                _natural_key = lambda s: [
                    int(t) if t.isdigit() else t.lower()
                    for t in _re.split(r'(\d+)', str(s))
                ]

            rgb_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
            files = sorted(
                self.rgb_dir.glob("*"), key=lambda p: _natural_key(p.stem)
            )
            files = [f for f in files if f.suffix.lower() in rgb_exts]
            if frame_idx < len(files):
                img = cv2.imread(str(files[frame_idx]))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return None

    def _get_sensor_depth(self, frame) -> Optional[np.ndarray]:
        """Returns HxW float64 depth in raw sensor units."""
        for attr in (
            'depth', 'depth_data', 'depth_image', 'depth_map', 'depth_array',
        ):
            d = _g(frame, attr)
            if isinstance(d, np.ndarray) and d.ndim == 2:
                return d.astype(np.float64)
        return None

    # ── DA3 inference ────────────────────────────────────────────────────────

    def _predict_da3(
        self, rgb_list: List[np.ndarray],
    ) -> dict:
        """
        Run DA3 model.inference() on a batch of RGB images.

        Returns dict with keys:
            depth       — list of HxW float32 metric depth (metres)
            conf        — list of HxW float32 confidence  (or None)
            extrinsics  — Nx3x4 float32 w2c poses          (or None)
            intrinsics  — Nx3x3 float32 camera intrinsics   (or None)
        """
        result = self._model.inference(
            rgb_list,
            export_dir=None,       # don't write files
            export_format=None,
        )

        depths = []
        confs = []

        N = result.depth.shape[0]
        for i in range(N):
            d = result.depth[i]                 # (H, W) float32
            c = result.conf[i] if result.conf is not None else None

            # Resize to match original RGB if needed
            h, w = rgb_list[i].shape[:2]
            if d.shape != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            if c is not None and c.shape != (h, w):
                c = cv2.resize(c, (w, h), interpolation=cv2.INTER_LINEAR)

            depths.append(d)
            confs.append(c)

        return {
            'depth': depths,
            'conf': confs,
            'extrinsics': result.extrinsics if hasattr(result, 'extrinsics') else None,
            'intrinsics': result.intrinsics if hasattr(result, 'intrinsics') else None,
        }

    def _predict_v2(
        self, rgb_list: List[np.ndarray],
    ) -> dict:
        """
        Fallback: Depth Anything V2 via transformers.
        Returns dict with same keys (conf/extrinsics/intrinsics = None).
        """
        import torch
        from PIL import Image

        pil_imgs = [Image.fromarray(img) for img in rgb_list]
        inputs = self._processor(
            images=pil_imgs, return_tensors="pt"
        ).to(self.device)

        if self.half and self.device == "cuda":
            inputs = {
                k: v.half() if v.dtype == torch.float32 else v
                for k, v in inputs.items()
            }

        with torch.no_grad():
            outputs = self._model(**inputs)

        pred = outputs.predicted_depth            # (B, H, W)

        depths = []
        for i, rgb in enumerate(rgb_list):
            h, w = rgb.shape[:2]
            d = pred[i].float().cpu().numpy()
            if d.shape != (h, w):
                d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            depths.append(d.astype(np.float32))

        return {
            'depth': depths,
            'conf': [None] * len(depths),
            'extrinsics': None,
            'intrinsics': None,
        }

    # ── Per-frame fusion ─────────────────────────────────────────────────────

    def _fuse_depth(
        self,
        sensor_raw: Optional[np.ndarray],    # HxW raw sensor units
        da3_metric: np.ndarray,              # HxW metres
        da3_conf: Optional[np.ndarray] = None,  # HxW confidence [0,1]
        fx: float = 500.0,
    ) -> np.ndarray:
        """
        Fuse sensor and DA3 depth according to self.mode.
        Returns HxW float64 in raw sensor units.
        """
        da3_sensor = da3_metric / self.depth_scale    # metres → sensor units

        if self.mode == "replace":
            return da3_sensor.astype(np.float64)

        if self.mode == "fill":
            if sensor_raw is None:
                return da3_sensor.astype(np.float64)
            result = sensor_raw.copy()
            missing = (sensor_raw <= 0) | ~np.isfinite(sensor_raw)
            result[missing] = da3_sensor[missing]
            return result.astype(np.float64)

        if self.mode == "conf_blend" and da3_conf is not None:
            # ── Confidence-weighted blend ────────────────────────────────
            if sensor_raw is None:
                return da3_sensor.astype(np.float64)

            # Normalise confidence to [0, 1]
            c = da3_conf.astype(np.float64)
            c_min, c_max = c.min(), c.max()
            if c_max > c_min:
                c = (c - c_min) / (c_max - c_min)
            else:
                c = np.full_like(c, 0.5)

            valid_sensor = (sensor_raw > 0) & np.isfinite(sensor_raw)
            valid_da3 = (da3_sensor > 0) & np.isfinite(da3_sensor)

            result = np.zeros_like(sensor_raw, dtype=np.float64)

            # Both valid: confidence-weighted
            both = valid_sensor & valid_da3
            w_da3 = c[both] * self.da3_w
            w_sensor = (1.0 - c[both]) * self.sensor_w
            w_total = w_da3 + w_sensor
            w_total[w_total == 0] = 1.0       # avoid /0
            result[both] = (w_da3 * da3_sensor[both]
                            + w_sensor * sensor_raw[both]) / w_total

            # Only one valid
            only_sensor = valid_sensor & ~valid_da3
            result[only_sensor] = sensor_raw[only_sensor]

            only_da3 = ~valid_sensor & valid_da3
            result[only_da3] = da3_sensor[only_da3]

            return result

        # ── mode == 'blend' (default) ────────────────────────────────────
        if sensor_raw is None:
            return da3_sensor.astype(np.float64)

        valid_sensor = (sensor_raw > 0) & np.isfinite(sensor_raw)
        valid_da3 = (da3_sensor > 0) & np.isfinite(da3_sensor)

        result = np.zeros_like(sensor_raw, dtype=np.float64)

        both = valid_sensor & valid_da3
        result[both] = (self.sensor_w * sensor_raw[both]
                        + self.da3_w * da3_sensor[both])

        only_sensor = valid_sensor & ~valid_da3
        result[only_sensor] = sensor_raw[only_sensor]

        only_da3 = ~valid_sensor & valid_da3
        result[only_da3] = da3_sensor[only_da3]

        return result

    # ── Main entry ───────────────────────────────────────────────────────────

    def refine(self, frames: list) -> list:
        n = min(len(frames), self.max_frames)
        log.info(f"═══ DA3Refiner: {n} frames  mode={self.mode}  "
                 f"sensor_w={self.sensor_w}  da3_w={self.da3_w} ═══")

        self._load_model()

        # Detect which API we have
        is_da3 = hasattr(self._model, 'inference')
        is_v2 = not is_da3
        if is_v2:
            log.warning("  Using DA3 V2 fallback — no confidence maps or poses")

        # Focal length from first frame
        try:
            from src.modules.dense_aligner import _get_K
            K = _get_K(frames[0]) if frames else None
        except ImportError:
            K = None
        fx = float(K[0, 0]) if K is not None else 500.0

        replaced = filled = blended = conf_blended = skipped = 0

        for batch_start in range(0, n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n)
            batch_indices = list(range(batch_start, batch_end))

            # Collect RGB for this batch
            rgb_batch: List[np.ndarray] = []
            valid_idx: List[int] = []
            for i in batch_indices:
                rgb = self._get_rgb(frames[i], i)
                if rgb is not None:
                    rgb_batch.append(rgb)
                    valid_idx.append(i)
                else:
                    skipped += 1
                    log.debug(f"  Frame {i}: no RGB — skipping DA3")

            if not rgb_batch:
                continue

            # Run inference
            try:
                if is_da3:
                    pred = self._predict_da3(rgb_batch)
                else:
                    pred = self._predict_v2(rgb_batch)
            except Exception as e:
                log.warning(f"  DA3 batch {batch_start}-{batch_end} failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Save DA3 poses (once, from first batch)
            if self.export_poses and pred.get('extrinsics') is not None:
                pose_path = self.output_dir / "da3_poses.npz"
                if not pose_path.exists():      # save only once
                    np.savez(
                        pose_path,
                        extrinsics=pred['extrinsics'],
                        intrinsics=pred['intrinsics'],
                    )
                    log.info(f"  Saved DA3 poses → {pose_path}")

            # Fuse each frame
            for local_i, frame_i in enumerate(valid_idx):
                da3_m = pred['depth'][local_i]
                da3_c = pred['conf'][local_i] if local_i < len(pred['conf']) else None

                sensor_raw = self._get_sensor_depth(frames[frame_i])
                fused = self._fuse_depth(sensor_raw, da3_m, da3_c, fx)

                _set(frames[frame_i], 'depth', fused)

                # Also store DA3 outputs on the frame for downstream use
                _set(frames[frame_i], 'da3_depth', da3_m)
                if da3_c is not None:
                    _set(frames[frame_i], 'da3_conf', da3_c)

                # Count
                if self.mode == "replace":      replaced += 1
                elif self.mode == "fill":       filled += 1
                elif self.mode == "conf_blend": conf_blended += 1
                else:                           blended += 1

            # Save depth maps to disk
            if self.export_depth:
                import shutil
                free_bytes = shutil.disk_usage(self.output_dir).free
                # Each depth map ≈ H*W*4 bytes; use 50 MB as a safety floor
                if free_bytes < 50 * 1024 * 1024:
                    log.warning(
                        f"  Low disk space ({free_bytes / 1e9:.2f} GB free) — "
                        f"disabling depth export to avoid write errors."
                    )
                    self.export_depth = False
                else:
                    for local_i, frame_i in enumerate(valid_idx):
                        dpath = self.output_dir / f"da3_depth_{frame_i:06d}.npy"
                        try:
                            np.save(dpath, pred['depth'][local_i])
                        except OSError as e:
                            log.warning(
                                f"  Failed to save {dpath.name}: {e} — "
                                f"disabling further depth exports."
                            )
                            self.export_depth = False
                            break

            if batch_end % 200 == 0 or batch_end == n:
                log.info(f"    {batch_end}/{n} frames processed")

        log.info(f"  DA3 done: replaced={replaced} blended={blended} "
                 f"conf_blended={conf_blended} filled={filled} "
                 f"skipped={skipped}")
        return frames