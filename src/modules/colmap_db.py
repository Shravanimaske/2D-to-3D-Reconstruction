"""
modules/colmap_db.py  — v6 FAST

Speed improvements:
  1. Subsampling: only keyframes go to COLMAP (caller controls this)
  2. GPU SIFT enabled when available (3-5× faster extraction)
  3. Reduced default features (2048 vs 4096)
  4. Reduced match overlap for sequential matcher
  5. Mapper runs with prior poses from ARKit → faster convergence
  6. pycolmap write_camera/write_image positional bool fix
"""

import logging
import os
import sqlite3
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

COLMAP_PATH_L = r"E:\Rohit AI\Colmap\bin\colmap.exe"
log = logging.getLogger(__name__)
COLMAP_PINHOLE = 1


def _try_import_pycolmap():
    try:
        import pycolmap
        return pycolmap
    except ImportError:
        return None


def _find_qt_plugin_path(colmap_bin: str) -> str | None:
    colmap_dir = Path(colmap_bin).resolve().parent
    candidates = [
        colmap_dir,
        colmap_dir / "plugins",
        colmap_dir.parent / "plugins",
        colmap_dir.parent / "lib" / "qt" / "plugins",
        colmap_dir.parent / "lib" / "plugins",
        colmap_dir / "qt" / "plugins",
        colmap_dir / "Qt" / "plugins",
    ]
    for base in candidates:
        platforms = base / "platforms"
        if platforms.is_dir():
            dlls = list(platforms.glob("q*.dll"))
            if dlls:
                return str(base)
    return None


def _run(cmd, timeout=None, stream_log=False):
    colmap_bin = cmd[0]
    env = os.environ.copy()

    qt_plugin_path = _find_qt_plugin_path(colmap_bin)
    if qt_plugin_path:
        env["QT_PLUGIN_PATH"] = qt_plugin_path
        env["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(Path(qt_plugin_path) / "platforms")

    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")

    if stream_log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, env=env)
        lines = []
        for line in proc.stdout:
            line = line.rstrip()
            lines.append(line)
            log.info(f"  [colmap] {line}")
        proc.wait()
        return proc.returncode, "\n".join(lines)
    else:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return r.returncode, r.stdout + r.stderr


def _probe_mapper_prefix(colmap_bin):
    try:
        _, txt = _run([colmap_bin, "mapper", "--help"], timeout=15)
        if "--mapper." in txt and "--Mapper." not in txt:
            return "mapper"
    except Exception:
        pass
    return "Mapper"


class COLMAPDatabaseBuilder:
    def __init__(self, output_dir: Path,
                 colmap_bin: str = COLMAP_PATH_L,
                 fast_mode: bool = True,
                 mapper_timeout_sec: int = 900,
                 max_features: int = 2048,       # ★ NEW — was hardcoded 4096
                 match_overlap: int = 4,          # ★ NEW — was auto-computed
                 use_gpu: bool = True):           # ★ NEW — try GPU by default
        self.output_dir = Path(output_dir)
        self.colmap_bin = colmap_bin
        self.fast_mode  = fast_mode
        self.mapper_timeout_sec = int(mapper_timeout_sec)
        self.max_features = max_features
        self.match_overlap = match_overlap
        self.use_gpu = use_gpu
        self.db_path    = self.output_dir / "colmap.db"
        self.sparse_dir = self.output_dir / "sparse"
        self.image_list = self.output_dir / "image_list.txt"

    def build(self, frames, skip_feature_matching: bool = False) -> Path:
        self.sparse_dir.mkdir(parents=True, exist_ok=True)

        # ★ FIX: Delete old DB before doing ANYTHING — prevents duplicate ID crash
        if self.db_path.exists():
            self.db_path.unlink()
            log.info(f"  Deleted old DB -> {self.db_path}")

        self._write_image_list(frames)

        pycolmap = _try_import_pycolmap()
        if pycolmap:
            self._build_with_pycolmap(frames, pycolmap)
        else:
            self._build_with_sqlite(frames)

        if not skip_feature_matching:
            self._run_colmap_pipeline(frames)
        else:
            log.info("Skipping COLMAP feature matching — using ARKit poses directly.")
            self._write_known_poses(frames)

        return self.db_path

    def _write_image_list(self, frames):
        with open(self.image_list, "w") as f:
            for fr in frames:
                f.write(fr.rgb_path.name + "\n")
        log.info(f"Image list -> {self.image_list}  ({len(frames)} images, relative names)")

    # ── pycolmap path (fast, no subprocess) ────────────────────────────────────

    def _build_with_pycolmap(self, frames, pycolmap):
        log.info("Using pycolmap to build database")

        # ★ FIX: DB file already deleted in build(), but double-check
        if self.db_path.exists():
            self.db_path.unlink()

        db = None
        try:
            db = pycolmap.Database(str(self.db_path))
        except TypeError:
            try:
                db = pycolmap.Database()
                db.open(str(self.db_path))
            except AttributeError:
                pass

        if db is None:
            self._build_with_sqlite(frames)
            return

        try:
            # ★ FIX: Group frames by intrinsics — shared camera_id
            # Multiple frames with same fx,fy,cx,cy,w,h share one camera
            cam_map = {}   # (w,h,fx,fy,cx,cy) -> camera_id
            cam_counter = 1

            for seq_id, fr in enumerate(frames):
                intr = fr.intrinsics
                key = (intr.width, intr.height,
                       round(intr.fx, 2), round(intr.fy, 2),
                       round(intr.cx, 2), round(intr.cy, 2))

                if key not in cam_map:
                    cam = pycolmap.Camera(
                        model="PINHOLE",
                        width=intr.width,
                        height=intr.height,
                        params=[intr.fx, intr.fy, intr.cx, intr.cy],
                        camera_id=cam_counter,
                    )
                    db.write_camera(cam, True)    # positional bool
                    cam_map[key] = cam_counter
                    cam_counter += 1

                cam_id = cam_map[key]

                # ★ FIX: Use sequential image_id (0-based) — NOT fr.idx
                # fr.idx has gaps (0, 8, 16...) from subsampling and may
                # collide with camera_ids
                img = pycolmap.Image(
                    name=fr.rgb_path.name,
                    camera_id=cam_id,
                    image_id=seq_id + 1,
                )
                db.write_image(img, True)         # positional bool

            log.info(f"  pycolmap: {len(frames)} images, "
                     f"{len(cam_map)} unique cameras")

        except Exception as e:
            log.warning(f"pycolmap DB write failed ({e}) — falling back to sqlite3")
            try:
                db.close()
            except Exception:
                pass
            self._build_with_sqlite(frames)
            return

        db.close()
        log.info(f"pycolmap DB -> {self.db_path}")

    def _build_with_sqlite(self, frames):
        log.info("Using sqlite3 to build COLMAP database")
        if self.db_path.exists():
            self.db_path.unlink()

        con = sqlite3.connect(str(self.db_path))
        cur = con.cursor()
        self._create_schema(cur)

        # ★ FIX: Same grouping logic — shared camera_id per unique intrinsics
        cam_map = {}   # (w,h,fx,fy,cx,cy) -> camera_id
        cam_counter = 1

        for seq_id, fr in enumerate(frames):
            intr = fr.intrinsics
            key = (intr.width, intr.height,
                   round(intr.fx, 2), round(intr.fy, 2),
                   round(intr.cx, 2), round(intr.cy, 2))

            if key not in cam_map:
                params_blob = _pack_doubles([intr.fx, intr.fy, intr.cx, intr.cy])
                cur.execute(
                    "INSERT OR REPLACE INTO cameras VALUES (?,?,?,?,?,1)",
                    (cam_counter, COLMAP_PINHOLE, intr.width,
                     intr.height, params_blob),
                )
                cam_map[key] = cam_counter
                cam_counter += 1

            # ★ FIX: Use seq_id for image_id, cam_map[key] for camera_id
            qw, qx, qy, qz, tx, ty, tz = _c2w_to_quat_trans(fr.c2w)
            cur.execute(
                "INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?,?)",
                (seq_id + 1, fr.rgb_path.name, cam_map[key],
                 qw, qx, qy, qz, tx, ty, tz),
            )

        con.commit()
        con.close()
        log.info(f"sqlite3 DB -> {self.db_path}  "
                 f"({len(frames)} images, {len(cam_map)} cameras)")

    @staticmethod
    def _create_schema(cur):
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS cameras (
            camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            model INTEGER NOT NULL, width INTEGER NOT NULL,
            height INTEGER NOT NULL, params BLOB,
            prior_focal_length INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS images (
            image_id  INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
            name      TEXT NOT NULL UNIQUE,
            camera_id INTEGER NOT NULL,
            prior_qw REAL, prior_qx REAL, prior_qy REAL, prior_qz REAL,
            prior_tx REAL, prior_ty REAL, prior_tz REAL
        );
        CREATE TABLE IF NOT EXISTS keypoints (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
        );
        CREATE TABLE IF NOT EXISTS descriptors (
            image_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
        );
        CREATE TABLE IF NOT EXISTS matches (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL, cols INTEGER NOT NULL, data BLOB
        );
        CREATE TABLE IF NOT EXISTS two_view_geometries (
            pair_id INTEGER PRIMARY KEY NOT NULL,
            rows INTEGER NOT NULL, cols INTEGER NOT NULL,
            data BLOB, config INTEGER,
            F BLOB, E BLOB, H BLOB, qvec BLOB, tvec BLOB
        );
        """)

    # ── COLMAP pipeline (FAST) ────────────────────────────────────────────────

    def _run_colmap_pipeline(self, frames):
        if not self._colmap_available():
            log.warning("COLMAP binary not found — writing known poses as fallback.")
            self._write_known_poses(frames)
            return

        image_dir = str(Path(frames[0].rgb_path).parent)
        n = len(frames)
        log.info(f"COLMAP pipeline: {n} frames  image_dir={image_dir}")

        sift_gpu_on, sift_gpu_off = self._detect_sift_gpu_flag()
        match_gpu_on, match_gpu_off = self._detect_matcher_gpu_flag()

        # ── Step 1: Feature extraction (GPU preferred) ────────────────────────
        feat_cmds = []

        if self.use_gpu and sift_gpu_on:
            feat_cmds.append(("GPU", [
                self.colmap_bin, "feature_extractor",
                "--database_path",                   str(self.db_path),
                "--image_path",                      image_dir,
                "--image_list_path",                 str(self.image_list),
                "--ImageReader.camera_model",        "PINHOLE",
                "--ImageReader.single_camera",       "1",
                "--SiftExtraction.max_num_features", str(self.max_features),
            ] + sift_gpu_on))

        cpu_base = [
            self.colmap_bin, "feature_extractor",
            "--database_path",                   str(self.db_path),
            "--image_path",                      image_dir,
            "--image_list_path",                 str(self.image_list),
            "--ImageReader.camera_model",        "PINHOLE",
            "--ImageReader.single_camera",       "1",
            "--SiftExtraction.max_num_features", str(self.max_features),
        ]
        if sift_gpu_off:
            feat_cmds.append(("CPU", cpu_base + sift_gpu_off))
        feat_cmds.append(("CPU-minimal", [
            self.colmap_bin, "feature_extractor",
            "--database_path",   str(self.db_path),
            "--image_path",      image_dir,
            "--image_list_path", str(self.image_list),
        ]))

        log.info(f"Step 1/3: Feature extraction (max_features={self.max_features})...")
        feat_ok = False
        for label, cmd in feat_cmds:
            log.info(f"  Trying [{label}]")
            rc, out = _run(cmd)
            if rc == 0:
                log.info(f"  Feature extraction OK [{label}]")
                feat_ok = True
                break
            else:
                log.warning(f"  [{label}] failed: {out[-400:]}")

        if not feat_ok:
            log.error("All feature_extractor variants failed — falling back to ARKit poses.")
            self._write_known_poses(frames)
            return

        # ── Step 2: Sequential matching ───────────────────────────────────────
        overlap = self.match_overlap
        log.info(f"Step 2/3: Sequential matching (overlap={overlap})...")

        match_cmd = [
            self.colmap_bin, "sequential_matcher",
            "--database_path",                     str(self.db_path),
            "--SequentialMatching.overlap",        str(overlap),
            "--SequentialMatching.loop_detection", "0",
        ] + match_gpu_off
        rc, out = _run(match_cmd)
        if rc != 0:
            log.error(f"sequential_matcher failed:\n{out[-2000:]}")
            self._write_known_poses(frames)
            return
        log.info("  Sequential matching OK")

        # ── Step 3: Point Triangulator / Mapper ───────────────────────────────
        sparse_0 = self.sparse_dir / "0"
        sparse_0.mkdir(parents=True, exist_ok=True)

        # Write known poses as COLMAP model for warm-start (uses pycolmap BIN)
        init_dir = self.sparse_dir / "init"
        self._write_known_poses_to_dir(frames, init_dir)

        log.info("Step 3/3: Point triangulator (using known poses — no slow SfM)...")

        tri_variants = [
            # Variant 1: Point Triangulator with known poses (FASTEST, no SfM)
            [
                self.colmap_bin, "point_triangulator",
                "--database_path", str(self.db_path),
                "--image_path",    image_dir,
                "--input_path",    str(init_dir),
                "--output_path",   str(sparse_0),
                "--clear_points",  "1",
            ],
            # Variant 2: Mapper with minimal BA (1 iteration prevents Matrix crash)
            [
                self.colmap_bin, "mapper",
                "--database_path", str(self.db_path),
                "--image_path",    image_dir,
                "--output_path",   str(self.sparse_dir),
                "--Mapper.ba_global_max_num_iterations", "1",
                "--Mapper.ba_local_max_num_iterations",  "1",
            ],
        ]

        tri_ok = False
        for i, cmd in enumerate(tri_variants):
            label = ["triangulator", "mapper-min-ba"][i]
            log.info(f"  Trying [{label}]")
            rc, out = _run(cmd, timeout=self.mapper_timeout_sec, stream_log=True)
            if rc == 0:
                log.info(f"  Point triangulation / Mapping OK [{label}]")
                tri_ok = True
                break
            else:
                log.warning(f"  [{label}] failed (rc={rc})")

        if not tri_ok:
            log.warning("All triangulation variants failed — using ARKit poses only.")
            self._write_known_poses(frames)
            return

        self._export_sparse_txt(sparse_0)
    def _export_sparse_txt(self, sparse_dir: Path):
        txt_dir = sparse_dir.parent / "0_txt"
        txt_dir.mkdir(exist_ok=True)
        cmd = [
            self.colmap_bin, "model_converter",
            "--input_path",  str(sparse_dir),
            "--output_path", str(txt_dir),
            "--output_type", "TXT",
        ]
        rc, out = _run(cmd)
        if rc == 0:
            log.info(f"  Sparse TXT model -> {txt_dir}")
        else:
            log.warning(f"model_converter failed (non-critical): {out[-300:]}")

    def _write_known_poses(self, frames):
        """Fallback: write ARKit poses as COLMAP images.txt without SfM."""
        images_txt  = self.sparse_dir / "0" / "images.txt"
        cameras_txt = self.sparse_dir / "0" / "cameras.txt"
        points_txt  = self.sparse_dir / "0" / "points3D.txt"
        images_txt.parent.mkdir(parents=True, exist_ok=True)

        with open(cameras_txt, "w") as f:
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            seen = set()
            for fr in frames:
                if fr.idx not in seen:
                    i = fr.intrinsics
                    f.write(f"{fr.idx+1} PINHOLE {i.width} {i.height} "
                            f"{i.fx} {i.fy} {i.cx} {i.cy}\n")
                    seen.add(fr.idx)

        with open(images_txt, "w") as f:
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for fr in frames:
                qw, qx, qy, qz, tx, ty, tz = _c2w_to_quat_trans(fr.c2w)
                f.write(f"{fr.idx+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                        f"{tx:.9f} {ty:.9f} {tz:.9f} {fr.idx+1} "
                        f"{fr.rgb_path.name}\n\n")

        points_txt.write_text("")
        log.info(f"Known poses written -> {images_txt.parent}")

    def _write_known_poses_to_dir(self, frames, out_dir: Path):
        """Write ARKit poses as COLMAP model for mapper warm-start."""
        out_dir.mkdir(parents=True, exist_ok=True)
        
        pycolmap = _try_import_pycolmap()
        if pycolmap:
            try:
                self._write_init_model_pycolmap(frames, out_dir, pycolmap)
                return
            except Exception as e:
                log.warning(f"pycolmap init model write failed ({e}) — falling back to TXT")

        # Fallback: TXT format with frames.txt (required for COLMAP 4.x)
        cam_map = {}
        cam_counter = 1

        cameras_txt = out_dir / "cameras.txt"
        with open(cameras_txt, "w") as f:
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            for fr in frames:
                intr = fr.intrinsics
                key = (intr.width, intr.height,
                       round(intr.fx, 2), round(intr.fy, 2),
                       round(intr.cx, 2), round(intr.cy, 2))
                if key not in cam_map:
                    cam_map[key] = cam_counter
                    f.write(f"{cam_counter} PINHOLE {intr.width} {intr.height} "
                            f"{intr.fx} {intr.fy} {intr.cx} {intr.cy}\n")
                    cam_counter += 1

        images_txt = out_dir / "images.txt"
        with open(images_txt, "w") as f:
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for seq_id, fr in enumerate(frames):
                intr = fr.intrinsics
                key = (intr.width, intr.height,
                       round(intr.fx, 2), round(intr.fy, 2),
                       round(intr.cx, 2), round(intr.cy, 2))
                qw, qx, qy, qz, tx, ty, tz = _c2w_to_quat_trans(fr.c2w)
                f.write(f"{seq_id+1} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                        f"{tx:.9f} {ty:.9f} {tz:.9f} {cam_map[key]} "
                        f"{fr.rgb_path.name}\n\n")

        points_txt = out_dir / "points3D.txt"
        points_txt.write_text("")
        
        # ★ Required for COLMAP 4.x to prevent RigId crash
        frames_txt = out_dir / "frames.txt"
        frames_txt.write_text("# FRAME_ID, RIG_ID, IMAGE_ID\n")

        log.info(f"  Init poses for mapper warm-start -> {out_dir}")

    def _write_init_model_pycolmap(self, frames, out_dir: Path, pycolmap):
        """Write ARKit poses as COLMAP BIN model using pycolmap (safest for 4.x)."""
        recon = pycolmap.Reconstruction()
        cam_map = {}
        cam_counter = 1

        for seq_id, fr in enumerate(frames):
            intr = fr.intrinsics
            key = (intr.width, intr.height,
                   round(intr.fx, 2), round(intr.fy, 2),
                   round(intr.cx, 2), round(intr.cy, 2))

            if key not in cam_map:
                cam = pycolmap.Camera(
                    model="PINHOLE",
                    width=intr.width,
                    height=intr.height,
                    params=[intr.fx, intr.fy, intr.cx, intr.cy],
                    camera_id=cam_counter,
                )
                recon.add_camera(cam)
                cam_map[key] = cam_counter
                cam_counter += 1

            cam_id = cam_map[key]
            qw, qx, qy, qz, tx, ty, tz = _c2w_to_quat_trans(fr.c2w)

            try:
                img = pycolmap.Image(
                    id=seq_id + 1,
                    name=fr.rgb_path.name,
                    camera_id=cam_id,
                    qvec=np.array([qw, qx, qy, qz]),
                    tvec=np.array([tx, ty, tz]),
                )
            except TypeError:
                img = pycolmap.Image(
                    id=seq_id + 1,
                    name=fr.rgb_path.name,
                    camera_id=cam_id,
                )
                img.qvec = np.array([qw, qx, qy, qz])
                img.tvec = np.array([tx, ty, tz])

            recon.add_image(img)

        recon.write(str(out_dir))
        log.info(f"  Init model written via pycolmap -> {out_dir}")

    def _detect_sift_gpu_flag(self):
        try:
            _, txt = _run([self.colmap_bin, "feature_extractor", "--help"], timeout=15)
        except Exception:
            return [], []
        if "--FeatureExtraction.use_gpu" in txt:
            return ["--FeatureExtraction.use_gpu", "1"], \
                ["--FeatureExtraction.use_gpu", "0"]
        if "--SiftExtraction.use_gpu" in txt:
            return ["--SiftExtraction.use_gpu", "1"], \
                ["--SiftExtraction.use_gpu", "0"]
        return [], []

    def _detect_matcher_gpu_flag(self):
        try:
            _, txt = _run([self.colmap_bin, "sequential_matcher", "--help"], timeout=15)
        except Exception:
            return [], []
        if "--FeatureMatching.use_gpu" in txt:
            return ["--FeatureMatching.use_gpu", "1"], \
                ["--FeatureMatching.use_gpu", "0"]
        if "--SiftMatching.use_gpu" in txt:
            return ["--SiftMatching.use_gpu", "1"], \
                ["--SiftMatching.use_gpu", "0"]
        return [], []

    def _colmap_available(self) -> bool:
        try:
            subprocess.run([self.colmap_bin, "help"],
                           capture_output=True, timeout=10)
            return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False


# ─── helpers ──────────────────────────────────────────────────────────────────

def _pack_doubles(values):
    return struct.pack(f"{len(values)}d", *values)


def _c2w_to_quat_trans(c2w: np.ndarray):
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s  = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2,1] - R[1,2]) * s
        qy = (R[0,2] - R[2,0]) * s
        qz = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        qw = (R[2,1] - R[1,2]) / s; qx = 0.25 * s
        qy = (R[0,1] + R[1,0]) / s; qz = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        qw = (R[0,2] - R[2,0]) / s; qx = (R[0,1] + R[1,0]) / s
        qy = 0.25 * s;               qz = (R[1,2] + R[2,1]) / s
    else:
        s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        qw = (R[1,0] - R[0,1]) / s; qx = (R[0,2] + R[2,0]) / s
        qy = (R[1,2] + R[2,1]) / s; qz = 0.25 * s
    return qw, qx, qy, qz, t[0], t[1], t[2]