"""
pipeline.py - v14 "evidence-filtered mesh"   (Phase 1 + 2 + 3)

Changes from v13 (Phase 3)
--------------------------
  - Evidence filter (mesh_evidence.py): faces that no camera can see are removed
    before texturing (25 % of the faces / 6 % of the area on Full_Refuge_data, with
    no change in coverage or accuracy).  MeshCleaner smoothing switched off.

Changes from v12 (Phase 2)
--------------------------
  - Voxel size, SDF truncation, border mask and depth-confidence level now come
    from measured depth statistics (smart_tuner.py); the 'boost truncation to 6x'
    rule and the 6x clamp are gone.  GLB/mesh stages are unchanged (Phases 3-4).

Changes from v11
----------------
  - DataLoader now reads the per-frame intrinsics from the pose file
    (it silently used fx = 0.8*width before) and its validation report is
    checked: structural errors stop the run.
  - POSE_SMOOTH default 0.  Measured on Full_Refuge_data (multiview depth
    agreement) neither +-2-frame smoothing nor ICP refinement improved the raw
    ARKit poses.  Set POSE_SMOOTH = 2 to re-enable.
  - PoseDiagnostics runs before/after pose modification; if dense alignment
    does not keep the agreement metric at least as good, the previous poses are
    restored automatically.  Reports -> <out>/diagnostics/pose_report_*.json
  - Stale outputs from earlier runs are moved to <out>/_previous_run_<time>/
    at start, so a failed stage can no longer be "rescued" by an old file
    (the 0-point TSDF used to continue with a stale mesh).
  - Stage gates: empty point cloud / missing or stale mesh / missing GLB stop
    the pipeline with a clear message instead of passing garbage on.
  - Paths are configurable:  python pipeline.py --data <dir> --colmap <exe>
    (defaults = your previous hardcoded paths, or env RECON_DATA_DIR /
    RECON_COLMAP_BIN).
  - Environment self-test at start-up (Open3D 0.20.0's ScalableTSDFVolume returns
    0 points; the check stops the run with the fix instead of failing downstream).
  - DepthRefiner / NormalEstimator / DA3Refiner are imported only when enabled.
  - COLMAP stage now receives the keyframe subset it was always meant to get.
"""

import argparse, logging, os, shutil, signal, sys, time
from pathlib import Path
import numpy as np

from src.modules.data_loader      import DataLoader
from src.modules.colmap_db        import COLMAPDatabaseBuilder
from src.modules.depth_fusion     import DepthFusion, _load_exr
from src.modules.reconstruction   import Reconstruction
from src.modules.meshing          import Meshing
from src.modules.pose_smoother    import PoseSmoother
from src.modules.texture_baker    import TextureBaker
from src.modules.frame_filter     import FrameFilter
from src.modules.smart_tuner      import AutoTuner
from src.modules.mesh_cleaner     import MeshCleaner
from src.modules.dense_aligner    import DenseAligner, detect_depth_scale
from src.modules.pose_diagnostics import PoseDiagnostics
from src.modules.env_check        import check_environment
from src.modules.mesh_metrics     import evaluate_mesh
from src.modules.mesh_evidence    import compute_evidence, evidence_summary, filter_by_evidence

os.makedirs("output", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("pipeline")

_STOP = False
def _sig(s, f):
    global _STOP
    if _STOP: os._exit(1)
    _STOP = True; log.warning("Ctrl+C - stopping. Press again to force.")
signal.signal(signal.SIGINT,  _sig)
signal.signal(signal.SIGTERM, _sig)
if hasattr(signal, "SIGBREAK"):
    signal.signal(signal.SIGBREAK, lambda s, f: os._exit(1))

def chk(st):
    if _STOP: log.info(f"Stop before {st}"); sys.exit(0)

# -- Defaults (override on the command line / environment) ---------------------
DATA_SUBDIR = os.environ.get("RECON_DATA_DIR",
                             r"E:\Shravani\3d_reconstruction\Data\Full_Refuge_data")
COLMAP_BIN  = os.environ.get("RECON_COLMAP_BIN",
                             r"F:\Rohit AI\Colmap\bin\colmap.exe")
OUT_NAME    = "no_refiner"

# -- Pipeline switches ----------------------------------------------------------
DEPTH_SCALE       = "auto"   # "auto" detects automatically; or set 1.0 / 0.001
FRAME_SKIP        = 0
POSE_SMOOTH       = 0        # was 2; no measured benefit on Full_Refuge_data
SKIP_COLMAP       = False
BAKE              = True
TEX_SIZE          = 8192
COLMAP_KEYFRAME_EVERY = 8
COLMAP_MAX_FEATURES   = 1024
COLMAP_MATCH_OVERLAP  = 4
USE_FRAME_FILTER     = True
USE_DEPTH_REFINER    = False
USE_NORMAL_ESTIMATOR = False
USE_MESH_CLEANER     = True
USE_EVIDENCE_FILTER  = True    # [P3] drop faces that no camera can see (cannot be textured)
EVIDENCE_MAX_REMOVED_AREA_PCT = 25.0   # safety: never let the filter delete more than this
USE_DENSE_ALIGNER    = True    # gated: reverted automatically if it does not help
USE_DA3_REFINER      = False
DA3_MODE             = "blend"  # 'blend' | 'fill' | 'replace'

# files produced by earlier runs that must never be reused silently
_STALE_PATTERNS = ["fused_pointcloud.ply", "fused_tsdf_mesh.ply", "mesh_*.ply",
                   "mesh_*.obj", "mesh_*.glb", "mesh_*.mtl", "refined_poses.txt"]


def find_root(start, sub):
    p = Path(sub)
    if p.is_absolute():                      # absolute data path: nothing to search
        if p.is_dir(): return Path(start).resolve()
        raise RuntimeError(f"Data directory not found: {p}")
    c = Path(start).resolve()
    for _ in range(5):
        if (c / sub).is_dir(): return c
        c = c.parent
    raise RuntimeError(f"Cannot find {sub}")

def find_d(root, names):
    for n in names:
        p = root / n
        if p.is_dir(): return p

def find_f(root, names):
    for n in names:
        p = root / n
        if p.is_file(): return p

def discover(root):
    r = find_d(root, ["rgb", "images", "color", "frames"])
    d = find_d(root, ["depth", "depths", "depth_maps"])
    p = find_f(root, ["poses.txt", "pose.txt", "transforms.txt", "trajectory.txt"])
    if not r or not d or not p:
        raise RuntimeError(f"Missing rgb/depth/poses in {root}")
    return r, d, p


# -- stage-gate helpers ---------------------------------------------------------

def archive_previous_outputs(out: Path) -> None:
    """Move intermediate files of an earlier run out of the way."""
    moved = []
    dest = out / f"_previous_run_{time.strftime('%Y%m%d_%H%M%S')}"
    for pat in _STALE_PATTERNS:
        for f in out.glob(pat):
            if f.is_file():
                dest.mkdir(exist_ok=True)
                shutil.move(str(f), str(dest / f.name)); moved.append(f.name)
    if moved:
        log.info(f"  Archived {len(moved)} stale file(s) from an earlier run -> {dest.name}")

def _fail(stage, msg):
    log.error(f"STAGE GATE FAILED [{stage}]: {msg}")
    raise RuntimeError(f"[{stage}] {msg}")

def gate_point_cloud(path, min_points=1000):
    import open3d as o3d
    p = Path(path)
    if not p.is_file():
        _fail("fusion", f"point cloud file missing: {p}")
    n = len(o3d.io.read_point_cloud(str(p)).points)
    if n < min_points:
        _fail("fusion", f"only {n} points (< {min_points}). The TSDF is empty or "
                        f"nearly empty - check depth range, intrinsics and poses. "
                        f"Nothing was passed to the mesh stage.")
    log.info(f"  gate OK: {n:,} points")

def gate_mesh(path, t_start, stage, min_faces=1000):
    import open3d as o3d
    p = Path(path)
    if not p.is_file():
        _fail(stage, f"mesh file missing: {p}")
    if p.stat().st_mtime < t_start - 1.0:
        _fail(stage, f"{p.name} is older than this run (stale file).")
    m = o3d.io.read_triangle_mesh(str(p))
    nf = len(m.triangles)
    if nf < min_faces:
        _fail(stage, f"mesh has only {nf} faces (< {min_faces}).")
    log.info(f"  gate OK: {len(m.vertices):,} verts, {nf:,} faces")


def report_mesh(path, frames, out_dir, label):
    """Measurable mesh-quality proxies -> diagnostics/mesh_report_<label>.json (never fatal)."""
    import json
    try:
        r = evaluate_mesh(path, frames, n_views=16, tol_m=0.03)
        t, d = r["topology"], r["depth_fidelity"]
        (Path(out_dir) / f"mesh_report_{label}.json").write_text(json.dumps(r, indent=2))
        log.info(f"  [{label}] faces={t['n_faces']:,} components={t['n_components']} "
                 f"boundary={t['boundary_length_m']:.0f}m | vs sensor depth: "
                 f"coverage={d['coverage']*100:.1f}%  median err={d['median_abs_err_m']*100:.2f}cm  "
                 f"within3cm={d['within_tol']*100:.1f}%  in-front-floaters={d['front_floaters']*100:.1f}%")
    except Exception as e:                                           # noqa: BLE001
        log.warning(f"  mesh report skipped: {type(e).__name__}: {e}")


def apply_evidence_filter(mesh_path, frames, out_dir, t_start):
    """
    [P3] Remove faces that no camera can see.  They cannot be textured correctly
    (the baker would leave them black or paint them from a wrong frame) and the
    sensor never observed that surface, so the limitation is reported instead of
    keeping fabricated geometry.  Returns the path of the mesh to texture.
    """
    import json
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    ev = compute_evidence(mesh, frames, step=2)
    before = evidence_summary(ev)
    filtered, st = filter_by_evidence(mesh, ev, require_visible=True)
    ok = (len(filtered.triangles) > 1000
          and st["area_removed_pct"] <= EVIDENCE_MAX_REMOVED_AREA_PCT)
    (Path(out_dir) / "evidence_report.json").write_text(
        json.dumps(dict(before=before, filter=st, applied=bool(ok)), indent=2))
    log.info(f"  before: {before['no_camera_sees_face_pct']:.1f}% of faces "
             f"({before['no_camera_sees_face_area_pct']:.1f}% of area) are seen by NO camera")
    if not ok:
        log.warning(f"  evidence filter NOT applied (would remove {st['area_removed_pct']:.1f}% "
                    f"of the area, limit {EVIDENCE_MAX_REMOVED_AREA_PCT}%) - keeping the mesh")
        return Path(mesh_path)
    out = Path(mesh_path).with_name(Path(mesh_path).stem + "_evidence.obj")
    o3d.io.write_triangle_mesh(str(out), filtered, write_vertex_normals=False)
    return out


def parse_args():
    ap = argparse.ArgumentParser(description="3D reconstruction pipeline")
    ap.add_argument("--data",   default=DATA_SUBDIR, help="dataset directory (rgb/, depth/, poses.txt)")
    ap.add_argument("--colmap", default=COLMAP_BIN,  help="path to colmap executable")
    ap.add_argument("--out-name", default=OUT_NAME,  help="output sub-folder name")
    ap.add_argument("--no-bake", action="store_true", help="skip texture baking")
    ap.add_argument("--tex-size", type=int, default=TEX_SIZE)
    ap.add_argument("--frame-skip", type=int, default=FRAME_SKIP)
    return ap.parse_args()


def main():
    args = parse_args()
    t_run = time.time()
    check_environment()        # fails fast on a broken Open3D TSDF build
    bake = BAKE and not args.no_bake
    root = find_root(Path(__file__).parent, args.data)
    src  = Path(args.data) if Path(args.data).is_absolute() else root / args.data
    out  = root / args.out_name
    out.mkdir(exist_ok=True)
    rgb_d, dep_d, pose_f = discover(src)
    log.info(f"Root:{root}  Data:{src}  Out:{out}")
    archive_previous_outputs(out)
    diag_dir = out / "diagnostics"

    # -- Stage 1: Load ------------------------------------------------------------
    chk("Load"); log.info("=== Stage 1/9 Load ===")
    loader = DataLoader(
        data_dir=src, rgb_subdir=rgb_d.name,
        depth_subdir=dep_d.name, pose_file=pose_f.name,
        depth_scale=1.0,  # raw scale; we fix it below
    )
    frames = loader.load_all()
    log.info(f"  {len(frames)} frames loaded")
    if loader.report.get("errors"):
        _fail("load", "input validation errors: " + "; ".join(loader.report["errors"]))

    # -- Stage 1b: Frame filter ---------------------------------------------------
    if USE_FRAME_FILTER:
        chk("Filter"); log.info("=== Stage 1b Frame Filter ===")
        frames = FrameFilter(
            sharpness_threshold=0.20,
            pose_jump_threshold=0.40,
        ).filter(frames)
        log.info(f"  {len(frames)} frames after filter")

    # -- Stage 1c: AUTO-TUNE ------------------------------------------------------
    chk("AutoTune"); log.info("=== Stage 1c Auto-Tune Parameters ===")
    cfg = AutoTuner(
        frames    = frames,
        rgb_dir   = src / rgb_d.name,
        depth_dir = src / dep_d.name,
        enable_verification = True,
        output_dir = out,
    ).compute()
    print(f"Auto tuned values : {cfg}")
    log.info(f"  Detected: env={cfg['env_type']}  "
             f"dynamic_objects={cfg['has_dynamic_objects']}  "
             f"span={cfg['scan_span_m']}m")

    # -- Stage 1d: AUTO DEPTH SCALE ----------------------------------------------
    chk("DepthScale"); log.info("=== Stage 1d Auto Depth-Scale Detection ===")
    if DEPTH_SCALE == "auto":
        DEPTH_SCALE_DETECTED = detect_depth_scale(
            frames,
            depth_dir=src / dep_d.name,
        )
    else:
        DEPTH_SCALE_DETECTED = float(DEPTH_SCALE)
    log.info(f"  Depth scale = {DEPTH_SCALE_DETECTED}")

    # -- Stage 1e: DA3 DEPTH REFINEMENT ------------------------------------------
    if USE_DA3_REFINER:
        from src.modules.da3_refiner import DA3Refiner
        chk("DA3Refine"); log.info("=== Stage 1e DA3 Depth Refinement ===")
        frames = DA3Refiner(
            output_dir   = out,
            depth_scale  = DEPTH_SCALE_DETECTED,
            mode         = DA3_MODE,
            sensor_weight= 0.6,
            da3_weight   = 0.4,
            model_name   = "depth-anything/DA3-LARGE",
            batch_size   = 4,
            device       = "auto",
            rgb_dir      = src / rgb_d.name,
            half_precision = True,
            export_depth=False
        ).refine(frames)
        log.info("  DA3 refinement complete")

    # -- Stage 2: Pose smoothing --------------------------------------------------
    chk("Smooth"); log.info("=== Stage 2/9 Smooth ===")
    if POSE_SMOOTH > 0:
        frames = PoseSmoother(window=POSE_SMOOTH).smooth(frames)
        log.info(f"  smoothed (window=+-{POSE_SMOOTH})")
    else:
        log.info("  skipped (POSE_SMOOTH = 0)")

    # -- Stage 2-diag: pose self-consistency before any pose modification ---------
    chk("PoseDiag"); log.info("=== Stage 2-diag Pose diagnostics (input) ===")
    diag = PoseDiagnostics(frames, depth_scale=DEPTH_SCALE_DETECTED, output_dir=diag_dir)
    pose_before = diag.run("input")

    # -- Stage 2a: DENSE ALIGNMENT (measured gate) --------------------------------
    if USE_DENSE_ALIGNER:
        chk("DenseAlign"); log.info("=== Stage 2a Dense ICP Pose Alignment (gated) ===")
        saved_c2w = [f.c2w.copy() for f in frames]

        env = cfg.get("env_type", "indoor")
        use_lc = env in ("indoor",) and cfg.get("scan_span_m", 0) > 3.0
        frames = DenseAligner(
            output_dir   = out,
            depth_scale  = DEPTH_SCALE_DETECTED,
            max_depth    = cfg.get("max_d", 8.0),
            voxel_downsample_size  = max(0.008, cfg.get("vox", 0.01) * 1.2),
            icp_max_correspondence = 0.10,
            min_icp_fitness        = 0.25,
            keyframe_interval      = 4,
            min_overlap_fraction   = 0.20,
            skip_if_poses_good     = False,
            pose_quality_threshold = 0.15,
            loop_closure = use_lc,
            loop_closure_spatial_dist = 0.5,
            loop_closure_min_frame_gap = 20,
            stop_flag = lambda: _STOP,
            depth_dir = src / dep_d.name,
            rgb_dir   = src / rgb_d.name,
        ).refine(frames)

        pose_after = diag.run("after_dense_alignment")
        if PoseDiagnostics.alignment_is_not_worse(pose_before, pose_after, tolerance=0.0):
            log.info("  Dense alignment kept (agreement metric not worse than input).")
        else:
            for f, T in zip(frames, saved_c2w):
                f.c2w = T
            log.warning("  Dense alignment made the agreement metric WORSE "
                        "-> reverted to the input poses.")

    # -- Stage 2b: Neural depth refinement (optional, lazy import) -----------------
    if USE_DEPTH_REFINER:
        from src.modules.depth_refiner1 import DepthRefiner
        frames = DepthRefiner(
            model="depth-anything-v2",
            lidar_weight=0.2,
            output_dir=out,
            ensemble_passes=4,
            half_precision=True,
        ).refine(frames)

    # -- Stage 3: COLMAP DB -------------------------------------------------------
    chk("COLMAP"); log.info("=== Stage 3/9 COLMAP DB ===")
    colmap_frames = frames[::COLMAP_KEYFRAME_EVERY]
    log.info(f"  COLMAP subset: {len(colmap_frames)}/{len(frames)} frames "
             f"(every {COLMAP_KEYFRAME_EVERY}th)")
    db = COLMAPDatabaseBuilder(
        output_dir=out,
        colmap_bin=args.colmap,
        fast_mode=True,
        mapper_timeout_sec=600,
        max_features=COLMAP_MAX_FEATURES,
        match_overlap=COLMAP_MATCH_OVERLAP,
    ).build(colmap_frames, skip_feature_matching=SKIP_COLMAP)
    log.info(f"  DB->{db.name}  (note: the COLMAP result is not used by later stages)")

    # -- Stage 4: TSDF Fusion -----------------------------------------------------
    chk("Fuse"); log.info("=== Stage 4/9 TSDF Fusion ===")
    # [P2] truncation, voxel size and confidence level are derived from measured
    # depth statistics by AutoTuner (see smart_tuner.py); no manual boost.
    sdf_trunc_m = cfg["sdf_trunc_m"]
    log.info(f"  depth=[{cfg['min_d']:.2f},{cfg['max_d']:.2f}]  vox={cfg['vox']:.4f}  "
             f"sdf_trunc={sdf_trunc_m:.4f}m  min_confidence={cfg['min_confidence']}  "
             f"noise_sigma={cfg.get('depth_noise_sigma_m', 0):.4f}m")

    fusion = DepthFusion(
        output_dir   = out,
        depth_scale  = DEPTH_SCALE_DETECTED,
        min_depth    = cfg["min_d"],
        max_depth    = cfg["max_d"],
        voxel_size   = cfg["vox"],
        frame_skip   = args.frame_skip,
        stop_flag    = lambda: _STOP,
        scene_mode   = cfg["scene"],
        sdf_trunc_m  = sdf_trunc_m,
        min_confidence = cfg["min_confidence"],

        use_bilateral              = cfg["use_bilateral"],
        use_border_mask            = True,
        border_frac                = cfg["border_frac"],
        use_sky_suppress           = cfg["use_sky_suppress"],
        sky_bright_thresh          = cfg["sky_bright_thresh"],
        sky_depth_min_m            = cfg["sky_depth_min_m"],
        use_flying_pixel_suppress  = True,
        flying_pixel_jump_thresh_m = cfg["flying_pixel_jump_thresh_m"],
        use_inpaint                = True,
        inpaint_radius_px          = 3,
        use_spike_filter           = True,
        use_glass_weight           = True,
        use_soft_consistency       = True,
        use_dynamic_weight         = True,
        dynamic_motion_thresh_m    = cfg["dynamic_motion_thresh_m"],
        dynamic_min_weight         = cfg["dynamic_min_weight"],
        use_planar_snap            = True,
        planar_ransac_thresh       = cfg["planar_ransac_thresh"],
        n_planes                   = cfg["n_planes"],
        planar_snap_dist           = cfg["planar_snap_dist"],
        planar_remove_dist         = cfg["planar_remove_dist"],
        use_hole_fill              = True,
        max_hole_size              = cfg["max_hole_size"],
    )
    pcd_path = fusion.fuse(frames)
    log.info(f"  Point cloud -> {pcd_path.name}")
    gate_point_cloud(pcd_path)                       # stop here if the TSDF is empty

    if USE_NORMAL_ESTIMATOR:
        from src.modules.normal_estimator import NormalEstimator
        chk("Normals"); log.info("=== Stage 4b Normal Estimation ===")
        pcd_path = NormalEstimator(
            output_dir=out, model="dsine", frame_skip=2,
        ).estimate(pcd_path, frames)
        log.info("  Done")

    log.info("=== Stage 5/9 Sparse (skipped) ===")

    # -- Stage 6: Mesh ------------------------------------------------------------
    chk("Mesh"); log.info("=== Stage 6/9 Mesh ===")
    mesher = Meshing(
        output_dir             = out,
        depth                  = cfg["mdepth"],
        target_faces           = cfg["mfaces"],
        density_quantile       = cfg["mq"],
        tsdf_voxel_size        = cfg["vox"],
        min_component_fraction = cfg["mcomp"],
        largest_component_ratio= cfg["mlargest"],
    )
    tsdf_mesh = getattr(fusion, "last_tsdf_mesh_path", None)
    if tsdf_mesh and Path(tsdf_mesh).is_file():
        mesh = mesher.cleanup_mesh(tsdf_mesh, frames=frames)
    else:
        mesh = mesher.mesh(pcd_path, frames=frames)
    log.info(f"  Raw mesh -> {mesh.name}")
    gate_mesh(mesh, t_run, "mesh")
    diag_dir.mkdir(exist_ok=True)
    report_mesh(mesh, frames, diag_dir, "after_meshing")

    # -- Stage 6b: Mesh Cleaning --------------------------------------------------
    if USE_MESH_CLEANER:
        chk("Clean"); log.info("=== Stage 6b/9 Mesh Cleaning ===")
        log.info(f"  PyMeshLab + Open3D repair  "
                 f"(smooth_passes={cfg['mesh_smooth_iter']}  "
                 f"dynamic={cfg['has_dynamic_objects']})")

        cleaner = MeshCleaner(
            output_dir          = out,
            use_outlier_removal = True,
            use_hole_fill       = True,
            use_smooth          = False,   # [P3] measured: 3rd smoothing pass has no benefit
            use_decimate        = True,
            use_watertight      = False,
            scan_span_m         = cfg["scan_span_m"],
            has_dynamic_objects = cfg["has_dynamic_objects"],
            target_faces        = cfg["mesh_target_faces"],
            smooth_iterations   = cfg["mesh_smooth_iter"],
            smooth_lambda       = cfg["mesh_smooth_lambda"],
        )
        mesh = cleaner.clean(mesh)
        log.info(f"  Cleaned mesh -> {mesh.name}")
        gate_mesh(mesh, t_run, "mesh_clean")
        report_mesh(mesh, frames, diag_dir, "after_cleaner")

    # -- Stage 6c: evidence filter (Phase 3) ------------------------------------
    if USE_EVIDENCE_FILTER:
        chk("Evidence"); log.info("=== Stage 6c Evidence filter (faces no camera can see) ===")
        mesh = apply_evidence_filter(mesh, frames, diag_dir, t_run)
        gate_mesh(mesh, t_run, "evidence")
    report_mesh(mesh, frames, diag_dir, "final_geometry")

    # -- Stage 7: Texture ---------------------------------------------------------
    chk("Texture"); log.info("=== Stage 7/9 Texture ===")
    if bake:
        t0  = time.time()
        tex = TextureBaker(
            output_dir   = out,
            texture_size = args.tex_size,
            xatlas_threads = 4,
            frame_batch  = 64,
            bake_faces   = 10_000_000,
        ).bake(mesh, frames, frame_skip=args.frame_skip)
        log.info(f"  Tex->{tex.name}  ({time.time()-t0:.1f}s)")
        glb = out / "mesh_texture.glb"
        if not glb.is_file() or glb.stat().st_mtime < t_run - 1.0:
            _fail("texture", "mesh_texture.glb was not (re)written by this run")
    else:
        tex = mesh

    log.info(f"Done -> {out}")
    log.info("Preview: sandbox.babylonjs.com")


if __name__ == "__main__":
    s1 = time.time()
    main()
    print(f"Total time: {time.time()-s1:.2f}s")
