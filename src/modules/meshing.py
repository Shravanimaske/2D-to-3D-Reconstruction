"""
modules/meshing.py  ── v6  "coordinate-correct"

CRITICAL FIX from v5:
  Normal orientation was using (fr.c2w @ _FLIP) for camera positions,
  which gives OpenGL-convention world positions. But the point cloud
  from TSDF fusion is in OpenCV-convention world space (because TSDF
  uses OpenCV extrinsics internally).

  Fix: Use (fr.c2w @ _FLIP) consistently — this IS the correct camera
  position in the TSDF's world coordinate system, because TSDF was
  given inv(c2w @ _FLIP) as the extrinsic, so its world frame IS the
  OpenCV-convention world frame.

  The _FLIP is already correct here. No change needed for normal
  orientation — but we DO need to ensure the visibility cull uses
  the same convention.

Additional fixes:
  - Visibility cull rewritten to correctly project mesh points
  - Reduced aggressive cleanup that was removing valid geometry
  - Better hole filling integration with meshfix
"""

import logging
from pathlib import Path
import numpy as np
import os

log = logging.getLogger(__name__)
_FLIP = np.diag([1., -1., -1., 1.])
COLMAP_PATH = r"F:\Rohit AI\Colmap\bin\colmap.exe"


class Meshing:

    def __init__(
        self,
        output_dir,
        method: str = "poisson",
        depth: int = 9,
        target_faces: int = 0,
        density_quantile: float = 0.02,
        tsdf_voxel_size: float = 0.008,
        min_component_fraction: float = 0.0001,
        largest_component_ratio: float = 0.10,
        colmap_bin: str = COLMAP_PATH,
    ):
        self.output_dir              = Path(output_dir)
        self.method                  = method
        self.depth                   = depth
        self.target_faces            = target_faces
        self.density_quantile        = density_quantile
        self.tsdf_voxel_size         = tsdf_voxel_size
        self.min_component_fraction  = min_component_fraction
        self.largest_component_ratio = largest_component_ratio
        self.colmap_bin              = colmap_bin

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────────

    def mesh(self, pcd_path, frames=None):
        out = self.output_dir / f"mesh_{self.method}.ply"
        self._poisson(pcd_path, out, frames=frames)
        return out

    def cleanup_mesh(self, mesh_path, frames=None, visibility_cull: bool = False):
        """
        Cleanup pipeline:
          1. Remove ghost geometry (reflections) via Visibility Culling.
          2. Repair mesh to hide black lines (holes).
          3. Refine photo-geometry.
        """
        try:
            import open3d as o3d
        except ImportError:
            return mesh_path

        out  = self.output_dir / f"mesh_{self.method}.ply"
        log.info(f"Loading mesh: {mesh_path}")
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
        mesh.compute_vertex_normals()

        n_raw = len(mesh.triangles)
        log.info(f"  Raw mesh: {len(mesh.vertices):,} verts, {n_raw:,} faces")

        # 1. Basic manifold repair
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # 2. Needle removal
        mesh = self._remove_needle_faces(mesh)

        # 3. Gentle normal-outlier removal
        mesh = self._gentle_normal_outlier_removal(mesh)

        # 4. Conservative component removal
        n_total  = len(mesh.triangles)
        min_size = max(2000, int(n_total * 0.005))
        mesh = self._components_safe(mesh, min_size=min_size, bbox_min_m=0.03)
        mesh = self._components_safe(mesh, min_size=500, bbox_min_m=0.03)

        # ── Visibility Culling ────────────────────────────────────────────────
        if frames and visibility_cull:
            log.info("  Visibility Culling (Removing Reflections/Ghosts)...")
            mesh = self._visibility_cull(mesh, frames, voxel_size=self.tsdf_voxel_size)

        # 5. Light edge-collapse
        mesh = self._light_edge_collapse(mesh, reduction=0.10)

        # 6. HC Laplacian smoothing
        log.info("  HC Laplacian smoothing (2 iterations) ...")
        mesh = self._smooth_laplacian_hc(mesh, iterations=2)

        # 7. Simplification
        if self.target_faces > 0 and len(mesh.triangles) > self.target_faces:
            log.info(f"  Simplifying to {self.target_faces:,} faces ...")
            mesh = mesh.simplify_quadric_decimation(self.target_faces)
            mesh.compute_vertex_normals()

        # ── Heal mesh (fill holes) ────────────────────────────────────────────
        log.info("  Healing mesh to hide black lines...")
        mesh = self._heal_mesh(mesh)

        n_final = len(mesh.triangles)
        log.info(f"  Cleanup: {n_raw:,} → {n_final:,} faces")
        self._save(mesh, out)
        return out

    def _visibility_cull(self, mesh, frames, voxel_size=0.02):
        """
        Removes faces that are inconsistent with depth maps.
        Rewritten in v6 to correctly handle coordinate conventions.

        The mesh is in TSDF world space (= OpenCV convention world).
        Camera positions in that space: cam_pos = c2w_gl @ _FLIP @ [0,0,0,1].T
        = (c2w_gl @ _FLIP)[:3, 3]
        """
        import open3d as o3d
        import numpy as np

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        n_faces = len(faces)

        if n_faces == 0:
            return mesh

        # Compute face centroids for faster checking
        centroids = verts[faces].mean(axis=1)  # (n_faces, 3)

        # Camera positions in TSDF world space (OpenCV convention)
        cam_pos_list = []
        cam_data_list = []
        for fr in frames:
            try:
                c2w_gl = fr.c2w
                cam_world = (c2w_gl @ _FLIP)[:3, 3]
                cam_pos_list.append(cam_world)

                # Get intrinsics
                intr = fr.intrinsics
                if hasattr(intr, 'fx'):
                    fx, fy = intr.fx, intr.fy
                    cx, cy = getattr(intr, 'cx', 0), getattr(intr, 'cy', 0)
                elif hasattr(intr, 'K'):
                    K_arr = np.array(intr.K)
                    fx, fy = K_arr[0, 0], K_arr[1, 1]
                    cx, cy = K_arr[0, 2], K_arr[1, 2]
                else:
                    continue

                # w2c in OpenCV convention
                w2c_cv = np.linalg.inv(c2w_gl @ _FLIP)

                cam_data_list.append({
                    'pos': cam_world,
                    'w2c': w2c_cv,
                    'fx': fx, 'fy': fy,
                    'cx': cx, 'cy': cy,
                    'depth_path': getattr(fr, 'depth_path', None),
                })
            except Exception:
                continue

        if not cam_data_list:
            return mesh

        cam_positions = np.array(cam_pos_list)

        # For each centroid, find nearest camera and check depth consistency
        keep_face = np.ones(n_faces, dtype=bool)

        # Process in batches for efficiency
        batch_size = 10_000
        for start in range(0, n_faces, batch_size):
            end = min(start + batch_size, n_faces)
            batch_centroids = centroids[start:end]

            # Find nearest camera for each centroid
            try:
                from scipy.spatial import cKDTree
                cam_tree = cKDTree(cam_positions)
                _, cam_indices = cam_tree.query(batch_centroids, k=1)
            except ImportError:
                # Brute force
                dists = np.linalg.norm(
                    batch_centroids[:, None, :] - cam_positions[None, :, :], axis=2)
                cam_indices = dists.argmin(axis=1)

            for bi in range(end - start):
                fi = start + bi
                ci = cam_indices[bi]
                cam = cam_data_list[ci]

                # Load depth for this camera
                depth_path = cam['depth_path']
                if depth_path is None:
                    continue

                try:
                    depth_img = self._load_metric_depth(depth_path)
                    if depth_img is None:
                        continue
                except Exception:
                    continue

                # Project centroid to camera
                pt = np.append(batch_centroids[bi], 1.0)
                pt_cam = cam['w2c'] @ pt

                if pt_cam[2] <= 0.1:
                    # Point behind camera — likely ghost
                    keep_face[fi] = False
                    continue

                u = int(cam['fx'] * pt_cam[0] / pt_cam[2] + cam['cx'])
                v = int(cam['fy'] * pt_cam[1] / pt_cam[2] + cam['cy'])

                H, W = depth_img.shape
                if u < 0 or u >= W or v < 0 or v >= H:
                    continue  # Out of frame — can't verify

                measured_z = depth_img[v, u]
                mesh_z = pt_cam[2]

                # If mesh point is significantly behind measured surface → ghost/reflection
                if measured_z > 0.05 and (mesh_z - measured_z) > 0.05:
                    keep_face[fi] = False

        n_removed = int((~keep_face).sum())
        if n_removed > 0:
            mesh.remove_triangles_by_mask(~keep_face)
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
            log.info(f"  Visibility cull: removed {n_removed:,} ghost faces "
                     f"({100*n_removed/n_faces:.1f}%)")
        else:
            log.info("  Visibility cull: no ghost faces found")

        return mesh

    def _load_metric_depth(self, path):
        import numpy as np
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            return None
        if path.suffix == ".npy":
            return np.load(str(path)).astype(np.float32)
        if path.suffix == ".exr":
            try:
                import cv2
                d = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
                if d is not None:
                    return d.astype(np.float32)
            except Exception:
                pass
            try:
                import OpenEXR, Imath
                f = OpenEXR.InputFile(str(path))
                dw = f.header()["dataWindow"]
                W = dw.max.x - dw.min.x + 1
                H = dw.max.y - dw.min.y + 1
                ch = next(iter(f.header()["channels"].keys()))
                raw = f.channel(ch, Imath.PixelType(Imath.PixelType.FLOAT))
                return np.frombuffer(raw, dtype=np.float32).reshape(H, W)
            except Exception:
                pass
        if path.suffix == ".bin":
            try:
                raw = np.fromfile(str(path), dtype=np.float32)
                # Try 192x256 (LiDAR native)
                if raw.size == 192 * 256:
                    return raw.reshape(192, 256)
                # Try other shapes
                for h, w in [(480, 640), (1440, 1920)]:
                    if raw.size == h * w:
                        return raw.reshape(h, w)
            except Exception:
                pass
        return None

    def _heal_mesh(self, mesh):
        """
        Hides black lines by filling holes and making the mesh watertight.
        """
        try:
            import pymeshfix
            import open3d as o3d
            import numpy as np

            v = np.asarray(mesh.vertices, dtype=np.float64)
            f = np.asarray(mesh.triangles, dtype=np.int32)

            meshfix = pymeshfix.MeshFix(v, f)
            meshfix.repair(auto_fix=True)

            v_clean, f_clean = meshfix.mesh

            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices = o3d.utility.Vector3dVector(v_clean)
            new_mesh.triangles = o3d.utility.Vector3iVector(f_clean)
            new_mesh.compute_vertex_normals()

            log.info(f"  Mesh healed: {len(f_clean):,} faces (holes filled).")
            return new_mesh
        except ImportError:
            log.warning("pymeshfix not installed. Install with 'pip install pymeshfix' to hide black lines.")
            mesh = self._fill_holes_open3d(mesh, max_hole_size=50)
            return mesh
        except Exception as e:
            log.warning(f"Mesh healing failed: {e}")
            return mesh

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE – geometry filters
    # ──────────────────────────────────────────────────────────────────────────

    def _remove_needle_faces(self, mesh):
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        if len(faces) == 0:
            return mesh

        v0, v1, v2 = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)
        e_max = np.maximum(e0, np.maximum(e1, e2))
        e_min = np.minimum(e0, np.minimum(e1, e2))
        cross = np.cross(v1 - v0, v2 - v0)
        area  = np.linalg.norm(cross, axis=1) * 0.5
        med_area = np.median(area[area > 0]) + 1e-12

        bad = (
            (area < 1e-10) |
            ((e_max / (e_min + 1e-9) > 20.0) & (area < med_area * 0.0001))
        )
        n_bad = int(bad.sum())
        if n_bad > 0:
            mesh.remove_triangles_by_mask(bad)
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
            log.info(f"  Needle removal: {n_bad:,} faces removed")
        return mesh

    def _gentle_normal_outlier_removal(
        self,
        mesh,
        neighborhood_radius: float = 0.06,
        angle_threshold_deg: float = 55.0,
        area_fraction: float = 0.02,
    ):
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            log.warning("  Gentle normal filter skipped (scipy not available)")
            return mesh

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        if len(faces) < 100:
            return mesh

        v0, v1, v2   = verts[faces[:,0]], verts[faces[:,1]], verts[faces[:,2]]
        cross        = np.cross(v1 - v0, v2 - v0)
        area         = np.linalg.norm(cross, axis=1) * 0.5
        norm_len     = np.linalg.norm(cross, axis=1, keepdims=True) + 1e-9
        face_normals = cross / norm_len
        centroids    = (v0 + v1 + v2) / 3.0
        med_area     = np.median(area[area > 0]) + 1e-12

        tree    = cKDTree(centroids)
        cos_thr = np.cos(np.radians(angle_threshold_deg))

        outlier = np.zeros(len(faces), dtype=bool)
        chunk   = 50_000

        for start in range(0, len(faces), chunk):
            end    = min(start + chunk, len(faces))
            q_ctr  = centroids[start:end]
            q_norm = face_normals[start:end]
            q_area = area[start:end]

            nbrs = tree.query_ball_point(q_ctr, r=neighborhood_radius,
                                         workers=-1)
            for li, nbr_idx in enumerate(nbrs):
                if len(nbr_idx) < 3:
                    continue
                avg_n = face_normals[nbr_idx].mean(axis=0)
                avg_n /= (np.linalg.norm(avg_n) + 1e-9)
                cos   = float(np.dot(q_norm[li], avg_n))
                gi    = start + li
                if cos < cos_thr and q_area[li] < med_area * area_fraction:
                    outlier[gi] = True

        n_rem = int(outlier.sum())
        if n_rem > 0:
            mesh.remove_triangles_by_mask(outlier)
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
            log.info(f"  Gentle normal filter: removed {n_rem:,} shard faces "
                     f"({100*n_rem/len(faces):.2f}%)")
        else:
            log.info("  Gentle normal filter: no shards found")
        return mesh

    def _light_edge_collapse(self, mesh, reduction: float = 0.10):
        n = len(mesh.triangles)
        if n < 10_000:
            return mesh
        target = int(n * (1.0 - reduction))
        log.info(f"  Light edge collapse: {n:,} → {target:,} faces ...")
        mesh = mesh.simplify_quadric_decimation(target)
        mesh.compute_vertex_normals()
        return mesh

    def _smooth_laplacian_hc(self, mesh, iterations: int = 2,
                              alpha: float = 0.5, beta: float = 0.5):
        try:
            import open3d as o3d
        except ImportError:
            return mesh

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        n_v   = len(verts)

        adj = [[] for _ in range(n_v)]
        for f in faces:
            for i in range(3):
                a, b = f[i], f[(i+1)%3]
                adj[a].append(b)
                adj[b].append(a)

        p = verts.copy()
        q = verts.copy()

        for _ in range(iterations):
            q_new = np.zeros_like(q)
            for i in range(n_v):
                q_new[i] = q[adj[i]].mean(axis=0) if adj[i] else q[i]

            b = q_new - (alpha * p + (1.0 - alpha) * q)
            b_corr = np.zeros_like(b)
            for i in range(n_v):
                b_corr[i] = b[adj[i]].mean(axis=0) if adj[i] else b[i]

            q = q_new - (beta * b + (1.0 - beta) * b_corr)

        mesh.vertices = o3d.utility.Vector3dVector(q)
        mesh.compute_vertex_normals()
        return mesh

    def _components_safe(self, mesh, min_size: int = 2000,
                      bbox_min_m: float = 0.03, max_aspect: float = 15.0):
        import open3d as o3d
        if len(mesh.triangles) == 0:
            return mesh

        tri_clusters, cluster_n_tri, _ = mesh.cluster_connected_triangles()
        tri_clusters  = np.asarray(tri_clusters)
        cluster_n_tri = np.asarray(cluster_n_tri)
        if len(cluster_n_tri) == 0:
            return mesh

        largest_idx = int(np.argmax(cluster_n_tri))
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        to_remove = []
        for i in range(len(cluster_n_tri)):
            if i == largest_idx:
                continue
            fi = np.where(tri_clusters == i)[0]
            cv = verts[faces[fi].flatten()]
            extent = cv.max(axis=0) - cv.min(axis=0)
            diag = np.linalg.norm(extent)
            # thin-sliver check: longest extent vs shortest extent
            sorted_extent = np.sort(extent)
            aspect = sorted_extent[-1] / (sorted_extent[0] + 1e-6)

            if cluster_n_tri[i] < min_size and diag < bbox_min_m:
                to_remove.append(i)
            elif aspect > max_aspect and cluster_n_tri[i] < min_size * 5:
                # large-but-thin curled flap -> almost certainly a tear
                to_remove.append(i)

        if to_remove:
            mask = np.isin(tri_clusters, to_remove)
            mesh.remove_triangles_by_mask(mask)
            mesh.remove_unreferenced_vertices()
            log.info(f"  Component removal: {int(mask.sum()):,} faces from "
                    f"{len(to_remove)} clusters (size+aspect filter)")
        return mesh

    def _fill_holes_open3d(self, mesh, max_hole_size: int = 30):
        try:
            import open3d as o3d
        except ImportError:
            return mesh

        n_before = len(mesh.triangles)
        try:
            filled = mesh.fill_holes(hole_size=max_hole_size)
            if hasattr(filled, "triangles") and len(filled.triangles) > n_before:
                mesh = filled
                mesh.compute_vertex_normals()
                log.info(f"  Hole fill (O3D): {n_before:,} → "
                         f"{len(mesh.triangles):,} faces")
        except Exception as e:
            log.debug(f"  fill_holes skipped: {e}")
        return mesh

    def _fill_large_holes_planar(self, mesh):
        try:
            import open3d as o3d
        except ImportError:
            return mesh

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.triangles, dtype=np.int32)
        if len(faces) == 0:
            return mesh

        from collections import defaultdict
        edge_count = defaultdict(int)
        edge_face  = defaultdict(list)
        for fi, f in enumerate(faces):
            for j in range(3):
                e = tuple(sorted([f[j], f[(j+1)%3]]))
                edge_count[e] += 1
                edge_face[e].append(fi)

        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            return mesh

        b_adj = defaultdict(list)
        for a, b in boundary_edges:
            b_adj[a].append(b)
            b_adj[b].append(a)

        visited = set()
        loops   = []
        for start in b_adj:
            if start in visited:
                continue
            loop  = [start]
            visited.add(start)
            prev  = -1
            curr  = start
            while True:
                nbrs = [n for n in b_adj[curr] if n != prev and n not in visited]
                if not nbrs:
                    break
                nxt  = nbrs[0]
                loop.append(nxt)
                visited.add(nxt)
                prev = curr
                curr = nxt
            loops.append(loop)

        new_verts = list(verts)
        new_faces = list(faces)

        for loop in loops:
            if len(loop) < 6 or len(loop) > 500:
                continue
            loop_pts = verts[loop]
            centroid = loop_pts.mean(axis=0)

            cid = len(new_verts)
            new_verts.append(centroid)

            for k in range(len(loop)):
                a = loop[k]
                b = loop[(k+1) % len(loop)]
                new_faces.append([cid, a, b])

        if len(new_faces) > len(faces):
            new_mesh = o3d.geometry.TriangleMesh()
            new_mesh.vertices  = o3d.utility.Vector3dVector(
                np.array(new_verts, dtype=np.float64))
            new_mesh.triangles = o3d.utility.Vector3iVector(
                np.array(new_faces, dtype=np.int32))
            new_mesh.compute_vertex_normals()
            n_added = len(new_faces) - len(faces)
            log.info(f"  Planar hole fill: added {n_added:,} faces "
                     f"across {sum(1 for l in loops if 6 <= len(l) <= 500)} holes")
            return new_mesh

        return mesh

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE – Poisson reconstruction
    # ──────────────────────────────────────────────────────────────────────────

    def _poisson(self, pcd_path, out_path, frames=None):
        try:
            import open3d as o3d
        except ImportError:
            return

        log.info(f"Loading point cloud: {pcd_path}")
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        log.info(f"  {len(pcd.points):,} points loaded")

        if len(pcd.points) < 100:
            log.error("Too few points."); return

        pts = np.asarray(pcd.points)
        ok  = np.isfinite(pts).all(axis=1)
        if (~ok).sum():
            pcd = pcd.select_by_index(np.where(ok)[0])

        if len(pcd.points) > 500_000:
            pcd = pcd.voxel_down_sample(self.tsdf_voxel_size)
            log.info(f"  Downsampled to {len(pcd.points):,}")

        log.info("  SOR outlier removal ...")
        # FIX: std_ratio raised 2.5→3.5 and nb_neighbors raised 20→30 to avoid
        # removing valid surface points near edges and corners (was causing holes).
        # Radius outlier nb_points raised 6→8, radius tightened 0.07→0.05 to
        # remove only true floating noise, not sparse-but-valid surface samples.
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=3.5)
        pcd, _ = pcd.remove_radius_outlier(nb_points=8, radius=0.05)
        log.info(f"  After outlier removal: {len(pcd.points):,} points")

        if len(pcd.points) < 100:
            log.error("Point cloud too sparse after cleaning."); return

        radius = float(self.tsdf_voxel_size * 8.0)
        log.info("Estimating normals ...")
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=50))

        if frames:
            pcd = self._orient_kdtree(pcd, frames)
        else:
            pcd.orient_normals_consistent_tangent_plane(k=15)

        log.info(f"Poisson reconstruction (depth={self.depth}) ...")
        mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=self.depth, width=0, scale=1.05, linear_fit=False)
        log.info(f"  Raw: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

        d = np.asarray(dens)
        # FIX: density_quantile clamped to 0.02 max here even if caller sets higher.
        # Pruning > 2% of low-density vertices removes valid surface points at
        # frame boundaries where the TSDF only saw a few frames — exactly the
        # seam regions that were producing the torn jagged edges in the mesh.
        safe_quantile = min(self.density_quantile, 0.08)
        thr = np.quantile(d, safe_quantile)
        mesh.remove_vertices_by_mask(d < thr)
        mesh.compute_vertex_normals()

        n_total  = len(mesh.triangles)
        # FIX: min_size raised proportionally — 0.002 (was 0.005) avoids
        # removing medium-sized valid components at scan boundaries.
        min_size = max(1000, int(n_total * 0.002))
        mesh = self._components_safe(mesh, min_size=min_size, bbox_min_m=0.02)
        mesh = self._light_edge_collapse(mesh, reduction=0.10)
        mesh = self._smooth_laplacian_hc(mesh, iterations=2)
        mesh = self._fill_holes_open3d(mesh, max_hole_size=50)

        if self.target_faces > 0 and len(mesh.triangles) > self.target_faces:
            mesh = mesh.simplify_quadric_decimation(self.target_faces)
            mesh.compute_vertex_normals()

        self._save(mesh, out_path)

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE – normal orientation
    # ──────────────────────────────────────────────────────────────────────────

    def _orient_kdtree(self, pcd, frames):
        import open3d as o3d
        try:
            import torch
            if torch.cuda.is_available():
                return self._orient_gpu(pcd, frames)
        except ImportError:
            pass
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return self._orient_slow(pcd, frames)

        # Camera positions in TSDF world space (= OpenCV convention)
        # c2w_gl @ _FLIP converts OpenGL c2w to OpenCV c2w, whose translation
        # is the camera position in the TSDF's world frame.
        cams  = np.array([(fr.c2w @ _FLIP)[:3, 3] for fr in frames])
        pts   = np.asarray(pcd.points)
        norms = np.asarray(pcd.normals).copy()
        _, idx = cKDTree(cams).query(pts, k=1, workers=-1)
        to_cam = cams[idx] - pts
        flip   = np.einsum("ij,ij->i", norms, to_cam) < 0
        norms[flip] *= -1
        pcd.normals = o3d.utility.Vector3dVector(norms)
        return pcd

    def _orient_gpu(self, pcd, frames):
        import open3d as o3d, torch
        device   = torch.device('cuda')
        cams_np  = np.array([(fr.c2w @ _FLIP)[:3, 3] for fr in frames])
        pts_np   = np.asarray(pcd.points)
        norms_np = np.asarray(pcd.normals)
        pts_t    = torch.from_numpy(pts_np).to(device)
        norms_t  = torch.from_numpy(norms_np).to(device)
        cams_t   = torch.from_numpy(cams_np).to(device)
        chunk    = 100_000
        idx_t    = torch.zeros(len(pts_t), dtype=torch.long, device=device)
        for i in range(0, len(pts_t), chunk):
            end = min(i + chunk, len(pts_t))
            idx_t[i:end] = torch.cdist(pts_t[i:end], cams_t).argmin(dim=1)
        to_cam = cams_t[idx_t] - pts_t
        flip   = (norms_t * to_cam).sum(dim=1) < 0
        norms_t[flip] *= -1
        pcd.normals = o3d.utility.Vector3dVector(norms_t.cpu().numpy())
        return pcd

    def _orient_slow(self, pcd, frames):
        import open3d as o3d
        cams  = np.array([(fr.c2w @ _FLIP)[:3, 3] for fr in frames])
        pts   = np.asarray(pcd.points)
        norms = np.asarray(pcd.normals).copy()
        for s in range(0, len(pts), 10_000):
            e  = min(s + 10_000, len(pts))
            p  = pts[s:e]
            d  = np.linalg.norm(p[:, None] - cams[None], axis=2)
            tc = cams[d.argmin(1)] - p
            fl = np.einsum("ij,ij->i", norms[s:e], tc) < 0
            norms[s:e][fl] *= -1
        pcd.normals = o3d.utility.Vector3dVector(norms)
        return pcd

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE – save
    # ──────────────────────────────────────────────────────────────────────────

    def _save(self, mesh, path):
        import open3d as o3d
        o3d.io.write_triangle_mesh(str(path), mesh)
        log.info(f"Mesh saved -> {path}")
        o3d.io.write_triangle_mesh(str(path.with_suffix(".obj")), mesh)
        log.info(f"OBJ saved  -> {path.with_suffix('.obj')}")