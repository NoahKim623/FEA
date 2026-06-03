"""
mesh_handler.py
===============

Geometry / mesh pipeline for the FEA app.

Responsibilities
----------------
1. **Import** a triangulated STL surface with ``meshio`` and turn it into a
   clean PyVista ``PolyData`` (coincident STL vertices are merged).
2. **Validate** that the surface is *watertight* (a closed two-manifold) -- a
   hard requirement before it can be filled with tetrahedra.  An optional
   ``fill_holes`` repair is offered.
3. **Tetrahedralise** the surface into a solid Tet4 volume mesh with TetGen.
4. **Assess mesh quality** (scaled Jacobian / aspect ratio) and warn about
   slivers or inverted cells.
5. **Select boundary faces** for boundary conditions, either by the six
   bounding-box extreme planes (robust, default) and compute *consistent nodal
   loads* for a traction or pressure applied to a face.

Units: all coordinates are converted to **metres** on import via ``unit_scale``
so that downstream forces/stresses come out in SI (N, Pa, N/m).
"""

from __future__ import annotations

import queue as _queue
import time as _time
import warnings
from collections import defaultdict
from multiprocessing import get_context

import numpy as np
import meshio
import pyvista as pv

try:
    import tetgen
except ImportError:  # pragma: no cover - surfaced to the user in the UI
    tetgen = None

try:
    import pymeshfix
except ImportError:  # pragma: no cover - repair just falls back to PyVista
    pymeshfix = None

import tetgen_worker


# Mapping of common STL units to the scale factor that converts to metres.
UNIT_SCALE = {"m": 1.0, "mm": 1.0e-3, "cm": 1.0e-2, "in": 0.0254}


# ---------------------------------------------------------------------------
# Surface <-> array helpers and crash-isolated TetGen driver
# ---------------------------------------------------------------------------
def _polydata_to_arrays(surf: "pv.PolyData"):
    """Return ``(points, tris)`` arrays from a triangulated PolyData surface."""
    surf = surf.triangulate()
    pts = np.ascontiguousarray(surf.points, dtype=np.float64)
    tris = surf.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
    return pts, tris


def _build_tet_grid(node: np.ndarray, elem: np.ndarray) -> "pv.UnstructuredGrid":
    """Build a PyVista Tet4 ``UnstructuredGrid`` from node/element arrays."""
    elem = np.asarray(elem, dtype=np.int64)
    n_cells = elem.shape[0]
    cells = np.hstack([np.full((n_cells, 1), 4, dtype=np.int64), elem]).ravel()
    celltypes = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, celltypes,
                               np.ascontiguousarray(node, dtype=np.float64))


def _run_tetgen_isolated(points, tris, kwargs, timeout: float = 180.0):
    """Run TetGen in a child process; return ``(node, elem)`` arrays.

    Running TetGen out-of-process means a hard C-level crash on pathological
    geometry (self-intersections, non-manifold junctions) only kills the child:
    we detect the non-zero exit code and raise :class:`MeshError` instead of
    letting the whole application disappear.

    Raises
    ------
    MeshError
        If TetGen reports an error, crashes, or exceeds ``timeout`` seconds.
    """
    ctx = get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=tetgen_worker.run, args=(points, tris, kwargs, q))
    proc.start()

    result = None
    deadline = _time.monotonic() + timeout
    while True:
        try:
            result = q.get(timeout=0.2)
            break
        except _queue.Empty:
            if not proc.is_alive():
                break                       # crashed without posting a result
            if _time.monotonic() > deadline:
                proc.terminate()
                proc.join(5)
                raise MeshError(
                    f"TetGen timed out after {timeout:.0f}s.  The surface is "
                    "probably too complex or has many self-intersections.")
    proc.join(5)

    if result is None:
        raise MeshError(
            "TetGen crashed while meshing this surface (it is most likely "
            "self-intersecting or non-manifold).  An automatic repair pass "
            "will be attempted.")
    tag = result[0]
    if tag == "ok":
        return result[1], result[2]
    raise MeshError(f"TetGen failed: {result[1]}")

# Named bounding-box faces: (axis index, side).  Axis 0=X, 1=Y, 2=Z.
FACE_OPTIONS = {
    "-X (min)": (0, "min"), "+X (max)": (0, "max"),
    "-Y (min)": (1, "min"), "+Y (max)": (1, "max"),
    "-Z (min)": (2, "min"), "+Z (max)": (2, "max"),
}


class MeshError(RuntimeError):
    """Raised for unrecoverable geometry / meshing problems."""


class MeshHandler:
    """Owns the surface mesh, the tetrahedral volume mesh and face queries."""

    def __init__(self):
        self.surface = None        # pv.PolyData  (input surface, in metres)
        self.volume = None         # pv.UnstructuredGrid (Tet4 volume mesh)
        self.nodes = None          # (n_nodes, 3) volume-mesh node coordinates
        self.tets = None           # (n_elem, 4) tetra connectivity
        self.unit_scale = 1.0
        self.source_path = None
        self.repaired = False      # True if the surface needed pymeshfix repair

        # Surface of the *volume* mesh (for load distribution / face picking).
        self._surf = None          # pv.PolyData
        self._surf_tris = None     # (n_tri, 3) triangles in GLOBAL node ids
        self._surf_node_ids = None # global node ids that lie on the surface
        self._surf_normals = None  # (n_tri, 3) outward unit normal per triangle
        self._surf_centroids = None# (n_tri, 3) centroid per surface triangle
        self._tri_adj = None       # list: edge-adjacent triangle indices

    # ==================================================================
    # 1. STL import
    # ==================================================================
    def load_stl(self, path: str, unit_scale: float = 1.0) -> dict:
        """Read an STL file with meshio and build a cleaned surface mesh.

        Parameters
        ----------
        path : path to a binary or ASCII ``.stl`` file.
        unit_scale : multiply coordinates by this to convert to metres
            (e.g. 1e-3 if the STL is in millimetres).

        Returns
        -------
        dict with basic surface info (n_points, n_faces, bounds, size).
        """
        try:
            m = meshio.read(path)
        except Exception as exc:  # meshio raises various exceptions
            raise MeshError(f"Could not read STL file:\n{exc}") from exc

        # Gather all triangle cells (STL is triangles-only, but be defensive).
        tri_blocks = [c.data for c in m.cells if c.type == "triangle"]
        if not tri_blocks:
            raise MeshError("No triangular facets found in the STL file.")
        tris = np.vstack(tri_blocks).astype(np.int64)
        points = np.asarray(m.points, dtype=float) * float(unit_scale)

        # Build a PyVista PolyData.  PyVista's face array is a flat list of
        # [n, i0, i1, ..., n, j0, ...]; for triangles n == 3 throughout.
        faces = np.hstack([np.full((tris.shape[0], 1), 3, dtype=np.int64),
                           tris]).ravel()
        surf = pv.PolyData(points, faces)

        # STL stores vertices per-facet, so the same corner is duplicated many
        # times.  clean() merges coincident points -> a proper connected mesh,
        # which is essential for watertightness checks and for TetGen.
        surf = surf.clean(tolerance=1e-8, absolute=False)
        if surf.n_points == 0 or surf.n_faces == 0:
            raise MeshError("STL surface is empty after cleaning.")

        self.surface = surf
        self.unit_scale = float(unit_scale)
        self.source_path = path
        # New geometry invalidates any previous volume mesh.
        self.volume = self.nodes = self.tets = None
        self._surf = self._surf_tris = self._surf_node_ids = None

        size = surf.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
        return {
            "n_points": surf.n_points,
            "n_faces": surf.n_faces,
            "bounds": size,
            "extent": (size[1] - size[0], size[3] - size[2], size[5] - size[4]),
        }

    # ==================================================================
    # 2. Watertightness
    # ==================================================================
    def check_watertight(self):
        """Return ``(is_watertight, n_open_edges)`` for the input surface.

        A surface is watertight when every edge is shared by exactly two
        facets, i.e. it has zero *boundary* (open) edges.  We count boundary
        edges with PyVista's feature-edge extraction, which is robust across
        versions.
        """
        if self.surface is None:
            raise MeshError("Load an STL before checking watertightness.")
        edges = self.surface.extract_feature_edges(
            boundary_edges=True, feature_edges=False,
            manifold_edges=False, non_manifold_edges=False,
        )
        n_open = int(edges.n_cells)
        return (n_open == 0), n_open

    def attempt_repair(self, hole_size_fraction: float = 0.5) -> int:
        """Try to make the input surface a clean watertight manifold.

        Prefers :mod:`pymeshfix` (which removes self-intersections and degenerate
        faces and closes holes, producing a manifold that TetGen can fill) and
        falls back to PyVista's ``fill_holes`` when pymeshfix is unavailable.

        Returns the number of open edges remaining afterwards.
        """
        if self.surface is None:
            raise MeshError("Load an STL first.")

        if pymeshfix is not None:
            pts, tris = _polydata_to_arrays(self.surface)
            rpts, rtris = self._repair_arrays_pymeshfix(pts, tris)
            faces = np.hstack([np.full((rtris.shape[0], 1), 3, dtype=np.int64),
                               rtris.astype(np.int64)]).ravel()
            self.surface = pv.PolyData(rpts, faces).clean()
        else:
            diag = float(np.linalg.norm(
                np.ptp(np.array(self.surface.bounds).reshape(3, 2), axis=1)))
            filled = self.surface.fill_holes(hole_size_fraction * diag)
            filled = filled.clean().triangulate()
            filled = filled.compute_normals(auto_orient_normals=True,
                                            consistent_normals=True)
            self.surface = filled

        _, n_open = self.check_watertight()
        return n_open

    @staticmethod
    def _repair_arrays_pymeshfix(points, tris):
        """Repair a triangle soup with pymeshfix; return ``(points, tris)``."""
        mf = pymeshfix.MeshFix(np.ascontiguousarray(points, dtype=np.float64),
                               np.ascontiguousarray(tris, dtype=np.int32))
        # joincomp keeps disconnected shells; remove_smallest_components off so
        # thin lattice struts are not discarded.
        mf.repair(joincomp=True, remove_smallest_components=False)
        return (np.ascontiguousarray(mf.points, dtype=np.float64),
                np.ascontiguousarray(mf.faces, dtype=np.int32))

    # ==================================================================
    # 3. Tetrahedralisation
    # ==================================================================
    def tetrahedralize(self, refinement: str = "medium",
                       min_radius_edge: float = 1.5,
                       min_dihedral: float = 10.0) -> dict:
        """Fill the surface with Tet4 elements using TetGen.

        Parameters
        ----------
        refinement : {"coarse", "medium", "fine"} controls the maximum tetra
            volume (as a fraction of the bounding-box volume).  ``coarse`` lets
            TetGen choose; finer levels cap the element volume to refine.
        min_radius_edge : TetGen ``-q`` radius/edge ratio bound (lower = better
            shaped but more elements; 1.1-2.0 is sensible).
        min_dihedral : minimum dihedral angle (deg) used as a quality target.

        Returns
        -------
        dict with n_nodes, n_elem.
        """
        if tetgen is None:
            raise MeshError(
                "The 'tetgen' package is not installed.  Install it with "
                "`pip install tetgen`."
            )
        if self.surface is None:
            raise MeshError("Load an STL before meshing.")

        # TetGen needs a clean, triangulated, manifold PLC surface.
        surf = self.surface.triangulate().clean()
        pts, tris = _polydata_to_arrays(surf)

        # Progressive kwargs: try the requested quality first, then fall back to
        # easier-to-satisfy settings so awkward geometry still produces a mesh
        # rather than failing outright.
        b = np.array(surf.bounds).reshape(3, 2)
        bbox_vol = float(np.prod(b[:, 1] - b[:, 0]))
        variants = []
        kw = dict(order=1, mindihedral=float(min_dihedral),
                  minratio=float(min_radius_edge))
        if refinement != "coarse":
            frac = {"medium": 1.0 / 4000.0, "fine": 1.0 / 30000.0}[refinement]
            kw["maxvolume"] = max(bbox_vol * frac, 1e-30)
        variants.append(kw)
        # Relaxed quality, no volume cap.
        variants.append(dict(order=1, mindihedral=10.0, minratio=2.0))
        # Last resort: a plain conforming tetrahedralisation (no quality pass).
        variants.append(dict(order=1, quality=False))

        node = elem = None
        last_err = None
        repaired = False
        # Round 0 = surface as imported; round 1 = after a pymeshfix repair.
        for _round in range(2):
            for kwargs in variants:
                try:
                    node, elem = _run_tetgen_isolated(pts, tris, kwargs)
                    break
                except MeshError as exc:
                    last_err = exc
                    node = elem = None
            if node is not None:
                break
            if not repaired and pymeshfix is not None:
                # The surface is likely self-intersecting / non-manifold; clean
                # it up and try the whole ladder again.
                pts, tris = self._repair_arrays_pymeshfix(pts, tris)
                repaired = True
            else:
                break

        if node is None:
            raise MeshError(
                "TetGen could not mesh this surface even after repair.  It is "
                "likely badly self-intersecting or has zero-thickness features."
                + (f"\n\nDetails: {last_err}" if last_err else ""))

        self.volume = _build_tet_grid(node, elem)
        self.nodes = np.ascontiguousarray(node, dtype=float)
        self.tets = np.ascontiguousarray(elem, dtype=np.int64)
        self.repaired = repaired
        self._build_surface_lookup()

        return {"n_nodes": self.nodes.shape[0], "n_elem": self.tets.shape[0],
                "repaired": repaired}

    def _build_surface_lookup(self):
        """Cache the boundary triangles of the volume mesh (global node ids)."""
        surf = self.volume.extract_surface(pass_pointid=True, pass_cellid=False,
                                           algorithm="dataset_surface")
        surf = surf.triangulate()
        orig = np.asarray(surf.point_data["vtkOriginalPointIds"], dtype=np.int64)
        faces = surf.faces.reshape(-1, 4)[:, 1:4]            # local tri ids
        self._surf = surf
        self._surf_tris = orig[faces]                        # global node ids
        self._surf_node_ids = np.unique(self._surf_tris)

        # Per-triangle geometry used for click-picking face regions.
        self._surf_normals = self._triangle_normals(self._surf_tris)
        self._surf_centroids = self.nodes[self._surf_tris].mean(axis=1)

        # Edge-adjacency: triangles sharing an edge are neighbours, used to
        # region-grow a connected near-coplanar patch from a clicked point.
        edge_map = defaultdict(list)
        for ti, (a, b, c) in enumerate(self._surf_tris):
            for u, v in ((a, b), (b, c), (c, a)):
                edge_map[(u, v) if u < v else (v, u)].append(ti)
        adj = [set() for _ in range(self._surf_tris.shape[0])]
        for sharing in edge_map.values():
            for i in sharing:
                for j in sharing:
                    if i != j:
                        adj[i].add(j)
        self._tri_adj = [np.fromiter(s, dtype=np.int64) for s in adj]

    # ==================================================================
    # Click-to-pick face region selection
    # ==================================================================
    def pick_face_region(self, point, angle_tol_deg: float = 45.0) -> dict:
        """Select a connected, near-coplanar surface patch around ``point``.

        Starting from the surface triangle nearest the clicked ``point``, this
        flood-fills across edge-adjacent triangles whose outward normal stays
        within ``angle_tol_deg`` of the seed triangle's normal.  The result is
        a CAD-like "face" (a flat or gently curved region), letting the user
        pick a load- or support-bearing face by clicking directly on the model.

        Returns
        -------
        dict with keys:
            ``node_ids``  - unique global node ids of the selected patch,
            ``normal``    - area-weighted mean outward unit normal (3,),
            ``centroid``  - area-weighted centroid of the patch (3,),
            ``n_tris``    - number of triangles selected.
        """
        if self._surf_tris is None:
            raise MeshError("Generate a volume mesh first.")
        pt = np.asarray(point, dtype=float).ravel()[:3]
        seed = int(np.argmin(((self._surf_centroids - pt) ** 2).sum(axis=1)))

        ref = self._surf_normals[seed]
        cos_tol = np.cos(np.radians(angle_tol_deg))
        seen = {seed}
        stack = [seed]
        while stack:
            t = stack.pop()
            for nb in self._tri_adj[t]:
                nb = int(nb)
                if nb in seen:
                    continue
                if float(self._surf_normals[nb] @ ref) >= cos_tol:
                    seen.add(nb)
                    stack.append(nb)

        tri_idx = np.fromiter(seen, dtype=np.int64)
        sel_tris = self._surf_tris[tri_idx]
        node_ids = np.unique(sel_tris).astype(np.int64)

        areas = self._triangle_areas(sel_tris)
        w = areas / (areas.sum() + 1e-30)
        normal = (self._surf_normals[tri_idx] * w[:, None]).sum(axis=0)
        nrm = np.linalg.norm(normal)
        normal = normal / nrm if nrm > 0 else ref
        centroid = (self._surf_centroids[tri_idx] * w[:, None]).sum(axis=0)
        return {"node_ids": node_ids, "tris": sel_tris.astype(np.int64),
                "normal": normal, "centroid": centroid,
                "n_tris": int(tri_idx.size)}

    # ==================================================================
    # 4. Mesh quality
    # ==================================================================
    def mesh_quality(self) -> dict:
        """Compute element quality statistics and human-readable warnings.

        Uses a robust, geometry-based *shape quality* metric computed directly
        (so it never returns NaN like some VTK measures can):

            q = 6*sqrt(2) * V / l_rms^3

        where V is the element volume and l_rms is the root-mean-square of the
        six edge lengths.  This is normalised so q = 1 for a regular (ideal)
        tetrahedron and q -> 0 for a flat "sliver".  Signed volumes additionally
        flag inverted (negative-volume) and degenerate (zero-volume) elements.
        Returns summary statistics plus a list of warning strings.
        """
        if self.volume is None:
            raise MeshError("Generate a volume mesh first.")

        warns = []
        from fea_solver import element_volumes_and_gradients
        vol, _, valid = element_volumes_and_gradients(self.nodes, self.tets)
        absvol = np.abs(vol)
        n_inverted = int(np.count_nonzero(vol <= 0.0))
        n_degenerate = int(np.count_nonzero(~valid))

        # Sum of the six squared edge lengths per tetra -> RMS edge length.
        p = self.nodes[self.tets]                            # (M, 4, 3)
        edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        l2 = np.zeros(self.tets.shape[0])
        for a, b in edges:
            l2 += np.sum((p[:, a] - p[:, b]) ** 2, axis=1)
        l_rms = np.sqrt(l2 / 6.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            q = 6.0 * np.sqrt(2.0) * absvol / np.power(l_rms, 3)
        q = np.clip(np.nan_to_num(q, nan=0.0), 0.0, 1.0)

        stats = {
            "n_elem": self.tets.shape[0],
            "n_nodes": self.nodes.shape[0],
            "vol_total": float(absvol.sum()),
            "vol_min": float(absvol.min()),
            "n_inverted": n_inverted,
            "n_degenerate": n_degenerate,
            "q_min": float(q.min()),
            "q_mean": float(q.mean()),
            "n_poor": int(np.count_nonzero(q < 0.1)),
        }
        if stats["q_min"] < 0.05:
            warns.append(
                f"Poor mesh quality: minimum shape quality {stats['q_min']:.3f} "
                "(< 0.05). Results near sliver elements may be inaccurate; "
                "consider a finer mesh.")
        elif stats["n_poor"]:
            warns.append(
                f"{stats['n_poor']} element(s) have low shape quality (< 0.1).")
        if n_inverted:
            warns.append(f"{n_inverted} inverted element(s) (negative volume).")
        if n_degenerate:
            warns.append(f"{n_degenerate} degenerate (zero-volume) element(s).")

        stats["warnings"] = warns
        return stats

    # ==================================================================
    # 5. Boundary-condition face selection & loads
    # ==================================================================
    def select_face_nodes(self, face_name: str, tol_fraction: float = 0.02) -> np.ndarray:
        """Return global node ids on a named bounding-box face.

        Parameters
        ----------
        face_name : a key of :data:`FACE_OPTIONS` (e.g. ``"+Z (max)"``).
        tol_fraction : selection band thickness as a fraction of that axis'
            total extent (nodes within this distance of the extreme plane are
            selected).

        Only *surface* nodes are returned (interior nodes that happen to lie on
        the plane are excluded).
        """
        if self.nodes is None:
            raise MeshError("Generate a volume mesh first.")
        if face_name not in FACE_OPTIONS:
            raise MeshError(f"Unknown face '{face_name}'.")
        axis, side = FACE_OPTIONS[face_name]

        coord = self.nodes[:, axis]
        extent = coord.max() - coord.min()
        tol = max(tol_fraction * extent, 1e-9)
        target = coord.min() if side == "min" else coord.max()

        on_plane = np.abs(coord - target) <= tol
        # Restrict to surface nodes.
        surf_mask = np.zeros(self.nodes.shape[0], dtype=bool)
        surf_mask[self._surf_node_ids] = True
        ids = np.where(on_plane & surf_mask)[0]
        if ids.size == 0:                      # fall back to any node on plane
            ids = np.where(on_plane)[0]
        return ids.astype(np.int64)

    def face_area(self, node_ids) -> float:
        """Total surface area of the boundary triangles fully inside node_ids."""
        tris = self._face_triangles(node_ids)
        if tris.shape[0] == 0:
            return 0.0
        return float(self._triangle_areas(tris).sum())

    def compute_face_loads(self, node_ids, magnitude: float,
                           direction=None, mode: str = "force") -> dict:
        """Distribute a load on a face into *consistent nodal forces*.

        For a constant traction t over a linear triangle of area A, the
        energetically-consistent nodal force is ``t * A / 3`` at each of the
        three corner nodes (the integral of each linear shape function over the
        triangle is A/3).  Summing over the face's triangles reproduces the
        intended resultant exactly.

        Parameters
        ----------
        node_ids : global node ids defining the loaded face.
        magnitude : in ``"force"`` mode the total force in newtons; in
            ``"pressure"`` mode the pressure in pascals.
        direction : (3,) load direction (only for ``"force"`` mode); it is
            normalised internally.
        mode : ``"force"`` (total force along *direction*) or ``"pressure"``
            (acts along each triangle's outward normal; positive = outward).

        Returns
        -------
        dict {global_node_id: np.array([fx, fy, fz])}
        """
        tris = self._face_triangles(node_ids)
        if tris.shape[0] == 0:
            raise MeshError(
                "No boundary triangles were found on the selected face.  Try a "
                "larger selection tolerance.")
        areas = self._triangle_areas(tris)
        loads = {}

        def _add(nid, f):
            if nid in loads:
                loads[nid] += f
            else:
                loads[nid] = f.copy()

        if mode == "force":
            d = np.asarray(direction, dtype=float)
            nrm = np.linalg.norm(d)
            if nrm == 0:
                raise MeshError("Load direction vector must be non-zero.")
            d = d / nrm
            total_area = float(areas.sum())
            traction = magnitude * d / total_area          # N/m^2 (vector)
            for tri, a in zip(tris, areas):
                f_node = traction * (a / 3.0)
                for nid in tri:
                    _add(int(nid), f_node)
        elif mode == "pressure":
            normals = self._triangle_normals(tris)
            for tri, a, n in zip(tris, areas, normals):
                f_node = magnitude * n * (a / 3.0)         # along outward normal
                for nid in tri:
                    _add(int(nid), f_node)
        else:
            raise MeshError(f"Unknown load mode '{mode}'.")
        return loads

    # -- internal triangle helpers -------------------------------------
    def _face_triangles(self, node_ids) -> np.ndarray:
        """Boundary triangles whose three vertices are all in ``node_ids``."""
        sel = np.zeros(self.nodes.shape[0], dtype=bool)
        sel[np.asarray(node_ids, dtype=np.int64)] = True
        m = sel[self._surf_tris].all(axis=1)
        return self._surf_tris[m]

    def _triangle_areas(self, tris) -> np.ndarray:
        p = self.nodes
        cross = np.cross(p[tris[:, 1]] - p[tris[:, 0]],
                        p[tris[:, 2]] - p[tris[:, 0]])
        return 0.5 * np.linalg.norm(cross, axis=1)

    def _triangle_normals(self, tris) -> np.ndarray:
        p = self.nodes
        cross = np.cross(p[tris[:, 1]] - p[tris[:, 0]],
                        p[tris[:, 2]] - p[tris[:, 0]])
        n = np.linalg.norm(cross, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return cross / n

    # ==================================================================
    # Misc accessors
    # ==================================================================
    @property
    def bbox_diagonal(self) -> float:
        """Length of the bounding-box diagonal of the volume mesh (m)."""
        if self.nodes is None:
            return 1.0
        ext = self.nodes.max(axis=0) - self.nodes.min(axis=0)
        return float(np.linalg.norm(ext))

    @property
    def characteristic_size(self) -> float:
        """A representative model size (mean bounding-box edge length)."""
        if self.nodes is None:
            return 1.0
        ext = self.nodes.max(axis=0) - self.nodes.min(axis=0)
        return float(ext.mean())
