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


def _tetgen_body_isolated(pts, tris, variants, timeout: float = 60.0):
    """Mesh one shell, trying each variant in its own isolated TetGen run."""
    last = None
    for kwargs in variants:
        try:
            return _run_tetgen_isolated(pts, tris, kwargs, timeout=timeout)
        except MeshError as exc:
            last = exc
    raise last or MeshError("TetGen failed on this shell.")


def _run_tetgen_many_isolated(bodies, variants, timeout_per_body: float = 60.0):
    """Mesh a list of ``(points, tris)`` shells; return ``{index: (node, elem)}``.

    The whole list is meshed in **one** child process (:func:`tetgen_worker.run_many`)
    which streams each body's result back as it finishes -- so the common case
    pays a single process-spawn cost no matter how many pieces an assembly has.
    If that child dies on a pathological body (hard C-level crash) any body that
    was neither completed nor explicitly rejected is re-meshed afterwards in full
    per-body isolation, so one bad shell can never sink the whole batch.
    """
    if not bodies:
        return {}
    ctx = get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=tetgen_worker.run_many, args=(bodies, variants, q))
    proc.start()

    results = {}        # index -> (node, elem)
    failed = set()      # indices TetGen explicitly rejected (don't retry)
    clean_finish = False
    deadline = _time.monotonic() + max(30.0, timeout_per_body * len(bodies))
    while True:
        try:
            msg = q.get(timeout=0.2)
        except _queue.Empty:
            if not proc.is_alive():
                break
            if _time.monotonic() > deadline:
                proc.terminate()
                break
            continue
        tag = msg[0]
        if tag == "item":
            results[msg[1]] = (msg[2], msg[3])
        elif tag == "itemerr":
            failed.add(msg[1])
        elif tag == "done":
            clean_finish = True
            break
        elif tag == "fatal":
            break
    proc.join(5)

    # Re-mesh anything lost to a crash/timeout, each in its own process.
    if not clean_finish:
        for i, (pts, tris) in enumerate(bodies):
            if i in results or i in failed:
                continue
            try:
                results[i] = _tetgen_body_isolated(pts, tris, variants,
                                                   timeout=timeout_per_body)
            except MeshError:
                failed.add(i)
    return results

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

    # Curvature-adaptive simplification tolerance per refinement level, as a
    # fraction of the bounding-box diagonal: the maximum the simplified surface
    # may deviate from the original.  Smaller = curved regions keep more
    # triangles; flat regions collapse regardless of the level.
    _REFINE_TOL = {"coarse": 0.012, "medium": 0.004, "fine": 0.0012}

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

        The repair strategy is chosen from the *kind* of defect:

        * **Multiple/overlapping shells or non-manifold junctions** (e.g. a base
          plate with separately-modelled features sitting in it) cannot be fixed
          by hole-filling -- the pieces overlap and self-intersect.  These are
          rebuilt as a single watertight *union* via :meth:`_remesh_union`.
        * **A single shell with holes** is repaired in place with :mod:`pymeshfix`
          (or PyVista's ``fill_holes`` fallback), which preserves the original
          geometry far more faithfully than a voxel rebuild.

        Returns the number of open edges remaining afterwards.
        """
        if self.surface is None:
            raise MeshError("Load an STL first.")

        surf = self.surface.triangulate().clean()
        n_components = len(self._split_components(surf))
        n_nonmanifold = surf.extract_feature_edges(
            boundary_edges=False, feature_edges=False,
            manifold_edges=False, non_manifold_edges=True).n_cells

        if n_components > 1 or n_nonmanifold > 0:
            try:
                self.surface = self._remesh_union(surf)
                self.repaired = True
                _, n_open = self.check_watertight()
                return n_open
            except MeshError:
                pass            # fall through to a soup repair as a last resort

        if pymeshfix is not None:
            pts, tris = _polydata_to_arrays(surf)
            rpts, rtris = self._repair_arrays_pymeshfix(pts, tris)
            faces = np.hstack([np.full((rtris.shape[0], 1), 3, dtype=np.int64),
                               rtris.astype(np.int64)]).ravel()
            self.surface = pv.PolyData(rpts, faces).clean()
        else:
            diag = float(np.linalg.norm(
                np.ptp(np.array(surf.bounds).reshape(3, 2), axis=1)))
            filled = surf.fill_holes(hole_size_fraction * diag)
            filled = filled.clean().triangulate()
            filled = filled.compute_normals(auto_orient_normals=True,
                                            consistent_normals=True)
            self.surface = filled

        self.repaired = True
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
        ref_b = np.array(surf.bounds).reshape(3, 2)
        ref_diag = float(np.linalg.norm(ref_b[:, 1] - ref_b[:, 0]))

        # Curvature-adaptive deviation budget for this refinement level.
        tol_frac = self._REFINE_TOL.get(refinement, 0.006)

        node = elem = None
        last_err = None
        repaired = False
        n_components = 1

        # 1. Exact path -- any clean, watertight, two-manifold surface (one shell
        #    OR several disjoint/nested shells) is fed straight to TetGen, which
        #    preserves the geometry and the curvature-adaptive surface exactly.
        #    Accepting *multiple* clean shells here (not just a single one) is
        #    what keeps flat plate / sheet / multi-part structures crisp: they no
        #    longer get diverted into the voxel rebuild below, which would smear
        #    their flat faces into a dense, un-flattenable triangulation.
        if self._is_closed_manifold(surf):
            try:
                msurf = self._adaptive_surface(surf, tol_frac)
                pts, tris = _polydata_to_arrays(msurf)
                node, elem = self._tetgen_ladder(
                    pts, tris, self._make_variants(
                        msurf, refinement, min_radius_edge, min_dihedral))
            except MeshError as exc:
                last_err = exc
                node = elem = None

        # 1b. Split-mesh-weld repair -- for assemblies of many shells and for
        #     single shells that self-touch at hinges (rotating-tile / re-entrant
        #     auxetic / lattice geometry), the surface is decomposed into its
        #     manifold pieces, volumetric overlaps are unioned away, each piece is
        #     meshed *exactly* by TetGen, and the pieces are welded back together
        #     at their shared nodes.  This keeps every joint connected and every
        #     feature at full size -- the voxel rebuild (step 3) instead severs
        #     joints thinner than its voxel and rounds sharp corners, which is the
        #     gap-at-the-joints failure this path exists to avoid.  It only runs
        #     when the surface actually decomposes into more than one piece.
        if node is None:
            try:
                a_node, a_elem, n_meshed, n_bodies = self._mesh_assembly(
                    surf, self._make_variants(
                        surf, refinement, min_radius_edge, min_dihedral),
                    ref_diag)
                a_diag = float(np.linalg.norm(
                    a_node.max(axis=0) - a_node.min(axis=0)))
                if ref_diag > 0 and a_diag >= 0.5 * ref_diag:
                    node, elem = a_node, a_elem
                    repaired = True
                    n_components = self._count_tet_components(node, elem)
                else:
                    last_err = MeshError(
                        "Split-mesh-weld spanned only "
                        f"{a_diag / ref_diag:.0%} of the model.")
            except MeshError as exc:
                last_err = exc
                node = elem = None

        # 2. Geometry-preserving repair -- pymeshfix stitches up holes,
        #    self-intersections and non-manifold junctions while keeping the
        #    original surface (and therefore its fillets and curved blends)
        #    essentially intact.  It is tried *before* the voxel rebuild so curved
        #    geometry survives whenever the repair yields a clean solid: the
        #    repaired surface then goes through the same curvature-adaptive +
        #    boundary-preserving path as a clean import.
        if node is None and pymeshfix is not None and surf.n_faces <= 200_000:
            try:
                pts, tris = _polydata_to_arrays(surf)
                pts, tris = self._repair_arrays_pymeshfix(pts, tris)
                faces = np.hstack([np.full((tris.shape[0], 1), 3, dtype=np.int64),
                                   tris.astype(np.int64)]).ravel()
                rsurf = pv.PolyData(pts, faces).clean()
                msurf = self._adaptive_surface(rsurf, tol_frac)
                pts, tris = _polydata_to_arrays(msurf)
                repaired = True
                node, elem = self._tetgen_ladder(
                    pts, tris, self._make_variants(
                        msurf, refinement, min_radius_edge, min_dihedral))
            except MeshError as exc:
                last_err = exc
                node = elem = None

        # 3. Last resort -- rebuild the solid as a watertight *union* of all its
        #    shells on a voxel grid.  This always produces a meshable surface for
        #    badly self-intersecting / non-manifold input, but it rounds sharp
        #    corners and re-tessellates uniformly (curves become facets), so it is
        #    only used when the geometry-preserving repair above cannot succeed.
        if node is None:
            try:
                rsurf = self._remesh_union(surf)
                rsurf = self._adaptive_surface(rsurf, tol_frac)
                repaired = True
                pts, tris = _polydata_to_arrays(rsurf)
                node, elem = self._tetgen_ladder(
                    pts, tris, self._make_variants(
                        rsurf, refinement, min_radius_edge, min_dihedral))
            except MeshError as exc:
                last_err = exc
                node = elem = None

        if node is None:
            raise MeshError(
                "TetGen could not mesh this surface even after repair.  It is "
                "likely badly self-intersecting or has zero-thickness features."
                + (f"\n\nDetails: {last_err}" if last_err else ""))

        # Guard against a repair that silently *collapsed* the geometry to a
        # small fragment (the previous failure mode on multi-shell STLs): the
        # meshed volume must still span essentially the whole model.
        node_diag = float(np.linalg.norm(node.max(axis=0) - node.min(axis=0)))
        if ref_diag > 0 and node_diag < 0.5 * ref_diag:
            raise MeshError(
                f"Meshing collapsed the geometry to a fragment "
                f"({node_diag / ref_diag:.0%} of the model size); the surface is "
                "too broken to mesh reliably.  Try exporting a cleaner STL.")

        self.volume = _build_tet_grid(node, elem)
        self.nodes = np.ascontiguousarray(node, dtype=float)
        self.tets = np.ascontiguousarray(elem, dtype=np.int64)
        self.repaired = repaired
        self._build_surface_lookup()

        return {"n_nodes": self.nodes.shape[0], "n_elem": self.tets.shape[0],
                "repaired": repaired, "n_components": n_components}

    # -- meshing helpers ------------------------------------------------
    @staticmethod
    def _adaptive_surface(surf, tol_frac):
        """Curvature-adaptive simplification of a watertight PLC surface.

        Collapses near-coplanar triangles so that *flat* regions are described
        by a few large triangles while *curved* regions keep enough triangles
        to capture their shape -- the geometric meaning of "flat surfaces stay
        flat, curved surfaces are approximated by triangles".

        ``vtkDecimatePro`` is driven here by a *maximum geometric error*
        (``tol_frac`` of the bounding-box diagonal) rather than a fixed
        reduction ratio, so the amount of decimation is decided locally by
        curvature: vertices are removed only while the accumulated deviation
        stays under the tolerance.  Topology is preserved so the result stays a
        watertight two-manifold shell that TetGen can fill.  If decimation ever
        breaks watertightness the original surface is returned unchanged.
        """
        surf = surf.triangulate().clean()
        # A closed shell cannot go below 4 triangles; nothing to gain below that.
        if surf.n_faces < 12 or tol_frac <= 0.0:
            return surf
        try:
            import vtk
            dec = vtk.vtkDecimatePro()
            dec.SetInputData(surf)
            # Allow essentially unlimited removal -- the *error gate* below, not a
            # reduction ratio, decides where to stop.  On a flat region the error
            # stays zero so the triangles merge all the way down to the minimum a
            # box needs (12 triangles / 6 sides); on a curved region removal stops
            # as soon as the deviation would exceed the tolerance.
            dec.SetTargetReduction(0.999)
            dec.PreserveTopologyOn()       # keep the shell watertight / manifold
            dec.SplittingOff()             # never tear the surface apart
            dec.BoundaryVertexDeletionOn()
            # The geometric error gate below -- not the feature angle -- is what
            # protects sharp corners (collapsing a true corner incurs a large
            # deviation and is rejected).  A *low* feature angle here would
            # instead lock whole rings of triangles around any crease (e.g. a
            # cylinder's cap edge), defeating refinement; 45 deg keeps genuine
            # creases without that lock-up.
            dec.SetFeatureAngle(45.0)
            dec.AccumulateErrorOn()        # bound the *total* deviation, not per step
            dec.SetErrorIsAbsolute(0)      # MaximumError is a fraction of the bbox diag
            dec.SetMaximumError(float(tol_frac))
            dec.Update()
            out = pv.wrap(dec.GetOutput()).triangulate().clean()
        except Exception:                  # pragma: no cover - keep raw on any failure
            return surf
        if out.n_faces and MeshHandler._is_closed_manifold(out):
            return out
        return surf

    @staticmethod
    def _make_variants(surf, refinement, min_radius_edge, min_dihedral):
        """Build the progressive list of TetGen kwargs (quality -> permissive).

        ``nobisect`` (TetGen's ``-Y``) forbids Steiner points *on the boundary*,
        so the curvature-adaptive **surface** triangulation is carried into the
        volume mesh unchanged: flat faces stay flat (a box stays 6 sides) and the
        element count is governed by the refinement-driven surface rather than a
        uniform volume cap that would re-subdivide every flat face.  TetGen may
        still add *interior* points to meet the quality target.

        The boundary-preserving (``nobisect``) variants are tried *first and
        exhausted* -- including a plain constrained tetrahedralisation with no
        quality target, which always succeeds for a valid surface -- before any
        boundary-refining variant is attempted.  This keeps the simplified
        surface intact in the volume mesh: TetGen only falls back to
        re-subdividing the boundary (re-densifying flat faces) if the surface is
        genuinely unmeshable as-is.
        """
        return [
            dict(order=1, mindihedral=float(min_dihedral),
                 minratio=float(min_radius_edge), nobisect=True),
            dict(order=1, mindihedral=10.0, minratio=2.0, nobisect=True),
            dict(order=1, nobisect=True, quality=False),   # plain CDT, boundary kept
            dict(order=1, mindihedral=10.0, minratio=2.0), # last resort: refine boundary
            dict(order=1, quality=False),
        ]

    @staticmethod
    def _tetgen_ladder(pts, tris, variants):
        """Try each TetGen variant in turn; return ``(node, elem)`` or raise."""
        last_err = None
        for kwargs in variants:
            try:
                return _run_tetgen_isolated(pts, tris, kwargs)
            except MeshError as exc:
                last_err = exc
        raise last_err or MeshError("TetGen failed on this surface.")

    # -- split / mesh / weld repair (preserves thin joints) -------------
    @staticmethod
    def _split_manifold_shells(surf):
        """Decompose a surface into closed, manifold sub-shells.

        Two levels of separation:

        1. disjoint connected components (separately-modelled parts), and
        2. *within* a component, patches that meet only along **non-manifold
           edges** -- edges shared by more than two facets.

        Level 2 is the key to keeping rotating-tile / lattice "joints" intact.
        Where two rigid cells touch at a hinge (a corner or a zero-thickness
        edge), extrusion turns that contact into an edge used by four facets.
        A solid mesher cannot place material across such a hinge, so the voxel
        fallback simply *severs* it.  Splitting there instead yields one clean
        prism per cell that TetGen meshes exactly; because the split keeps each
        piece's original vertex coordinates, a later coordinate weld
        (:meth:`_weld_assembly`) re-joins the cells at the shared hinge nodes --
        reconnecting the joint without inventing or rounding any geometry.

        Returns a list of ``(points, tris)`` arrays, one per sub-shell.
        """
        surf = surf.triangulate().clean()
        pts = np.ascontiguousarray(surf.points, dtype=np.float64)
        tris = surf.faces.reshape(-1, 4)[:, 1:4].astype(np.int64)
        if tris.shape[0] == 0:
            return []

        # Face adjacency across manifold (exactly-two-facet) edges only; a
        # non-manifold edge is therefore a "wall" the flood fill will not cross.
        edge_faces = defaultdict(list)
        for ti, (a, b, c) in enumerate(tris):
            for u, v in ((a, b), (b, c), (c, a)):
                edge_faces[(u, v) if u < v else (v, u)].append(ti)
        adj = defaultdict(list)
        for fs in edge_faces.values():
            if len(fs) == 2:
                adj[fs[0]].append(fs[1])
                adj[fs[1]].append(fs[0])

        seen = np.zeros(tris.shape[0], dtype=bool)
        pieces = []
        for start in range(tris.shape[0]):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            group = []
            while stack:
                f = stack.pop()
                group.append(f)
                for nb in adj[f]:
                    if not seen[nb]:
                        seen[nb] = True
                        stack.append(nb)
            face_idx = np.asarray(group, dtype=np.int64)
            used = np.unique(tris[face_idx])
            remap = np.full(pts.shape[0], -1, dtype=np.int64)
            remap[used] = np.arange(used.shape[0])
            pieces.append((np.ascontiguousarray(pts[used]),
                           np.ascontiguousarray(remap[tris[face_idx]])))
        return pieces

    @staticmethod
    def _resolve_overlaps(pieces):
        """Union *volumetrically overlapping* pieces into clean solids.

        Parts exported from CAD as separate bodies often interpenetrate (a
        strut sunk into a plate, fillets that overlap).  Meshing those as-is
        would leave doubled, interlocking tetrahedra.  When :mod:`manifold3d`
        is available the pieces are unioned with a robust (exact-predicate)
        boolean so overlaps are resolved into single solids while sharp
        geometry is preserved -- unlike the voxel fallback, which rounds it.

        Pieces that merely *touch* (hinges, coincident faces) are left as
        separate bodies; if there is no real overlap, or :mod:`manifold3d` is
        not installed, the original pieces are returned unchanged so the exact
        hinge vertices survive for the later weld.
        """
        try:
            import manifold3d as m3
            from manifold3d import Manifold, Mesh64, OpType
        except Exception:                       # pragma: no cover - optional dep
            return pieces

        mans, leftover, vol_sum = [], [], 0.0
        for lpts, ltris in pieces:
            try:
                mesh = Mesh64(
                    vert_properties=np.ascontiguousarray(lpts, dtype=np.float64),
                    tri_verts=np.ascontiguousarray(ltris, dtype=np.uint64))
                mesh.merge()
                man = Manifold(mesh)
                if not man.is_empty() and man.status() == m3.Error.NoError:
                    mans.append(man)
                    vol_sum += abs(man.volume())
                    continue
            except Exception:
                pass
            leftover.append((lpts, ltris))       # keep un-convertible pieces as-is

        if len(mans) < 2:
            return pieces
        try:
            u = Manifold.batch_boolean(mans, OpType.Add)
            uvol = abs(u.volume())
        except Exception:                         # pragma: no cover
            return pieces
        if uvol >= 0.999 * vol_sum:
            return pieces                         # negligible overlap -> keep exact pieces

        out = list(leftover)
        try:
            bodies = u.decompose() or [u]
        except Exception:                         # pragma: no cover
            bodies = [u]
        for bd in bodies:
            try:
                me = bd.to_mesh()
                v = np.asarray(me.vert_properties)[:, :3].astype(np.float64)
                f = np.asarray(me.tri_verts).astype(np.int64)
                if f.shape[0]:
                    out.append((np.ascontiguousarray(v), np.ascontiguousarray(f)))
            except Exception:
                pass
        return out or pieces

    @staticmethod
    def _weld_assembly(meshes, ref_diag, tol_frac: float = 3e-4):
        """Concatenate per-body tet meshes and weld near-coincident nodes.

        Welding merges nodes that fall in the same cell of a grid of size
        ``tol_frac * ref_diag`` (a few ten-thousandths of the model), so the
        hinge / contact vertices that two bodies share collapse to one node and
        the bodies become a single connected mesh.  The tolerance is far below
        the smallest real feature, so distinct parts are never fused by
        accident.  Tetrahedra that degenerate (gain a repeated node) under the
        weld are dropped.
        """
        offset = 0
        all_nodes, all_elem = [], []
        for node, elem in meshes:
            all_nodes.append(np.asarray(node, dtype=np.float64))
            all_elem.append(np.asarray(elem, dtype=np.int64) + offset)
            offset += node.shape[0]
        nodes = np.vstack(all_nodes)
        elem = np.vstack(all_elem)

        tol = max(ref_diag * tol_frac, 1e-12)
        keys = np.round(nodes / tol).astype(np.int64)
        uniq, inv = np.unique(keys, axis=0, return_inverse=True)
        inv = inv.ravel()
        new_nodes = np.zeros((uniq.shape[0], 3), dtype=np.float64)
        counts = np.zeros(uniq.shape[0], dtype=np.float64)
        np.add.at(new_nodes, inv, nodes)
        np.add.at(counts, inv, 1.0)
        new_nodes /= counts[:, None]

        elem = inv[elem]
        ok = np.ones(elem.shape[0], dtype=bool)
        for a in range(4):
            for b in range(a + 1, 4):
                ok &= elem[:, a] != elem[:, b]
        elem = elem[ok]
        if elem.shape[0] == 0:
            raise MeshError("Welding the assembly produced no valid elements.")

        # Snapping nodes can flip a tetra's orientation; restore a positive
        # signed volume by swapping two vertices (the element is unchanged
        # geometrically, only its node order), so no element is left inverted.
        p = new_nodes[elem]
        signed = np.einsum("ij,ij->i",
                           np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                           p[:, 3] - p[:, 0])
        flip = signed < 0.0
        elem[flip] = elem[flip][:, [0, 1, 3, 2]]
        return np.ascontiguousarray(new_nodes), np.ascontiguousarray(elem)

    def _mesh_bodies(self, bodies, variants, ref_diag):
        """Mesh every shell in ``bodies`` and weld the result into one mesh."""
        body_arrays = [(np.ascontiguousarray(p, dtype=np.float64),
                        np.ascontiguousarray(t, dtype=np.int32))
                       for p, t in bodies]
        results = _run_tetgen_many_isolated(body_arrays, variants)
        meshes = [results[i] for i in sorted(results)]
        if not meshes:
            raise MeshError("None of the decomposed shells could be meshed.")
        node, elem = self._weld_assembly(meshes, ref_diag)
        return node, elem, len(meshes)

    def _mesh_assembly(self, surf, variants, ref_diag):
        """Repair by decomposition: split into manifold shells, mesh each body,
        then weld the shared nodes back together.

        Two attempts, cleaner first:

        * **overlap-resolved** -- if :mod:`manifold3d` unions interpenetrating
          pieces into clean solids, those are meshed (correct material in the
          overlap, no doubled tetrahedra);
        * **raw pieces** -- otherwise (or when the union is too degenerate for
          TetGen, which happens when bodies merely touch at faces) every original
          manifold piece is meshed independently and welded.  Overlapping pieces
          then interpenetrate, but nothing is lost and the joints still connect --
          strictly better than failing outright.

        Returns ``(node, elem, n_meshed, n_bodies)`` or raises :class:`MeshError`.
        """
        pieces = self._split_manifold_shells(surf)
        if len(pieces) < 2:
            raise MeshError("Surface does not decompose into multiple shells.")

        attempts = []
        resolved = self._resolve_overlaps(pieces)
        if resolved is not pieces:
            attempts.append(resolved)      # overlap-resolved (cleaner) first
        attempts.append(pieces)            # raw pieces (robust) fallback

        last_err = None
        for bodies in attempts:
            try:
                node, elem, n_meshed = self._mesh_bodies(bodies, variants, ref_diag)
            except MeshError as exc:
                last_err = exc
                continue
            a_diag = float(np.linalg.norm(node.max(axis=0) - node.min(axis=0)))
            if ref_diag > 0 and a_diag < 0.5 * ref_diag:
                last_err = MeshError(
                    f"Assembly spanned only {a_diag / ref_diag:.0%} of the model.")
                continue
            return node, elem, n_meshed, len(bodies)
        raise last_err or MeshError("Assembly meshing failed.")

    @staticmethod
    def _count_tet_components(node, elem):
        """Number of node-connected groups of tetrahedra (union-find)."""
        n = node.shape[0]
        parent = np.arange(n)

        def find(x):
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        for tet in elem:
            r0 = find(int(tet[0]))
            for k in (1, 2, 3):
                rk = find(int(tet[k]))
                if rk != r0:
                    parent[rk] = r0
        used = np.unique(elem)
        roots = {find(int(u)) for u in used}
        return len(roots)

    @staticmethod
    def _split_components(surf):
        """Split a surface into its connected-component shells (PolyData list)."""
        surf = surf.triangulate()
        if surf.n_cells == 0:
            return []
        conn = surf.connectivity()
        rid = np.asarray(conn.cell_data["RegionId"], dtype=np.int64)
        comps = []
        for r in range(int(rid.max()) + 1):
            sub = conn.threshold([r - 0.5, r + 0.5], scalars="RegionId")
            sub = sub.extract_surface(algorithm="dataset_surface")
            sub = sub.triangulate().clean()
            if sub.n_faces:
                comps.append(sub)
        return comps

    def _is_simple_solid(self, surf) -> bool:
        """True when ``surf`` is a single, watertight, two-manifold shell."""
        if surf.extract_feature_edges(
                boundary_edges=True, feature_edges=False,
                manifold_edges=False, non_manifold_edges=False).n_cells:
            return False
        if surf.extract_feature_edges(
                boundary_edges=False, feature_edges=False,
                manifold_edges=False, non_manifold_edges=True).n_cells:
            return False
        if surf.n_cells == 0:
            return False
        rid = np.asarray(surf.connectivity().cell_data["RegionId"], dtype=np.int64)
        return rid.size > 0 and int(rid.max()) == 0

    @staticmethod
    def _is_closed_manifold(surf) -> bool:
        """True when ``surf`` has no open and no non-manifold edges.

        Unlike :meth:`_is_simple_solid` this allows *several* disjoint closed
        shells, which TetGen meshes happily -- it is the condition a remeshed
        surface must keep after smoothing/decimation.
        """
        if surf.n_cells == 0:
            return False
        if surf.extract_feature_edges(
                boundary_edges=True, feature_edges=False,
                manifold_edges=False, non_manifold_edges=False).n_cells:
            return False
        if surf.extract_feature_edges(
                boundary_edges=False, feature_edges=False,
                manifold_edges=False, non_manifold_edges=True).n_cells:
            return False
        return True

    @staticmethod
    def _count_components(surf) -> int:
        """Number of connected components in a surface."""
        if surf.n_cells == 0:
            return 0
        rid = np.asarray(surf.connectivity().cell_data["RegionId"], dtype=np.int64)
        return int(rid.max()) + 1 if rid.size else 0

    @staticmethod
    def _enclosed_mask(grid, shell) -> np.ndarray:
        """Boolean mask of ``grid`` points inside ``shell`` (robust ray-cast).

        Uses VTK's enclosed-points test, whose sign does not depend on the
        shell's (often inconsistent) facet normals -- unlike a signed-distance
        sign, which a single inward-facing shell would invert.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sel = grid.select_enclosed_points(shell, tolerance=0.0,
                                              check_surface=False)
        return np.asarray(sel["SelectedPoints"]).astype(bool)

    def _remesh_union(self, surf, n_div: int = 110,
                      max_cells: int = 5_000_000) -> "pv.PolyData":
        """Rebuild a watertight manifold as the *union* of every input shell.

        Voxel signed-distance reconstruction on a regular grid covering the
        model: each connected shell contributes a signed distance field whose
        *sign* comes from a robust ray-cast inside test (:meth:`_enclosed_mask`)
        and whose *magnitude* is the unsigned distance to that shell.  Taking the
        per-cell minimum unions the shells -- a cell is inside the result when it
        is inside *any* shell -- and contouring the field at zero yields a
        single, watertight, non-self-intersecting surface TetGen can always fill.

        The trade-off is geometric: sharp corners are slightly rounded and
        features closer together than the voxel size merge.  ``n_div`` sets the
        grid resolution along the bounding-box diagonal; ``max_cells`` caps the
        grid so very large models stay within memory/time budgets.
        """
        comps = self._split_components(surf)
        if not comps:
            raise MeshError("Surface has no triangles to remesh.")

        b = np.array(surf.bounds).reshape(3, 2)
        ext = b[:, 1] - b[:, 0]
        diag = float(np.linalg.norm(ext))
        if not np.isfinite(diag) or diag <= 0:
            raise MeshError("Degenerate (zero-size) surface; nothing to mesh.")

        h = diag / float(n_div)
        pad = 3.0 * h
        # Keep the grid within a sane size; coarsen if a thin, large model would
        # otherwise blow past the cell budget.
        while np.prod(np.ceil((ext + 2 * pad) / h) + 1) > max_cells:
            h *= 1.25
            pad = 3.0 * h
        lo = b[:, 0] - pad
        dims = np.maximum(np.ceil((ext + 2 * pad) / h).astype(int) + 1, 2)
        grid = pv.ImageData(dimensions=tuple(int(d) for d in dims),
                            spacing=(float(h), float(h), float(h)),
                            origin=tuple(float(x) for x in lo))

        inside = np.zeros(grid.n_points, dtype=bool)
        unsigned = np.full(grid.n_points, np.inf)
        for c in comps:
            inside |= self._enclosed_mask(grid, c)
            d = np.abs(np.asarray(
                grid.compute_implicit_distance(c)["implicit_distance"]))
            unsigned = np.minimum(unsigned, d)
        if not inside.any():
            raise MeshError(
                "Could not identify the solid interior of this surface (it may "
                "be an open, non-enclosing shell).")

        grid["sdf"] = np.where(inside, -unsigned, unsigned)
        out = grid.contour([0.0], scalars="sdf").triangulate().clean()
        if out.n_points == 0 or out.n_faces == 0:
            raise MeshError("Voxel remesh produced an empty surface.")

        # If the reconstruction shattered into far more pieces than the input
        # had, the model has features thinner than the voxel size (an open
        # lattice / porous solid, or a badly broken mesh).  A faithful solid
        # mesh is impossible at this resolution -- fail clearly rather than feed
        # thousands of fragments to TetGen (which would explode or crash).
        n_out = self._count_components(out)
        if n_out > max(20, 10 * len(comps)):
            raise MeshError(
                f"This surface reconstructs into {n_out} disconnected fragments, "
                "meaning it has features thinner than can be resolved (a porous "
                "or open-lattice solid, or a very broken mesh).  It cannot be "
                "meshed reliably -- please supply a cleaner, watertight STL.")

        # Marching cubes emits a dense, "staircased" triangulation full of
        # slivers.  A volume-preserving Taubin smoothing pass relaxes those into
        # well-shaped triangles (Laplacian would shrink the model); this both
        # removes near-degenerate facets and slashes the downstream tet count.
        try:
            sm = out.smooth_taubin(n_iter=20, pass_band=0.05).clean()
            if sm.n_faces and self._is_closed_manifold(sm):
                out = sm
        except Exception:                          # pragma: no cover - keep raw
            pass

        # The uniform marching-cubes triangulation is thinned later by the
        # curvature-adaptive pass in ``tetrahedralize`` (flat regions collapse,
        # curved ones are kept), so no fixed-ratio decimation is done here.
        return out

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
