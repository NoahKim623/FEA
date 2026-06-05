"""
tetgen_worker.py
================

A *deliberately tiny* module that runs TetGen in a separate process.

TetGen is a C++ library; on pathological input (self-intersecting,
non-manifold, or zero-thickness surfaces such as the auxetic lattice in the
sample set) it can abort the whole interpreter with a hard crash that no
Python ``try/except`` can catch.  By running it in a child process spawned by
:mod:`multiprocessing`, such a crash only kills the child -- the parent GUI
detects the non-zero exit code and reports a clean error instead of dying.

This module intentionally imports **only numpy and tetgen** (no PyVista / Qt)
so the spawned child starts quickly and carries no GUI state.  The volume grid
is rebuilt from the returned node/element arrays back in the parent process.
"""

from __future__ import annotations

import numpy as np


def run(points, tris, kwargs, queue):
    """Tetrahedralise ``(points, tris)`` and push the result onto ``queue``.

    Parameters
    ----------
    points : (n, 3) float array of surface vertex coordinates (metres).
    tris   : (m, 3) int array of triangle connectivity into ``points``.
    kwargs : dict of keyword arguments forwarded to ``TetGen.tetrahedralize``.
    queue  : a ``multiprocessing.Queue`` used to return the outcome as one of
             ``("ok", node, elem)`` or ``("err", message)``.

    The function never raises across the process boundary; any exception is
    converted to an ``("err", ...)`` message.  A hard C-level crash leaves the
    queue empty, which the parent interprets from the child's exit code.
    """
    try:
        import tetgen

        pts = np.ascontiguousarray(points, dtype=np.float64)
        tri = np.ascontiguousarray(tris, dtype=np.int32)
        tg = tetgen.TetGen(pts, tri)
        tg.tetrahedralize(**kwargs)

        node = np.ascontiguousarray(tg.node, dtype=np.float64)
        elem = np.ascontiguousarray(tg.elem, dtype=np.int64)
        if node.size == 0 or elem.size == 0:
            queue.put(("err", "TetGen produced an empty volume mesh."))
            return
        queue.put(("ok", node, elem))
    except BaseException as exc:                      # noqa: BLE001 - report all
        queue.put(("err", f"{type(exc).__name__}: {exc}"))


def run_many(bodies, variants, queue):
    """Mesh a *list* of independent closed shells in a single child process.

    This is the fast path for the split-and-weld repair: an assembly of many
    small manifold pieces (auxetic prisms, lattice cells, separate tiles) is
    meshed body-by-body without paying the process-spawn cost once per body.

    Parameters
    ----------
    bodies : list of ``(points, tris)`` tuples, one per closed shell.
    variants : list of TetGen kwargs dicts, tried in order for each body until
        one succeeds (the same quality->permissive ladder used elsewhere).
    queue : results are *streamed* so a hard C-level crash on a pathological
        body only costs that one body -- everything meshed before it is already
        in the parent's hands:

        * ``("item", i, node, elem)`` - body ``i`` meshed successfully,
        * ``("itemerr", i, message)`` - body ``i`` could not be meshed,
        * ``("done", n)``            - all ``n`` bodies attempted (clean finish).

    The parent reads until it sees ``done`` or the child exits; any body with
    neither an ``item`` nor an ``itemerr`` (lost to a crash) is re-meshed in
    full isolation by the caller.
    """
    try:
        import tetgen
    except BaseException as exc:                      # noqa: BLE001
        queue.put(("fatal", f"{type(exc).__name__}: {exc}"))
        return

    for i, (points, tris) in enumerate(bodies):
        pts = np.ascontiguousarray(points, dtype=np.float64)
        tri = np.ascontiguousarray(tris, dtype=np.int32)
        last = "no variant produced a mesh"
        meshed = False
        for kwargs in variants:
            try:
                tg = tetgen.TetGen(pts.copy(), tri.copy())
                tg.tetrahedralize(**kwargs)
                node = np.ascontiguousarray(tg.node, dtype=np.float64)
                elem = np.ascontiguousarray(tg.elem, dtype=np.int64)
                if node.size and elem.size:
                    queue.put(("item", i, node, elem))
                    meshed = True
                    break
                last = "empty mesh"
            except BaseException as exc:              # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
        if not meshed:
            queue.put(("itemerr", i, last))
    queue.put(("done", len(bodies)))
