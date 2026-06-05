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
