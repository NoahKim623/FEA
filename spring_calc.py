"""
spring_calc.py
==============

Effective spring-constant (stiffness) extraction for a meshed part.

A linear-elastic structure behaves, between a fixed support and a loaded face,
exactly like a Hookean spring:

        F = k * x          =>      k = F / x         [N/m]

where ``F`` is the magnitude of the applied force resultant and ``x`` is the
resulting maximum displacement.  This module:

1. Sweeps the load from 0 to F_max in several increments (a "load case"
   sweep), solving the FE system at each step.
2. Records the force-displacement pairs (F_i, x_i).
3. Fits the slope of the F-vs-x line, which is the spring constant k.

Because the model is *linear*, displacement is exactly proportional to load, so
the same factorised stiffness matrix is reused for every increment (only the
right-hand side is rescaled).  The load steps therefore trace a straight line
through the origin whose slope is k -- but solving incrementally keeps the code
honest (and ready for a future non-linear extension) while costing almost
nothing thanks to the cached LU factorisation in
:meth:`fea_solver.FEASolver.solve_with_load`.
"""

from __future__ import annotations

import numpy as np


class SpringCalculator:
    """Run a multi-step load sweep and extract the spring constant k.

    Parameters
    ----------
    solver : fea_solver.FEASolver
        An *assembled* solver that already has its fixed nodes and full-load
        (F = F_max) load vector set.
    progress : optional callable(percent:int, msg:str) for UI feedback.
    """

    def __init__(self, solver, progress=None):
        self.solver = solver
        self._progress = progress
        self.forces = None          # (n_steps+1,) applied force magnitudes (N)
        self.displacements = None   # (n_steps+1,) max displacement per step (m)
        self.k = None               # fitted spring constant (N/m)
        self.k_direct = None        # simple F_max / x_max estimate (N/m)
        self.r_squared = None       # goodness of the linear fit

    # ------------------------------------------------------------------
    def run(self, n_steps: int = 8) -> dict:
        """Perform the load sweep and compute k.

        Parameters
        ----------
        n_steps : number of load increments from 0 to F_max (5-10 typical).

        Returns
        -------
        dict with keys: forces, displacements, k, k_direct, r_squared,
        F_max, x_max.
        """
        n_steps = int(max(2, n_steps))
        F_full = self.solver.F.copy()
        F_max = self.solver.applied_force_magnitude
        if F_max <= 0.0:
            raise ValueError("No load is applied; cannot compute a spring constant.")

        self._report(5, "Factorising stiffness for load sweep...")
        self.solver.prepare_factorization()

        # Load fractions 0, 1/n, 2/n, ... 1  (include the origin point).
        fractions = np.linspace(0.0, 1.0, n_steps + 1)
        forces = np.zeros(n_steps + 1)
        disps = np.zeros(n_steps + 1)

        for i, frac in enumerate(fractions):
            if frac == 0.0:
                forces[i] = 0.0
                disps[i] = 0.0
            else:
                u = self.solver.solve_with_load(F_full * frac)
                umax = np.linalg.norm(u.reshape(-1, 3), axis=1).max()
                forces[i] = frac * F_max
                disps[i] = umax
            pct = 10 + int(85 * (i + 1) / (n_steps + 1))
            self._report(pct, f"Load step {i}/{n_steps}: "
                              f"F = {forces[i]:.3g} N, x = {disps[i]:.3g} m")

        self.forces = forces
        self.displacements = disps

        # --- fit k as the slope of F vs x, constrained through the origin ---
        # For y = k*x with no intercept, the least-squares slope is
        #     k = sum(x*y) / sum(x*x).
        x = disps
        y = forces
        denom = np.dot(x, x)
        self.k = float(np.dot(x, y) / denom) if denom > 0 else float("inf")

        # Simple endpoint estimate, and an R^2 for the straight-line quality.
        x_max = disps[-1]
        self.k_direct = float(F_max / x_max) if x_max > 0 else float("inf")
        ss_res = float(np.sum((y - self.k * x) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        self.r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

        self._report(100, f"Spring constant k = {self.k:.4g} N/m")
        return {
            "forces": forces,
            "displacements": disps,
            "k": self.k,
            "k_direct": self.k_direct,
            "r_squared": self.r_squared,
            "F_max": F_max,
            "x_max": x_max,
        }

    # ------------------------------------------------------------------
    def _report(self, pct, msg):
        if self._progress is not None:
            self._progress(int(pct), msg)

    # ------------------------------------------------------------------
    @staticmethod
    def format_k(k: float) -> str:
        """Human-friendly formatting of a stiffness value with unit prefix."""
        if not np.isfinite(k):
            return "infinite (rigid / under-constrained)"
        for scale, unit in ((1e9, "GN/m"), (1e6, "MN/m"),
                            (1e3, "kN/m"), (1.0, "N/m")):
            if abs(k) >= scale:
                return f"{k / scale:.4g} {unit}"
        return f"{k:.4g} N/m"
