"""
postprocessor.py
================

Stress and displacement post-processing for a solved Tet4 model.

Given the nodal displacement vector ``u`` returned by
:class:`fea_solver.FEASolver`, this module recovers, for every element, the
constant strain and stress, then derives engineering scalars used for the
colour-mapped results plots:

* **Element strain**   eps = B u_e                       (Voigt, 6 components)
* **Element stress**   sig = D eps = D B u_e              (Voigt, 6 components)
* **von Mises stress** an equivalent scalar stress used for yield assessment
* **Principal stresses** sigma_1 >= sigma_2 >= sigma_3, the eigenvalues of the
  3x3 Cauchy stress tensor (the stress state in its principal axes)

Because a Tet4 is a constant-strain element, each of these quantities is a
single value per element.  For smooth Abaqus/Ansys-style contour plots the
per-element field can optionally be averaged to the nodes.

The von Mises stress is

    s_vm = sqrt( 0.5 [ (sxx-syy)^2 + (syy-szz)^2 + (szz-sxx)^2 ]
                 + 3 ( txy^2 + tyz^2 + tzx^2 ) )

which is the scalar that, compared against the material yield strength,
predicts the onset of plastic yielding under the von Mises (J2) criterion.
"""

from __future__ import annotations

import numpy as np

from fea_solver import (
    constitutive_matrix,
    element_dofs,
    element_volumes_and_gradients,
    strain_displacement_matrices,
)


class PostProcessor:
    """Recover strains, stresses and displacement scalars from a solution.

    Parameters
    ----------
    nodes : (n_nodes, 3) nodal coordinates.
    tets  : (n_elem, 4) element connectivity.
    u     : (3*n_nodes,) global displacement vector from the solver.
    E, nu : material constants (must match those used in the solve).
    """

    def __init__(self, nodes, tets, u, E, nu):
        self.nodes = np.ascontiguousarray(nodes, dtype=float)
        self.tets = np.ascontiguousarray(tets, dtype=np.int64)
        self.u = np.asarray(u, dtype=float).ravel()
        self.E = float(E)
        self.nu = float(nu)
        self.D = constitutive_matrix(E, nu)

        self.n_nodes = self.nodes.shape[0]
        self.n_elem = self.tets.shape[0]

        self._compute()

    # ------------------------------------------------------------------
    def _compute(self):
        """Compute all element and nodal result fields (called on init)."""
        # --- nodal displacement field -------------------------------------
        self.u_node = self.u.reshape(self.n_nodes, 3)         # (N, 3)
        self.disp_mag = np.linalg.norm(self.u_node, axis=1)   # (N,)

        # --- element strains and stresses ---------------------------------
        _, grad, _ = element_volumes_and_gradients(self.nodes, self.tets)
        B = strain_displacement_matrices(grad)                # (M, 6, 12)
        edof = element_dofs(self.tets)                        # (M, 12)
        u_e = self.u[edof]                                    # (M, 12)

        # eps_e = B_e u_e ;   sig_e = D eps_e
        self.strain = np.einsum("mij,mj->mi", B, u_e)         # (M, 6)
        self.stress = np.einsum("ij,mj->mi", self.D, self.strain)  # (M, 6)

        # --- derived scalar fields ----------------------------------------
        self.von_mises = self._von_mises(self.stress)         # (M,)
        self.principal = self._principal_stresses(self.stress)  # (M, 3) desc

    # ------------------------------------------------------------------
    @staticmethod
    def _von_mises(stress: np.ndarray) -> np.ndarray:
        """Von Mises equivalent stress from Voigt stress [sxx,syy,szz,txy,tyz,tzx]."""
        sxx, syy, szz, txy, tyz, tzx = stress.T
        return np.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
            + 3.0 * (txy ** 2 + tyz ** 2 + tzx ** 2)
        )

    @staticmethod
    def _principal_stresses(stress: np.ndarray) -> np.ndarray:
        """Principal stresses (eigenvalues of the Cauchy tensor), descending.

        The symmetric 3x3 stress tensor is assembled from the Voigt vector and
        diagonalised; the eigenvalues are the principal stresses and are
        returned sorted so column 0 is sigma_1 (max) and column 2 is sigma_3.
        """
        sxx, syy, szz, txy, tyz, tzx = stress.T
        n = stress.shape[0]
        T = np.empty((n, 3, 3))
        T[:, 0, 0] = sxx
        T[:, 1, 1] = syy
        T[:, 2, 2] = szz
        T[:, 0, 1] = T[:, 1, 0] = txy
        T[:, 1, 2] = T[:, 2, 1] = tyz
        T[:, 0, 2] = T[:, 2, 0] = tzx
        # eigvalsh returns ascending eigenvalues for symmetric matrices.
        eig = np.linalg.eigvalsh(T)
        return eig[:, ::-1]                                  # descending

    # ------------------------------------------------------------------
    def element_to_nodal(self, field: np.ndarray) -> np.ndarray:
        """Average a per-element field onto the nodes for smooth contouring.

        Each element contributes its (constant) value to its four nodes; the
        nodal value is the simple average of all incident elements.  This is
        the standard "nodal averaging" used by FEA post-processors to turn the
        piecewise-constant Tet4 stress into a continuous contour.
        """
        field = np.asarray(field, dtype=float)
        nodal = np.zeros(self.n_nodes)
        count = np.zeros(self.n_nodes)
        flat_nodes = self.tets.ravel()
        np.add.at(nodal, flat_nodes, np.repeat(field, 4))
        np.add.at(count, flat_nodes, 1.0)
        count[count == 0] = 1.0
        return nodal / count

    # ------------------------------------------------------------------
    # Convenience summary accessors used by the results panel.
    # ------------------------------------------------------------------
    @property
    def max_displacement(self) -> float:
        """Maximum nodal displacement magnitude (m)."""
        return float(self.disp_mag.max())

    @property
    def max_displacement_node(self) -> int:
        """Index of the node with the largest displacement magnitude."""
        return int(np.argmax(self.disp_mag))

    @property
    def max_von_mises(self) -> float:
        """Maximum element von Mises stress (Pa)."""
        return float(self.von_mises.max())

    @property
    def max_principal(self) -> float:
        """Largest (most tensile) principal stress over the model (Pa)."""
        return float(self.principal[:, 0].max())

    @property
    def min_principal(self) -> float:
        """Smallest (most compressive) principal stress over the model (Pa)."""
        return float(self.principal[:, 2].min())

    def summary(self) -> dict:
        """Return a dict of headline results for the UI results panel."""
        return {
            "max_displacement": self.max_displacement,
            "max_displacement_node": self.max_displacement_node,
            "max_von_mises": self.max_von_mises,
            "max_principal": self.max_principal,
            "min_principal": self.min_principal,
        }
