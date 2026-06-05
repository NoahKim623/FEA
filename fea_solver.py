"""
fea_solver.py
=============

Core linear-elastic finite element engine for 4-node tetrahedral (Tet4)
elements in 3D.

The mathematics implemented here
--------------------------------
A Tet4 element has 4 nodes and 3 translational degrees of freedom (DOF) per
node (u_x, u_y, u_z), giving 12 DOF per element.  Because the shape functions
of a linear tetrahedron are *linear* in (x, y, z), their spatial derivatives
are constant over the element.  Consequently the strain is constant inside each
element -- the 3D analogue of the "constant strain triangle" (CST).  This makes
the element stiffness matrix exact with a single (constant) integration point.

For a node i the shape function is

        N_i(x, y, z) = a_i + b_i*x + c_i*y + d_i*z

with N_i = 1 at node i and 0 at the other three nodes.  Collecting the four
shape functions and requiring N_i(node_j) = delta_ij leads to

        C * P^T = I        with   C = [[1, x1, y1, z1],
                                       [1, x2, y2, z2],
                                       [1, x3, y3, z3],
                                       [1, x4, y4, z4]]

so the coefficient matrix is P = inv(C)^T.  Rows 1..3 of inv(C) are therefore
exactly dN_i/dx, dN_i/dy and dN_i/dz -- a clean, robust way to obtain the
shape-function gradients numerically.  The signed element volume is det(C)/6.

The strain-displacement matrix B (6 x 12, Voigt order
[exx, eyy, ezz, gxy, gyz, gzx]) maps nodal displacements to element strain,
the isotropic constitutive matrix D (6 x 6) maps strain to stress, and the
element stiffness is the (constant-integrand) volume integral

        Ke = V * B^T * D * B          (12 x 12)

Element matrices are scattered into the global sparse stiffness matrix K, the
Dirichlet (fixed) boundary conditions are imposed by *direct elimination* of
the constrained DOF (a robust penalty option is also provided), and the linear
system  K u = F  is solved with a sparse direct solver.

All quantities are in SI units: lengths in metres, E in pascals (N/m^2),
forces in newtons, displacements in metres, stresses in pascals.
"""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, splu


class SingularStiffnessError(RuntimeError):
    """Raised when the (constrained) stiffness matrix is singular.

    In practice this almost always means the model is *under-constrained*:
    the applied displacement boundary conditions do not remove all six
    rigid-body modes (3 translations + 3 rotations), so the structure can
    move or spin without storing strain energy and K has no unique inverse.
    """


# ---------------------------------------------------------------------------
# Constitutive (material) matrix
# ---------------------------------------------------------------------------
def constitutive_matrix(E: float, nu: float) -> np.ndarray:
    """Return the 6x6 isotropic linear-elastic constitutive matrix D.

    Uses the Lame parameters

        lam = E*nu / ((1+nu)(1-2nu))      (first Lame parameter)
        mu  = E / (2(1+nu))               (shear modulus, second Lame param.)

    in the Voigt ordering [sxx, syy, szz, txy, tyz, tzx].  The shear rows use
    the *engineering* shear strain (gamma = 2*epsilon), which is why the shear
    diagonal is simply mu (not 2*mu).

    Raises
    ------
    ValueError
        If nu is outside the physically admissible open interval (-1, 0.5).
        nu -> 0.5 is the incompressible limit where lam -> infinity.
    """
    if not (-1.0 < nu < 0.5):
        raise ValueError(
            f"Poisson's ratio nu={nu} is non-physical; it must satisfy "
            "-1 < nu < 0.5 (nu = 0.5 is the incompressible limit)."
        )
    if E <= 0.0:
        raise ValueError(f"Young's modulus E={E} must be positive.")

    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))

    D = np.zeros((6, 6), dtype=float)
    D[:3, :3] = lam                      # fill the normal-stress block with lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu     # shear terms
    return D


# ---------------------------------------------------------------------------
# Element geometry: volumes and shape-function gradients (vectorised)
# ---------------------------------------------------------------------------
def element_volumes_and_gradients(nodes: np.ndarray, tets: np.ndarray):
    """Compute signed volumes and shape-function gradients for *all* elements.

    Parameters
    ----------
    nodes : (n_nodes, 3) float array of nodal coordinates.
    tets  : (n_elem, 4) int array of element connectivity (node indices).

    Returns
    -------
    vol   : (n_elem,) signed element volumes (det(C)/6).
    grad  : (n_elem, 4, 3) shape-function gradients; grad[e, i] = [dNi/dx,
            dNi/dy, dNi/dz] for node i of element e.
    valid : (n_elem,) bool mask, False for degenerate (near-zero-volume)
            elements whose gradient matrix could not be inverted reliably.
    """
    coords = nodes[tets]                              # (M, 4, 3)
    n_elem = tets.shape[0]

    # Build C for every element: column 0 is ones, columns 1..3 are coords.
    C = np.empty((n_elem, 4, 4), dtype=float)
    C[:, :, 0] = 1.0
    C[:, :, 1:] = coords

    det = np.linalg.det(C)
    vol = det / 6.0

    # An element is usable only if its volume is not vanishingly small relative
    # to the mesh scale (a degenerate "sliver" would give an ill-conditioned
    # gradient and pollute the global matrix).
    scale = max(np.abs(vol).max(), 1e-300)
    valid = np.abs(vol) > 1e-12 * scale

    invC = np.zeros((n_elem, 4, 4), dtype=float)
    if np.any(valid):
        invC[valid] = np.linalg.inv(C[valid])

    # Rows 1..3 of inv(C) are dN/dx, dN/dy, dN/dz (direction-major).  Transpose
    # the last two axes so the result is node-major: grad[e, node, direction].
    grad = invC[:, 1:4, :].transpose(0, 2, 1).copy()
    return vol, grad, valid


def strain_displacement_matrices(grad: np.ndarray) -> np.ndarray:
    """Assemble the 6x12 strain-displacement matrix B for every element.

    The Voigt strain ordering is [exx, eyy, ezz, gxy, gyz, gzx].  For node i
    (with gradients b=dNi/dx, c=dNi/dy, d=dNi/dz) the 6x3 sub-block is

            [ b  0  0 ]
            [ 0  c  0 ]
            [ 0  0  d ]
            [ c  b  0 ]      (gamma_xy = du/dy + dv/dx)
            [ 0  d  c ]      (gamma_yz = dv/dz + dw/dy)
            [ d  0  b ]      (gamma_zx = dw/dx + du/dz)

    Parameters
    ----------
    grad : (n_elem, 4, 3) shape-function gradients from
           :func:`element_volumes_and_gradients`.

    Returns
    -------
    B : (n_elem, 6, 12) array; the 12 columns are ordered
        [u0, v0, w0, u1, v1, w1, u2, v2, w2, u3, v3, w3].
    """
    n_elem = grad.shape[0]
    B = np.zeros((n_elem, 6, 12), dtype=float)
    bx, by, bz = grad[:, :, 0], grad[:, :, 1], grad[:, :, 2]   # each (M, 4)
    for i in range(4):
        col = 3 * i
        B[:, 0, col + 0] = bx[:, i]
        B[:, 1, col + 1] = by[:, i]
        B[:, 2, col + 2] = bz[:, i]
        B[:, 3, col + 0] = by[:, i]
        B[:, 3, col + 1] = bx[:, i]
        B[:, 4, col + 1] = bz[:, i]
        B[:, 4, col + 2] = by[:, i]
        B[:, 5, col + 0] = bz[:, i]
        B[:, 5, col + 2] = bx[:, i]
    return B


def element_dofs(tets: np.ndarray) -> np.ndarray:
    """Map element node connectivity to global DOF indices.

    Global DOF for node n, component c (0=x,1=y,2=z) is ``3*n + c``.

    Returns
    -------
    edof : (n_elem, 12) int array of global DOF indices, ordered
           [3n0, 3n0+1, 3n0+2, 3n1, ...] to match the columns of B.
    """
    return (3 * tets[:, :, None] + np.arange(3)[None, None, :]).reshape(tets.shape[0], 12)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------
class FEASolver:
    """Assemble and solve the linear elasticity problem K u = F for a Tet4 mesh.

    Typical usage::

        solver = FEASolver(nodes, tets, E=2.1e11, nu=0.3, rho=7850)
        solver.assemble()
        solver.set_fixed_nodes(fixed_node_ids)
        solver.set_loads({node_id: [fx, fy, fz], ...})
        u = solver.solve()
    """

    def __init__(self, nodes, tets, E, nu, rho=0.0, progress=None):
        self.nodes = np.ascontiguousarray(nodes, dtype=float)
        self.tets = np.ascontiguousarray(tets, dtype=np.int64)
        self.E = float(E)
        self.nu = float(nu)
        self.rho = float(rho)
        self._progress = progress      # optional callback(percent:int, msg:str)

        self.n_nodes = self.nodes.shape[0]
        self.n_elem = self.tets.shape[0]
        self.ndof = 3 * self.n_nodes

        self.D = constitutive_matrix(E, nu)

        # Filled in by assemble()/set_*()/solve().
        self.K = None                  # global stiffness (CSR), unconstrained
        self.F = np.zeros(self.ndof)   # global load vector
        self.fixed_dofs = np.array([], dtype=np.int64)
        self.free_dofs = np.arange(self.ndof)
        self.u = None                  # solution displacement vector
        self.residual = None           # relative solve residual
        self.element_volumes = None
        self.applied_force_magnitude = 0.0
        self.resultant_force = np.zeros(3)
        self._lu = None                # cached LU factorisation of K_ff
        self._penalty = None

    # -- progress helper -----------------------------------------------------
    def _report(self, pct, msg):
        if self._progress is not None:
            self._progress(int(pct), msg)

    # -- assembly ------------------------------------------------------------
    def assemble(self):
        """Assemble the global sparse stiffness matrix K.

        Builds every element matrix ``Ke = V * B^T D B`` in a vectorised batch
        and scatters the 144 entries of each into a COO triplet list, which is
        summed into a CSR matrix (duplicate (row, col) pairs are accumulated,
        which is exactly the finite-element assembly operation).
        """
        self._report(2, "Computing element geometry...")
        vol, grad, valid = element_volumes_and_gradients(self.nodes, self.tets)
        self.element_volumes = vol
        n_bad = int(np.count_nonzero(~valid))
        if n_bad:
            warnings.warn(
                f"{n_bad} degenerate (near-zero-volume) element(s) were found "
                "and excluded from assembly; check mesh quality.",
                RuntimeWarning,
            )

        self._report(10, "Building strain-displacement matrices...")
        B = strain_displacement_matrices(grad)               # (M, 6, 12)

        self._report(25, "Forming element stiffness matrices...")
        # DB = D @ B   then   Ke = B^T @ DB  scaled by the (positive) volume.
        DB = np.einsum("ij,mjk->mik", self.D, B)             # (M, 6, 12)
        Ke = np.einsum("mji,mjk->mik", B, DB)                # (M, 12, 12)
        Ke *= np.abs(vol)[:, None, None]
        Ke[~valid] = 0.0                                     # drop bad elements

        self._report(55, "Scattering into global sparse matrix...")
        edof = element_dofs(self.tets)                       # (M, 12)
        rows = np.repeat(edof, 12, axis=1).ravel()           # a-major
        cols = np.tile(edof, (1, 12)).ravel()                # b fastest
        data = Ke.ravel()
        K = sp.coo_matrix((data, (rows, cols)),
                          shape=(self.ndof, self.ndof))
        K.sum_duplicates()                                   # the assembly sum
        self.K = K.tocsr()

        # Stash for stress recovery / re-use.
        self._B = B
        self._edof = edof
        self._valid = valid
        self._lu = None                                      # invalidate cache
        self._report(70, "Assembly complete.")
        return self.K

    # -- boundary conditions -------------------------------------------------
    def set_fixed_nodes(self, node_ids):
        """Fully fix (u = 0 in x, y, z) the given nodes.

        Implemented later via direct elimination of the corresponding DOF.
        """
        node_ids = np.unique(np.asarray(node_ids, dtype=np.int64))
        fixed = (3 * node_ids[:, None] + np.arange(3)[None, :]).ravel()
        self.fixed_dofs = np.unique(fixed)
        mask = np.ones(self.ndof, dtype=bool)
        mask[self.fixed_dofs] = False
        self.free_dofs = np.where(mask)[0]
        self._lu = None
        return self.fixed_dofs

    def set_fixed_dofs(self, dof_ids):
        """Fix an explicit list of global DOF indices (per-DOF constraint).

        Lower-level than :meth:`set_fixed_nodes`; useful for roller/symmetry
        boundary conditions where only some components of a node are held.
        """
        self.fixed_dofs = np.unique(np.asarray(dof_ids, dtype=np.int64))
        mask = np.ones(self.ndof, dtype=bool)
        mask[self.fixed_dofs] = False
        self.free_dofs = np.where(mask)[0]
        self._lu = None
        return self.fixed_dofs

    def set_loads(self, loads):
        """Assemble the global load vector from per-node force vectors.

        Parameters
        ----------
        loads : dict {node_id: (fx, fy, fz)} of consistent nodal forces (N).
        """
        F = np.zeros(self.ndof)
        resultant = np.zeros(3)
        for nid, fvec in loads.items():
            fvec = np.asarray(fvec, dtype=float)
            F[3 * nid:3 * nid + 3] += fvec
            resultant += fvec
        self.F = F
        self.resultant_force = resultant
        # "Applied force" used for the spring constant: magnitude of the
        # resultant of all external nodal loads.
        self.applied_force_magnitude = float(np.linalg.norm(resultant))
        return self.F

    def set_load_vector(self, F):
        """Set the global load vector F directly (length 3*n_nodes)."""
        F = np.asarray(F, dtype=float).ravel()
        if F.shape[0] != self.ndof:
            raise ValueError("Load vector length must equal 3*n_nodes.")
        self.F = F
        r = F.reshape(-1, 3).sum(axis=0)
        self.resultant_force = r
        self.applied_force_magnitude = float(np.linalg.norm(r))
        return self.F

    # -- solve ---------------------------------------------------------------
    def solve(self, method="elimination", penalty_scale=1e8):
        """Solve K u = F and return the global displacement vector u.

        Parameters
        ----------
        method : {"elimination", "penalty"}
            "elimination" removes the constrained DOF from the system and
            solves the reduced problem K_ff u_f = F_f (robust, exact).
            "penalty" adds a large stiffness to each fixed DOF's diagonal.
        penalty_scale : multiplier on max|diag(K)| for the penalty method.

        Raises
        ------
        SingularStiffnessError
            If the reduced/penalised system is singular (under-constrained).
        """
        if self.K is None:
            raise RuntimeError("Call assemble() before solve().")
        if self.fixed_dofs.size == 0:
            raise SingularStiffnessError(
                "No fixed boundary conditions are defined.  A 3D model needs "
                "constraints that remove all six rigid-body modes; otherwise "
                "the stiffness matrix is singular."
            )

        self._report(75, "Applying boundary conditions and solving...")
        if method == "penalty":
            u = self._solve_penalty(penalty_scale)
        else:
            u = self._solve_elimination()

        self.u = u

        # Residual check -- a reliable singularity / accuracy diagnostic.
        # Equilibrium K u = F holds only on the FREE DOF; on the fixed DOF the
        # left-hand side equals the (non-zero) support reactions, so those rows
        # must be excluded or the residual would spuriously look like ~1.
        free = self.free_dofs
        r_free = self.K[free] @ u - self.F[free]
        f_norm = np.linalg.norm(self.F[free]) + 1e-30
        self.residual = float(np.linalg.norm(r_free) / f_norm)
        if not np.all(np.isfinite(u)) or self.residual > 1e-6:
            raise SingularStiffnessError(
                "The stiffness matrix appears singular or the solution did not "
                f"converge (relative residual = {self.residual:.2e}).  This "
                "usually means the model is under-constrained -- add more fixed "
                "supports so the part cannot translate or rotate freely."
            )
        self._report(95, "Solve complete.")
        return u

    def _solve_elimination(self):
        """Direct-elimination solve of the reduced free-free system."""
        free = self.free_dofs
        # K_ff is the stiffness restricted to the free DOF.  With u on the
        # fixed DOF equal to zero, the reduced system is simply K_ff u_f = F_f.
        Kff = self.K[free][:, free].tocsc()
        Ff = self.F[free]
        u = np.zeros(self.ndof)
        try:
            self._lu = splu(Kff)               # cache for load-stepping reuse
            uf = self._lu.solve(Ff)
        except (RuntimeError, ValueError) as exc:
            raise SingularStiffnessError(
                "Sparse factorisation failed -- the constrained stiffness "
                "matrix is singular (under-constrained model)."
            ) from exc
        u[free] = uf
        return u

    def _solve_penalty(self, penalty_scale):
        """Penalty-method solve: stiffen fixed DOF instead of removing them."""
        K = self.K.tolil(copy=True)
        big = penalty_scale * abs(self.K.diagonal()).max()
        self._penalty = big
        F = self.F.copy()
        for d in self.fixed_dofs:
            K[d, d] += big          # huge diagonal -> u[d] ~ F[d]/big ~ 0
            F[d] = 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            u = spsolve(K.tocsc(), F)
        return u

    # -- load stepping support (used by spring_calc) -------------------------
    def prepare_factorization(self):
        """Factorise the reduced stiffness once for fast repeated solves.

        Because the structure is *linear*, the same factorisation can be reused
        for every load increment -- only the right-hand side changes.  This is
        what makes the multi-step spring-constant sweep cheap.
        """
        if self.K is None:
            raise RuntimeError("Call assemble() first.")
        if self.fixed_dofs.size == 0:
            raise SingularStiffnessError("No fixed DOF; system is singular.")
        Kff = self.K[self.free_dofs][:, self.free_dofs].tocsc()
        try:
            self._lu = splu(Kff)
        except (RuntimeError, ValueError) as exc:
            raise SingularStiffnessError(
                "Cannot factorise the constrained stiffness matrix "
                "(under-constrained model)."
            ) from exc
        return self._lu

    def solve_with_load(self, F):
        """Solve for a given global load vector reusing the cached LU factor."""
        if self._lu is None:
            self.prepare_factorization()
        u = np.zeros(self.ndof)
        u[self.free_dofs] = self._lu.solve(np.asarray(F)[self.free_dofs])
        return u
