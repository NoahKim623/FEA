# PyFEA — Desktop Finite Element Analysis

A self-contained desktop application for **linear-static structural FEA** on
solid geometry, in the spirit of SimScale / Abaqus CAE but small enough to read
in an afternoon. Import an STL, auto-mesh it into tetrahedra, apply a fixed
support and a load, solve `[K]{u} = {F}`, and visualize the deformed shape,
displacement and von Mises stress — plus extract an effective **spring
constant** from a multi-step load sweep.

---

## Features

- **STL import** (`meshio`) with automatic vertex merging and a **watertight**
  check; optional hole-filling repair for leaky surfaces.
- **Automatic tetrahedral meshing** of the surface into a solid Tet4 volume
  mesh (`tetgen`), with coarse/medium/fine refinement and **mesh-quality**
  reporting (sliver / inverted-element warnings).
- **Material model**: isotropic linear elasticity from Young's modulus `E`,
  Poisson's ratio `ν`, density `ρ`, with presets (steel, aluminium, titanium,
  copper).
- **Boundary conditions** through the UI: pick a **fixed face** (zero
  displacement) and a **loaded face** with either a total **force** (magnitude +
  direction) or a **pressure** (along the face normal). Loads are converted to
  energetically **consistent nodal forces**.
- **Solver**: vectorized global stiffness assembly into a SciPy sparse matrix,
  Dirichlet BCs by **direct elimination** (penalty method also implemented),
  and a sparse direct solve (`scipy.sparse.linalg.splu`).
- **Post-processing**: per-element strain, stress, **von Mises** and
  **principal stresses**, plus nodal displacement magnitude.
- **Visualization** (PyVista): colour-mapped displacement / von Mises overlays
  on the **deformed** mesh, toggle original vs. deformed, an undeformed
  reference outline, element edges, and a **deformation-scale slider**.
- **Spring constant**: runs 2–20 load increments from 0 → F_max, plots the
  **force–displacement curve** (matplotlib) and annotates the slope
  `k = F / x` in N/m.
- **Robust error handling**: non-watertight STL gate, poor-mesh-quality
  warnings, and a clear *"under-constrained / singular stiffness"* message
  instead of a silent garbage result.
- **Responsive UI**: assembly/solve/sweep run in a background thread with a
  live progress bar.

---

## Install

```powershell
# (optional) create a virtual environment first
python -m pip install -r requirements.txt
```

All dependencies ship as binary wheels (including `tetgen`'s `abi3` wheel and
`vtk`), so no C++ compiler is required. Tested on **Python 3.14, Windows 11**.

---

## Run

```powershell
python ui_main.py
```

### Workflow

1. **Import STL** — choose the file and its **units** (m / mm / cm / in). The
   surface is checked for watertightness and then auto-meshed into tetrahedra.
2. **Material** — pick a preset or type in `E`, `ν`, `ρ`.
3. **Boundary conditions**
   - *Fixed face*: choose one of the six bounding-box faces (±X/±Y/±Z) and
     click **Set Fixed Face**.
   - *Load face*: choose a face, a type (Force N / Pressure Pa), a magnitude and
     (for force) a direction vector, then **Set Load Face**.
   - The *selection tolerance* (% of model size) controls how thick the picked
     plane band is. Fixed nodes show **red**, loaded nodes **blue**, with a
     magenta load arrow.
4. **Solve** `[K]{u}={F}` for displacement + stress, **or Compute Spring
   Constant** to additionally run the load sweep and plot `F` vs `x`.
5. **Visualize** — switch the field (displacement / von Mises), toggle the
   deformed shape, and drag the deformation-scale slider.

### Try the included sample

```powershell
python generate_sample_stl.py     # writes sample_beam.stl (0.1 x 0.02 x 0.02 m)
python ui_main.py                 # Import sample_beam.stl (units = m)
```
Fix **−X**, load **+X** with `1e5` N along `(1,0,0)`, choose Steel, and Compute
Spring Constant → `k ≈ 8.4 × 10⁸ N/m` (matches the analytical bar stiffness
`A·E/L`).

---

## The FEM, briefly

Each **linear tetrahedron (Tet4)** has 4 nodes × 3 DOF = 12 DOF. Because the
shape functions are linear, strain is *constant* in each element (the 3D
constant-strain tetrahedron). For node `i`,
`N_i = a_i + b_i x + c_i y + d_i z`; requiring `N_i = δ_ij` at the nodes gives
the gradients directly from the inverse of the coordinate matrix `C`, and the
signed volume is `det(C)/6`.

- **Strain–displacement** `B` (6×12, Voigt `[εxx εyy εzz γxy γyz γzx]`) maps
  nodal displacements to strain.
- **Constitutive** `D` (6×6) from Lamé parameters `λ, μ` maps strain to stress.
- **Element stiffness** `Ke = V · Bᵀ D B` (exact, single integration point).
- **Assembly** scatters every `Ke` into a sparse global `K`; **direct
  elimination** removes the fixed DOF; `K_ff u_f = F_f` is solved sparsely.
- **Stress recovery**: `σ_e = D B u_e`; von Mises and principal stresses follow.

A linear structure obeys `F = k·x`, so the spring sweep traces a straight line
through the origin whose slope is the stiffness `k` (the LU factorisation is
reused across load steps, so the sweep is nearly free).

---

## Modules

| File | Responsibility |
|------|----------------|
| `mesh_handler.py` | STL import, watertight check/repair, TetGen meshing, mesh quality, face selection, consistent nodal loads |
| `fea_solver.py`   | Tet4 element math, vectorized sparse assembly, BC application, sparse solve, singular-matrix detection |
| `postprocessor.py`| Strain/stress recovery, von Mises, principal stresses, nodal averaging |
| `spring_calc.py`  | Load-step sweep, force–displacement data, least-squares `k` |
| `ui_main.py`      | PyQt5 window, PyVista viewport, matplotlib plot, threaded solve worker |
| `generate_sample_stl.py` | Writes a sample beam STL for testing |
| `test_fea.py`     | Headless verification (run `python test_fea.py`) |

---

## Verification

`python test_fea.py` runs without a GUI and checks:

- **Symmetry** of the global stiffness matrix.
- **Exactly six rigid-body modes** (no spurious zero-energy modes).
- A **constant-stress patch test** — prescribing an exact uniform-strain field
  reproduces it in the interior and recovers the exact uniaxial stress to
  **machine precision** (≈1e-15).
- The **full STL→mesh→solve→spring pipeline**, with the computed spring
  constant matching the analytical `k = A·E/L` to within a few percent (the
  small excess is the physical stiffening from the fully-clamped end), a
  force–displacement line through the origin with `R² = 1`, and a
  `SingularStiffnessError` raised for an under-constrained model.

---

## Units & assumptions

- **SI throughout**: metres, pascals, newtons → `k` in N/m. Set the STL units on
  import so coordinates are converted to metres.
- **Linear static, small strain**: results scale linearly with load; the
  deformation-scale slider only *exaggerates* the (typically tiny) real
  displacements for visualization.
- Face selection uses the six **bounding-box planes**, which is robust for the
  prismatic/bracket-type parts typical of stiffness studies. (Loads are still
  distributed over the real surface triangles on that plane.)

## Troubleshooting

- *"Surface not watertight"* — accept the repair prompt, or clean the STL in
  your CAD/mesh tool first. TetGen needs a closed manifold.
- *"Under-constrained / singular stiffness"* — the fixed face doesn't remove all
  rigid-body motion; fix a face that fully restrains the part.
- *Poor mesh quality warning* — increase the refinement level.
- *Blank viewport / OpenGL error* — update your GPU drivers; PyVista needs a
  working OpenGL context (this is unavailable in headless/SSH sessions).
