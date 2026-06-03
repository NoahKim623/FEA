"""
ui_main.py
==========

PyQt5 desktop front-end for the tetrahedral FEA engine, laid out in the style
of SimScale / Abaqus CAE:

    +-----------------------------------------------------------------------+
    |  menu bar                                                             |
    +-------------------+-------------------------------+-------------------+
    | LEFT panel        |        CENTER viewport        |  RIGHT panel      |
    |  1 Geometry/Mesh  |   (PyVista 3D render window)   |  Results summary  |
    |  2 Material       |   colour-mapped deformed mesh  |  Visualization    |
    |  3 Boundary cond. |                                |  F-x spring plot  |
    |  4 Solve          |                                |                   |
    +-------------------+-------------------------------+-------------------+
    |  status bar:  message .......................   [progress bar]        |
    +-----------------------------------------------------------------------+

The heavy numerical work (assembly / solve / load sweep) runs in a background
QThread so the GUI stays responsive and can show solver progress; all PyVista
rendering happens on the main thread in the worker's completion slot.

Run:  python ui_main.py
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor
import pyvista as pv

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from mesh_handler import MeshHandler, MeshError, FACE_OPTIONS, UNIT_SCALE
from fea_solver import FEASolver, SingularStiffnessError
from postprocessor import PostProcessor
from spring_calc import SpringCalculator

# Material presets: name -> (E [Pa], nu, rho [kg/m^3]).
MATERIALS = {
    "Steel":     (2.10e11, 0.30, 7850.0),
    "Aluminium": (6.90e10, 0.33, 2700.0),
    "Titanium":  (1.16e11, 0.32, 4500.0),
    "Copper":    (1.10e11, 0.34, 8960.0),
    "Custom":    (None, None, None),
}


# ===========================================================================
# Background worker: assemble -> constrain -> solve -> post-process (-> spring)
# ===========================================================================
class SolveWorker(QtCore.QThread):
    """Runs the FE pipeline off the UI thread and reports progress."""

    progress = QtCore.pyqtSignal(int, str)
    done = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, nodes, tets, E, nu, rho, fixed_nodes, loads,
                 do_spring=False, n_steps=8):
        super().__init__()
        self.nodes, self.tets = nodes, tets
        self.E, self.nu, self.rho = E, nu, rho
        self.fixed_nodes, self.loads = fixed_nodes, loads
        self.do_spring, self.n_steps = do_spring, n_steps

    def run(self):
        try:
            cb = lambda p, m: self.progress.emit(int(p), m)
            solver = FEASolver(self.nodes, self.tets, self.E, self.nu,
                               self.rho, progress=cb)
            solver.assemble()
            solver.set_fixed_nodes(self.fixed_nodes)
            solver.set_loads(self.loads)
            u = solver.solve()
            cb(96, "Recovering stresses...")
            post = PostProcessor(self.nodes, self.tets, u, self.E, self.nu)

            result = {"u": u, "post": post, "solver": solver, "spring": None,
                      "residual": solver.residual,
                      "applied_force": solver.applied_force_magnitude}

            if self.do_spring:
                cb(0, "Running load steps for spring constant...")
                sc = SpringCalculator(solver, progress=cb)
                result["spring"] = sc.run(self.n_steps)

            cb(100, "Done.")
            self.done.emit(result)
        except SingularStiffnessError as exc:
            self.failed.emit(str(exc))
        except Exception:                       # pragma: no cover
            self.failed.emit(traceback.format_exc())


# ===========================================================================
# Main window
# ===========================================================================
class FEAMainWindow(QtWidgets.QMainWindow):
    """The application's main window and controller."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyFEA - Tetrahedral Finite Element Analysis")
        self.resize(1480, 880)

        # --- application state ------------------------------------------
        self.mesh = MeshHandler()
        self.fixed_nodes = None        # accumulated union of all fixed-face nodes
        self.fixed_faces = []          # list of {"tris","normal","node_ids"} per face
        self.load_nodes = None         # nodes the load is actually applied to
        self.load_dict = None
        self.load_face_tris = None     # full picked load-face triangles
        self.load_tris = None          # loaded region (full face or targeted spot)
        self.load_normal = None        # outward unit normal of the load face
        self.load_centroid = None      # centroid of the full load face
        self.load_target = None        # targeted spot on the face (or None)
        self.load_dir = np.array([1.0, 0.0, 0.0])
        self.results = None            # dict from SolveWorker
        self.has_results = False
        self.spring_current = None     # latest spring run shown on the chart
        self.overlay_images = []       # imported chart PNGs overlaid for compare
        self.auto_scale = 1.0
        self._worker = None
        self._pick_mode = None         # None | "fixed" | "load" | "target"
        self._saved_camera = None      # camera to restore after target-force mode

        self._build_ui()
        self._set_enabled_stages()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        self._build_menu()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_viewport())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([360, 780, 340])
        self.setCentralWidget(splitter)

        # --- status bar with progress ----------------------------------
        self.status_label = QtWidgets.QLabel("Ready. Import an STL file to begin.")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

    def _build_menu(self):
        bar = self.menuBar()
        filemenu = bar.addMenu("&File")
        act_import = filemenu.addAction("&Import STL...")
        act_import.setShortcut("Ctrl+O")
        act_import.triggered.connect(self.on_import)
        filemenu.addSeparator()
        act_quit = filemenu.addAction("E&xit")
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)

        helpmenu = bar.addMenu("&Help")
        helpmenu.addAction("&About / Usage", self.on_about)

    # -- left panel -----------------------------------------------------
    def _build_left_panel(self):
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setSpacing(8)

        # 1. Geometry / mesh -------------------------------------------
        g = QtWidgets.QGroupBox("1.  Geometry / Mesh")
        form = QtWidgets.QFormLayout(g)
        self.btn_import = QtWidgets.QPushButton("Import STL...")
        self.btn_import.clicked.connect(self.on_import)
        self.lbl_file = QtWidgets.QLabel("(no file loaded)")
        self.lbl_file.setWordWrap(True)
        self.combo_units = QtWidgets.QComboBox()
        self.combo_units.addItems(list(UNIT_SCALE.keys()))
        self.combo_refine = QtWidgets.QComboBox()
        self.combo_refine.addItems(["coarse", "medium", "fine"])
        self.combo_refine.setCurrentText("medium")
        self.btn_mesh = QtWidgets.QPushButton("Generate Mesh")
        self.btn_mesh.clicked.connect(lambda: self._do_mesh(reset_camera=True))
        self.lbl_mesh = QtWidgets.QLabel("mesh: -")
        self.lbl_mesh.setWordWrap(True)
        form.addRow(self.btn_import)
        form.addRow("File:", self.lbl_file)
        form.addRow("STL units:", self.combo_units)
        form.addRow("Refinement:", self.combo_refine)
        form.addRow(self.btn_mesh)
        form.addRow(self.lbl_mesh)
        v.addWidget(g)

        # 2. Material ---------------------------------------------------
        g2 = QtWidgets.QGroupBox("2.  Material")
        form2 = QtWidgets.QFormLayout(g2)
        self.combo_mat = QtWidgets.QComboBox()
        self.combo_mat.addItems(list(MATERIALS.keys()))
        self.combo_mat.currentTextChanged.connect(self._apply_material_preset)
        self.edit_E = QtWidgets.QLineEdit("2.1e11")
        self.edit_nu = QtWidgets.QLineEdit("0.3")
        self.edit_rho = QtWidgets.QLineEdit("7850")
        form2.addRow("Preset:", self.combo_mat)
        form2.addRow("E  [Pa]:", self.edit_E)
        form2.addRow("nu  [-]:", self.edit_nu)
        form2.addRow("rho [kg/m^3]:", self.edit_rho)
        v.addWidget(g2)

        # 3. Boundary conditions ---------------------------------------
        g3 = QtWidgets.QGroupBox("3.  Boundary Conditions")
        form3 = QtWidgets.QFormLayout(g3)

        # -- Fixed supports (green; click to add, click again to remove) --
        self.btn_pick_fixed = QtWidgets.QPushButton("Pick Fixed Faces (click model)")
        self.btn_pick_fixed.setCheckable(True)
        self.btn_pick_fixed.setToolTip(
            "Click a face to fix it (turns green); click a green face again "
            "to unselect it.  Multiple fixed faces are allowed.")
        self.btn_pick_fixed.clicked.connect(
            lambda chk: self._toggle_pick("fixed", chk))
        # Indicator only: shows which bounding-box face the click matched.
        self.combo_fixed_face = QtWidgets.QComboBox()
        self.combo_fixed_face.addItems(list(FACE_OPTIONS.keys()))
        self.combo_fixed_face.setEnabled(False)
        self.btn_clear_fixed = QtWidgets.QPushButton("Clear Fixed")
        self.btn_clear_fixed.clicked.connect(self.on_clear_fixed)
        self.lbl_fixed = QtWidgets.QLabel("fixed: none  (click a face)")
        self.lbl_fixed.setWordWrap(True)
        form3.addRow(self.btn_pick_fixed)
        form3.addRow("Detected face:", self.combo_fixed_face)
        form3.addRow(self.btn_clear_fixed)
        form3.addRow(self.lbl_fixed)

        sep = QtWidgets.QFrame(); sep.setFrameShape(QtWidgets.QFrame.HLine)
        form3.addRow(sep)

        # -- Applied load (yellow face + purple arrow, single face) ----
        self.btn_pick_load = QtWidgets.QPushButton("Pick Load Face (click model)")
        self.btn_pick_load.setCheckable(True)
        self.btn_pick_load.clicked.connect(
            lambda chk: self._toggle_pick("load", chk))
        # Indicator only: shows which bounding-box face the click matched.
        self.combo_load_face = QtWidgets.QComboBox()
        self.combo_load_face.addItems(list(FACE_OPTIONS.keys()))
        self.combo_load_face.setEnabled(False)
        self.combo_load_mode = QtWidgets.QComboBox()
        self.combo_load_mode.addItems(["Force (N)", "Pressure (Pa)"])
        self.edit_mag = QtWidgets.QLineEdit("1e5")
        self.edit_dx = QtWidgets.QLineEdit("1")
        self.edit_dy = QtWidgets.QLineEdit("0")
        self.edit_dz = QtWidgets.QLineEdit("0")
        dirw = QtWidgets.QWidget(); dl = QtWidgets.QHBoxLayout(dirw)
        dl.setContentsMargins(0, 0, 0, 0)
        for e in (self.edit_dx, self.edit_dy, self.edit_dz):
            e.setMaximumWidth(48); dl.addWidget(e)
        self.btn_load = QtWidgets.QPushButton("Apply Load")
        self.btn_load.clicked.connect(self.on_apply_load)
        # Target-force: align the camera to the face and click the exact spot.
        self.btn_target = QtWidgets.QPushButton("Target force...")
        self.btn_target.setToolTip(
            "Look straight at the load face and click the spot where the force "
            "should act, then Confirm.")
        self.btn_target.clicked.connect(self.on_target_force)
        self.btn_confirm_target = QtWidgets.QPushButton("Confirm")
        self.btn_confirm_target.clicked.connect(self.on_confirm_target)
        self.btn_confirm_target.setVisible(False)
        tgtw = QtWidgets.QWidget(); tl = QtWidgets.QHBoxLayout(tgtw)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addWidget(self.btn_target); tl.addWidget(self.btn_confirm_target)
        self.lbl_load = QtWidgets.QLabel("load: none  (click a face)")
        self.lbl_load.setWordWrap(True)
        form3.addRow(self.btn_pick_load)
        form3.addRow("Detected face:", self.combo_load_face)
        form3.addRow("Load type:", self.combo_load_mode)
        form3.addRow("Magnitude:", self.edit_mag)
        form3.addRow("Direction xyz:", dirw)
        form3.addRow(self.btn_load)
        form3.addRow(tgtw)
        form3.addRow(self.lbl_load)
        v.addWidget(g3)

        # 4. Solve ------------------------------------------------------
        g4 = QtWidgets.QGroupBox("4.  Solve")
        form4 = QtWidgets.QFormLayout(g4)
        # One button: solves [K]{u}={F} for displacement/stress AND runs the
        # load sweep for the spring constant k (they share the same solve).
        self.btn_solve = QtWidgets.QPushButton("Solve  [K]{u} = {F}  &  k")
        self.btn_solve.clicked.connect(lambda: self.on_solve(do_spring=True))
        self.spin_steps = QtWidgets.QSpinBox()
        self.spin_steps.setRange(2, 20); self.spin_steps.setValue(8)
        form4.addRow(self.btn_solve)
        form4.addRow("Load steps:", self.spin_steps)
        v.addWidget(g4)

        v.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        return scroll

    # -- center viewport ------------------------------------------------
    def _build_viewport(self):
        frame = QtWidgets.QFrame()
        lay = QtWidgets.QVBoxLayout(frame)
        lay.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(frame)
        self.plotter.set_background("white", top="lightblue")
        self.plotter.add_axes()
        self._apply_camera_style()
        lay.addWidget(self.plotter)
        return frame

    def _apply_camera_style(self):
        """Left-drag rotates, right-drag (and middle) pans, scroll zooms."""
        try:
            self.plotter.enable_custom_trackball_style(
                left="rotate", middle="pan", right="pan")
        except Exception:                           # pragma: no cover
            pass

    # -- right panel ----------------------------------------------------
    def _build_right_panel(self):
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        v.setSpacing(8)

        # Results summary ----------------------------------------------
        g = QtWidgets.QGroupBox("Results Summary")
        form = QtWidgets.QFormLayout(g)
        self.lbl_res_disp = QtWidgets.QLabel("-")
        self.lbl_res_vm = QtWidgets.QLabel("-")
        self.lbl_res_p1 = QtWidgets.QLabel("-")
        self.lbl_res_p3 = QtWidgets.QLabel("-")
        self.lbl_res_k = QtWidgets.QLabel("-")
        self.lbl_res_resid = QtWidgets.QLabel("-")
        big = QtGui.QFont(); big.setPointSize(big.pointSize() + 1); big.setBold(True)
        self.lbl_res_k.setFont(big)
        form.addRow("Max displacement:", self.lbl_res_disp)
        form.addRow("Max von Mises:", self.lbl_res_vm)
        form.addRow("Max principal s1:", self.lbl_res_p1)
        form.addRow("Min principal s3:", self.lbl_res_p3)
        form.addRow("Spring constant k:", self.lbl_res_k)
        form.addRow("Solve residual:", self.lbl_res_resid)
        v.addWidget(g)

        # Visualization controls ---------------------------------------
        g2 = QtWidgets.QGroupBox("Visualization")
        form2 = QtWidgets.QFormLayout(g2)
        self.combo_field = QtWidgets.QComboBox()
        self.combo_field.addItems(["Displacement magnitude", "von Mises stress",
                                   "Mesh only"])
        self.combo_field.currentIndexChanged.connect(lambda *_: self._update_view())
        self.chk_deformed = QtWidgets.QCheckBox("Show deformed shape")
        self.chk_deformed.setChecked(True)
        self.chk_deformed.stateChanged.connect(lambda *_: self._update_view())
        self.chk_outline = QtWidgets.QCheckBox("Show undeformed outline")
        self.chk_outline.setChecked(True)
        self.chk_outline.stateChanged.connect(lambda *_: self._update_view())
        self.chk_edges = QtWidgets.QCheckBox("Show element edges")
        self.chk_edges.stateChanged.connect(lambda *_: self._update_view())
        self.slider_scale = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_scale.setRange(0, 100); self.slider_scale.setValue(50)
        self.slider_scale.valueChanged.connect(self._on_scale_change)
        self.lbl_scale = QtWidgets.QLabel("scale: -")
        form2.addRow("Field:", self.combo_field)
        form2.addRow(self.chk_deformed)
        form2.addRow(self.chk_outline)
        form2.addRow(self.chk_edges)
        form2.addRow("Deform. scale:", self.slider_scale)
        form2.addRow("", self.lbl_scale)
        v.addWidget(g2)

        # Force-displacement (spring) plot -----------------------------
        g3 = QtWidgets.QGroupBox("Force - Displacement  (Spring)")
        l3 = QtWidgets.QVBoxLayout(g3)
        self.fig = Figure(figsize=(3.4, 3.0), constrained_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(260)
        l3.addWidget(self.canvas)

        ctrl = QtWidgets.QWidget(); cl = QtWidgets.QHBoxLayout(ctrl)
        cl.setContentsMargins(0, 0, 0, 0)
        self.btn_compare = QtWidgets.QPushButton("Compare...")
        self.btn_compare.setToolTip(
            "Import previously-saved chart PNG(s) and overlay them on top of "
            "the current chart to compare.")
        self.btn_compare.clicked.connect(self.on_import_overlay)
        self.btn_clear_overlays = QtWidgets.QPushButton("Clear overlays")
        self.btn_clear_overlays.clicked.connect(self.on_clear_overlays)
        self.btn_save_chart = QtWidgets.QPushButton("Save chart...")
        self.btn_save_chart.clicked.connect(self.on_save_chart)
        cl.addWidget(self.btn_compare, 1)
        cl.addWidget(self.btn_clear_overlays)
        cl.addWidget(self.btn_save_chart)
        l3.addWidget(ctrl)
        v.addWidget(g3)
        self._plot_chart()

        v.addStretch(1)
        panel.setMinimumWidth(330)
        panel.setMaximumWidth(420)
        return panel

    # ------------------------------------------------------------------
    # Stage enabling (greys out steps that aren't ready yet)
    # ------------------------------------------------------------------
    def _set_enabled_stages(self):
        has_mesh = self.mesh.nodes is not None
        has_bc = (self.fixed_nodes is not None) and (self.load_dict is not None)
        for w in (self.btn_mesh, self.combo_refine):
            w.setEnabled(self.mesh.surface is not None)
        in_target = self._pick_mode == "target"
        self.btn_load.setEnabled(has_mesh and self.load_face_tris is not None
                                 and not in_target)
        self.btn_pick_fixed.setEnabled(has_mesh and not in_target)
        self.btn_pick_load.setEnabled(has_mesh and not in_target)
        self.btn_clear_fixed.setEnabled(has_mesh and self.fixed_nodes is not None
                                        and not in_target)
        self.btn_target.setEnabled(self.load_face_tris is not None and not in_target)
        self.btn_solve.setEnabled(has_bc and not in_target)
        viz = self.has_results
        for w in (self.combo_field, self.chk_deformed, self.chk_outline,
                  self.chk_edges, self.slider_scale):
            w.setEnabled(viz)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_float(self, edit, name, positive=False, allow_zero=False):
        txt = edit.text().strip()
        try:
            val = float(txt)
        except ValueError:
            raise ValueError(f"{name} must be a number (got '{txt}').")
        if positive and not allow_zero and val <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        return val

    def _material(self):
        """Read and validate the material inputs."""
        E = self._get_float(self.edit_E, "Young's modulus E", positive=True)
        nu = self._get_float(self.edit_nu, "Poisson's ratio nu")
        rho = self._get_float(self.edit_rho, "Density rho", positive=True)
        if not (-1.0 < nu < 0.5):
            raise ValueError("Poisson's ratio must satisfy -1 < nu < 0.5.")
        return E, nu, rho

    def _apply_material_preset(self, name):
        vals = MATERIALS.get(name)
        if not vals or vals[0] is None:
            return
        E, nu, rho = vals
        self.edit_E.setText(f"{E:g}")
        self.edit_nu.setText(f"{nu:g}")
        self.edit_rho.setText(f"{rho:g}")

    def _warn(self, title, msg):
        QtWidgets.QMessageBox.warning(self, title, msg)

    def _error(self, title, msg):
        QtWidgets.QMessageBox.critical(self, title, msg)

    def _set_status(self, msg, pct=None):
        self.status_label.setText(msg)
        if pct is not None:
            self.progress.setValue(int(pct))

    # ------------------------------------------------------------------
    # Actions: import & mesh
    # ------------------------------------------------------------------
    def on_import(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import STL file", "", "STL files (*.stl);;All files (*)")
        if not path:
            return
        unit_scale = UNIT_SCALE[self.combo_units.currentText()]
        try:
            info = self.mesh.load_stl(path, unit_scale)
        except MeshError as exc:
            self._error("Import failed", str(exc))
            return

        ext = info["extent"]
        self.lbl_file.setText(
            f"{path.split('/')[-1]}\n{info['n_points']} pts, "
            f"{info['n_faces']} facets\nbbox {ext[0]:.3g} x {ext[1]:.3g} x "
            f"{ext[2]:.3g} m")

        # --- watertightness gate --------------------------------------
        watertight, n_open = self.mesh.check_watertight()
        if not watertight:
            ans = QtWidgets.QMessageBox.question(
                self, "Surface not watertight",
                f"The STL surface has {n_open} open (boundary) edge(s) and is "
                "not a closed solid.\n\nTetrahedral meshing requires a "
                "watertight surface.  Attempt automatic repair now?\n\n"
                "(If you skip this, meshing will still try to repair the "
                "surface automatically if it fails.)",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
                | QtWidgets.QMessageBox.Cancel)
            if ans == QtWidgets.QMessageBox.Cancel:
                self._set_status("Import cancelled (surface not watertight).")
                return
            if ans == QtWidgets.QMessageBox.Yes:
                try:
                    remaining = self.mesh.attempt_repair()
                except MeshError as exc:
                    self._error("Repair failed", str(exc)); return
                if remaining > 0:
                    self._warn("Repair incomplete",
                               f"{remaining} open edge(s) remain after repair. "
                               "Meshing may still fail.")

        # Reset downstream state and clear the view.
        self._cancel_pick_mode()
        self.fixed_nodes = self.load_nodes = self.load_dict = None
        self.fixed_faces = []
        self.load_face_tris = self.load_tris = self.load_normal = None
        self.load_centroid = self.load_target = None
        self.results = None
        self.has_results = False
        self.lbl_fixed.setText("fixed: none  (click a face)")
        self.lbl_load.setText("load: none  (click a face)")
        self._clear_results_labels()

        # Auto-convert to a volumetric tetra mesh, as specified.
        self._do_mesh(reset_camera=True)

    def _do_mesh(self, reset_camera=True):
        if self.mesh.surface is None:
            self._warn("No geometry", "Import an STL file first.")
            return
        refinement = self.combo_refine.currentText()
        self._set_status(f"Meshing ({refinement})...", 10)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()
        try:
            minfo = self.mesh.tetrahedralize(refinement=refinement)
            q = self.mesh.mesh_quality()
        except MeshError as exc:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._error("Meshing failed", str(exc))
            self._set_status("Meshing failed.", 0)
            return
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        # Re-meshing invalidates any node-id-based boundary conditions/results.
        self._cancel_pick_mode()
        self.fixed_nodes = self.load_nodes = self.load_dict = None
        self.fixed_faces = []
        self.load_face_tris = self.load_tris = self.load_normal = None
        self.load_centroid = self.load_target = None
        self.results = None
        self.has_results = False
        self.lbl_fixed.setText("fixed: none  (click a face)")
        self.lbl_load.setText("load: none  (click a face)")
        self._clear_results_labels()

        repaired = " (auto-repaired)" if minfo.get("repaired") else ""
        self.lbl_mesh.setText(
            f"{minfo['n_nodes']} nodes, {minfo['n_elem']} tets{repaired}\n"
            f"quality min={q['q_min']:.2f} mean={q['q_mean']:.2f}")
        if q["warnings"]:
            self._set_status("Mesh warning: " + "; ".join(q["warnings"]), 100)
        else:
            self._set_status(
                f"Mesh ready: {minfo['n_elem']} tetrahedra, "
                f"min quality {q['q_min']:.2f}.", 100)

        self._set_enabled_stages()
        self._update_view(reset=reset_camera)

    # ------------------------------------------------------------------
    # Actions: boundary conditions
    # ------------------------------------------------------------------
    _PICK_ANGLE_DEG = 45.0     # region-grow tolerance for a clicked face patch

    def _cancel_pick_mode(self):
        """Turn off any active click-to-pick mode (e.g. on re-mesh/import)."""
        self.btn_pick_fixed.setChecked(False)
        self.btn_pick_load.setChecked(False)
        if self._pick_mode == "target":
            self._exit_target_mode(restore=True)
        elif self._pick_mode is not None:
            self._pick_mode = None
            try:
                self.plotter.disable_picking()
            except Exception:                       # pragma: no cover
                pass
            self._apply_camera_style()

    # -- click-to-pick mode management ---------------------------------
    def _toggle_pick(self, mode, checked):
        """Enter/leave click-to-pick mode for a fixed or load face."""
        if self.mesh.nodes is None:
            self._warn("No mesh", "Generate a volume mesh first.")
            (self.btn_pick_fixed if mode == "fixed"
             else self.btn_pick_load).setChecked(False)
            return
        # Keep the two pick buttons mutually exclusive.
        other_btn = self.btn_pick_load if mode == "fixed" else self.btn_pick_fixed
        other_btn.setChecked(False)

        if checked:
            self._pick_mode = mode
            # Draw the undeformed surface first so picks map to true node
            # positions (also clears any deformed result actors), then arm the
            # surface picker on the freshly-rendered geometry actor.
            self._update_view(force_geometry=True)
            try:
                self.plotter.enable_surface_point_picking(
                    callback=self._on_pick, show_message=False,
                    show_point=False, left_clicking=True)
            except Exception:                       # pragma: no cover
                pass
            if mode == "fixed":
                self._set_status("Pick mode: click faces to fix (green); click "
                                 "a green face to unselect. Button again to end.")
            else:
                self._set_status("Pick mode: click the load face on the model. "
                                 "Click the button again to finish.")
        else:
            self._pick_mode = None
            try:
                self.plotter.disable_picking()
            except Exception:                       # pragma: no cover
                pass
            self._apply_camera_style()
            self._set_status("Pick mode off.")
            self._update_view()

    def _on_pick(self, point, *args):
        """Callback fired when the user clicks the model in pick mode."""
        if self._pick_mode is None or point is None:
            return
        if self._pick_mode == "target":
            self._on_target_pick(point)
            return
        try:
            region = self.mesh.pick_face_region(
                point, angle_tol_deg=self._PICK_ANGLE_DEG)
        except MeshError as exc:
            self._error("Pick failed", str(exc)); return
        if region["node_ids"].size == 0:
            return
        if self._pick_mode == "fixed":
            self._apply_fixed(region)
        else:
            self._apply_load(region)

    @staticmethod
    def _nearest_box_face(normal):
        """Return the FACE_OPTIONS key whose outward direction best matches."""
        best, best_dot = None, -np.inf
        for name, (axis, side) in FACE_OPTIONS.items():
            v = np.zeros(3)
            v[axis] = -1.0 if side == "min" else 1.0
            d = float(np.asarray(normal) @ v)
            if d > best_dot:
                best, best_dot = name, d
        return best

    # -- fixed supports (green; click to add, click again to remove) ---
    def _rebuild_fixed_nodes(self):
        if self.fixed_faces:
            self.fixed_nodes = np.unique(np.concatenate(
                [f["node_ids"] for f in self.fixed_faces]))
        else:
            self.fixed_nodes = None

    def _update_fixed_label(self):
        if self.fixed_faces:
            self.lbl_fixed.setText(f"fixed: {len(self.fixed_faces)} face(s), "
                                   f"{self.fixed_nodes.size} nodes")
        else:
            self.lbl_fixed.setText("fixed: none  (click a face)")

    def _apply_fixed(self, region):
        """Toggle a clicked face in the fixed-support set (green = fixed)."""
        ids = np.asarray(region["node_ids"], dtype=np.int64)
        # Clicking an already-fixed face unselects it.
        for i, f in enumerate(self.fixed_faces):
            common = np.intersect1d(ids, f["node_ids"]).size
            if common >= 0.5 * min(ids.size, f["node_ids"].size):
                del self.fixed_faces[i]
                self._rebuild_fixed_nodes()
                self._update_fixed_label()
                self.has_results = False
                self._set_status("Fixed support removed.")
                self._set_enabled_stages()
                self._update_view()
                return
        self.fixed_faces.append({"tris": region["tris"],
                                 "normal": region["normal"], "node_ids": ids})
        self._rebuild_fixed_nodes()
        axis = self._nearest_box_face(region["normal"])
        self.combo_fixed_face.setCurrentText(axis)
        self._update_fixed_label()
        self.has_results = False
        self._set_status(f"Fixed support added on {axis} face "
                         f"({self.fixed_nodes.size} nodes total).")
        self._set_enabled_stages()
        self._update_view()

    def on_clear_fixed(self):
        self.fixed_nodes = None
        self.fixed_faces = []
        self.lbl_fixed.setText("fixed: none  (click a face)")
        self.has_results = False
        self._set_status("Cleared all fixed supports.")
        self._set_enabled_stages()
        self._update_view()

    # -- applied load (yellow face + purple arrow; single face) --------
    def _apply_load(self, region):
        """Store a clicked load face and seed direction from its inward normal."""
        self.load_face_tris = region["tris"]
        self.load_tris = region["tris"]
        self.load_normal = np.asarray(region["normal"], dtype=float)
        self.load_centroid = np.asarray(region["centroid"], dtype=float)
        self.load_target = None
        self.load_nodes = np.unique(region["tris"]).astype(np.int64)
        axis = self._nearest_box_face(region["normal"])
        self.combo_load_face.setCurrentText(axis)
        # Default the force direction to the inward face normal (pressing on the
        # face); the user can still edit the components and re-apply.
        d = -self.load_normal
        self.edit_dx.setText(f"{d[0]:.4g}")
        self.edit_dy.setText(f"{d[1]:.4g}")
        self.edit_dz.setText(f"{d[2]:.4g}")
        self.on_apply_load()

    def on_apply_load(self):
        """(Re)compute consistent nodal loads on the current load region."""
        if self.load_face_tris is None or self.load_nodes is None:
            self._warn("No load face", "Click a load face on the model first.")
            return
        ids = self.load_nodes
        try:
            mag = self._get_float(self.edit_mag, "Load magnitude", positive=True)
            if self.combo_load_mode.currentText().startswith("Force"):
                d = np.array([self._get_float(self.edit_dx, "dx"),
                              self._get_float(self.edit_dy, "dy"),
                              self._get_float(self.edit_dz, "dz")])
                loads = self.mesh.compute_face_loads(ids, mag, d, mode="force")
            else:
                loads = self.mesh.compute_face_loads(ids, mag, mode="pressure")
        except (ValueError, MeshError) as exc:
            self._error("Invalid load", str(exc)); return

        self.load_dict = loads
        resultant = np.sum(list(loads.values()), axis=0)
        rmag = float(np.linalg.norm(resultant))
        self.load_dir = (resultant / rmag) if rmag > 0 else np.array([1.0, 0, 0])
        axis = self.combo_load_face.currentText()
        spot = "  (targeted spot)" if self.load_target is not None else ""
        self.lbl_load.setText(
            f"load: {axis}{spot}  ({ids.size} nodes)\n|F| = {rmag:.4g} N")
        self.has_results = False
        self._set_status(f"Load applied on {axis} face: resultant {rmag:.4g} N.")
        self._set_enabled_stages()
        self._update_view()

    # -- target force: aim the camera at the face and click the spot ---
    @staticmethod
    def _perp(n):
        """A unit vector perpendicular to ``n`` (for the camera 'up')."""
        n = np.asarray(n, dtype=float)
        ref = np.array([0.0, 0.0, 1.0])
        if abs(float(n @ ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        up = ref - float(ref @ n) * n
        nrm = np.linalg.norm(up)
        return up / nrm if nrm > 0 else np.array([0.0, 1.0, 0.0])

    def _targeted_region(self, spot):
        """Triangles of the load face within a small radius of ``spot``."""
        ftris = self.load_face_tris
        cents = self.mesh.nodes[ftris].mean(axis=1)
        fnodes = self.mesh.nodes[np.unique(ftris)]
        diag = float(np.linalg.norm(np.ptp(fnodes, axis=0)))
        radius = max(0.18 * diag, 1e-12)
        d2 = ((cents - spot) ** 2).sum(axis=1)
        sel = d2 <= radius * radius
        if not sel.any():
            sel = np.zeros(len(ftris), dtype=bool)
            sel[int(np.argmin(d2))] = True
        return ftris[sel]

    def on_target_force(self):
        if self.load_face_tris is None:
            self._warn("No load face", "Pick a load face first.")
            return
        self._cancel_pick_mode()                    # leave any add-face picking
        self._saved_camera = self.plotter.camera_position
        c = self.load_centroid
        n = np.asarray(self.load_normal, dtype=float)
        fnodes = self.mesh.nodes[np.unique(self.load_face_tris)]
        diag = float(np.linalg.norm(np.ptp(fnodes, axis=0))) or \
            self.mesh.characteristic_size
        pos = c + n * (1.8 * diag + 1e-9)
        up = self._perp(n)

        self._pick_mode = "target"
        self._update_view(force_geometry=True)
        try:
            self.plotter.enable_parallel_projection()
            self.plotter.camera_position = [tuple(pos), tuple(c), tuple(up)]
        except Exception:                           # pragma: no cover
            pass
        try:
            self.plotter.enable_surface_point_picking(
                callback=self._on_pick, show_message=False,
                show_point=True, left_clicking=True)
        except Exception:                           # pragma: no cover
            pass
        self.btn_confirm_target.setVisible(True)
        self._set_status("Target mode: click the spot on the face for the "
                         "force, then press Confirm.")
        self._set_enabled_stages()

    def _on_target_pick(self, point):
        spot = np.asarray(point, dtype=float).ravel()[:3]
        self.load_target = spot
        self.load_tris = self._targeted_region(spot)
        self.load_nodes = np.unique(self.load_tris).astype(np.int64)
        self.on_apply_load()                        # recompute on the sub-region

    def _exit_target_mode(self, restore=True):
        self.btn_confirm_target.setVisible(False)
        try:
            self.plotter.disable_picking()
        except Exception:                           # pragma: no cover
            pass
        self._apply_camera_style()
        try:
            self.plotter.disable_parallel_projection()
        except Exception:                           # pragma: no cover
            pass
        self._pick_mode = None
        if restore and self._saved_camera is not None:
            try:
                self.plotter.camera_position = self._saved_camera
            except Exception:                       # pragma: no cover
                pass
        self._saved_camera = None
        self._set_enabled_stages()
        self._update_view()

    def on_confirm_target(self):
        if self._pick_mode != "target":
            return
        had_spot = self.load_target is not None
        self._exit_target_mode(restore=True)
        self._set_status("Force target confirmed." if had_spot
                         else "Target mode closed (no spot selected).")

    # ------------------------------------------------------------------
    # Actions: solve / spring
    # ------------------------------------------------------------------
    def on_solve(self, do_spring=False):
        if self.mesh.nodes is None:
            self._warn("No mesh", "Generate a volume mesh first."); return
        if self.fixed_nodes is None:
            self._warn("No support", "Define a fixed face first."); return
        if not self.load_dict:
            self._warn("No load", "Define an applied load first."); return
        try:
            E, nu, rho = self._material()
        except ValueError as exc:
            self._error("Invalid material", str(exc)); return

        # Disable controls and start the background worker.
        self._set_controls_busy(True)
        self._set_status("Starting solver...", 0)
        self._worker = SolveWorker(
            self.mesh.nodes, self.mesh.tets, E, nu, rho,
            self.fixed_nodes, self.load_dict,
            do_spring=do_spring, n_steps=self.spin_steps.value())
        self._worker.progress.connect(self._on_progress)
        self._worker.done.connect(self._on_solve_done)
        self._worker.failed.connect(self._on_solve_failed)
        self._worker.finished.connect(lambda: self._set_controls_busy(False))
        self._worker.start()

    def _on_progress(self, pct, msg):
        self._set_status(msg, pct)

    def _on_solve_failed(self, message):
        self.progress.setValue(0)
        self._set_status("Solve failed.")
        self._error("Solver error", message)

    def _on_solve_done(self, result):
        self.results = result
        self.has_results = True
        post = result["post"]

        # Auto deformation scale: make the peak deflection ~15% of model size.
        max_disp = post.max_displacement
        char = self.mesh.characteristic_size
        self.auto_scale = (0.15 * char / max_disp) if max_disp > 0 else 1.0

        # Update results panel.
        s = post.summary()
        self.lbl_res_disp.setText(f"{s['max_displacement']:.4g} m")
        self.lbl_res_vm.setText(f"{s['max_von_mises']:.4g} Pa")
        self.lbl_res_p1.setText(f"{s['max_principal']:.4g} Pa")
        self.lbl_res_p3.setText(f"{s['min_principal']:.4g} Pa")
        self.lbl_res_resid.setText(f"{result['residual']:.2e}")

        if result.get("spring"):
            sp = dict(result["spring"])
            self.lbl_res_k.setText(SpringCalculator.format_k(sp["k"]))
            fmax = float(result.get("applied_force", 0.0))
            sp["label"] = f"{self.combo_mat.currentText()}, |F|={fmax:.3g} N"
            self.spring_current = sp
            self._plot_chart()
            self._set_status(
                f"Done. Max disp {s['max_displacement']:.3g} m, "
                f"k = {SpringCalculator.format_k(sp['k'])}.", 100)
        else:
            self._set_status(
                f"Done. Max disp {s['max_displacement']:.3g} m, "
                f"max von Mises {s['max_von_mises']:.3g} Pa.", 100)

        self._set_enabled_stages()
        self._on_scale_change()                 # refresh scale label
        self._update_view(reset=True)

    def _set_controls_busy(self, busy):
        if busy:
            self._cancel_pick_mode()
        for w in (self.btn_solve, self.btn_import, self.btn_mesh, self.btn_load,
                  self.btn_pick_fixed, self.btn_pick_load, self.btn_clear_fixed,
                  self.btn_target):
            w.setEnabled(not busy)
        if not busy:
            self._set_enabled_stages()

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    def _current_scale(self):
        # Slider 0..100 maps linearly to 0..2*auto_scale (50 == auto).
        return self.auto_scale * (self.slider_scale.value() / 50.0)

    def _on_scale_change(self):
        scale = self._current_scale()
        self.lbl_scale.setText(f"scale: {scale:,.0f}x"
                               if scale >= 1 else f"scale: {scale:.2g}x")
        if self.has_results:
            self._update_view()

    def _update_view(self, reset=False, force_geometry=False):
        """Redraw the 3D scene, preserving the camera unless reset=True.

        While a click-to-pick mode is active we update only the boundary-
        condition highlights (without clearing the scene), so the pickable
        geometry actor and the picking observer stay intact.
        """
        if self._pick_mode and not force_geometry:
            self._refresh_pick_highlights()
            return
        cam = None if reset else self.plotter.camera_position
        self.plotter.clear()
        if self.has_results and not force_geometry:
            self._add_result_actors()
        else:
            self._add_geometry_actors()
        if reset:
            self.plotter.reset_camera()
        elif cam is not None:
            self.plotter.camera_position = cam
        self.plotter.render()

    def _refresh_pick_highlights(self):
        """Update BC highlight actors in place (used during pick mode).

        Named actors are removed and re-added so the geometry actor and the
        active picking observer are left untouched.
        """
        for nm in ("fixed_face", "load_face", "load_arrow"):
            try:
                self.plotter.remove_actor(nm, render=False)
            except Exception:                       # pragma: no cover
                pass
        self._add_bc_highlights()
        self.plotter.render()

    def _add_geometry_actors(self):
        if self.mesh.volume is not None:
            self.plotter.add_mesh(self.mesh.volume, color="lightsteelblue",
                                  show_edges=True, edge_color="gray",
                                  name="geom")
        elif self.mesh.surface is not None:
            self.plotter.add_mesh(self.mesh.surface, color="lightsteelblue",
                                  show_edges=True, name="geom")
        self._add_bc_highlights()

    def _node_positions(self):
        """Current node coordinates: deformed if a deformed result is shown."""
        if (self.has_results and not self._pick_mode
                and self.chk_deformed.isChecked()):
            post = self.results["post"]
            return self.mesh.nodes + self._current_scale() * post.u_node
        return self.mesh.nodes

    def _face_polydata(self, tris, coords, normal=None):
        """Compact PolyData of the given triangles, nudged out along ``normal``.

        The small outward offset keeps the coloured highlight from z-fighting
        with the underlying model surface.
        """
        tris = np.asarray(tris, dtype=np.int64)
        uniq, inv = np.unique(tris, return_inverse=True)
        pts = coords[uniq].astype(float, copy=True)
        if normal is not None:
            pts += np.asarray(normal) * (0.004 * self.mesh.characteristic_size)
        local = inv.reshape(tris.shape)
        faces = np.hstack([np.full((local.shape[0], 1), 3, dtype=np.int64),
                           local]).ravel()
        return pv.PolyData(pts, faces)

    def _add_bc_highlights(self):
        if self.mesh.nodes is None:
            return
        coords = self._node_positions()

        # Fixed faces -> solid green (fixed nodes don't move, so they stay put).
        if self.fixed_faces:
            green = None
            for f in self.fixed_faces:
                pd = self._face_polydata(f["tris"], coords, f["normal"])
                green = pd if green is None else green.merge(pd)
            self.plotter.add_mesh(green, color="limegreen", name="fixed_face",
                                  show_edges=False, pickable=False)

        # Load face -> solid yellow patch + a big purple arrow.
        if self.load_tris is not None and len(self.load_tris):
            pd = self._face_polydata(self.load_tris, coords, self.load_normal)
            self.plotter.add_mesh(pd, color="yellow", name="load_face",
                                  show_edges=False, pickable=False)
            centroid = coords[np.unique(self.load_tris)].mean(axis=0)
            arrow_len = 0.45 * self.mesh.characteristic_size
            d = np.asarray(self.load_dir, dtype=float)
            outward = np.asarray(self.load_normal, dtype=float)
            # Keep the whole arrow OUTSIDE the mesh whatever the load sign:
            #  - pressing in (d points inward): head on the face, tail outside.
            #  - pulling out (d points outward): tail on the face, head outside.
            if float(d @ outward) < 0.0:
                tail = centroid - d * arrow_len
            else:
                tail = centroid
            self.plotter.add_arrows(tail[None, :], d[None, :], mag=arrow_len,
                                    color="purple", name="load_arrow",
                                    pickable=False)

    def _add_result_actors(self):
        post = self.results["post"]
        grid = self.mesh.volume.copy()
        scale = self._current_scale() if self.chk_deformed.isChecked() else 0.0
        grid.points = self.mesh.nodes + scale * post.u_node

        field = self.combo_field.currentText()
        show_edges = self.chk_edges.isChecked()
        if field.startswith("Displacement"):
            grid.point_data["Displacement [m]"] = post.disp_mag
            self.plotter.add_mesh(
                grid, scalars="Displacement [m]", cmap="jet",
                show_edges=show_edges, name="result",
                scalar_bar_args=dict(title="Displacement [m]", n_labels=5,
                                     fmt="%.2e", vertical=True))
        elif field.startswith("von Mises"):
            grid.cell_data["von Mises [Pa]"] = post.von_mises
            self.plotter.add_mesh(
                grid, scalars="von Mises [Pa]", cmap="jet",
                show_edges=show_edges, name="result",
                scalar_bar_args=dict(title="von Mises [Pa]", n_labels=5,
                                     fmt="%.2e", vertical=True))
        else:
            self.plotter.add_mesh(grid, color="lightsteelblue",
                                  show_edges=True, name="result")

        # Undeformed reference outline.
        if self.chk_outline.isChecked() and scale > 0:
            self.plotter.add_mesh(self.mesh.volume, style="wireframe",
                                  color="gray", opacity=0.25, name="outline")
        self._add_bc_highlights()

    # ------------------------------------------------------------------
    # Matplotlib spring plot
    # ------------------------------------------------------------------
    def _style_spring_axes(self, ax):
        ax.set_xlabel("Max displacement (mm)", fontsize=8)
        ax.set_ylabel("Applied force F (N)", fontsize=8)
        ax.set_title("Force - Displacement", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.locator_params(axis="x", nbins=5)
        ax.locator_params(axis="y", nbins=6)
        ax.grid(True, alpha=0.3)

    def _plot_chart(self):
        """Draw the current spring curve plus any imported PNG overlays."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        sp = self.spring_current
        if sp:
            x_mm = sp["displacements"] * 1.0e3          # m -> mm for readability
            k_txt = SpringCalculator.format_k(sp["k"])
            ax.plot(x_mm, sp["forces"], "o-", color="tab:blue", markersize=3.5,
                    linewidth=1.6, label=f"{sp.get('label', 'current')}\nk={k_txt}")
            ax.plot(x_mm, sp["k"] * sp["displacements"], "--", color="0.4",
                    linewidth=1.0, alpha=0.8,
                    label=f"fit  R$^2$={sp['r_squared']:.4f}")
            ax.legend(fontsize=7, loc="lower right", framealpha=0.9,
                      handlelength=1.4, borderpad=0.4, labelspacing=0.3)
        else:
            ax.text(0.5, 0.5, "Run Solve to compute k", ha="center",
                    va="center", transform=ax.transAxes, color="gray",
                    fontsize=9)
        self._style_spring_axes(ax)

        # Imported chart PNGs overlaid (semi-transparent) on top, full-figure.
        for img in self.overlay_images:
            ov = self.fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ov.set_in_layout(False)                 # don't disturb the main axes
            ov.imshow(img, aspect="auto", alpha=0.5, zorder=10)
            ov.axis("off")
        self.canvas.draw()

    def on_import_overlay(self):
        """Import one or more saved chart images and overlay them."""
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Import chart image(s) to overlay", "",
            "Images (*.png *.jpg *.jpeg);;All files (*)")
        if not paths:
            return
        import matplotlib.image as mpimg
        added = 0
        for p in paths:
            try:
                self.overlay_images.append(mpimg.imread(p))
                added += 1
            except Exception as exc:                # pragma: no cover
                self._warn("Could not load image", f"{p}\n{exc}")
        if added:
            self._plot_chart()
            self._set_status(f"Overlaid {added} chart image(s) "
                             f"({len(self.overlay_images)} total).")

    def on_clear_overlays(self):
        """Remove imported overlays, leaving only the current chart."""
        self.overlay_images = []
        self._plot_chart()
        self._set_status("Cleared chart overlays.")

    def on_save_chart(self):
        """Save the current Force-Displacement chart to an image/PDF file."""
        if self.spring_current is None and not self.overlay_images:
            self._warn("Nothing to save", "Run a solve to produce a chart first.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Force-Displacement chart", "force_displacement.png",
            "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)")
        if not path:
            return
        try:
            # Save the full canvas (no tight crop) so the plot area keeps a
            # consistent position -- this lets saved charts overlay/align when
            # re-imported via Compare.
            self.fig.savefig(path, dpi=200)
        except Exception as exc:                    # pragma: no cover
            self._error("Save failed", str(exc)); return
        self._set_status(f"Chart saved to {path}.")

    def _clear_results_labels(self):
        for lbl in (self.lbl_res_disp, self.lbl_res_vm, self.lbl_res_p1,
                    self.lbl_res_p3, self.lbl_res_k, self.lbl_res_resid):
            lbl.setText("-")
        self.spring_current = None
        self._plot_chart()

    # ------------------------------------------------------------------
    def on_about(self):
        QtWidgets.QMessageBox.information(
            self, "About PyFEA",
            "<b>PyFEA</b> - linear-static FEA with Tet4 elements.<br><br>"
            "<b>Workflow</b><br>"
            "1. Import an STL (set its units); it is auto-meshed into tetrahedra. "
            "Self-intersecting or non-watertight surfaces are repaired "
            "automatically and meshing runs in a crash-isolated process.<br>"
            "2. Pick a material preset or enter E, nu, rho.<br>"
            "3. Define supports and loads by <b>clicking faces directly on the "
            "model</b>.  Fixed faces turn <b>green</b> (click a green face again "
            "to unselect; multiple allowed); the load face turns <b>yellow</b> "
            "with a purple force arrow.  Use <b>Target force...</b> to look "
            "straight at the load face and click the exact spot for the force.<br>"
            "4. <b>Solve</b> computes displacement &amp; stress and the spring "
            "constant k (load sweep) in one step.<br><br>"
            "Camera: left-drag rotates, right-drag pans, scroll zooms.  "
            "Use the visualization controls to switch displacement / von Mises "
            "fields and scale the deformed shape.  <b>Compare...</b> overlays "
            "saved chart images on the current plot, <b>Clear overlays</b> "
            "removes them, and <b>Save chart...</b> exports the plot.<br><br>"
            "Units are SI: metres, pascals, newtons; k is reported in N/m.")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        # Cleanly shut down the VTK render window to avoid a hang on exit.
        try:
            self.plotter.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    win = FEAMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
