"""
generate_sample_stl.py
=======================

Create a simple rectangular *beam* STL to use as test geometry for the FEA app.

The beam is 100 mm x 20 mm x 20 mm, written in METRES (0.1 x 0.02 x 0.02 m), so
it can be loaded with unit scale = "m".  Fixing one end face and pulling the
opposite end face makes the analytical axial stiffness easy to check:

        k = A * E / L          (A = cross-section area, L = length)

Run:  python generate_sample_stl.py
"""

import numpy as np
import pyvista as pv

# Beam dimensions in metres.
L, W, H = 0.10, 0.02, 0.02


def main(path: str = "sample_beam.stl") -> str:
    # pv.Box with level>0 and quads=False yields a nicely tessellated,
    # watertight triangular surface (more surface nodes per face = better load
    # distribution and a smoother result).
    box = pv.Box(bounds=(0.0, L, 0.0, W, 0.0, H), level=6, quads=False)
    box = box.triangulate().clean()
    box.save(path)
    print(f"Wrote {path}")
    print(f"  dimensions  : {L} x {W} x {H} m")
    print(f"  surface pts : {box.n_points}, facets: {box.n_faces}")
    print(f"  cross-sect A: {W * H:.3e} m^2, length L: {L} m")
    print(f"  For steel (E=2.1e11 Pa): analytical k = A*E/L = "
          f"{(W * H) * 2.1e11 / L:.4e} N/m")
    return path


if __name__ == "__main__":
    main()
