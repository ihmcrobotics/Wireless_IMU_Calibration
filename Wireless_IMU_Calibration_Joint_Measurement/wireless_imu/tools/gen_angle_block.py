#!/usr/bin/env python3
"""
gen_angle_block.py  -  printable multi-angle test fixture for Test A.

DESIGN
------
Not a stack of wedges. A single prism whose cross-section is a fan of flat
facets, each cut at a known angle to the sensor mounting face. The IMU sits
in a channel on the mounting face; you rest the block on facet k and the
sensor is tilted by exactly theta_k.

Why this shape:
  * No hinge, no stack, nothing to settle or slip between holds.
  * Every angle is fixed by the same piece of plastic, so the angles are
    related to each other exactly even if the printer is slightly off.
  * 90 degrees is trivial - it is just another facet.
  * Print it with the prism axis VERTICAL. Then every facet and the mounting
    face are vertical walls, so the angles between them come from X-Y motion,
    which is the printer's accurate axis. Angles printed as stacked layers
    inherit the layer-height staircase; these do not.
  * No magnets anywhere. A magnetic base would wreck the magnetometer.
"""

import numpy as np
from stl import mesh

# ------------------------------------------------------------------ parameters
ANGLES = [0, 15, 30, 45, 75, 90]     # add 60 here for an extra point
R = 60.0        # mm, facet distance from centre - sets overall size
TOP = 55.0      # mm, mounting face offset
BACK = 55.0     # mm, rear face offset
WIDTH = 40.0    # mm, prism width (this becomes the PRINT HEIGHT)

CH_W = 28.0     # mm, sensor channel width  - measure your board and adjust
CH_D = 2.5      # mm, sensor channel depth

OUT_STL = "angle_block.stl"


# ------------------------------------------------------------------ geometry
def clip(poly, n, d):
    """Sutherland-Hodgman: keep the half-plane  n . p <= d."""
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        da, db = np.dot(n, a) - d, np.dot(n, b) - d
        if da <= 0:
            out.append(a)
        if (da > 0) != (db > 0):
            out.append(a + (b - a) * (da / (da - db)))
    return out


def build_profile():
    big = 400.0
    poly = [np.array([-big, -big]), np.array([big, -big]),
            np.array([big, big]), np.array([-big, big])]

    for th in ANGLES:                      # the fan of seating facets
        t = np.radians(th)
        poly = clip(poly, np.array([np.sin(t), -np.cos(t)]), R)
    poly = clip(poly, np.array([0.0, 1.0]), TOP)     # mounting face
    poly = clip(poly, np.array([-1.0, 0.0]), BACK)   # rear face

    poly = [p for i, p in enumerate(poly)
            if np.linalg.norm(p - poly[i - 1]) > 1e-6]

    # channel for the sensor, cut into the mounting face, full prism width
    xs = [p[0] for p in poly if abs(p[1] - TOP) < 1e-6]
    cx = 0.5 * (min(xs) + max(xs))
    x0, x1 = cx - CH_W / 2, cx + CH_W / 2

    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        out.append(a)
        if abs(a[1] - TOP) < 1e-6 and abs(b[1] - TOP) < 1e-6:
            lo, hi = (x0, x1) if b[0] > a[0] else (x1, x0)
            out += [np.array([lo, TOP]), np.array([lo, TOP - CH_D]),
                    np.array([hi, TOP - CH_D]), np.array([hi, TOP])]
    return out


# ------------------------------------------------------------------ meshing
def area2(p):
    return sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
               for i in range(len(p)))


def inside(a, b, c, q):
    d = ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
    if abs(d) < 1e-12:
        return False
    u = ((b[1] - c[1]) * (q[0] - c[0]) + (c[0] - b[0]) * (q[1] - c[1])) / d
    v = ((c[1] - a[1]) * (q[0] - c[0]) + (a[0] - c[0]) * (q[1] - c[1])) / d
    return u > 1e-9 and v > 1e-9 and (1 - u - v) > 1e-9


def earclip(poly):
    """Triangulate a simple polygon (handles the non-convex channel)."""
    p = list(poly)
    if area2(p) < 0:
        p.reverse()
    idx = list(range(len(p)))
    tris, guard = [], 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = p[i0], p[i1], p[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                continue
            if any(inside(a, b, c, p[j]) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    return p, tris


def extrude(poly, w):
    p, tris = earclip(poly)
    n = len(p)
    V = ([np.array([q[0], q[1], 0.0]) for q in p] +
         [np.array([q[0], q[1], w]) for q in p])
    F = []
    for a, b, c in tris:
        F.append([a, c, b])                       # bottom cap, normal -Z
        F.append([a + n, b + n, c + n])           # top cap, normal +Z
    for i in range(n):
        j = (i + 1) % n
        F.append([i, j, j + n])
        F.append([i, j + n, i + n])
    return np.array(V), np.array(F)


# ------------------------------------------------------------------ main
prof = build_profile()
V, F = extrude(prof, WIDTH)

m = mesh.Mesh(np.zeros(len(F), dtype=mesh.Mesh.dtype))
for i, f in enumerate(F):
    for j in range(3):
        m.vectors[i][j] = V[f[j]]
m.save(OUT_STL)

xs = [p[0] for p in prof]
ys = [p[1] for p in prof]
print(f"facets      {ANGLES}")
print(f"profile     {len(prof)} vertices, {len(F)} triangles")
print(f"bounding box {max(xs)-min(xs):.1f} x {max(ys)-min(ys):.1f} x {WIDTH:.1f} mm")
print(f"channel     {CH_W:.1f} wide x {CH_D:.1f} deep, full width")
print(f"saved       {OUT_STL}")

# facet lengths tell you how much seating area each angle actually has
print("\n facet   seat length")
for th in ANGLES:
    t = np.radians(th)
    n = np.array([np.sin(t), -np.cos(t)])
    on = [p for p in prof if abs(np.dot(n, p) - R) < 1e-6]
    if len(on) >= 2:
        L = max(np.linalg.norm(a - b) for a in on for b in on)
        flag = "  <-- narrow, check stability" if L < 15 else ""
        print(f"  {th:3d} deg  {L:6.1f} mm{flag}")
