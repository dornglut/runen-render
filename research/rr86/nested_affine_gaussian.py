#!/usr/bin/env python3
"""RR86 independent parent-frame affine + finite Gaussian proof (stdlib).

Implements a specific F3F semantic case, not the RunenRender runtime:
local rectangle -> parent-scale -> shadow offset -> normalized finite
separable Gaussian in immediate-parent space -> ancestor affine.
Compares against a WRONG candidate that first transforms the caster
and then applies the same isotropic Gaussian in root space.
"""
from math import erf, exp, pi, sqrt

SIGMA = 0.25
CUT = 3.0 * SIGMA
NORM = erf(3.0 / sqrt(2.0))
# Parent-frame caster: local [0,1]x[0,1] after group-local scale (2,0.5)
RECT = (0.0, 2.0, 0.0, 0.5)
OFFSET = (0.2, -0.1)
# root_x=a*x+c*y+tx, root_y=b*x+d*y+ty. Nonuniform/sheared.
A = (1.8, -0.35, 0.55, 0.9, 0.4, -0.25)


def forward(p):
    a, b, c, d, tx, ty = A
    return (a * p[0] + c * p[1] + tx, b * p[0] + d * p[1] + ty)


def inverse(p):
    a, b, c, d, tx, ty = A
    det = a * d - b * c
    assert det > 0
    x, y = p[0] - tx, p[1] - ty
    return ((d * x - c * y) / det, (-b * x + a * y) / det)


def rect_shifted():
    x0, x1, y0, y1 = RECT
    dx, dy = OFFSET
    return (x0 + dx, x1 + dx, y0 + dy, y1 + dy)


def expanded_rect():
    x0, x1, y0, y1 = rect_shifted()
    return (x0 - CUT, x1 + CUT, y0 - CUT, y1 + CUT)


def within(p, rect):
    x0, x1, y0, y1 = rect
    return x0 <= p[0] <= x1 and y0 <= p[1] <= y1


def one_dim_mass(lo, hi, center):
    """Normalized N(0,sigma) mass over [lo,hi], truncated at +/- 3 sigma."""
    lo = max(lo, center - CUT)
    hi = min(hi, center + CUT)
    if hi <= lo:
        return 0.0
    t = SIGMA * sqrt(2.0)
    return (erf((hi - center) / t) - erf((lo - center) / t)) / (2 * NORM)


def correct_coverage(root_p):
    parent = inverse(root_p)
    x0, x1, y0, y1 = rect_shifted()
    return one_dim_mass(x0, x1, parent[0]) * one_dim_mass(y0, y1, parent[1])


def correct_support(root_p):
    """Support set is affine image of parent square-cutoff Minkowski sum."""
    return within(inverse(root_p), expanded_rect())


def corners(rect):
    x0, x1, y0, y1 = rect
    return [forward((x0, y0)), forward((x1, y0)),
            forward((x1, y1)), forward((x0, y1))]


def naive_support(root_p):
    """Incorrect: transformed caster AABB + untransformed root-space halo."""
    poly = corners(rect_shifted())
    x0 = min(v[0] for v in poly) - CUT
    x1 = max(v[0] for v in poly) + CUT
    y0 = min(v[1] for v in poly) - CUT
    y1 = max(v[1] for v in poly) + CUT
    return within(root_p, (x0, x1, y0, y1))


def interval_at_y(poly, y):
    """Horizontal slice through convex affine-transformed rectangle."""
    intersections = []
    for i in range(4):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % 4]
        if (y0 <= y < y1) or (y1 <= y < y0):
            t = (y - y0) / (y1 - y0)
            intersections.append(x0 + t * (x1 - x0))
    if len(intersections) < 2:
        return None
    return min(intersections), max(intersections)


def root_gaussian_density(t):
    if abs(t) > CUT:
        return 0.0
    return exp(-0.5 * (t / SIGMA) ** 2) / (SIGMA * sqrt(2 * pi) * NORM)


def incorrect_root_isotropic_blur(root_p, n=16384):
    """Independent numeric convolution in root frame over actual polygon.

    Integrate along root y with Simpson; conditional x Gaussian mass from erf.
    This faithfully implements the *incorrect frame choice*, not a bbox mask.
    """
    assert n % 2 == 0
    poly = corners(rect_shifted())
    lo = max(root_p[1] - CUT, min(y for _, y in poly))
    hi = min(root_p[1] + CUT, max(y for _, y in poly))
    if hi <= lo:
        return 0.0
    dy = (hi - lo) / n
    accum = 0.0
    for i in range(n + 1):
        y = lo + i * dy
        span = interval_at_y(poly, y)
        if span is None:
            value = 0.0
        else:
            value = root_gaussian_density(y - root_p[1]) * one_dim_mass(
                span[0], span[1], root_p[0])
        accum += value * (1 if i == 0 or i == n else 4 if i % 2 else 2)
    return accum * dy / 3.0


def test_support():
    # Query the *same root samples*, including offscreen/blur fringe and shear.
    both = only_correct = only_naive = neither = 0
    for iy in range(-25, 51):
        for ix in range(-30, 101):
            # Parent-basis sampling makes deterministic threshold cases legible.
            parent = (ix / 25.0, iy / 25.0)
            root = forward(parent)
            c = correct_support(root)
            n = naive_support(root)
            if c and n:
                both += 1
            elif c:
                only_correct += 1
            elif n:
                only_naive += 1
            else:
                neither += 1
            alpha = correct_coverage(root)
            assert 0.0 <= alpha <= 1.0 + 1e-12
            if not c:
                assert alpha < 1e-12, (root, parent, alpha)
    assert both + only_correct + only_naive + neither == 76 * 131
    assert only_correct > 0 and only_naive > 0, (only_correct, only_naive)
    print("Support samples=", 76 * 131, "both=", both,
          "false negatives=", only_correct, "false positives=", only_naive,
          "neither=", neither)
    return (only_correct, only_naive)


def test_blur():
    probes = [(-0.3, 0.2), (0.05, 0.2), (0.2, -0.1),
              (1.9, 0.4), (2.3, 0.2), (1.0, 0.25)]
    max_err = 0.0
    # Two Simpson resolutions provide an empirical convergence check, not
    # a mathematically certified absolute numerical error bound.
    for p in probes:
        root = forward(p)
        correct = correct_coverage(root)
        wrong = incorrect_root_isotropic_blur(root, n=8192)
        fine = incorrect_root_isotropic_blur(root, n=32768)
        quadrature_delta = abs(wrong - fine)
        assert quadrature_delta < 4e-5, (p, quadrature_delta)
        max_err = max(max_err, abs(correct - fine))
        print(f"parent={p} correct={correct:.9f} wrong_root={fine:.9f}"
              f" absolute_error={abs(correct-fine):.9f} quadrature_delta={quadrature_delta:.2g}")
    assert max_err > 0.05, max_err
    return max_err


if __name__ == "__main__":
    # A transformed point must round-trip for every probe.
    for p in ((0, 0), (0.3, 0.8), (2.0, -1.0)):
        q = inverse(forward(p))
        assert max(abs(p[i] - q[i]) for i in range(2)) < 1e-12
    fn, fp = test_support()
    err = test_blur()
    print(f"PASS: support false negatives={fn}, false positives={fp}; "
          f"max incorrect-frame alpha error={err:.6f}.")
