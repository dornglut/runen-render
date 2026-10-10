#!/usr/bin/env python3
"""RR86 independent orthogonal CSG oracle and signed-field candidate comparator.

Finite rectangles and nested union/intersection/difference produce an exact
orthogonal region. Source leaf data are geometry, NOT sampled paint alpha.
Boundary segments are constructed from exact induced arrangement cells;
Euclidean point-to-segment distance and Gaussian no-spread coverage then have
independent, non-SDF reference implementations.

Requires only Python 3 standard library; not F1/Rust/Vulkan conformance.
"""
from __future__ import annotations

from math import erf, hypot, sqrt
from time import perf_counter

# (left, bottom, right, top) in one logical frame.
A = (-2.0, -0.4, 2.0, 0.4)
B = (-0.4, -2.0, 0.4, 2.0)
C = (-1.3, -1.5, 1.25, 1.6)
HOLE = (-0.22, -0.24, 0.27, 0.28)

def rect(r):
    return ("rect", r)

def pair(op, left, right):
    return (op, left, right)

CASES = {
    "cross_union": pair("union", rect(A), rect(B)),
    "cross_clip": pair("intersection", pair("union", rect(A), rect(B)), rect(C)),
    "clipped_hole": pair("difference", rect(C), rect(HOLE)),
    "nested_csg": pair("difference", pair("intersection", pair("union", rect(A), rect(B)), rect(C)), rect(HOLE)),
    "overlap_intersection": pair("intersection", rect((-1.5, -1.5, 0.5, 1.0)), rect((-0.5, -0.3, 1.5, 1.5))),
}


def leaves(expr):
    if expr[0] == "rect":
        yield expr[1]
    else:
        yield from leaves(expr[1])
        yield from leaves(expr[2])


def rect_inside(r, p):
    return r[0] <= p[0] <= r[2] and r[1] <= p[1] <= r[3]


def inside(expr, p):
    op = expr[0]
    if op == "rect":
        return rect_inside(expr[1], p)
    a = inside(expr[1], p)
    b = inside(expr[2], p)
    return (a or b) if op == "union" else ((a and b) if op == "intersection" else (a and not b))


def rectangle_signed_distance(r, p):
    x0, y0, x1, y1 = r
    x, y = p
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    if dx > 0 or dy > 0:
        return hypot(dx, dy)
    return -min(x - x0, x1 - x, y - y0, y1 - y)


def naive_csg_field(expr, p):
    op = expr[0]
    if op == "rect":
        return rectangle_signed_distance(expr[1], p)
    a = naive_csg_field(expr[1], p)
    b = naive_csg_field(expr[2], p)
    return min(a, b) if op == "union" else (max(a, b) if op == "intersection" else max(a, -b))


def arrangement(expr):
    """Disjoint axis-aligned inside cells; exposed boundary segments."""
    xs = sorted(set(v for r in leaves(expr) for v in (r[0], r[2])))
    ys = sorted(set(v for r in leaves(expr) for v in (r[1], r[3])))
    cells = {}
    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            x0, x1, y0, y1 = xs[ix], xs[ix + 1], ys[iy], ys[iy + 1]
            cells[(ix, iy)] = inside(expr, ((x0+x1)/2, (y0+y1)/2))
    present = [(xs[ix], xs[ix+1], ys[iy], ys[iy+1]) for (ix, iy), v in cells.items() if v]
    segments = []
    for (ix, iy), v in cells.items():
        if not v:
            continue
        x0, x1, y0, y1 = xs[ix], xs[ix+1], ys[iy], ys[iy+1]
        if not cells.get((ix-1, iy), False):
            segments.append(((x0,y0),(x0,y1)))
        if not cells.get((ix+1, iy), False):
            segments.append(((x1,y0),(x1,y1)))
        if not cells.get((ix,iy-1), False):
            segments.append(((x0,y0),(x1,y0)))
        if not cells.get((ix,iy+1), False):
            segments.append(((x0,y1),(x1,y1)))
    return present, segments


def segment_distance(p, seg):
    (ax, ay), (bx, by) = seg
    ux, uy = bx-ax, by-ay
    q = max(0., min(1., ((p[0]-ax)*ux+(p[1]-ay)*uy)/(ux*ux+uy*uy)))
    return hypot(p[0]-ax-q*ux, p[1]-ay-q*uy)


def exact_signed_distance(expr, segments, p):
    if not segments:
        raise ValueError("Empty region has no finite signed-distance oracle")
    distance = min(segment_distance(p, e) for e in segments)
    # Probes avoid exact boundaries to avoid closed-vs-open set conventions.
    if distance <= 1.e-10:
        return 0.0
    return -distance if inside(expr,p) else distance


def partial_certificate(field, spread):
    """1-Lipschitz signed field: only certain offset classifications returned."""
    if spread > 0:
        return True if field <= 0 else False if field > spread else None
    if spread < 0:
        return False if field >= 0 else True if field < spread else None
    return field <= 0


def gaussian_axis(a, b, center, sigma):
    if sigma == 0:
        return 1.0 if a <= center <= b else 0.0
    radius = 3.0*sigma
    left, right = max(a, center-radius), min(b, center+radius)
    if right <= left:
        return 0.0
    norm = erf(3.0/sqrt(2.0))
    unit = sigma*sqrt(2.0)
    return (erf((right-center)/unit)-erf((left-center)/unit))/(2.0*norm)


def exact_gaussian_no_spread(cells, p, sigma):
    """Integrated normalized finite square Gaussian over disjoint cells.

    Does NOT imply a formula for nonzero signed Euclidean spread: curved
    disk-offset boundaries require a separate evaluator/integration oracle.
    """
    return sum(gaussian_axis(x0,x1,p[0],sigma)*gaussian_axis(y0,y1,p[1],sigma)
               for x0,x1,y0,y1 in cells)


def intersect_rect(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return (x0,y0,x1,y1) if x0 < x1 and y0 < y1 else None


def inclusion_exclusion_terms(expr):
    """Second, algebraically independent CSG Gaussian area oracle.

    A term is (integer sign, rectangle). This uses the CSG indicator
    algebra, not arrangement cells or exposed-boundary construction.
    """
    if expr[0] == "rect":
        return [(1,expr[1])]
    a = inclusion_exclusion_terms(expr[1])
    b = inclusion_exclusion_terms(expr[2])
    intersections = []
    for sign_a,ra in a:
        for sign_b,rb in b:
            ri = intersect_rect(ra, rb)
            if ri is not None:
                intersections.append((sign_a*sign_b, ri))
    if expr[0]=="intersection":
        return intersections
    if expr[0]=="union":
        return a+b+[(-sign, r) for sign,r in intersections]
    assert expr[0]=="difference"
    return a+[(-sign, r) for sign,r in intersections]


def gaussian_independent_csg(expr,p,sigma):
    return sum(sign*gaussian_axis(r[0],r[2],p[0],sigma)
               *gaussian_axis(r[1],r[3],p[1],sigma)
               for sign,r in inclusion_exclusion_terms(expr))


def one_case(name, expr):
    cells, segments = arrangement(expr)
    assert cells and segments, name
    radii=(-0.45,-0.2,0.0,0.12,0.45)
    counts={r:{"examined":0,"naive_wrong":0,"certain":0,"uncertain":0,
               "certificate_wrong":0,"fallback_wrong":0} for r in radii}
    # 46x39 fixed off-grid samples: avoids uncertain zero-boundary ties.
    probes=[(-2.27+ix*0.103,-1.98+iy*0.109) for iy in range(39) for ix in range(46)]
    t0=perf_counter()
    for p in probes:
        f=naive_csg_field(expr,p)
        d=exact_signed_distance(expr,segments,p)
        if abs(d) > 1.e-10:
            assert (f < 0)==(d < 0) or abs(f)<1.e-10, (name,p,f,d)
            assert abs(f) <= abs(d)+1.e-9, (name,p,f,d)
        for radius in radii:
            c=counts[radius]
            c["examined"]+=1
            exact=(d <= radius)
            naive=(f <= radius)
            cert=partial_certificate(f,radius)
            hybrid = exact if cert is None else cert
            c["naive_wrong"]+=(naive != exact)
            c["certain"]+=(cert is not None)
            c["uncertain"]+=(cert is None)
            c["certificate_wrong"]+=(cert is not None and cert != exact)
            c["fallback_wrong"]+=(hybrid != exact)
    wall=perf_counter()-t0
    assert all(c["fallback_wrong"]==0 and c["certificate_wrong"]==0 for c in counts.values())
    assert all(c["examined"]==c["certain"]+c["uncertain"] for c in counts.values())
    # Gaussian: evaluate the exact continuous (zero-spread) convolution
    # of inside arrangement cells against a normalized 3-sigma square.
    for sigma in (0.05,0.3):
        for p in ((0.113,0.179),(-0.303,0.487),(0.509,-0.417),(-1.73,0.033)):
            alpha=exact_gaussian_no_spread(cells,p,sigma)
            assert -1.e-12 <= alpha <= 1.+1.e-12, (name,p,sigma,alpha)
            independent = gaussian_independent_csg(expr,p,sigma)
            assert abs(alpha-independent) <= 1.e-12, (name,p,sigma,alpha,independent)
    print(f"{name}: leaves={len(list(leaves(expr)))} inside_cells={len(cells)} "
          f"boundary_segments={len(segments)} probes={len(probes)} "
          f"python_ref_wall_s={wall:.3f} (NOT a Rust/GPU benchmark)")
    for r,stat in counts.items():
        print(f"  spread={r:+.2f} {stat}")
    return counts


def main():
    total_wrong=0
    total_uncertain=0
    total_certified=0
    total_examined=0
    for name,expr in CASES.items():
        counts=one_case(name,expr)
        total_wrong+=sum(c["naive_wrong"] for c in counts.values())
        total_uncertain+=sum(c["uncertain"] for c in counts.values())
        total_certified+=sum(c["certain"] for c in counts.values())
        total_examined+=sum(c["examined"] for c in counts.values())
    assert total_wrong>0, "adversarial corpus should falsify min/max-as-metric"
    assert total_uncertain>0, "partial field certificates must expose uncertainty"
    assert total_certified+total_uncertain==total_examined
    print(f"TOTAL examined={total_examined}, naive_wrong={total_wrong}, "
          f"definite_certificates={total_certified}, uncertain={total_uncertain}, "
          "wrong_definite=0, wrong_exact_fallback=0")
    print("PASS: independent exact exposed-boundary oracle, correct partial certificates, "
          "identity-spread finite Gaussian on disjoint cells")

if __name__=="__main__":
    main()
