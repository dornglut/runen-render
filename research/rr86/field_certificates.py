#!/usr/bin/env python3
"""RR86 independent CSG field certification experiment (stdlib only).

Two intersecting radius-1 disks centered at (-0.5, 0), (+0.5, 0).
Reference: analytic *exposed circle arcs*, not the min/max SDF field.
Negative means inside, positive outside. Units are logical coordinates.

Run: python3 research/rr86/field_certificates.py
No Rust/RunenGPU/Vulkan code is imported or executed.
"""

from math import hypot, sqrt
from time import perf_counter

R = 1.0
CENTERS = ((-0.5, 0.0), (0.5, 0.0))
SEAM = ((0.0, sqrt(0.75)), (0.0, -sqrt(0.75)))
TOL = 1e-12


def disk_sd(p, c):
    return hypot(p[0] - c[0], p[1] - c[1]) - R


def approximate_field(p, op):
    a, b = (disk_sd(p, c) for c in CENTERS)
    return min(a, b) if op == "union" else max(a, b)


def in_region(p, op):
    a, b = (disk_sd(p, c) <= 0 for c in CENTERS)
    return a or b if op == "union" else a and b


def exposed_circle_distances(p, op):
    """Exact distance candidates to exposed arcs of the two circles.

    For each arc, a nearest point is either its unconstrained radial
    projection (if on that arc) or an arc endpoint at the circle seam.
    """
    distances = [hypot(p[0] - q[0], p[1] - q[1]) for q in SEAM]
    for i, c in enumerate(CENTERS):
        dx, dy = p[0] - c[0], p[1] - c[1]
        length = hypot(dx, dy)
        if length == 0:
            # Seam endpoints are sufficient for this center-symmetric case.
            continue
        q = (c[0] + dx / length, c[1] + dy / length)
        other = CENTERS[1 - i]
        other_sd = disk_sd(q, other)
        is_exposed = (other_sd >= -TOL) if op == "union" else (other_sd <= TOL)
        if is_exposed:
            distances.append(abs(length - R))
    return distances


def exact_signed_distance(p, op):
    d = min(exposed_circle_distances(p, op))
    return -d if in_region(p, op) else d


def predict_naively(p, op, spread):
    return approximate_field(p, op) <= spread


def predict_exact(p, op, spread):
    return exact_signed_distance(p, op) <= spread


def certify_using_1_lipschitz(p, op, spread):
    """Return True/False only when the field *proves* membership.

    For 1-Lipschitz signed fields with an exact zero set, |f| is a
    lower bound to distance from the closest boundary. At positive
    spread original interior stays interior; at negative spread original
    exterior stays exterior. Unknown MUST NOT be silently treated as zero.
    """
    f = approximate_field(p, op)
    if spread > 0:
        if f <= 0:
            return True
        if f > spread:
            return False
        return None
    if spread < 0:
        if f >= 0:
            return False
        if f < spread:
            return True
        return None
    return f <= 0


def reference_checks():
    p = (0.0, 0.0)
    union_f = approximate_field(p, "union")
    union_e = exact_signed_distance(p, "union")
    assert abs(union_f + 0.5) < TOL
    assert abs(union_e + sqrt(0.75)) < TOL
    assert predict_naively(p, "union", -0.6) is False
    assert predict_exact(p, "union", -0.6) is True
    assert certify_using_1_lipschitz(p, "union", -0.6) is None

    p = (0.0, 0.9)
    inter_f = approximate_field(p, "intersection")
    inter_e = exact_signed_distance(p, "intersection")
    assert abs(inter_f - (hypot(0.5, 0.9) - 1.0)) < TOL
    assert abs(inter_e - (0.9 - sqrt(0.75))) < TOL
    assert predict_naively(p, "intersection", 0.031) is True
    assert predict_exact(p, "intersection", 0.031) is False
    assert certify_using_1_lipschitz(p, "intersection", 0.031) is None

    # Certifiers must never contradict the analytic reference.
    examined = certain = uncertain = 0
    for op ("union", "intersection"):
        for spread in (-0.6, -0.2, 0.0, 0.031, 0.2, 0.6):
            for iy in range(-40, 41):
                for ix in range(-55, 56):
                    p = (ix / 40.0, iy / 40.0)
                    cert = certify_using_1_lipschitz(p, op, spread)
                    examined += 1
                    if cert is None:
                        uncertain += 1
                    else:
                        certain += 1
                    if cert is not None:
                        expected = predict_exact(p, op, spread)
                        assert cert == expected, (op, spread, p, cert, expected)
    return examined, certain, uncertain


def scan_case(op, spread):
    counts = {"samples": 0, "naive_disagreements": 0, "certified": 0,
              "uncertain": 0, "certifier_disagreements": 0, "exact_inside": 0,
              "naive_false_inside": 0, "naive_false_outside": 0}
    # 12,705 deterministic points: x=-1.5..1.5; y=-1.3..1.3, step .025.
    started = perf_counter()
    for iy in range(-52, 53):
        for ix in range(-60, 61):
            p = (ix / 40.0, iy / 40.0)
            naive = predict_naively(p, op, spread)
            exact = predict_exact(p, op, spread)
            cert = certify_using_1_lipschitz(p, op, spread)
            counts["samples"] += 1
            if exact:
                counts["exact_inside"] += 1
            if naive != exact:
                counts["naive_disagreements"] += 1
                counts["naive_false_inside" if naive else "naive_false_outside"] += 1
            if cert is None:
                counts["uncertain"] += 1
            else:
                counts["certified"] += 1
                if cert != exact:
                    counts["certifier_disagreements"] += 1
    counts["wall_seconds_python"] = round(perf_counter() - started, 3)
    return counts


if __name__ == "__main__":
    checked = reference_checks()
    print(f"Independent reference: {checked[0]} points examined; "
          f"{checked[1]} definite certificates checked; {checked[2]} uncertain")
    print("RR86 exact exposed-arc CSG oracle; Python stdlib; 10 Oct 2026")
    for op, spread in (("union", -0.6), ("union", 0.2),
                       ("intersection", 0.031), ("intersection", -0.2)):
        counts = scan_case(op, spread)
        print(f"{op:12s} spread={spread:+.3f}: {counts}")
    print("Reference assertions and all certified classifications PASS")
