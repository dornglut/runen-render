#!/usr/bin/env python3
"""RR86 exact finite-Gaussian counterexamples after CSG morphology.

Two unit disks centered (-1/2,0) and (+1/2,0). Exact counterexample
certificates rely only on Euclidean geometry and 1-Lipschitz distances,
not sampled approximations or RunenRender/GPU code.
"""
from math import hypot, sqrt

CENTERS = ((-0.5, 0.0), (0.5, 0.0))
CROSSING_Y = sqrt(3.0) / 2.0
# Ancestor nonuniform/sheared map; effects remain in the parent frame.
AFFINE = (2.0, -0.4, 0.7, 0.9, 0.3, -0.2)


def affine(p):
    a, b, c, d, tx, ty = AFFINE
    return (a * p[0] + c * p[1] + tx, b * p[0] + d * p[1] + ty)


def inverse(q):
    a, b, c, d, tx, ty = AFFINE
    det = a * d - b * c
    assert det > 0.0
    x, y = q[0] - tx, q[1] - ty
    return ((d * x - c * y) / det, (-b * x + a * y) / det)


def primitive_signed_disk(p, center):
    return hypot(p[0] - center[0], p[1] - center[1]) - 1.0


def naive_boolean_field(p, operation):
    a, b = (primitive_signed_disk(p, c) for c in CENTERS)
    return min(a, b) if operation == "union" else max(a, b)


def support_probe(op, spread, sigma, point):
    """Return exact and naive full-coverage certificates at root query.

    For a Gaussian whose strictly positive 2D kernel is confined to the
    parent square [-3sigma,3sigma]^2, entire-square inside => alpha 1,
    entire-square outside => alpha 0 after normalized integration.
    """
    root = affine(point)
    recovered = inverse(root)
    assert max(abs(recovered[i] - point[i]) for i in range(2)) < 1e-14
    displacement = sqrt(2.0) * 3.0 * sigma
    field = naive_boolean_field(recovered, op)
    if op == "union":
        assert recovered == (0.0, 0.0)
        assert spread < 0.0
        # The true distance from origin to exposed UNION boundary is
        # sqrt(3)/2. Erosion radius |spread| leaves a full disk of
        # radius (sqrt(3)/2 - |spread|) inside the true eroded union.
        exact_inside_margin = CROSSING_Y + spread
        # Naive min-SDF erosion equals union of two independently eroded
        # disks; their nearest points are at |x|=0.5-(1+spread).
        naive_outside_margin = field - spread
        assert exact_inside_margin > displacement
        assert naive_outside_margin > displacement
        # For any point within displacement, each primitive disk's
        # signed distance changes by at most displacement.
        assert field - spread - displacement > 0.0
        return {"case": "union erosion", "root": root, "spread": spread,
                "sigma": sigma, "kernel_max_displacement": displacement,
                "true_coverage": 1, "naive_coverage": 0,
                "true_margin": exact_inside_margin,
                "naive_field_at_center": field}
    assert op == "intersection" and spread > 0.0
    assert max(abs(recovered[i] - point[i]) for i in range(2)) < 1e-14
    # Closest point on lens-shaped intersection is top seam vertex.
    true_exterior_distance = recovered[1] - CROSSING_Y
    exact_outside_margin = true_exterior_distance - spread
    # Naive max-SDF dilation is intersection of individually dilated
    # disks: field <= spread, and 1-Lipschitz bounds remain inside.
    naive_inside_margin = spread - field
    assert exact_outside_margin > displacement
    assert naive_inside_margin > displacement
    return {"case": "intersection dilation", "root": root,
            "spread": spread, "sigma": sigma,
            "kernel_max_displacement": displacement,
            "true_coverage": 0, "naive_coverage": 1,
            "true_margin": exact_outside_margin,
            "naive_field_at_center": field}


if __name__ == "__main__":
    # Even though finite kernels are normalized, complete inclusion/exclusion
    # of the entire cutoff square makes these alpha results EXACT (0 or 1).
    cases = [support_probe("union", -0.6, 0.02, (0.0, 0.0)),
             support_probe("intersection", 0.031, 0.0001, (0.0, 0.9))]
    for c in cases:
        assert c["true_coverage"] != c["naive_coverage"]
        print(c)
    print("PASS: exact nested source-neutral Gaussian alpha flips in both cases")
