#!/usr/bin/env python3
"""RR86 adversarial arrangement and fallback scaling (research, not GPU).

Tests growth of reference orthogonal CSG evaluator and a private-work budget
analogue. Only Python stdlib. The bound is experimental, not an adopted
RunenRender resource limit or valid equivalent renderer implementation.
"""
from __future__ import annotations

from math import isfinite
from time import perf_counter
from tracemalloc import start, stop, get_traced_memory

from orthogonal_csg_reference import (
    arrangement, exact_signed_distance, naive_csg_field,
    partial_certificate, pair, rect,
)

# The crosshatch union forces both axes to accumulate leaf coordinates.
# Alternate vertical and horizontal narrow strips, preserving finite bounds.
def crosshatch(leaves: int):
    assert leaves >= 4 and leaves % 2 == 0
    shape = None
    for i in range(leaves):
        pos = -0.96 + 1.92 * ((i // 2) + 0.5) / (leaves // 2)
        width = 0.2 / (leaves // 2)
        leaf = (pos - width, -1.0, pos + width, 1.0) if i % 2 == 0 else (
            -1.0, pos - width, 1.0, pos + width
        )
        shape = rect(leaf) if shape is None else pair("union", shape, rect(leaf))
    return shape


# An intentionally simple material-region shape with many leaf boundaries.
# Arrangement prep cost and retained cells are distinct from semantic
# geometry complexity and should never silently become a public limit.
def measure(leaves: int):
    expr = crosshatch(leaves)
    start()
    t0 = perf_counter()
    cells, boundaries = arrangement(expr)
    prep_ms = (perf_counter() - t0) * 1000
    _current, peak = get_traced_memory()
    stop()
    assert cells and boundaries

    # Fixed 29x29 output queries for comparability across input sizes.
    points = [(-1.03 + x * 2.06/28, -1.03 + y * 2.06/28)
              for x in range(29) for y in range(29)]
    out = {}
    for spread in (-0.025, +0.05):
        uncertain = 0
        mismatch = 0
        oracle_t0 = perf_counter()
        truth = [exact_signed_distance(expr, boundaries, p) <= spread for p in points]
        exact_ms = (perf_counter() - oracle_t0) * 1000
        hybrid_t0 = perf_counter()
        for p, exact in zip(points, truth):
            f = naive_csg_field(expr, p)
            admission = partial_certificate(f, spread)
            uncertain += admission is None
            if admission is None:
                admission = exact_signed_distance(expr, boundaries, p) <= spread
            mismatch += admission != exact
        hybrid_ms = (perf_counter() - hybrid_t0) * 1000
        assert not mismatch
        out[spread] = dict(uncertain=uncertain, exact_ms=round(exact_ms,2),
                           hybrid_ms=round(hybrid_ms,2))
    result = dict(leaves=leaves, cells=len(cells), segments=len(boundaries),
                  prep_ms=round(prep_ms,2), python_traced_peak_bytes=peak,
                  queries=len(points), spreads=out)
    assert isfinite(result["prep_ms"])
    return result


def demonstrate_budget_gate(leaves:int, max_cells:int, max_segments:int):
    """Fail closed by a cheap *upper* bound BEFORE allocating an arrangement.

    This is intentionally conservative and may reject feasible shapes. A
    future implementation needs its own exact cost/coverage eligibility law.
    """
    # Each rectangle contributes <= two unique edges along each axis.
    # Therefore the coordinate arrangement has <= (2N-1)^2 cells.
    max_grid_cells = (2*leaves - 1)**2
    # Each occupied cell may have at most four unshared boundary segments.
    max_boundary_segments = 4 * max_grid_cells
    admitted = max_grid_cells <= max_cells and max_boundary_segments <= max_segments
    return dict(leaves=leaves, cell_bound=max_grid_cells,
                segment_bound=max_boundary_segments, admitted=admitted)


if __name__ == "__main__":
    print("RR86 cold/preparation and warm query scaling (Python stdlib only)")
    data = []
    for n in (4,8,16,32,48,64):
        r = measure(n)
        data.append(r)
        print(r)
    # Validate monotonic observed growth on adversarial complexity,
    # rather than attributing any one machine's elapsed time to RunenGPU.
    assert all(data[i]["cells"] < data[i+1]["cells"] for i in range(len(data)-1))
    assert all(data[i]["segments"] < data[i+1]["segments"] for i in range(len(data)-1))
    for n in (4,8,16,32,48,64,128):
        gate = demonstrate_budget_gate(n, 10_000, 40_000)
        print("ILLUSTRATIVE preflight upper-bound gate",gate)
    assert not demonstrate_budget_gate(64, 10_000,40_000)["admitted"]
    print("PASS: scaling/certified-query/bounded conservative preflight")
