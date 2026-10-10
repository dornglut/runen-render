#!/usr/bin/env python3
"""RR86 boundary BVH versus unindexed exact oracle on identical CSG queries.

Python standard library, research only. Input is the already exact orthogonal
exposed-boundary segment list; this is not an accelerated general F1 renderer.
"""
from __future__ import annotations

from math import hypot, sqrt
from time import perf_counter
from dataclasses import dataclass
from orthogonal_csg_reference import (
    arrangement, exact_signed_distance, inside, naive_csg_field,
    partial_certificate, segment_distance,
)
from preflight_scaling import crosshatch


def box_of_segment(edge):
    a, b = edge
    return (min(a[0],b[0]), min(a[1],b[1]), max(a[0],b[0]), max(a[1],b[1]))


def union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


@dataclass(frozen=True)
class Bvh:
    box: tuple
    left: 'Bvh | None'
    right: 'Bvh | None'
    edges: tuple


def build_bvh(segments, leaf_size=8):
    assert 0 < leaf_size <= 32
    if not segments:
        raise ValueError('empty region lacks a finite exposed-boundary metric')
    def build(edges):
        box=union_box([box_of_segment(e) for e in edges])
        if len(edges) <= leaf_size:
            return Bvh(box, None,None, tuple(edges))
        # Sort by segment center along longest span (deterministic).
        axis = int((box[3]-box[1]) > (box[2]-box[0]))
        edges = sorted(edges, key=lambda e: (e[0][axis]+e[1][axis], e[0], e[1]))
        mid = len(edges)//2
        return Bvh(box, build(edges[:mid]),build(edges[mid:]), tuple())
    return build(list(segments))


def squared_box_lower_bound(p,box):
    dx=max(box[0]-p[0],0.,p[0]-box[2])
    dy=max(box[1]-p[1],0.,p[1]-box[3])
    return dx*dx+dy*dy


def nearest_bvh(p,root):
    best2=float('inf')
    visited_nodes=0
    segments_evaluated=0
    stack=[root]
    while stack:
        node=stack.pop()
        visited_nodes+=1
        lower=squared_box_lower_bound(p,node.box)
        if lower > best2:
            continue
        if node.edges:
            for seg in node.edges:
                segments_evaluated+=1
                d=segment_distance(p,seg)
                best2=min(best2,d*d)
        else:
            assert node.left is not None and node.right is not None
            l=node.left
            r=node.right
            dl=squared_box_lower_bound(p,l.box)
            dr=squared_box_lower_bound(p,r.box)
            # Process closer child first: push farther child first.
            if dl<=dr:
                stack.extend((r,l))
            else:
                stack.extend((l,r))
    return sqrt(best2), visited_nodes,segments_evaluated


def signed_indexed(expr,root,p):
    d,vis,evals=nearest_bvh(p,root)
    return (-d if inside(expr,p) else d),vis,evals


def run(n):
    expr=crosshatch(n)
    _,edges=arrangement(expr)
    t0=perf_counter()
    root=build_bvh(edges)
    build_ms=1000*(perf_counter()-t0)
    probes=[(-1.03+i*2.06/28,-1.03+j*2.06/28)
            for j in range(29) for i in range(29)]
    t0=perf_counter()
    direct=[exact_signed_distance(expr,edges,p) for p in probes]
    scan_ms=1000*(perf_counter()-t0)
    t0=perf_counter()
    indexed=[signed_indexed(expr,root,p) for p in probes]
    indexed_ms=1000*(perf_counter()-t0)
    assert all(abs(a-b[0])<1e-12 for a,b in zip(direct,indexed))
    evals=sum(b[2] for b in indexed)
    nodes=sum(b[1] for b in indexed)
    total=len(edges)*len(probes)
    assert 0<evals<=total
    spreads=(-0.025,0.05)
    result={}
    for spread in spreads:
        t0=perf_counter()
        outcomes=[]
        fallback=0
        indexed_examined=0
        for p,d in zip(probes,direct):
            f=naive_csg_field(expr,p)
            certain=partial_certificate(f,spread)
            if certain is None:
                fallback+=1
                signed,_,sampled=signed_indexed(expr,root,p)
                indexed_examined+=sampled
                certain=signed<=spread
            outcomes.append(certain)
            assert certain==(d<=spread)
        hybrid_ms=1000*(perf_counter()-t0)
        result[spread]={"fallbacks":fallback,"hybrid_indexed_ms":round(hybrid_ms,2),
                        "exact_edges_visited_in_fallbacks":indexed_examined}
    print(dict(leaves=n,edges=len(edges),points=len(probes),
               indexed_build_ms=round(build_ms,2),brute_query_ms=round(scan_ms,2),
               indexed_query_ms=round(indexed_ms,2),
               exact_segments_tested_brute=total, exact_segments_tested_bvh=evals,
               bvh_node_visits=nodes,spread_comparisons=result))
    return scan_ms,indexed_ms,total,evals


if __name__=='__main__':
    results=[run(n) for n in (8,16,32,64)]
    assert all(samples<total for _,_,total,samples in results)
    print('PASS: exact indexed boundary and certified-field fallback agree with independent oracle')
