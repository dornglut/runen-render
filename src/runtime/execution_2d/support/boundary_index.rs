//! Bounded reusable spatial index over the TRUE exposed neutral-union boundary.
//!
//! Only source-neutral segments certified by `union_exterior_boundary` enter this
//! tree. An index box is an exclusion/lower-distance certificate, NEVER a
//! substitute region, a sampled SDF or an erosion membership oracle.
use super::{
    UnionBoundaryEdge, charge_shadow_preparation_work, distance_squared_to_segment, mask_failure,
};
use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};

const LEAF_SIZE: usize = 2;
const MAX_INDEX_BYTES: usize = 16 * 1024 * 1024;
const MAX_GLOBAL_QUERY_WORK: usize = 16_777_216;

#[derive(Clone, Copy)]
struct Node {
    bounds: [f64; 4],
    range: [usize; 2],
    children: Option<[usize; 2]>,
}

pub(super) struct ExposedBoundaryIndex<'a> {
    edges: &'a [UnionBoundaryEdge],
    order: Vec<usize>,
    nodes: Vec<Node>,
}

fn edge_bounds(edge: UnionBoundaryEdge) -> [f64; 4] {
    [
        edge[0][0].min(edge[1][0]),
        edge[0][1].min(edge[1][1]),
        edge[0][0].max(edge[1][0]),
        edge[0][1].max(edge[1][1]),
    ]
}

fn box_gap(a: [f64; 4], b: [f64; 4]) -> f64 {
    let dx = (a[0] - b[2]).max(b[0] - a[2]).max(0.0);
    let dy = (a[1] - b[3]).max(b[1] - a[3]).max(0.0);
    dx.hypot(dy)
}

fn point_gap_squared(p: [f64; 2], bounds: [f64; 4]) -> f64 {
    let dx = (bounds[0] - p[0]).max(p[0] - bounds[2]).max(0.0);
    let dy = (bounds[1] - p[1]).max(p[1] - bounds[3]).max(0.0);
    dx.mul_add(dx, dy * dy)
}

fn resource(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail)
}

fn charge_query(work: &mut usize, path: &[usize]) -> Result<(), Render2dExecutionError> {
    let next = work
        .checked_add(1)
        .ok_or_else(|| resource(path, "indexed neutral-boundary query work overflow"))?;
    if next > MAX_GLOBAL_QUERY_WORK {
        return Err(resource(
            path,
            "indexed neutral-boundary query exceeds bounded whole-scene work",
        ));
    }
    *work = next;
    Ok(())
}

impl<'a> ExposedBoundaryIndex<'a> {
    pub(super) fn new(
        edges: &'a [UnionBoundaryEdge],
        path: &[usize],
        work: &mut usize,
    ) -> Result<Self, Render2dExecutionError> {
        if edges.is_empty() || !edges.iter().flatten().flatten().all(|v| v.is_finite()) {
            return Err(mask_failure(
                path,
                Render2dSampleSpaceError::PrecisionLimit,
                "indexed neutral-boundary payload is empty or nonfinite",
            ));
        }
        // The whole tree, its permutation and conservative build traversal
        // are admitted BEFORE reserving any of its temporary host storage.
        let maximum_nodes = edges
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_sub(1))
            .ok_or_else(|| resource(path, "indexed neutral-boundary node count overflow"))?;
        let bytes = maximum_nodes
            .checked_mul(std::mem::size_of::<Node>())
            .and_then(|n| {
                edges
                    .len()
                    .checked_mul(std::mem::size_of::<usize>())
                    .and_then(|order| n.checked_add(order))
            })
            .ok_or_else(|| resource(path, "indexed neutral-boundary bytes overflow"))?;
        if bytes > MAX_INDEX_BYTES {
            return Err(resource(
                path,
                "indexed neutral-boundary peak storage exceeds bounded admission",
            ));
        }
        let levels = (usize::BITS - edges.len().leading_zeros()) as usize + 1;
        // Every level scans its participating edges to form node boxes;
        // stable axis partition sorts can do O(n log(n)^2) comparisons.
        let build_work = edges
            .len()
            .checked_mul(levels)
            .and_then(|n| n.checked_mul(levels))
            .ok_or_else(|| resource(path, "indexed neutral-boundary build work overflow"))?;
        charge_shadow_preparation_work(work, build_work, path)?;
        let mut order = Vec::new();
        order.try_reserve_exact(edges.len()).map_err(|_| {
            resource(
                path,
                "indexed neutral-boundary permutation allocation failed",
            )
        })?;
        order.extend(0..edges.len());
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(maximum_nodes)
            .map_err(|_| resource(path, "indexed neutral-boundary node allocation failed"))?;
        let mut index = Self {
            edges,
            order,
            nodes,
        };
        index.build(0, edges.len());
        Ok(index)
    }

    fn build(&mut self, start: usize, end: usize) -> usize {
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for &edge in &self.order[start..end] {
            let b = edge_bounds(self.edges[edge]);
            bounds[0] = bounds[0].min(b[0]);
            bounds[1] = bounds[1].min(b[1]);
            bounds[2] = bounds[2].max(b[2]);
            bounds[3] = bounds[3].max(b[3]);
        }
        let at = self.nodes.len();
        self.nodes.push(Node {
            bounds,
            range: [start, end],
            children: None,
        });
        if end - start > LEAF_SIZE {
            let axis = usize::from(bounds[3] - bounds[1] > bounds[2] - bounds[0]);
            self.order[start..end].sort_unstable_by(|&i, &j| {
                let a = edge_bounds(self.edges[i]);
                let b = edge_bounds(self.edges[j]);
                (a[axis] + a[axis + 2])
                    .total_cmp(&(b[axis] + b[axis + 2]))
                    .then(i.cmp(&j))
            });
            let mid = start + (end - start) / 2;
            let left = self.build(start, mid);
            let right = self.build(mid, end);
            self.nodes[at].children = Some([left, right]);
        }
        at
    }

    /// False is only "uncertain". True proves the entire cell is farther
    /// than radius from every exposed exterior edge, including hole edges.
    pub(super) fn far_from_exterior(
        &self,
        cell: [f64; 4],
        radius: f64,
        work: &mut usize,
        path: &[usize],
    ) -> Result<bool, Render2dExecutionError> {
        self.far_node(0, cell, radius, work, path)
    }

    fn far_node(
        &self,
        at: usize,
        cell: [f64; 4],
        radius: f64,
        work: &mut usize,
        path: &[usize],
    ) -> Result<bool, Render2dExecutionError> {
        charge_query(work, path)?;
        let node = self.nodes[at];
        if box_gap(node.bounds, cell) > radius {
            return Ok(true);
        }
        if let Some([left, right]) = node.children {
            return Ok(self.far_node(left, cell, radius, work, path)?
                && self.far_node(right, cell, radius, work, path)?);
        }
        for &index in &self.order[node.range[0]..node.range[1]] {
            charge_query(work, path)?;
            if box_gap(edge_bounds(self.edges[index]), cell) <= radius {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Exact segment distance; tree bounds only eliminate segments which
    /// cannot improve the currently proven minimum.
    pub(super) fn nearest_distance(
        &self,
        p: [f64; 2],
        work: &mut usize,
        path: &[usize],
    ) -> Result<f64, Render2dExecutionError> {
        Ok(self.nearest_node(0, p, f64::INFINITY, work, path)?.sqrt())
    }

    fn nearest_node(
        &self,
        at: usize,
        p: [f64; 2],
        mut nearest: f64,
        work: &mut usize,
        path: &[usize],
    ) -> Result<f64, Render2dExecutionError> {
        charge_query(work, path)?;
        let node = self.nodes[at];
        if point_gap_squared(p, node.bounds) >= nearest {
            return Ok(nearest);
        }
        if let Some([left, right]) = node.children {
            let a = point_gap_squared(p, self.nodes[left].bounds);
            let b = point_gap_squared(p, self.nodes[right].bounds);
            let (first, second) = if a <= b { (left, right) } else { (right, left) };
            nearest = self.nearest_node(first, p, nearest, work, path)?;
            return self.nearest_node(second, p, nearest, work, path);
        }
        for &index in &self.order[node.range[0]..node.range[1]] {
            charge_query(work, path)?;
            let [a, b] = self.edges[index];
            nearest = nearest.min(distance_squared_to_segment(p, a, b));
        }
        Ok(nearest)
    }
}

#[cfg(test)]
mod tests {
    use super::super::union_exterior_boundary;
    use super::*;

    #[test]
    fn indexed_exterior_respects_hole_edges_and_exact_distance() {
        // Closed outer square and disjoint inner hole boundary. We verify
        // distances to the SAME original certified segments, never boxes.
        let edges = [
            [[0.0, 0.0], [8.0, 0.0]],
            [[8.0, 0.0], [8.0, 8.0]],
            [[8.0, 8.0], [0.0, 8.0]],
            [[0.0, 8.0], [0.0, 0.0]],
            [[3.0, 3.0], [5.0, 3.0]],
            [[5.0, 3.0], [5.0, 5.0]],
            [[5.0, 5.0], [3.0, 5.0]],
            [[3.0, 5.0], [3.0, 3.0]],
        ];
        let mut work = 0;
        let index = ExposedBoundaryIndex::new(&edges, &[2, 1], &mut work).unwrap();
        for point in [[4.0, 4.0], [1.0, 4.0], [2.5, 3.5], [7.5, 7.5]] {
            let reference = edges
                .iter()
                .map(|&[a, b]| distance_squared_to_segment(point, a, b))
                .fold(f64::INFINITY, f64::min)
                .sqrt();
            let actual = index.nearest_distance(point, &mut work, &[2, 1]).unwrap();
            assert!((actual - reference).abs() < 1e-12);
        }
        assert!(
            index
                .far_from_exterior([0.5, 0.5, 1.0, 1.0], 0.25, &mut work, &[2, 1])
                .unwrap()
        );
        assert!(
            !index
                .far_from_exterior([2.9, 3.2, 3.1, 3.4], 0.25, &mut work, &[2, 1])
                .unwrap()
        );
    }

    #[test]
    fn indexed_boundary_from_overlap_matches_independent_exhaustive_queries() {
        let triangles = [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [0.0, 4.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [2.0, 2.0],
            [6.0, 2.0],
            [2.0, 6.0],
        ];
        let edges = union_exterior_boundary(&triangles, &[7], None).unwrap();
        let mut work = 0;
        let index = ExposedBoundaryIndex::new(&edges, &[7], &mut work).unwrap();
        for x in -2..=9 {
            for y in -2..=9 {
                let p = [f64::from(x) + 0.37, f64::from(y) + 0.23];
                let reference = edges
                    .iter()
                    .map(|&[a, b]| distance_squared_to_segment(p, a, b))
                    .fold(f64::INFINITY, f64::min)
                    .sqrt();
                let actual = index.nearest_distance(p, &mut work, &[7]).unwrap();
                assert!((actual - reference).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn index_prunes_distant_segments_without_changing_exact_distance() {
        let edges: Vec<UnionBoundaryEdge> = (0..128)
            .map(|x| {
                let x = f64::from(x) * 10.0;
                [[x, 0.0], [x, 10.0]]
            })
            .collect();
        let mut work = 0;
        let index = ExposedBoundaryIndex::new(&edges, &[8], &mut work).unwrap();
        let before = work;
        let nearest = index.nearest_distance([0.5, 5.0], &mut work, &[8]).unwrap();
        assert!((nearest - 0.5).abs() < 1e-12);
        assert!(
            work - before < edges.len() / 2,
            "distant certified exterior must not be rescanned per query"
        );
        let before = work;
        assert!(
            index
                .far_from_exterior([0.4, 4.0, 0.6, 6.0], 0.1, &mut work, &[8])
                .unwrap()
        );
        assert!(work - before < edges.len() / 2);
    }

    #[test]
    fn index_rejects_storage_before_allocation_with_authored_path() {
        let edges = vec![[[0.0, 0.0], [1.0, 0.0]]; 200_000];
        let mut work = 0;
        assert!(matches!(
            ExposedBoundaryIndex::new(&edges, &[3, 4], &mut work),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [3, 4]
        ));
        assert_eq!(work, 0);
    }
}
