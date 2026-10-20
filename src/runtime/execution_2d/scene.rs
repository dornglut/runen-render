//! One bounded, source-neutral F1 painter tree traversal for private GPU lowering.
//!
//! Source occurrence, group isolation, clip coordinates and derived transforms
//! are retained without inventing another scene semantic authority.
use crate::composition_2d::{
    Render2dAffineTransform, Render2dComposition, Render2dEntry, Render2dGroup, Render2dItem,
};
use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};

// Private admission budgets. Every authored entry is counted, including
// discarded empty descendants; none can hide unbounded traversal work.
const MAX_DEPTH: usize = 64;
const MAX_VISITED_ENTRIES: usize = 1_048_576;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Affine([f64; 6]);

impl Affine {
    pub(super) const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    pub(super) fn from_source(value: Render2dAffineTransform) -> Self {
        Self(value.components())
    }

    pub(super) fn compose(
        self,
        child: Self,
        path: &[usize],
    ) -> Result<Self, Render2dExecutionError> {
        let [a, b, c, d, x, y] = self.0;
        let [e, f, g, h, u, v] = child.0;
        let product = [
            a.mul_add(e, c * f),
            b.mul_add(e, d * f),
            a.mul_add(g, c * h),
            b.mul_add(g, d * h),
            a.mul_add(u, c.mul_add(v, x)),
            b.mul_add(u, d.mul_add(v, y)),
        ];
        if product.into_iter().all(f64::is_finite) {
            Ok(Self(product))
        } else {
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path.to_vec()),
                detail: "cumulative nested affine exceeds finite representation".to_owned(),
            })
        }
    }

    pub(super) const fn coefficients(self) -> [f64; 6] {
        self.0
    }
}

/// One renderer-private occurrence. All content, opacity and clipping still
/// come from the borrowed F1 item/group; transforms are derived, not authored.
#[derive(Debug)]
pub(super) enum Event<'a> {
    Item {
        path: Vec<usize>,
        item: &'a Render2dItem,
        parent_to_root: Affine,
        to_root: Affine,
    },
    BeginGroup {
        path: Vec<usize>,
        group: &'a Render2dGroup,
        parent_to_root: Affine,
    },
    EndGroup,
}

#[derive(Debug)]
pub(super) struct Plan<'a> {
    pub(super) events: Vec<Event<'a>>,
    composition: &'a Render2dComposition,
    retained_bytes: usize,
}

impl<'a> Plan<'a> {
    pub(super) fn composition(&self) -> &'a Render2dComposition {
        self.composition
    }

    /// Renderer-owned retained occurrence storage. Original F1 payloads are
    /// borrowed; spare event/path capacities still belong to this plan.
    pub(super) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }
}

/// A borrowed lexical occurrence, before any cumulative physical affine.
/// Paths borrow the cursor and are valid until its next advance.
pub(super) enum Occurrence<'a, 'path> {
    Item {
        path: &'path [usize],
        item: &'a Render2dItem,
    },
    BeginGroup {
        path: &'path [usize],
        group: &'a Render2dGroup,
    },
    EndGroup,
}

fn limit(path: &[usize], problem: &'static str) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::ResourceLimit,
        path: Some(path.to_vec()),
        detail: problem.to_owned(),
    }
}

fn buffer<T>(
    capacity: usize,
    path: &[usize],
    problem: &'static str,
) -> Result<Vec<T>, Render2dExecutionError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|_| limit(path, problem))?;
    if values.capacity() > capacity {
        return Err(limit(path, problem));
    }
    Ok(values)
}

struct CursorFrame<'a> {
    entries: &'a [Render2dEntry],
    next: usize,
    closing: Option<&'a Render2dGroup>,
}

/// One child is scheduled at a time; neither width nor cloned subtree sharing
/// increases the frontier. Inert descendants are still visited and admitted.
pub(super) struct OccurrenceCursor<'a> {
    frontier: Vec<CursorFrame<'a>>,
    path: Vec<usize>,
    pop_item: bool,
    visited: usize,
    failed: bool,
}

impl<'a> OccurrenceCursor<'a> {
    /// Fixed frontier/path capacities coexist only while the cursor is alive.
    pub(super) const fn storage_bytes() -> usize {
        size_of::<Self>() + (MAX_DEPTH + 1) * (size_of::<CursorFrame<'a>>() + size_of::<usize>())
    }

    pub(super) fn new(
        composition: &'a Render2dComposition,
        admit: &mut impl FnMut(usize, &[usize]) -> Result<(), Render2dExecutionError>,
    ) -> Result<Self, Render2dExecutionError> {
        // Two empty buffer reservations/capacity checks, the root-frame store,
        // scalar/header initialization and return bookkeeping. There is no
        // width-dependent initialization or copy in either empty allocation.
        admit(128, &[])?;
        // One root frame plus at most MAX_DEPTH group frames. The extra path
        // slot retains the exact first over-depth owner without growing.
        let mut frontier = buffer(MAX_DEPTH + 1, &[], "scene cursor allocation failed")?;
        let path = buffer(MAX_DEPTH + 1, &[], "scene cursor path allocation failed")?;
        frontier.push(CursorFrame {
            entries: composition.root_entries(),
            next: 0,
            closing: None,
        });
        Ok(Self {
            frontier,
            path,
            pop_item: false,
            visited: 0,
            failed: false,
        })
    }

    pub(super) fn next(
        &mut self,
        admit: &mut impl FnMut(usize, &[usize]) -> Result<(), Render2dExecutionError>,
    ) -> Result<Option<Occurrence<'a, '_>>, Render2dExecutionError> {
        if self.failed {
            return Ok(None);
        }
        // Invocation/terminal checks and optional previous-item path pop.
        // Rejection is terminal and changes no unpaid traversal state.
        if let Err(error) = admit(32, &self.path) {
            self.failed = true;
            return Err(error);
        }
        if self.pop_item {
            self.path.pop();
            self.pop_item = false;
        }
        loop {
            // Bound the frame lookup/exhaustion checks; either closing-frame
            // and path pops/activity dispatch, or child index/load/increment,
            // path push, visit/depth checks and one item/group dispatch plus
            // a fixed-size frame store. Pay EACH iteration: next() can skip
            // many inert authored entries without yielding an occurrence.
            // Its owner is the current containing lexical path, before the
            // next child's authored data or index has been inspected.
            if let Err(error) = admit(64, &self.path) {
                self.failed = true;
                return Err(error);
            }
            let Some(frame) = self.frontier.last_mut() else {
                return Ok(None);
            };
            if frame.next == frame.entries.len() {
                let closing = frame.closing;
                self.frontier.pop();
                if let Some(group) = closing {
                    self.path.pop();
                    if group.is_structurally_active() {
                        return Ok(Some(Occurrence::EndGroup));
                    }
                }
                continue;
            }
            let index = frame.next;
            let entry = &frame.entries[index];
            frame.next += 1;
            self.path.push(index);
            // A failed cursor is terminal, so the previous count is at most
            // MAX_VISITED_ENTRIES and this increment cannot overflow.
            self.visited += 1;
            if self.visited > MAX_VISITED_ENTRIES {
                self.failed = true;
                return Err(limit(
                    &self.path,
                    "semantic entry count exceeds compiler budget",
                ));
            }
            if self.path.len() > MAX_DEPTH {
                self.failed = true;
                return Err(limit(
                    &self.path,
                    "nested group depth exceeds compiler budget",
                ));
            }
            match entry {
                Render2dEntry::Item(item) => {
                    self.pop_item = true;
                    return Ok(Some(Occurrence::Item {
                        path: &self.path,
                        item,
                    }));
                }
                Render2dEntry::Group(group) => {
                    self.frontier.push(CursorFrame {
                        entries: group.entries(),
                        next: 0,
                        closing: Some(group),
                    });
                    if group.is_structurally_active() {
                        return Ok(Some(Occurrence::BeginGroup {
                            path: &self.path,
                            group,
                        }));
                    }
                }
            }
        }
    }
}

fn copy_path(path: &[usize]) -> Result<Vec<usize>, Render2dExecutionError> {
    let mut owned = buffer(path.len(), path, "scene occurrence path allocation failed")?;
    owned.extend_from_slice(path);
    Ok(owned)
}

fn validate_clips(
    owner_clips: &[crate::composition_2d::Render2dClip],
    parent_to_root: Affine,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    for clip in owner_clips {
        // The clip is in the owner's immediate parent space: NEVER multiply
        // the owner's own local-to-parent transform into its own clip.
        parent_to_root.compose(Affine::from_source(clip.clip_to_parent()), path)?;
    }
    Ok(())
}

/// Validate every authored occurrence before physical affine admission. A second
/// pass over the SAME borrowed cursor emits the active lexical stream, keeping
/// structural-limit errors ahead of affine errors without retaining a raw IR.
/// Neither pass copies paths or allocates a width-sized scheduling queue.
pub(super) fn analyze(
    composition: &Render2dComposition,
) -> Result<Plan<'_>, Render2dExecutionError> {
    let event_count = {
        // Ordinary no-shadow scenes retain their existing entry/depth policy;
        // this traversal does not impose the F3F numerical work ceiling.
        let mut cursor = OccurrenceCursor::new(composition, &mut |_, _| Ok(()))?;
        let mut count = 0_usize;
        while cursor.next(&mut |_, _| Ok(()))?.is_some() {
            count = count
                .checked_add(1)
                .ok_or_else(|| limit(&[], "scene event count overflow"))?;
        }
        count
    };
    let mut events = buffer(event_count, &[], "scene event allocation failed")?;
    let mut retained_bytes = events
        .capacity()
        .checked_mul(size_of::<Event<'_>>())
        .and_then(|bytes| bytes.checked_add(size_of::<Plan<'_>>()))
        .ok_or_else(|| limit(&[], "scene retained storage overflow"))?;
    let mut frames = buffer(MAX_DEPTH + 1, &[], "scene affine frame allocation failed")?;
    frames.push(Affine::IDENTITY);
    let mut cursor = OccurrenceCursor::new(composition, &mut |_, _| Ok(()))?;
    while let Some(event) = cursor.next(&mut |_, _| Ok(()))? {
        match event {
            Occurrence::Item { path, item } => {
                let parent_to_root = *frames.last().expect("root affine frame");
                validate_clips(item.clips(), parent_to_root, path)?;
                let to_root =
                    parent_to_root.compose(Affine::from_source(item.local_to_parent()), path)?;
                let owned_path = copy_path(path)?;
                retained_bytes = retained_bytes
                    .checked_add(owned_path.capacity() * size_of::<usize>())
                    .ok_or_else(|| limit(path, "scene retained storage overflow"))?;
                events.push(Event::Item {
                    path: owned_path,
                    item,
                    parent_to_root,
                    to_root,
                });
            }
            Occurrence::BeginGroup { path, group } => {
                let parent_to_root = *frames.last().expect("root affine frame");
                validate_clips(group.clips(), parent_to_root, path)?;
                let to_root =
                    parent_to_root.compose(Affine::from_source(group.local_to_parent()), path)?;
                let owned_path = copy_path(path)?;
                retained_bytes = retained_bytes
                    .checked_add(owned_path.capacity() * size_of::<usize>())
                    .ok_or_else(|| limit(path, "scene retained storage overflow"))?;
                events.push(Event::BeginGroup {
                    path: owned_path,
                    group,
                    parent_to_root,
                });
                frames.push(to_root);
            }
            Occurrence::EndGroup => {
                frames.pop().expect("balanced affine group frames");
                events.push(Event::EndGroup);
            }
        }
    }
    debug_assert_eq!(frames.len(), 1);
    debug_assert_eq!(events.len(), event_count);
    Ok(Plan {
        events,
        composition,
        retained_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::{
        Render2dBrush, Render2dClip, Render2dColorRgba8, Render2dDropShadow, Render2dGroup,
        Render2dOpacity, Render2dPrimitive, Render2dRect, Render2dShape,
    };

    fn affine(values: [f64; 6]) -> Render2dAffineTransform {
        Render2dAffineTransform::new(
            values[0], values[1], values[2], values[3], values[4], values[5],
        )
        .unwrap()
    }

    fn item(transform: Render2dAffineTransform) -> Render2dEntry {
        Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 1.0, 1.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::WHITE),
            },
            transform,
            Vec::new(),
            Render2dOpacity::OPAQUE,
        ))
    }

    fn group(
        entries: Vec<Render2dEntry>,
        transform: Render2dAffineTransform,
        clips: Vec<Render2dClip>,
    ) -> Render2dEntry {
        Render2dEntry::group(Render2dGroup::new(
            entries,
            transform,
            clips,
            Render2dOpacity::OPAQUE,
            Vec::new(),
        ))
    }

    #[test]
    fn nested_affine_and_traversal_budgets_preserve_machine_readable_paths() {
        let overflow = Affine([1e308, 0.0, 0.0, 1e308, 0.0, 0.0]);
        assert!(matches!(
            overflow.compose(overflow, &[2, 3, 5]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [2, 3, 5]
        ));
        assert!(matches!(
            limit(&[0, 4], "bounded traversal"),
            Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            } if path == [0, 4]
        ));
    }

    #[test]
    fn grouping_is_preorder_and_child_transform_composes_in_parent_order() {
        let tree = Render2dComposition::new(vec![
            item(Render2dAffineTransform::IDENTITY),
            group(
                vec![
                    item(affine([1.0, 0.0, 0.0, 1.0, 1.0, 0.0])),
                    group(
                        vec![item(affine([1.0, 0.0, 0.0, 1.0, 2.0, 0.0]))],
                        affine([1.0, 0.0, 0.0, 1.0, 1.0, 0.0]),
                        Vec::new(),
                    ),
                ],
                affine([2.0, 0.0, 0.0, 1.0, 4.0, 0.0]),
                Vec::new(),
            ),
        ])
        .unwrap();
        let plan = analyze(&tree).unwrap();
        assert_eq!(plan.events.len(), 7);
        let paths = plan
            .events
            .iter()
            .filter_map(|event| match event {
                Event::Item { path, to_root, .. } => Some((path.clone(), *to_root)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(paths.len(), 3);
        assert_eq!(paths[0].0, vec![0]);
        assert_eq!(paths[1].0, vec![1, 0]);
        assert_eq!(paths[2].0, vec![1, 1, 0]);
        assert_eq!(paths[1].1.coefficients(), [2.0, 0.0, 0.0, 1.0, 6.0, 0.0]);
        assert_eq!(paths[2].1.coefficients(), [2.0, 0.0, 0.0, 1.0, 10.0, 0.0]);
    }

    #[test]
    fn own_parent_space_clip_never_uses_owners_local_transform() {
        let own_transform = affine([2.0, 0.0, 0.0, 1.0, 4.0, 0.0]);
        let clip = Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(10.0, 0.0, 0.5, 1.0).unwrap()),
            Render2dAffineTransform::IDENTITY,
        );
        let parent = Affine::IDENTITY;
        let clipped = parent
            .compose(Affine::from_source(clip.clip_to_parent()), &[0])
            .unwrap();
        assert_eq!(clipped.coefficients(), Affine::IDENTITY.coefficients());
        let content = parent
            .compose(Affine::from_source(own_transform), &[0])
            .unwrap();
        assert_eq!(content.coefficients(), [2.0, 0.0, 0.0, 1.0, 4.0, 0.0]);
        let tree = Render2dComposition::new(vec![group(
            vec![item(Render2dAffineTransform::IDENTITY)],
            own_transform,
            vec![clip],
        )])
        .unwrap();
        assert_eq!(analyze(&tree).unwrap().events.len(), 3);
    }

    #[test]
    fn nonfinite_composed_affine_is_a_structured_preparation_error() {
        let huge = affine([1.0e308, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let tree =
            Render2dComposition::new(vec![group(vec![item(huge)], huge, Vec::new())]).unwrap();
        assert!(matches!(
            analyze(&tree),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [0, 0]
        ));
    }

    #[test]
    fn empty_nested_groups_do_not_require_representable_physical_transforms() {
        let huge = affine([1.0e308, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let tree = Render2dComposition::new(vec![group(
            vec![group(Vec::new(), huge, Vec::new())],
            huge,
            Vec::new(),
        )])
        .unwrap();
        assert!(analyze(&tree).unwrap().events.is_empty());
    }

    #[test]
    fn inactive_descendants_are_counted_against_depth_limits() {
        let mut subtree = group(Vec::new(), Render2dAffineTransform::IDENTITY, Vec::new());
        for _ in 0..MAX_DEPTH {
            subtree = group(vec![subtree], Render2dAffineTransform::IDENTITY, Vec::new());
        }
        let tree = Render2dComposition::new(vec![subtree]).unwrap();
        assert!(matches!(
            analyze(&tree),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path.len() == MAX_DEPTH + 1 && path.iter().all(|index| *index == 0)
        ));
    }

    #[test]
    fn last_admitted_item_depth_has_balanced_bounded_frames() {
        let mut subtree = item(Render2dAffineTransform::IDENTITY);
        for _ in 1..MAX_DEPTH {
            subtree = group(vec![subtree], Render2dAffineTransform::IDENTITY, Vec::new());
        }
        let tree = Render2dComposition::new(vec![subtree]).unwrap();
        let plan = analyze(&tree).unwrap();
        assert_eq!(plan.events.len(), MAX_DEPTH * 2 - 1);
        assert!(matches!(
            &plan.events[MAX_DEPTH - 1],
            Event::Item { path, .. }
                if path.len() == MAX_DEPTH && path.iter().all(|index| *index == 0)
        ));
    }

    #[test]
    fn physical_clip_admission_uses_parent_before_owner_transform() {
        let huge = affine([1.0e308, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let clip = Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(0.0, 0.0, 1.0, 1.0).unwrap()),
            huge,
        );
        // Both owner and own clip are finite in their shared parent frame.
        // Incorrectly multiplying them together would overflow the x scale.
        let tree = Render2dComposition::new(vec![group(
            vec![item(Render2dAffineTransform::IDENTITY)],
            huge,
            vec![clip],
        )])
        .unwrap();
        assert_eq!(analyze(&tree).unwrap().events.len(), 3);
    }

    #[test]
    fn active_group_keeps_its_f1_opacity_clips_and_parent_affine() {
        let transform = affine([2.0, 0.0, 0.0, 1.0, 4.0, 0.0]);
        let child = item(Render2dAffineTransform::IDENTITY);
        let tree =
            Render2dComposition::new(vec![group(vec![child], transform, Vec::new())]).unwrap();
        let plan = analyze(&tree).unwrap();
        assert!(matches!(
            plan.events.as_slice(),
            [
                Event::BeginGroup { group, path, .. },
                Event::Item { to_root, .. },
                Event::EndGroup
            ] if group.opacity() == Render2dOpacity::OPAQUE
                && path == &[0]
                && to_root.coefficients() == [2.0, 0.0, 0.0, 1.0, 4.0, 0.0]
        ));
    }

    #[test]
    fn empty_nested_groups_elide_and_ordinary_groups_remain_ordered() {
        let empty = group(
            vec![group(
                Vec::new(),
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
            )],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
        );
        let painted = item(Render2dAffineTransform::IDENTITY);
        let tree = Render2dComposition::new(vec![empty, painted]).unwrap();
        let plan = analyze(&tree).unwrap();
        assert!(matches!(
            plan.events.as_slice(),
            [Event::Item { path, .. }] if path == &[1]
        ));
    }

    #[test]
    fn cursor_replays_shared_subtrees_in_authored_occurrence_order() {
        let shared = group(
            vec![item(Render2dAffineTransform::IDENTITY)],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
        );
        let tree = Render2dComposition::new(vec![shared.clone(), shared]).unwrap();
        let mut cursor = OccurrenceCursor::new(&tree, &mut |_, _| Ok(())).unwrap();
        for root in 0..2 {
            assert!(matches!(
                cursor.next(&mut |_, _| Ok(())).unwrap(),
                Some(Occurrence::BeginGroup { path, .. }) if path == [root]
            ));
            assert!(matches!(
                cursor.next(&mut |_, _| Ok(())).unwrap(),
                Some(Occurrence::Item { path, .. }) if path == [root, 0]
            ));
            assert!(matches!(
                cursor.next(&mut |_, _| Ok(())).unwrap(),
                Some(Occurrence::EndGroup)
            ));
        }
        assert!(cursor.next(&mut |_, _| Ok(())).unwrap().is_none());
        assert_eq!(cursor.visited, 4);
        let plan = analyze(&tree).unwrap();
        assert_eq!(plan.events.len(), 6);
        // Source payload sharing never merges two painter occurrences.
        let (Event::Item { item: first, .. }, Event::Item { item: second, .. }) =
            (&plan.events[1], &plan.events[4])
        else {
            panic!("expected repeated item occurrences");
        };
        assert!(std::ptr::eq(*first, *second));
    }

    #[test]
    fn wide_inert_subtrees_use_only_the_bounded_cursor_frontier() {
        let leaf = group(Vec::new(), Render2dAffineTransform::IDENTITY, Vec::new());
        let branch = group(
            vec![leaf; 1023],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
        );
        // Each of 1024 occurrences visits one branch plus 1023 inert leaves:
        // exactly the existing 1,048,576-entry ceiling with shared source data.
        let tree = Render2dComposition::new(vec![branch; 1024]).unwrap();
        let mut cursor = OccurrenceCursor::new(&tree, &mut |_, _| Ok(())).unwrap();
        assert!(cursor.next(&mut |_, _| Ok(())).unwrap().is_none());
        assert_eq!(cursor.visited, MAX_VISITED_ENTRIES);
        assert_eq!(cursor.frontier.capacity(), MAX_DEPTH + 1);
        assert_eq!(cursor.path.capacity(), MAX_DEPTH + 1);
        let plan = analyze(&tree).unwrap();
        assert!(plan.events.is_empty());
        assert_eq!(plan.events.capacity(), 0);
    }

    #[test]
    fn cursor_prepays_each_inert_step_and_rejection_cannot_resume() {
        let leaf = group(Vec::new(), Render2dAffineTransform::IDENTITY, Vec::new());
        let branch = group(
            vec![leaf; 1000],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
        );
        let tree = Render2dComposition::new(vec![branch]).unwrap();
        // Setup, next-call bookkeeping, then exactly three scheduler steps:
        // root group, first inert child, that child's closing frame.
        let mut fuel = 128_usize + 32 + 3 * 64;
        let mut admit = |cost, path: &[usize]| {
            fuel = fuel
                .checked_sub(cost)
                .ok_or_else(|| limit(path, "test cursor work exhausted"))?;
            Ok(())
        };
        let mut cursor = OccurrenceCursor::new(&tree, &mut admit).unwrap();
        assert!(matches!(
            cursor.next(&mut admit),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [0]
        ));
        assert_eq!(fuel, 0);
        assert_eq!(cursor.visited, 2);
        assert_eq!(cursor.path, [0]);
        assert_eq!(cursor.frontier.last().unwrap().next, 1);
        // Even a different admission callback cannot revive this rejected
        // cursor or silently give its partially consumed work a fresh budget.
        assert!(cursor.next(&mut |_, _| Ok(())).unwrap().is_none());
        assert_eq!(cursor.visited, 2);
        let mut fresh = OccurrenceCursor::new(&tree, &mut |_, _| Ok(())).unwrap();
        assert!(fresh.next(&mut |_, _| Ok(())).unwrap().is_none());
        assert_eq!(fresh.visited, 1001);
    }

    #[test]
    fn cursor_setup_rejects_before_frontier_allocation() {
        let tree = Render2dComposition::new(Vec::<Render2dEntry>::new()).unwrap();
        let mut calls = 0;
        assert!(matches!(
            OccurrenceCursor::new(&tree, &mut |cost, path| {
                calls += 1;
                assert_eq!(cost, 128);
                Err(limit(path, "test setup rejected"))
            }),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path.is_empty()
        ));
        assert_eq!(calls, 1);
    }

    #[test]
    fn plan_caches_exact_live_capacity_and_original_composition() {
        let tree = Render2dComposition::new(vec![group(
            vec![item(Render2dAffineTransform::IDENTITY)],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
        )])
        .unwrap();
        let plan = analyze(&tree).unwrap();
        let path_bytes = plan
            .events
            .iter()
            .map(|event| match event {
                Event::Item { path, .. } | Event::BeginGroup { path, .. } => {
                    path.capacity() * size_of::<usize>()
                }
                Event::EndGroup => 0,
            })
            .sum::<usize>();
        assert_eq!(
            plan.retained_bytes(),
            size_of_val(&plan) + plan.events.capacity() * size_of::<Event<'_>>() + path_bytes
        );
        assert!(std::ptr::eq(plan.composition(), &tree));
    }

    #[test]
    fn inert_authored_entries_cannot_hide_the_visit_limit() {
        let leaf = group(Vec::new(), Render2dAffineTransform::IDENTITY, Vec::new());
        let branch = group(
            vec![leaf.clone(); 1023],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
        );
        let mut roots = vec![branch; 1024];
        roots.push(leaf);
        let tree = Render2dComposition::new(roots).unwrap();
        let mut cursor = OccurrenceCursor::new(&tree, &mut |_, _| Ok(())).unwrap();
        assert!(matches!(
            cursor.next(&mut |_, _| Ok(())),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [1024]
        ));
        assert_eq!(cursor.visited, MAX_VISITED_ENTRIES + 1);
        // A failed traversal cannot resume into unadmitted authored entries.
        assert!(cursor.next(&mut |_, _| Ok(())).unwrap().is_none());
    }

    #[test]
    fn transparent_effects_and_items_keep_structural_ancestors_active() {
        let effect =
            Render2dDropShadow::new(0.0, 0.0, 0.1, 0.0, Render2dColorRgba8::new(255, 0, 0, 0))
                .unwrap();
        let effects = Render2dEntry::group(Render2dGroup::new(
            Vec::new(),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::new(0.0).unwrap(),
            vec![effect],
        ));
        let transparent_item = Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 0.0, 0.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::new(0, 0, 0, 0)),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::new(0.0).unwrap(),
        ));
        let tree = Render2dComposition::new(vec![group(
            vec![effects, transparent_item],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
        )])
        .unwrap();
        let plan = analyze(&tree).unwrap();
        assert!(matches!(
            plan.events.as_slice(),
            [
                Event::BeginGroup { path: outer, .. },
                Event::BeginGroup { path: inner, group, .. },
                Event::EndGroup,
                Event::Item { path: item, .. },
                Event::EndGroup,
            ] if outer == &[0] && inner == &[0, 0] && item == &[0, 1]
                && group.shadows().len() == 1
        ));
    }

    #[test]
    fn structural_limit_errors_precede_earlier_physical_affine_errors() {
        let huge = affine([1.0e308, 0.0, 0.0, 1.0, 0.0, 0.0]);
        let mut inert = group(Vec::new(), Render2dAffineTransform::IDENTITY, Vec::new());
        for _ in 0..MAX_DEPTH {
            inert = group(vec![inert], Render2dAffineTransform::IDENTITY, Vec::new());
        }
        let tree = Render2dComposition::new(vec![group(vec![item(huge)], huge, Vec::new()), inert])
            .unwrap();
        assert!(matches!(
            analyze(&tree),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path.len() == MAX_DEPTH + 1 && path[0] == 1
        ));
    }
}
