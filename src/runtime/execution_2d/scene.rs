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
}

enum Pending<'a> {
    Visit {
        entry: &'a Render2dEntry,
        path: Vec<usize>,
    },
    EndGroup,
}

enum RawEvent<'a> {
    Item {
        path: Vec<usize>,
        item: &'a Render2dItem,
    },
    BeginGroup {
        path: Vec<usize>,
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

/// Builds each authored occurrence exactly once, then marks contributing
/// groups in reverse lexical order. This replaces repeated subtree searches
/// with a linear postorder pass and charges even transparent/empty children
/// against explicit depth and work budgets before any GPU preparation.
///
/// Physical affine admission is a separate forward pass so genuinely inert
/// subtrees cannot spuriously reject unrepresentable, unused transforms.
pub(super) fn analyze(
    composition: &Render2dComposition,
) -> Result<Plan<'_>, Render2dExecutionError> {
    let mut pending = Vec::new();
    for (index, entry) in composition.root_entries().iter().enumerate().rev() {
        pending.push(Pending::Visit {
            entry,
            path: vec![index],
        });
    }
    let mut raw = Vec::new();
    let mut visited = 0_usize;
    while let Some(next) = pending.pop() {
        match next {
            Pending::EndGroup => raw.push(RawEvent::EndGroup),
            Pending::Visit { entry, path } => {
                visited = visited
                    .checked_add(1)
                    .ok_or_else(|| limit(&path, "visit count overflow"))?;
                if visited > MAX_VISITED_ENTRIES {
                    return Err(limit(&path, "semantic entry count exceeds compiler budget"));
                }
                if path.len() > MAX_DEPTH {
                    return Err(limit(&path, "nested group depth exceeds compiler budget"));
                }
                match entry {
                    Render2dEntry::Item(item) => raw.push(RawEvent::Item { path, item }),
                    Render2dEntry::Group(group) => {
                        raw.push(RawEvent::BeginGroup {
                            path: path.clone(),
                            group,
                        });
                        pending.push(Pending::EndGroup);
                        for (index, child) in group.entries().iter().enumerate().rev() {
                            let mut child_path = path.clone();
                            child_path.push(index);
                            pending.push(Pending::Visit {
                                entry: child,
                                path: child_path,
                            });
                        }
                    }
                }
            }
        }
    }

    // Reverse walking a balanced group stream yields a group-activity stack.
    // An authored item is always significant for source identity even when it
    // has zero opacity or emits no pixels. An authored shadow is an effect,
    // currently unsupported, and MUST NOT silently be treated as transparent.
    let mut active = vec![false; raw.len()];
    let mut group_activity = Vec::<bool>::new();
    for (index, event) in raw.iter().enumerate().rev() {
        match event {
            RawEvent::EndGroup => group_activity.push(false),
            RawEvent::Item { .. } => {
                if let Some(parent) = group_activity.last_mut() {
                    *parent = true;
                }
            }
            RawEvent::BeginGroup { group, .. } => {
                let descendants = group_activity
                    .pop()
                    .expect("balanced group events emitted by one compiler");
                let contributes = descendants || !group.shadows().is_empty();
                active[index] = contributes;
                if contributes && let Some(parent) = group_activity.last_mut() {
                    *parent = true;
                }
            }
        }
    }
    debug_assert!(group_activity.is_empty());

    // Only active groups need transform/clip admission. The group frames are
    // derived from one F1 tree and are never stored as another semantic model.
    let mut events = Vec::new();
    let mut frames = vec![Affine::IDENTITY];
    let mut group_is_active = Vec::new();
    for (index, event) in raw.into_iter().enumerate() {
        match event {
            RawEvent::Item { path, item } => {
                let parent_to_root = *frames.last().expect("root affine frame");
                validate_clips(item.clips(), parent_to_root, &path)?;
                let to_root =
                    parent_to_root.compose(Affine::from_source(item.local_to_parent()), &path)?;
                events.push(Event::Item {
                    path,
                    item,
                    parent_to_root,
                    to_root,
                });
            }
            RawEvent::BeginGroup { path, group } => {
                let parent_to_root = *frames.last().expect("root affine frame");
                let contributes = active[index];
                group_is_active.push(contributes);
                if contributes {
                    validate_clips(group.clips(), parent_to_root, &path)?;
                    let to_root = parent_to_root
                        .compose(Affine::from_source(group.local_to_parent()), &path)?;
                    events.push(Event::BeginGroup {
                        path,
                        group,
                        parent_to_root,
                    });
                    frames.push(to_root);
                } else {
                    // Its children contain no items/effects. Keep balanced
                    // frames without evaluating unused physical transforms.
                    frames.push(parent_to_root);
                }
            }
            RawEvent::EndGroup => {
                frames.pop().expect("balanced affine group frames");
                if group_is_active.pop().expect("balanced group state") {
                    events.push(Event::EndGroup);
                }
            }
        }
    }
    debug_assert_eq!(frames.len(), 1);
    debug_assert!(group_is_active.is_empty());
    Ok(Plan { events })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::{
        Render2dBrush, Render2dClip, Render2dColorRgba8, Render2dGroup, Render2dOpacity,
        Render2dPrimitive, Render2dRect, Render2dShape,
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
            Err(Render2dExecutionError::Gpu {
                stage: "F3E nested affine admission",
                ..
            })
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
            Err(Render2dExecutionError::Gpu {
                stage: "F3E bounded semantic traversal",
                ..
            })
        ));
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
}
