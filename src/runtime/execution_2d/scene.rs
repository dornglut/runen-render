//! Private F3E painter-tree admission in one immutable F1 composition domain.
//!
//! This traversal does not paint or create a second semantic scene. It retains
//! borrowed F1 item references, source order, and exactly composed transforms.
//! The private execution compiler consumes it; no identity crosses the public
//! RunenGPU work/evidence boundary.

use crate::composition_2d::{
    Render2dAffineTransform, Render2dComposition, Render2dEntry, Render2dGroup, Render2dItem,
};
use crate::execution_2d::Render2dExecutionError;

// These are bounded compiler traversal resources, not public scene limits.
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
            Err(Render2dExecutionError::Gpu {
                stage: "F3E nested affine admission",
                detail: format!("entry path {path:?} exceeds finite affine representation"),
            })
        }
    }

    pub(super) const fn coefficients(self) -> [f64; 6] {
        self.0
    }
}

/// One lexically ordered semantic occurrence; grouping changes the
/// compositing operation, never the source representation's ownership.
#[derive(Debug)]
pub(super) enum Event<'a> {
    Item {
        path: Vec<usize>,
        item: &'a Render2dItem,
        to_root: Affine,
    },
    BeginGroup {
        path: Vec<usize>,
        has_shadows: bool,
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
        parent_to_root: Affine,
    },
    EndGroup,
}

fn limit(path: &[usize], problem: &'static str) -> Render2dExecutionError {
    Render2dExecutionError::Gpu {
        stage: "F3E bounded semantic traversal",
        detail: format!("entry path {path:?}: {problem}"),
    }
}

/// A no-paint, no-effect subtree must never be forced through physical
/// transform/clip admission: mathematically its contribution is transparent.
/// The walk is iterative so even hostile nesting does not grow the call stack.
fn is_structurally_empty(group: &Render2dGroup) -> bool {
    let mut pending = vec![group];
    while let Some(current) = pending.pop() {
        if !current.shadows().is_empty() {
            return false;
        }
        for entry in current.entries() {
            match entry {
                Render2dEntry::Item(_) => return false,
                Render2dEntry::Group(child) => pending.push(child),
            }
        }
    }
    true
}

fn validate_clips(
    owner_clips: &[crate::composition_2d::Render2dClip],
    parent_to_root: Affine,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    for clip in owner_clips {
        // A clip is in its owner's *parent* space, never transformed by
        // the owning item's or group's local_to_parent a second time.
        parent_to_root.compose(Affine::from_source(clip.clip_to_parent()), path)?;
    }
    Ok(())
}

/// Builds a bounded, nonrecursive event stream. No clipping, blending, resource
/// identity or placement policy is invented here: F1 remains the sole source.
pub(super) fn analyze(
    composition: &Render2dComposition,
) -> Result<Plan<'_>, Render2dExecutionError> {
    let mut pending = Vec::new();
    for (index, entry) in composition.root_entries().iter().enumerate().rev() {
        pending.push(Pending::Visit {
            entry,
            path: vec![index],
            parent_to_root: Affine::IDENTITY,
        });
    }

    let mut events = Vec::new();
    let mut visits = 0_usize;
    while let Some(next) = pending.pop() {
        match next {
            Pending::EndGroup => events.push(Event::EndGroup),
            Pending::Visit {
                entry,
                path,
                parent_to_root,
            } => {
                visits = visits
                    .checked_add(1)
                    .ok_or_else(|| limit(&path, "visit count overflow"))?;
                if visits > MAX_VISITED_ENTRIES {
                    return Err(limit(&path, "semantic entry count exceeds compiler budget"));
                }
                if path.len() > MAX_DEPTH {
                    return Err(limit(&path, "nested group depth exceeds compiler budget"));
                }
                match entry {
                    Render2dEntry::Item(item) => {
                        validate_clips(item.clips(), parent_to_root, &path)?;
                        let to_root = parent_to_root
                            .compose(Affine::from_source(item.local_to_parent()), &path)?;
                        events.push(Event::Item {
                            path,
                            item,
                            to_root,
                        });
                    }
                    Render2dEntry::Group(group) => {
                        if is_structurally_empty(group) {
                            continue;
                        }
                        validate_clips(group.clips(), parent_to_root, &path)?;
                        let to_root = parent_to_root
                            .compose(Affine::from_source(group.local_to_parent()), &path)?;
                        events.push(Event::BeginGroup {
                            path: path.clone(),
                            has_shadows: !group.shadows().is_empty(),
                        });
                        pending.push(Pending::EndGroup);
                        for (index, child) in group.entries().iter().enumerate().rev() {
                            let mut child_path = path.clone();
                            child_path.push(index);
                            pending.push(Pending::Visit {
                                entry: child,
                                path: child_path,
                                parent_to_root: to_root,
                            });
                        }
                    }
                }
            }
        }
    }
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
