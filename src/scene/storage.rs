//! Private persistent scene storage.
//!
//! This module owns the structurally shared radix tree used by the public scene store/snapshot
//! semantics. The storage shape is an implementation detail and must not become part of the
//! public `runen_render::scene` contract.

use super::{RenderObjectId, RenderObjectState};
use crate::participation::RenderObjectParticipation;
use std::collections::BTreeMap;
use std::sync::Arc;
#[cfg(test)]
use std::sync::Weak;

const RADIX_BITS: usize = 4;
const RADIX_MASK: u64 = (1 << RADIX_BITS) - 1;
pub(super) const RADIX_DEPTH: usize = u64::BITS as usize / RADIX_BITS;

#[derive(Debug, Clone, PartialEq, Eq)]
struct SceneNode {
    count: usize,
    terminal: bool,
    state: Option<Arc<RenderObjectState>>,
    participation: Option<Arc<RenderObjectParticipation>>,
    children: BTreeMap<u8, Arc<SceneNode>>,
}

impl SceneNode {
    fn empty() -> Self {
        Self {
            count: 0,
            terminal: false,
            state: None,
            participation: None,
            children: BTreeMap::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SceneObjects {
    root: Arc<SceneNode>,
}

impl Default for SceneObjects {
    fn default() -> Self {
        Self {
            root: Arc::new(SceneNode::empty()),
        }
    }
}

impl SceneObjects {
    pub(super) fn len(&self) -> usize {
        self.root.count
    }

    pub(super) fn is_empty(&self) -> bool {
        self.root.count == 0
    }

    fn leaf(&self, object_id: RenderObjectId) -> Option<&SceneNode> {
        let mut node = self.root.as_ref();
        let raw = object_id.raw();
        for depth in 0..RADIX_DEPTH {
            let key = radix_digit(raw, depth);
            node = node.children.get(&key)?.as_ref();
        }
        node.terminal.then_some(node)
    }

    pub(super) fn contains(&self, object_id: RenderObjectId) -> bool {
        self.leaf(object_id).is_some()
    }

    pub(super) fn object_state(&self, object_id: RenderObjectId) -> Option<&RenderObjectState> {
        self.leaf(object_id)?.state.as_deref()
    }

    pub(super) fn object_participation(
        &self,
        object_id: RenderObjectId,
    ) -> Option<&RenderObjectParticipation> {
        self.leaf(object_id)?.participation.as_deref()
    }

    pub(super) fn inserted(
        &self,
        object_id: RenderObjectId,
        state: Option<RenderObjectState>,
    ) -> (Self, usize) {
        debug_assert!(!self.contains(object_id));
        let state = state.map(Arc::new);
        let (root, copied_nodes) = insert_node(&self.root, object_id.raw(), 0, &state);
        (Self { root }, copied_nodes)
    }

    pub(super) fn removed(&self, object_id: RenderObjectId) -> (Self, usize) {
        debug_assert!(self.contains(object_id));
        let (root, copied_nodes) = remove_node(&self.root, object_id.raw(), 0);
        (Self { root }, copied_nodes)
    }

    pub(super) fn replaced_facets(
        &self,
        object_id: RenderObjectId,
        state: Option<RenderObjectState>,
        participation: Option<Option<RenderObjectParticipation>>,
    ) -> (Self, usize) {
        debug_assert!(self.contains(object_id));
        debug_assert!(state.is_some() || participation.is_some());
        let state = state.map(Arc::new);
        let participation = participation.map(|participation| participation.map(Arc::new));
        let (root, copied_nodes) = replace_facets_node(
            &self.root,
            object_id.raw(),
            0,
            &state,
            &participation,
        );
        (Self { root }, copied_nodes)
    }

    pub(super) fn object_ids(&self) -> Vec<RenderObjectId> {
        let mut object_ids = Vec::with_capacity(self.len());
        collect_object_ids(&self.root, 0, 0, &mut object_ids);
        object_ids
    }
}

fn radix_digit(raw: u64, depth: usize) -> u8 {
    debug_assert!(depth < RADIX_DEPTH);
    let shift = (RADIX_DEPTH - depth - 1) * RADIX_BITS;
    ((raw >> shift) & RADIX_MASK) as u8
}

fn insert_node(
    node: &Arc<SceneNode>,
    raw: u64,
    depth: usize,
    state: &Option<Arc<RenderObjectState>>,
) -> (Arc<SceneNode>, usize) {
    let mut updated = node.as_ref().clone();
    updated.count += 1;

    if depth == RADIX_DEPTH {
        debug_assert!(!updated.terminal);
        updated.terminal = true;
        updated.state = state.clone();
        updated.participation = None;
        return (Arc::new(updated), 1);
    }

    let key = radix_digit(raw, depth);
    let child = node
        .children
        .get(&key)
        .cloned()
        .unwrap_or_else(|| Arc::new(SceneNode::empty()));
    let (updated_child, copied_nodes) = insert_node(&child, raw, depth + 1, state);
    updated.children.insert(key, updated_child);
    (Arc::new(updated), copied_nodes + 1)
}

fn remove_node(node: &Arc<SceneNode>, raw: u64, depth: usize) -> (Arc<SceneNode>, usize) {
    let mut updated = node.as_ref().clone();
    updated.count -= 1;

    if depth == RADIX_DEPTH {
        debug_assert!(updated.terminal);
        updated.terminal = false;
        updated.state = None;
        updated.participation = None;
        return (Arc::new(updated), 1);
    }

    let key = radix_digit(raw, depth);
    let child = node
        .children
        .get(&key)
        .expect("validated scene object removal path must exist");
    let (updated_child, copied_nodes) = remove_node(child, raw, depth + 1);
    if updated_child.count == 0 {
        updated.children.remove(&key);
    } else {
        updated.children.insert(key, updated_child);
    }
    (Arc::new(updated), copied_nodes + 1)
}

fn replace_facets_node(
    node: &Arc<SceneNode>,
    raw: u64,
    depth: usize,
    state: &Option<Arc<RenderObjectState>>,
    participation: &Option<Option<Arc<RenderObjectParticipation>>>,
) -> (Arc<SceneNode>, usize) {
    let mut updated = node.as_ref().clone();

    if depth == RADIX_DEPTH {
        debug_assert!(updated.terminal);
        if let Some(state) = state {
            updated.state = Some(state.clone());
        }
        if let Some(participation) = participation {
            updated.participation = participation.clone();
        }
        return (Arc::new(updated), 1);
    }

    let key = radix_digit(raw, depth);
    let child = node
        .children
        .get(&key)
        .expect("validated scene object replacement path must exist");
    let (updated_child, copied_nodes) =
        replace_facets_node(child, raw, depth + 1, state, participation);
    updated.children.insert(key, updated_child);
    (Arc::new(updated), copied_nodes + 1)
}

fn collect_object_ids(
    node: &Arc<SceneNode>,
    depth: usize,
    prefix: u64,
    output: &mut Vec<RenderObjectId>,
) {
    if depth == RADIX_DEPTH {
        if node.terminal {
            output.push(
                RenderObjectId::from_raw(prefix)
                    .expect("renderer scene objects must contain only non-zero object IDs"),
            );
        }
        return;
    }

    for (digit, child) in &node.children {
        collect_object_ids(
            child,
            depth + 1,
            (prefix << RADIX_BITS) | u64::from(*digit),
            output,
        );
    }
}

#[cfg(test)]
#[derive(Debug, Clone)]
pub(super) struct SceneStorageContinuity {
    root: Weak<SceneNode>,
}

#[cfg(test)]
impl SceneStorageContinuity {
    pub(super) fn same_position(&self, other: &Self) -> bool {
        Weak::ptr_eq(&self.root, &other.root)
    }
}

#[cfg(test)]
impl SceneObjects {
    pub(super) fn continuity(&self) -> SceneStorageContinuity {
        SceneStorageContinuity {
            root: Arc::downgrade(&self.root),
        }
    }
}
