use runen_render::scene::{RenderSceneCommitError, RenderSceneStore, RenderSceneUpdate};

#[test]
fn duplicate_same_kind_operations_are_same_object_conflicts() {
    let mut scene = RenderSceneStore::new();
    let insert_id = scene
        .allocate_object_id()
        .expect("renderer identity should allocate");

    let mut duplicate_insert = RenderSceneUpdate::new();
    duplicate_insert.insert(insert_id).insert(insert_id);
    assert_eq!(
        scene.commit(duplicate_insert),
        Err(RenderSceneCommitError::ConflictingOperations {
            object_id: insert_id,
        })
    );
    assert!(scene.snapshot().is_empty());

    let mut insert = RenderSceneUpdate::new();
    insert.insert(insert_id);
    scene
        .commit(insert)
        .expect("test setup insertion should succeed");
    let before_duplicate_remove = scene.snapshot();

    let mut duplicate_remove = RenderSceneUpdate::new();
    duplicate_remove.remove(insert_id).remove(insert_id);
    assert_eq!(
        scene.commit(duplicate_remove),
        Err(RenderSceneCommitError::ConflictingOperations {
            object_id: insert_id,
        })
    );
    assert_eq!(scene.snapshot(), before_duplicate_remove);
}
