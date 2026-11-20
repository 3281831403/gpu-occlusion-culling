"""Red tests for the GPU occlusion-culling exercise.

Every test calls one of the stubs in ``cull/occlusion.py``.  The file is
intentionally fixed: implement the module, not the tests.
"""

import pytest

from cull.occlusion import (
    CullError,
    classify,
    cull_batch,
    early_z_ratio,
    merge_queries,
    occluded,
    project_aabb,
    pyramid_level,
    sort_front_to_back,
)


IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def test_project_aabb_clips_geometry_behind_near_plane():
    scene = {
        "objects": [
            {"id": "behind", "aabb": (-1.0, -1.0, -2.0, 1.0, 1.0, -2.0)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[10.0, 10.0], [10.0, 10.0]]],
    }
    assert project_aabb(
        scene["objects"][0]["aabb"], scene["camera"]["view_proj"]
    ) == ()


def test_pyramid_level_rounds_up_to_covering_level():
    # A 9x9 rect needs the 16-pixel texel at level 4, not the finer 8-pixel
    # texel a floor(log2(...)) implementation would choose.
    assert pyramid_level((0.0, 0.0, 9.0, 9.0), 5) == 4


def test_occluded_does_not_cull_coplanar_surface():
    scene = {
        "objects": [
            {"id": "coplanar", "aabb": (-1.0, -1.0, -1.0, 1.0, 1.0, -1.0)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[0.5, 0.5], [0.5, 0.5]]],
    }
    assert occluded((0.0, 0.0, 2.0, 2.0), 0,
                    scene["depth_pyramid"], 0.5) is False


def test_classify_conservative_keeps_unknown_visible():
    scene = {
        "objects": [
            {"id": "behind", "aabb": (-1.0, -1.0, -3.0, 1.0, 1.0, -2.0)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[10.0, 10.0], [10.0, 10.0]]],
    }
    result = classify(
        scene["objects"], scene["camera"], scene["depth_pyramid"],
        "conservative",
    )
    assert result == {
        "visible": ["behind"],
        "occluded": [],
        "unknown": ["behind"],
    }


def test_sort_front_to_back_is_stable_for_equal_distances():
    scene = {
        "objects": [
            {"id": "a", "aabb": (-2.0, -2.0, -5.0, -1.0, -1.0, -4.0)},
            {"id": "b", "aabb": (1.0, 1.0, -5.0, 2.0, 2.0, -4.0)},
            {"id": "c", "aabb": (-0.5, -0.5, -1.5, 0.5, 0.5, -0.5)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[100.0, 100.0], [100.0, 100.0]]],
    }
    ordered = sort_front_to_back(scene["objects"], scene["camera"])
    assert [obj["id"] for obj in ordered] == ["c", "a", "b"]


def test_merge_queries_deduplicates_identical_rects():
    rects = [
        (0.0, 0.0, 4.0, 4.0),
        (0.0, 0.0, 4.0, 4.0),
        (10.0, 10.0, 12.0, 12.0),
    ]
    assert merge_queries(rects) == [
        (0.0, 0.0, 4.0, 4.0),
        (10.0, 10.0, 12.0, 12.0),
    ]


def test_early_z_ratio_weights_pixels_not_objects():
    visible = [
        {"id": "v", "pixels": 90},
    ]
    occluded_objects = [
        {"id": "o1", "pixels": 10},
        {"id": "o2", "pixels": 10},
        {"id": "o3", "pixels": 10},
    ]
    assert early_z_ratio(visible, occluded_objects) == pytest.approx(0.75)


def test_cull_batch_keeps_per_frame_results_separate():
    first = {
        "objects": [
            {"id": "a", "aabb": (-0.5, -0.5, -0.8, 0.5, 0.5, -0.2)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[1000.0, 1000.0], [1000.0, 1000.0]]],
    }
    second = {
        "objects": [
            {"id": "b", "aabb": (-1.0, -1.0, -0.9, -0.2, 0.2, -0.3)},
            {"id": "c", "aabb": (0.2, -0.2, -0.9, 1.0, 1.0, -0.3)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[1000.0, 1000.0], [1000.0, 1000.0]]],
    }
    result = cull_batch([first, second], {"mode": "conservative"})
    assert result["frames"][0]["visible"] == ["a"]
    assert result["frames"][1]["visible"] == ["b", "c"]


def test_project_aabb_rejects_degenerate_aabb():
    scene = {
        "objects": [
            {"id": "bad", "aabb": (0.0, 0.0, 0.0, -1.0, 1.0, 1.0)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[10.0, 10.0], [10.0, 10.0]]],
    }
    with pytest.raises(CullError):
        project_aabb(scene["objects"][0]["aabb"],
                     scene["camera"]["view_proj"])


def test_pyramid_level_clamps_when_levels_are_exhausted():
    # ceil(log2(1024)) is 10, but only three levels exist.
    assert pyramid_level((0.0, 0.0, 1024.0, 1024.0), 3) == 2


def test_classify_empty_scene_returns_three_empty_lists():
    scene = {
        "objects": [],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [],
    }
    assert classify(
        scene["objects"], scene["camera"], scene["depth_pyramid"],
        "conservative",
    ) == {"visible": [], "occluded": [], "unknown": []}


def test_classify_rejects_unknown_mode():
    scene = {
        "objects": [
            {"id": "a", "aabb": (-0.5, -0.5, -0.8, 0.5, 0.5, -0.2)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[1000.0, 1000.0], [1000.0, 1000.0]]],
    }
    with pytest.raises(CullError):
        classify(
            scene["objects"], scene["camera"], scene["depth_pyramid"],
            "balanced",
        )


def test_conservative_mode_does_not_drop_unknown():
    scene = {
        "objects": [
            {"id": "near", "aabb": (-0.5, -0.5, -0.8, 0.5, 0.5, -0.6)},
            {"id": "behind", "aabb": (-1.0, -1.0, -3.0, 1.0, 1.0, -2.0)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[1000.0, 1000.0], [1000.0, 1000.0]]],
    }
    result = classify(
        scene["objects"], scene["camera"], scene["depth_pyramid"],
        "conservative",
    )
    assert result["visible"] == ["near", "behind"]
    assert result["unknown"] == ["behind"]
    assert result["occluded"] == []


def test_sort_front_to_back_preserves_tie_order():
    scene = {
        "objects": [
            {"id": "a", "aabb": (-1.5, -0.5, -0.5, -0.5, 0.5, 0.5)},
            {"id": "b", "aabb": (0.5, -0.5, -0.5, 1.5, 0.5, 0.5)},
            {"id": "c", "aabb": (-0.5, 0.5, -0.5, 0.5, 1.5, 0.5)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[100.0, 100.0], [100.0, 100.0]]],
    }
    ordered = sort_front_to_back(scene["objects"], scene["camera"])
    assert [obj["id"] for obj in ordered] == ["a", "b", "c"]


def test_cull_batch_preserves_frame_order_and_rolls_back_on_invalid_frame():
    frames = [
        {
            "objects": [
                {"id": "first", "aabb": (-0.6, -0.6, -0.8, 0.6, 0.6, -0.2)},
            ],
            "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
            "depth_pyramid": [[[1000.0, 1000.0], [1000.0, 1000.0]]],
        },
        {
            "objects": [
                {"id": "second", "aabb": (-0.7, -0.7, -0.8, 0.7, 0.7, -0.2)},
            ],
            "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
            "depth_pyramid": [[[1000.0, 1000.0], [1000.0, 1000.0]]],
        },
        {
            "objects": [
                {"id": "third", "aabb": (-0.8, -0.8, -0.8, 0.8, 0.8, -0.2)},
            ],
            "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
            "depth_pyramid": [[[1000.0, 1000.0], [1000.0, 1000.0]]],
        },
    ]
    result = cull_batch(frames, {"mode": "aggressive"})
    assert [item["frame_index"] for item in result["frames"]] == [0, 1, 2]
    assert [
        item["visible"] for item in result["frames"]
    ] == [["first"], ["second"], ["third"]]

    bad_frame = {
        "objects": [
            {"id": "bad", "aabb": (0.0, 0.0, 0.0, -1.0, 1.0, 1.0)},
        ],
        "camera": {"position": (0.0, 0.0, 0.0), "view_proj": IDENTITY},
        "depth_pyramid": [[[1000.0, 1000.0], [1000.0, 1000.0]]],
    }
    with pytest.raises(CullError):
        cull_batch([frames[0], bad_frame], {"mode": "aggressive"})
