"""Occlusion-culling primitives for the exercise.

Depth values are camera-space distances: larger values are farther away.
The depth pyramid stores the farthest depth per texel, so a conservative
occlusion test can reject a rectangle only when *every* covered texel is
strictly nearer than the object.
"""

from __future__ import annotations

import math


class CullError(Exception):
    """Raised for invalid culling input or an unknown culling mode."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def _validate_aabb(aabb):
    """Raise CullError unless *aabb* is finite with correctly ordered extents.

    A flat box (zero extent on an axis) is still a valid plane; only an
    inverted box (min > max) or a non-finite coordinate is rejected so a
    projection can never produce NaN.
    """
    if not isinstance(aabb, (tuple, list)) or len(aabb) != 6:
        raise CullError("AABB must be a 6-element sequence")
    if not all(_is_finite_number(v) for v in aabb):
        raise CullError("AABB contains non-finite coordinates")
    min_x, min_y, min_z, max_x, max_y, max_z = aabb
    if min_x > max_x or min_y > max_y or min_z > max_z:
        raise CullError("AABB mins must not exceed its maxs")


def _validate_matrix(view_proj):
    if not isinstance(view_proj, (tuple, list)) or len(view_proj) != 4:
        raise CullError("view_proj must be a 4x4 matrix")
    for row in view_proj:
        if not isinstance(row, (tuple, list)) or len(row) != 4:
            raise CullError("view_proj must be a 4x4 matrix")
        if not all(_is_finite_number(v) for v in row):
            raise CullError("view_proj contains non-finite entries")


def _validate_rect(rect):
    if not isinstance(rect, (tuple, list)) or len(rect) != 4:
        raise CullError("rect must be a 4-element sequence")
    x0, y0, x1, y1 = rect
    if not all(_is_finite_number(v) for v in rect):
        raise CullError("rect contains non-finite coordinates")
    if x0 >= x1 or y0 >= y1:
        raise CullError("rect must have a positive width and height")


def _aabb_center(aabb):
    min_x, min_y, min_z, max_x, max_y, max_z = aabb
    return (
        (min_x + max_x) * 0.5,
        (min_y + max_y) * 0.5,
        (min_z + max_z) * 0.5,
    )


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

# The 12 edges of a cube, indexed by the corner ordering built in
# _clip_space_corners.
_CUBE_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _clip_space_corners(aabb, view_proj):
    min_x, min_y, min_z, max_x, max_y, max_z = aabb
    points = (
        (min_x, min_y, min_z, 1.0),
        (max_x, min_y, min_z, 1.0),
        (min_x, max_y, min_z, 1.0),
        (max_x, max_y, min_z, 1.0),
        (min_x, min_y, max_z, 1.0),
        (max_x, min_y, max_z, 1.0),
        (min_x, max_y, max_z, 1.0),
        (max_x, max_y, max_z, 1.0),
    )
    corners = []
    for x, y, z, w in points:
        cx = (view_proj[0][0] * x + view_proj[0][1] * y
              + view_proj[0][2] * z + view_proj[0][3] * w)
        cy = (view_proj[1][0] * x + view_proj[1][1] * y
              + view_proj[1][2] * z + view_proj[1][3] * w)
        cz = (view_proj[2][0] * x + view_proj[2][1] * y
              + view_proj[2][2] * z + view_proj[2][3] * w)
        cw = (view_proj[3][0] * x + view_proj[3][1] * y
              + view_proj[3][2] * z + view_proj[3][3] * w)
        corners.append((cx, cy, cz, cw))
    return corners


def project_aabb(aabb, view_proj):
    """Project an axis-aligned box into a normalized screen rectangle.

    Parameters
    ----------
    aabb:
        ``(min_x, min_y, min_z, max_x, max_y, max_z)`` in world space.
    view_proj:
        A 4x4 row-major view-projection matrix as nested sequences.

    Returns
    -------
    tuple
        ``(x0, y0, x1, y1)`` with ``0 <= x0 <= x1 <= 1`` and
        ``0 <= y0 <= y1 <= 1`` in top-left-origin screen coordinates.
        A box completely behind the near plane returns ``()``.

    Notes
    -----
    Clip the box against the near plane *before* the perspective divide.
    With this module's convention a corner is in front of the near plane
    when ``z_clip >= -w_clip`` and ``w_clip > 0``.  A degenerate or
    non-finite AABB raises :class:`CullError`; producing NaN coordinates
    is never allowed.
    """
    _validate_aabb(aabb)
    _validate_matrix(view_proj)

    corners = _clip_space_corners(aabb, view_proj)

    # In front of the near plane iff w_clip > 0 and z_clip >= -w_clip.
    def in_front(corner):
        return corner[3] > 0.0 and corner[2] + corner[3] >= 0.0

    if not any(in_front(c) for c in corners):
        return ()

    # Clip every cube edge against the near plane in clip space so the
    # perspective divide never runs on a point behind the camera.
    clipped = []
    for i, j in _CUBE_EDGES:
        p, q = corners[i], corners[j]
        p_in = in_front(p)
        q_in = in_front(q)
        if p_in:
            clipped.append(p)
        if p_in != q_in:
            fp, fq = p[2] + p[3], q[2] + q[3]
            denom = fp - fq
            if denom == 0.0:
                continue
            t = fp / denom
            clipped.append(tuple(
                p[k] + t * (q[k] - p[k]) for k in range(4)
            ))
        if q_in:
            clipped.append(q)

    us = []
    vs = []
    for x, y, _z, w in clipped:
        if w <= 0.0:
            continue
        ndc_x = x / w
        ndc_y = y / w
        if not (math.isfinite(ndc_x) and math.isfinite(ndc_y)):
            raise CullError("projection produced non-finite coordinates")
        # Top-left-origin screen coordinates in the unit square.
        us.append(max(0.0, min(1.0, (ndc_x + 1.0) * 0.5)))
        vs.append(max(0.0, min(1.0, (1.0 - ndc_y) * 0.5)))

    if not us:
        return ()

    x0, x1 = min(us), max(us)
    y0, y1 = min(vs), max(vs)
    if not all(math.isfinite(v) for v in (x0, y0, x1, y1)):
        raise CullError("projection produced non-finite coordinates")
    return (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Depth pyramid
# ---------------------------------------------------------------------------

def pyramid_level(rect, levels):
    """Choose the minimal depth-pyramid level whose texel covers *rect*.

    ``rect`` is ``(x0, y0, x1, y1)`` in level-0 pixels with half-open
    bounds.  ``levels`` is the number of available pyramid levels.  Level
    ``L`` has texels of size ``2 ** L`` level-0 pixels.  Return the
    smallest ``L`` whose texel covers the rectangle's larger side, clamped
    to ``levels - 1``.  In other words, round
    ``log2(max(width, height))`` up, never down.  An undersized pyramid
    falls back to its last level instead of reading out of bounds.

    A non-positive rectangle or a non-positive ``levels`` raises
    :class:`CullError`.
    """
    _validate_rect(rect)
    if not isinstance(levels, int) or isinstance(levels, bool) or levels <= 0:
        raise CullError("levels must be a positive integer")

    x0, y0, x1, y1 = rect
    larger_side = max(x1 - x0, y1 - y0)
    if not math.isfinite(larger_side):
        raise CullError("rect has a non-finite extent")

    # ceil(log2(larger_side)) via exact integer stepping: the smallest L
    # with 2 ** L >= larger_side.
    level = 0
    texel_size = 1
    while texel_size < larger_side:
        texel_size *= 2
        level += 1

    if level > levels - 1:
        level = levels - 1
    return level


def occluded(rect, level, pyramid, depth):
    """Conservatively test whether *rect* is hidden at object *depth*.

    ``pyramid`` is a sequence indexed by level; each level is a row-major
    2-D sequence of maximum depths.  Level ``L`` has texel size
    ``2 ** L`` level-0 pixels.  The test covers every texel touched by
    the half-open rectangle (clamped to the grid) and returns ``True``
    only when all those texels are strictly nearer than ``depth``.
    Equal depth means coplanar surfaces, so the object is not occluded.
    If ``level`` is past the end, use the last available level; an empty
    pyramid cannot prove occlusion and returns ``False``.

    Boundary texels are selected conservatively from the half-open
    rectangle and clamped to the grid, so a pixel just outside the
    rectangle cannot make the result incorrectly occluded.  Invalid
    arguments raise :class:`CullError`.
    """
    _validate_rect(rect)
    if not isinstance(pyramid, (tuple, list)) or len(pyramid) == 0:
        # An empty pyramid can never prove occlusion.
        return False
    if not isinstance(level, int) or isinstance(level, bool) or level < 0:
        raise CullError("level must be a non-negative integer")
    if not _is_finite_number(depth):
        raise CullError("depth must be finite")

    # Fall back to the last available level rather than reading out of bounds.
    level = min(level, len(pyramid) - 1)
    grid = pyramid[level]
    if not isinstance(grid, (tuple, list)) or len(grid) == 0:
        return False
    rows = len(grid)
    cols = len(grid[0])
    if cols <= 0:
        return False

    texel = 2 ** level
    x0, y0, x1, y1 = rect

    # Half-open texel coverage, clamped to the grid.  Using floor for the
    # start and ceil for the end keeps boundary texels conservative.
    col_start = max(0, math.floor(x0 / texel))
    col_end = min(cols, math.ceil(x1 / texel))
    row_start = max(0, math.floor(y0 / texel))
    row_end = min(rows, math.ceil(y1 / texel))

    if col_start >= col_end or row_start >= row_end:
        return False

    for row in range(row_start, row_end):
        grid_row = grid[row]
        for col in range(col_start, col_end):
            texel_depth = grid_row[col]
            if not _is_finite_number(texel_depth):
                raise CullError("pyramid contains non-finite depth")
            # Strictly nearer only: a coplanar (equal-depth) texel must
            # not reject the object.
            if not (texel_depth < depth):
                return False
    return True


# ---------------------------------------------------------------------------
# Frame classification
# ---------------------------------------------------------------------------

def _grid_dimensions(pyramid):
    """Return (rows, cols) of level 0, or None when the pyramid is empty."""
    if not pyramid:
        return None
    level0 = pyramid[0]
    if not level0:
        return None
    rows = len(level0)
    cols = len(level0[0])
    return rows, cols


def classify(objects, camera, pyramid, mode):
    """Classify objects as visible, occluded, or unknown.

    ``objects`` is a sequence of ``{"id": ..., "aabb": ...}`` mappings.
    ``camera`` is ``{"position": (x, y, z), "view_proj": matrix}`` and
    ``pyramid`` is the depth pyramid used by :func:`occluded`.

    Returns ``{"visible": [ids], "occluded": [ids], "unknown": [ids]}``
    preserving input order in each list.  An object that cannot be
    projected, or whose depth cannot be established, is *unknown*.  In
    ``"conservative"`` mode unknown objects are also kept in ``visible``
    so rendering cannot pop; in ``"aggressive"`` mode they are only in
    ``"unknown"``.  Any other mode raises :class:`CullError`.
    """
    if mode not in ("conservative", "aggressive"):
        raise CullError(f"unknown culling mode: {mode!r}")

    view_proj = camera["view_proj"]
    cam_pos = camera["position"]
    dims = _grid_dimensions(pyramid)

    visible_ids = []
    occluded_ids = []
    unknown_ids = []

    for obj in objects:
        obj_id = obj["id"]
        aabb = obj["aabb"]

        # A box that cannot be validated or projected stays unknown rather
        # than leaking NaN screen coordinates into the visible list.
        try:
            _validate_aabb(aabb)
            rect_n = project_aabb(aabb, view_proj)
        except CullError:
            rect_n = None

        if not rect_n or dims is None:
            unknown_ids.append(obj_id)
            if mode == "conservative":
                visible_ids.append(obj_id)
            continue

        rows, cols = dims
        pixel_rect = (
            rect_n[0] * cols,
            rect_n[1] * rows,
            rect_n[2] * cols,
            rect_n[3] * rows,
        )
        if pixel_rect[0] >= pixel_rect[2] or pixel_rect[1] >= pixel_rect[3]:
            unknown_ids.append(obj_id)
            if mode == "conservative":
                visible_ids.append(obj_id)
            continue

        center = _aabb_center(aabb)
        depth = math.sqrt(
            (center[0] - cam_pos[0]) ** 2
            + (center[1] - cam_pos[1]) ** 2
            + (center[2] - cam_pos[2]) ** 2
        )
        if not math.isfinite(depth):
            unknown_ids.append(obj_id)
            if mode == "conservative":
                visible_ids.append(obj_id)
            continue

        level = pyramid_level(pixel_rect, len(pyramid))
        if occluded(pixel_rect, level, pyramid, depth):
            occluded_ids.append(obj_id)
        else:
            visible_ids.append(obj_id)

    return {
        "visible": visible_ids,
        "occluded": occluded_ids,
        "unknown": unknown_ids,
    }


def sort_front_to_back(visible, camera):
    """Return a new list sorted by squared distance to the camera.

    Distance is measured from ``camera["position"]`` to the AABB center.
    The sort is stable: objects at the same distance retain their input
    order.  The input list is not mutated.
    """
    cam_pos = camera["position"]

    def key(obj):
        center = _aabb_center(obj["aabb"])
        return (
            (center[0] - cam_pos[0]) ** 2
            + (center[1] - cam_pos[1]) ** 2
            + (center[2] - cam_pos[2]) ** 2
        )

    # Python's sorted() is stable, so equal-distance objects keep order.
    return sorted(visible, key=key)


def merge_queries(rects):
    """Merge overlapping query rectangles and remove duplicates.

    Rectangles are half-open ``(x0, y0, x1, y1)`` values.  Two rectangles
    merge when their intersection has positive area; merely touching
    edges do not.  Merge repeatedly until no pair overlaps, then return
    the union boxes sorted lexicographically by
    ``(x0, y0, x1, y1)``.  This reduces depth-pyramid reads compared with
    querying each duplicate separately.
    """
    merged = []
    for rect in rects:
        _validate_rect(rect)
        merged.append(tuple(rect))

    # Repeatedly fold one overlapping pair until no pair overlaps.  Using
    # positive-area overlap means touching edges stay separate and exact
    # duplicates collapse into a single query.
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                ax0, ay0, ax1, ay1 = merged[i]
                bx0, by0, bx1, by1 = merged[j]
                overlaps = (
                    ax0 < bx1 and bx0 < ax1
                    and ay0 < by1 and by0 < ay1
                )
                if overlaps:
                    union = (
                        min(ax0, bx0),
                        min(ay0, by0),
                        max(ax1, bx1),
                        max(ay1, by1),
                    )
                    merged = [
                        rect for k, rect in enumerate(merged)
                        if k not in (i, j)
                    ]
                    merged.append(union)
                    changed = True
                    break
            if changed:
                break

    return sorted(merged)


def early_z_ratio(visible, occluded):
    """Return the pixel-weighted early-depth pass ratio.

    Each object contributes its ``"pixels"`` count.  The ratio is
    ``visible_pixels / (visible_pixels + occluded_pixels)`` in ``[0, 1]``;
    an empty denominator yields ``0.0``.  Counting objects instead of
    pixels is incorrect.
    """
    visible_pixels = sum(obj["pixels"] for obj in visible)
    occluded_pixels = sum(obj["pixels"] for obj in occluded)
    total = visible_pixels + occluded_pixels
    if total <= 0:
        return 0.0
    return visible_pixels / total


def cull_batch(frames, config):
    """Cull several frames atomically.

    ``frames`` is a sequence of ``{"objects", "camera", "depth_pyramid"}``
    scenes and ``config`` contains ``"mode"``.  Return::

        {
          "frames": [
            {"frame_index": i, "visible": [...],
             "occluded": [...], "unknown": [...]},
            ...
          ],
          "totals": {"visible": n, "occluded": n, "unknown": n},
        }

    Every frame gets fresh result lists; results must not accumulate from
    one frame to the next.  Validate all frames and all AABBs before
    publishing anything.  If any frame is invalid, raise
    :class:`CullError` and leave the inputs unchanged; no partial batch is
    returned.
    """
    mode = config.get("mode")
    if mode not in ("conservative", "aggressive"):
        raise CullError(f"unknown culling mode: {mode!r}")

    # Validate every frame and every AABB up front so a failure rolls the
    # whole batch back instead of publishing partial frame results.
    for frame in frames:
        if not isinstance(frame, dict) or not all(
            key in frame for key in ("objects", "camera", "depth_pyramid")
        ):
            raise CullError("frame is missing required scene keys")
        _validate_matrix(frame["camera"]["view_proj"])
        for obj in frame["objects"]:
            _validate_aabb(obj["aabb"])

    result_frames = []
    totals = {"visible": 0, "occluded": 0, "unknown": 0}

    for index, frame in enumerate(frames):
        # Fresh lists per frame: nothing accumulates across iterations.
        classification = classify(
            frame["objects"], frame["camera"],
            frame["depth_pyramid"], mode,
        )
        result_frames.append({
            "frame_index": index,
            "visible": list(classification["visible"]),
            "occluded": list(classification["occluded"]),
            "unknown": list(classification["unknown"]),
        })
        for key in totals:
            totals[key] += len(classification[key])

    return {"frames": result_frames, "totals": totals}
