"""Occlusion-culling primitives for the exercise.

Depth values are camera-space distances: larger values are farther away.
The depth pyramid stores the farthest depth per texel, so a conservative
occlusion test can reject a rectangle only when *every* covered texel is
strictly nearer than the object.
"""

from __future__ import annotations

import heapq
import math


class CullError(Exception):
    """Raised for invalid culling input or an unknown culling mode."""


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    ``unknown``.  Any other mode raises :class:`CullError`.
    """
    raise NotImplementedError


def sort_front_to_back(visible, camera):
    """Return a new list sorted by squared distance to the camera.

    Distance is measured from ``camera["position"]`` to the AABB center.
    The sort is stable: objects at the same distance retain their input
    order.  The input list is not mutated.
    """
    raise NotImplementedError


def merge_queries(rects):
    """Merge overlapping query rectangles and remove duplicates.

    Rectangles are half-open ``(x0, y0, x1, y1)`` values.  Two rectangles
    merge when their intersection has positive area; merely touching
    edges do not.  Merge repeatedly until no pair overlaps, then return
    the union boxes sorted lexicographically by
    ``(x0, y0, x1, y1)``.  This reduces depth-pyramid reads compared with
    querying each duplicate separately.
    """
    raise NotImplementedError


def early_z_ratio(visible, occluded):
    """Return the pixel-weighted early-depth pass ratio.

    Each object contributes its ``"pixels"`` count.  The ratio is
    ``visible_pixels / (visible_pixels + occluded_pixels)`` in ``[0, 1]``;
    an empty denominator yields ``0.0``.  Counting objects instead of
    pixels is incorrect.
    """
    raise NotImplementedError


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
    raise NotImplementedError
