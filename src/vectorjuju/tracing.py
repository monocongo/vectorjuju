"""Trace boundary linework into straight runs.

Raster -> Otsu threshold -> optional text exclusion -> thin to a 1 px centreline
-> contract the skeleton into chains between junctions and pair those chains by
straightest continuation -> merge collinear dashes -> split at corners -> drop
linework that is not a boundary stroke.

Runs are simplified polylines in raster pixels (top-left origin, y down);
``convert()`` maps them to CAD units. Rejection is geometric, never
content-based: strokes thinner than a boundary stroke are right-of-way/setback
lines and curve-table borders, and runs too small to be a boundary stroke are
glyphs and monument marks. ``join_corners`` closes the skeleton's shortfall at
a shared corner, which is what makes a traced run the length of the edge its
bound call measures.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import pairwise

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

_REFERENCE_DPI = 200.0

# Skeleton junction vertices stop ~5-8 px short of a corner at the reference
# dpi (the suite's end-reach slop), and the neighbouring run is short too, so
# closing an end may travel up to the sum of both shortfalls. 16 px covers
# that with headroom while staying far inside the distance between real
# boundary runs.
_CORNER_CLOSE_PX = 16.0
# A neighbour whose centreline is nearly parallel is a merged dash or an
# offset line, not a corner; the tracer's own corner threshold is 15 degrees.
_CORNER_SIN = math.sin(math.radians(15.0))

# Thresholds are px at the reference DPI and scale with the run's dpi, so the
# same figure traces the same way at 100 and 400 dpi. 3.0 px separates the
# synthetic sheet's 1.4 pt boundary strokes (~3.9 px) from its 0.6-1.0 pt
# distractors (~1.7-2.8 px); the fixture test pins both sides of it.
_MIN_STROKE_PX = 3.0
_MIN_EXTENT_PX = 22.0
_CHAIN_PRUNE_PX = 4.0
_DASH_GAP_PX = 10.0
_DASH_ANGLE_DEG = 12.0
_DASH_COLLINEAR_PX = 1.5
_SIMPLIFY_PX = 1.5
_CORNER_DEG = 15.0
_JOIN_DEG = 75.0
_DUPLICATE_PX = 2.0
_WIDTH_STEP_PX = 0.1
_MAX_INDEX_CELLS = 4096
# Per endpoint, not per cell: a crowded cell (hatching, a glyph cluster) would
# otherwise enumerate every pair inside it and in the eight cells around it.
_MAX_NEIGHBOURS = 8

_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass(frozen=True, eq=False)
class Run:
    """One traced straight run: a simplified polyline in raster px, y down."""

    points_px: np.ndarray  # (N, 2) float, x = column, y = row


def trace_runs(
    image: Image.Image | np.ndarray,
    dpi: float = _REFERENCE_DPI,
    *,
    exclude_mask: np.ndarray | None = None,
) -> list[Run]:
    """Trace boundary runs from a raster.

    ``image`` is a PIL image or a 2-D uint8 grayscale array. ``exclude_mask``
    is a boolean array the size of the raster; true pixels are removed before
    tracing (text boxes go here once OCR lands).
    """
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError(f"dpi must be a positive finite number, got {dpi!r}")
    if isinstance(image, Image.Image):
        gray = np.asarray(image.convert("L"))
    else:
        gray = np.asarray(image)
        if gray.ndim != 2 or gray.dtype != np.uint8:
            raise ValueError(f"image array must be 2-D uint8 grayscale, got shape {gray.shape} dtype {gray.dtype}")
    if gray.size == 0:
        return []
    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask)
        if exclude_mask.shape != gray.shape:
            raise ValueError(f"exclude_mask shape {exclude_mask.shape} does not match image shape {gray.shape}")
    mask = exclude_mask.astype(bool) if exclude_mask is not None else None
    binary = _binarize(gray)
    if mask is not None:
        binary &= ~mask
    # Nothing to trace; a near-solid page is a scan of something other than
    # linework, and thinning it costs minutes of CPU for no centreline.
    if not binary.any() or binary.mean() > 0.7:
        return []
    # Ink coverage, not a binary width: anti-aliasing quantises px counts, but
    # the integrated coverage of a cross-section is the true stroke width.
    coverage = (255.0 - gray.astype(np.float32)) / 255.0
    if mask is not None:
        coverage[mask] = 0.0  # excluded ink must not measure as part of a remaining stroke

    min_stroke = _scaled(_MIN_STROKE_PX, dpi)
    measured = [
        (path, _stroke_width(path, coverage, dpi))
        for path in _trace_paths(skeletonize(binary), _scaled(_CHAIN_PRUNE_PX, dpi))
    ]
    paths = _drop_duplicates([(path, width) for path, width in measured if width >= min_stroke], dpi)
    spans = [
        (path[span], width)
        for path, width in _merge_dashes(paths, coverage, dpi)
        for span in _split_at_corners(path, dpi)
    ]
    # Merge again after splitting: a corner the trace crossed leaves a span
    # pair whose join is interior to the merged path.
    runs: list[Run] = []
    for span, width in _merge_dashes(_drop_duplicates(spans, dpi), coverage, dpi):
        if _is_distractor(span, width, dpi):
            continue
        runs.append(Run(points_px=_simplify(span, dpi)))
    return runs


def join_corners(runs: Sequence[Run], dpi: float = _REFERENCE_DPI) -> list[Run]:
    """Close run ends onto the corner the neighbouring run's centreline makes.

    Thinning puts a junction where the two inked strokes overlap, not at the
    geometric corner: each run's extreme vertex sits a stroke-width-dependent
    distance short of the edge it traces. That shortfall is ~2 % of a parcel
    edge -- the same edge whose length calibrates feet-per-pixel and whose
    corner an ARC's endpoint is supposed to reach -- so emitted geometry is
    closed the same way ``calibrate_scale`` closes it for measurement:
    intersect each end's ray with the neighbouring run's centreline and take
    the nearest crossing inside ``_CORNER_CLOSE_PX``. A parallel neighbour is
    rejected by the angle test, so a right-of-way line never captures a
    boundary run's end.
    """
    if len(runs) < 2:
        return list(runs)
    cap = _scaled(_CORNER_CLOSE_PX, dpi)
    cell = 2 * cap
    index = _cell_index(runs, cell)
    joined: list[Run] = []
    for run_index, run in enumerate(runs):
        points = run.points_px
        moved = points.copy()
        ends: list[int] = []
        for at_start in (True, False):
            end = 0 if at_start else -1
            closed = _closes_end(points, at_start, _nearby_runs(index, cell, points[end], runs, run_index), cap)
            if not np.array_equal(closed, points[end]):
                moved[end] = closed
                ends.append(end)
        if ends:
            moved = _seat_curve_ends(points, moved, ends, dpi)
            joined.append(Run(points_px=moved))
        else:
            joined.append(run)
    return joined


def _seat_curve_ends(points: np.ndarray, moved: np.ndarray, ends: list[int], dpi: float) -> np.ndarray:
    """Pull a curved run's closed ends back onto its own fitted circle.

    A corner sits on the neighbouring centreline and on the curve's circle;
    closing the end onto the neighbour's centreline alone can land off the
    circle, and two off-circle endpoints out of six skew the writer's own
    least-squares fit -- measured at ~9 % radius error on the fixture's clean
    48-vertex arc. Projecting each moved end radially back onto the circle fit
    through the traced points keeps the radius the tracer earned and still
    puts the ARC's endpoint on the corner.
    """
    from vectorjuju.curves import classify  # local: curves imports this module

    kind, fit = classify(points, dpi)
    if kind != "curve" or fit is None:
        return moved
    centre = np.asarray(fit.center_px)
    for end in ends:
        vector = moved[end] - centre
        moved[end] = centre + vector * (fit.radius_px / float(np.hypot(*vector)))
    return moved


def _closest_point(point: np.ndarray, points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Distance to a polyline, its nearest point, and that segment's unit direction."""
    best = (math.inf, point, np.zeros(2))
    for a, b in pairwise(points):
        edge = b - a
        length = float(np.hypot(*edge))
        if length == 0:
            continue
        t = max(0.0, min(1.0, float((point - a) @ edge) / (length * length)))
        nearest = a + t * edge
        distance = float(np.linalg.norm(point - nearest))
        if distance < best[0]:
            best = (distance, nearest, edge / length)
    return best


def _closes_end(points: np.ndarray, at_start: bool, neighbours: Sequence[np.ndarray], cap: float) -> np.ndarray:
    """Extend one run end to its corner.

    Intersects the run's outward end ray with each neighbour's centreline and
    takes the smallest valid crossing; the end is returned unchanged when no
    neighbour meets it inside the cap.
    """
    end = points[0] if at_start else points[-1]
    direction = _end_direction(points, at_start)
    if direction is None:
        return end
    best: tuple[float, np.ndarray | None] = (math.inf, None)
    for neighbour in neighbours:
        distance, nearest, edge = _closest_point(end, neighbour)
        if distance > 2 * cap or abs(direction[0] * edge[1] - direction[1] * edge[0]) < _CORNER_SIN:
            continue
        try:
            along, across = np.linalg.solve(
                np.array([[direction[0], -edge[0]], [direction[1], -edge[1]]]), nearest - end
            )
        except np.linalg.LinAlgError:  # parallel, already guarded; keep the degenerate safe
            continue
        if 0.0 < along <= cap and abs(across) <= cap and along < best[0]:
            best = (float(along), end + along * direction)
    return best[1] if best[1] is not None else end


def _cell_index(runs: Sequence[Run], cell_px: float) -> dict[tuple[int, int], list[int]]:
    """Run indices by raster cell, so closure looks up neighbours instead of
    scanning every other run's polyline at every end."""
    index: dict[tuple[int, int], list[int]] = {}
    for i, run in enumerate(runs):
        for a, b in pairwise(run.points_px):
            x0, x1 = min(float(a[0]), float(b[0])), max(float(a[0]), float(b[0]))
            y0, y1 = min(float(a[1]), float(b[1])), max(float(a[1]), float(b[1]))
            for cx in range(int(x0 // cell_px), int(x1 // cell_px) + 1):
                for cy in range(int(y0 // cell_px), int(y1 // cell_px) + 1):
                    index.setdefault((cx, cy), []).append(i)
    return index


def _nearby_runs(
    index: dict[tuple[int, int], list[int]], cell_px: float, point: np.ndarray, runs: Sequence[Run], skip: int
) -> list[np.ndarray]:
    """Polylines of the runs in the 3x3 cell block around ``point``.

    The cell is twice the closure cap, so every run whose centreline comes
    within reach of ``point`` -- the only ones ``_closes_end`` can use -- is
    inside that block.
    """
    cx, cy = int(point[0] // cell_px), int(point[1] // cell_px)
    near: set[int] = set()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            near.update(index.get((cx + dx, cy + dy), ()))
    near.discard(skip)
    return [runs[i].points_px for i in sorted(near)]


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Ink true, via Otsu. Dark linework on a light background."""
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binary.astype(bool)


def _trace_paths(skeleton: np.ndarray, min_chain_px: float) -> list[np.ndarray]:
    """Trace the skeleton into long pixel paths, as (x, y) polylines.

    A per-pixel walk shatters wherever two strokes touch or a diagonal stroke
    steps: thinning leaves junctions there. Instead, contract the pixels between
    junctions into chains, pair chains at each junction by which continuation is
    straightest, then follow the pairings from every unpaired end. Rung chains
    that no pairing wants are pruned first.
    """
    degree = _degrees(skeleton)
    chains = [chain for chain in _chains(skeleton, degree) if _arc_length(chain) >= min_chain_px]
    return _follow_chains(chains, _pair_chains(chains))


def _degrees(skeleton: np.ndarray) -> np.ndarray:
    height, width = skeleton.shape
    degree = np.zeros_like(skeleton, dtype=np.uint8)
    padded = np.pad(skeleton, 1)
    for dr, dc in _OFFSETS:
        degree += padded[1 + dr : height + 1 + dr, 1 + dc : width + 1 + dc]
    return degree


def _neighbours(skeleton: np.ndarray, row: int, col: int) -> list[tuple[int, int]]:
    height, width = skeleton.shape
    return [
        (row + dr, col + dc)
        for dr, dc in _OFFSETS
        if 0 <= row + dr < height and 0 <= col + dc < width and skeleton[row + dr, col + dc]
    ]


def _chains(skeleton: np.ndarray, degree: np.ndarray) -> list[np.ndarray]:
    """Pixel chains between junctions and endpoints, plus standalone cycles."""
    visited = np.zeros_like(skeleton, dtype=bool)
    used_node_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    chains: list[np.ndarray] = []

    def walk(start: tuple[int, int], first: tuple[int, int]) -> list[tuple[int, int]]:
        path = [start, first]
        if degree[first] == 2:
            visited[first] = True
        previous, current = start, first
        while degree[current] == 2:
            options = [pixel for pixel in _neighbours(skeleton, *current) if pixel != previous]
            if not options:
                break
            following = options[0] if len(options) == 1 else _straightest(path[-4:], current, options)
            if degree[following] == 2:
                visited[following] = True
            path.append(following)
            previous, current = current, following
        return path

    for node in (tuple(int(v) for v in pixel) for pixel in np.argwhere(skeleton & (degree != 2))):
        for first in _neighbours(skeleton, *node):
            if degree[first] == 2:
                if visited[first]:
                    continue
            else:
                edge = (node, first) if node <= first else (first, node)
                if edge in used_node_edges:
                    continue
                used_node_edges.add(edge)
            chains.append(_as_points(walk(node, first)))
    for pixel in (tuple(int(v) for v in p) for p in np.argwhere(skeleton)):
        if degree[pixel] != 2 or visited[pixel]:
            continue
        path = [pixel]
        visited[pixel] = True
        previous, current = None, pixel
        while True:
            options = [p for p in _neighbours(skeleton, *current) if p != previous and not visited[p]]
            if not options:
                break
            following = options[0] if len(options) == 1 else _straightest(path[-4:], current, options)
            visited[following] = True
            path.append(following)
            previous, current = current, following
        if len(path) > 3 and _pixel_gap(path[0], path[-1]) <= 1.5:
            path.append(path[0])
        chains.append(_as_points(path))
    return chains


def _as_points(path: list[tuple[int, int]]) -> np.ndarray:
    """(row, col) walk -> (x, y) points."""
    return np.array([(col, row) for row, col in path], dtype=float)


def _pixel_gap(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _straightest(
    recent: list[tuple[int, int]], current: tuple[int, int], options: list[tuple[int, int]]
) -> tuple[int, int]:
    """Pick the option that continues the current heading best."""
    heading = (current[0] - recent[0][0], current[1] - recent[0][1])
    norm = float(np.hypot(*heading))
    if norm == 0:
        return options[0]
    best, best_score = options[0], -np.inf
    for option in options:
        vector = (option[0] - current[0], option[1] - current[1])
        score = (vector[0] * heading[0] + vector[1] * heading[1]) / float(np.hypot(*vector))
        if score > best_score:
            best, best_score = option, score
    return best


def _pair_chains(chains: list[np.ndarray]) -> dict[tuple[int, bool], tuple[int, bool]]:
    """Pair chain ends meeting at the same junction when they continue straight.

    An end is ``(chain index, True)`` at the chain's start and ``(chain index,
    False)`` at its end; the value is the end it pairs with.
    """
    ends: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    for index, chain in enumerate(chains):
        for at_start in (True, False):
            node = tuple(int(v) for v in (chain[0] if at_start else chain[-1]))
            ends.setdefault(node, []).append((index, at_start))
    limit = -np.cos(np.radians(_JOIN_DEG))
    pairs: dict[tuple[int, bool], tuple[int, bool]] = {}
    for node_ends in ends.values():
        remaining = list(node_ends)
        while len(remaining) >= 2:
            best: tuple[float, int, int] | None = None
            for a in range(len(remaining)):
                for b in range(a + 1, len(remaining)):
                    first, second = remaining[a], remaining[b]
                    if first[0] == second[0]:
                        continue  # the two ends of one small loop
                    dot = float(
                        np.dot(
                            _chain_direction(chains[first[0]], first[1]), _chain_direction(chains[second[0]], second[1])
                        )
                    )
                    if best is None or dot < best[0]:
                        best = (dot, a, b)
            if best is None or best[0] > limit:
                break
            _, a, b = best
            first, second = remaining[a], remaining[b]
            pairs[first] = second
            pairs[second] = first
            remaining = [end for index, end in enumerate(remaining) if index not in (a, b)]
    return pairs


def _chain_direction(chain: np.ndarray, at_start: bool, window: int = 10) -> np.ndarray:
    """Unit vector pointing into the chain from one of its ends."""
    span = chain[: min(len(chain), window)] if at_start else chain[-min(len(chain), window) :]
    vector = (span[-1] - span[0]) if at_start else (span[0] - span[-1])
    norm = float(np.hypot(*vector))
    return vector / norm if norm else np.zeros(2)


def _follow_chains(chains: list[np.ndarray], pairs: dict[tuple[int, bool], tuple[int, bool]]) -> list[np.ndarray]:
    """Walk chain-to-chain through the pairings, from every unpaired end."""
    used: set[tuple[int, bool]] = set()

    def build(index: int, enter_start: bool) -> np.ndarray:
        path = chains[index] if enter_start else chains[index][::-1]
        used.add((index, enter_start))
        while True:
            far = (index, not enter_start)
            used.add(far)
            partner = pairs.get(far)
            if partner is None or partner in used:
                return path
            index, enter_start = partner
            used.add((index, enter_start))
            segment = chains[index] if enter_start else chains[index][::-1]
            path = np.vstack([path, segment[1:]])

    order = [(index, at_start) for index in range(len(chains)) for at_start in (True, False)]
    paths = []
    for start in (end for end in order if end not in pairs):  # open paths first, so they claim their chains
        if start not in used:
            paths.append(build(*start))
    for start in order:  # a chain cycle whose every end is paired still has to be walked
        if start not in used:
            paths.append(build(*start))
    return paths


def _merge_dashes(
    entries: list[tuple[np.ndarray, float]], coverage: np.ndarray, dpi: float
) -> list[tuple[np.ndarray, float]]:
    """Join collinear paths separated by a small gap into one run.

    Candidate pairs come from an endpoint grid and are capped to the nearest
    endpoints around each one, so neither a crowded cell nor a long chain of
    fragments makes the candidate count explode. The tightest collinear gap is
    merged first, so a dash chain cannot absorb a merely nearby stroke; the cap
    keeps the tightest gaps, which are the ones greedy merging wants. Pending
    candidates sit in a heap and only pairs touching a merged path are rebuilt,
    so a chain of n fragments is not rescanned n times.
    """
    paths = [np.asarray(path, dtype=float) for path, _ in entries]
    widths = [width for _, width in entries]
    grid = _scaled(_DASH_GAP_PX, dpi) + (max(widths) if widths else 0.0)
    alive = [True] * len(paths)
    # Endpoint coordinates, one row per fragment, mirrored into the append below.
    ends = np.array([(path[0, 0], path[0, 1], path[-1, 0], path[-1, 1]) for path in paths]).reshape(-1, 4)

    def cell(point: np.ndarray) -> tuple[int, int]:
        return (int(point[0] // grid), int(point[1] // grid))

    buckets: dict[tuple[int, int], set[int]] = {}

    def index_bucket(index: int) -> None:
        for point in (paths[index][0], paths[index][-1]):
            buckets.setdefault(cell(point), set()).add(index)

    def drop_bucket(index: int) -> None:
        for point in (paths[index][0], paths[index][-1]):
            key = cell(point)
            bucket = buckets.get(key)
            if bucket is not None:
                bucket.discard(index)
                if not bucket:
                    del buckets[key]

    def neighbours(index: int) -> set[int]:
        """Candidate fragment indices around ``index``, nearest endpoints first.

        Capped per endpoint: a crowded cell would otherwise enumerate every
        pair inside it and the eight cells around it. The cap is chosen without
        sorting the cell, because a crowded cell is exactly the one that would
        otherwise sort hundreds of candidates once per endpoint.
        """
        found: set[int] = set()
        for point in (paths[index][0], paths[index][-1]):
            col, row = cell(point)
            candidates: set[int] = set()
            for dc in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    candidates.update(buckets.get((col + dc, row + dr), ()))
            candidates.discard(index)
            if len(candidates) > _MAX_NEIGHBOURS:
                ids = np.fromiter(candidates, dtype=int, count=len(candidates))
                near = ends[ids]
                distance = np.minimum(
                    np.hypot(near[:, 0] - point[0], near[:, 1] - point[1]),
                    np.hypot(near[:, 2] - point[0], near[:, 3] - point[1]),
                )
                candidates = set(ids[np.argpartition(distance, _MAX_NEIGHBOURS)[:_MAX_NEIGHBOURS]].tolist())
            found.update(candidates)
        return found

    def merge_result(i: int, j: int) -> tuple[float, np.ndarray, np.ndarray] | None:
        return _merge_pair(paths[i], paths[j], dpi, min(widths[i], widths[j]))

    heap: list[tuple[float, int, int]] = []
    for index in range(len(paths)):
        index_bucket(index)
    for index in range(len(paths)):
        for other in neighbours(index):
            result = merge_result(index, other)
            if result is not None:
                heappush(heap, (result[0], index, other))
    while heap:
        _, i, j = heappop(heap)
        if not (alive[i] and alive[j]):
            continue
        result = merge_result(i, j)
        if result is None:
            continue
        _, first, second = result
        alive[i] = alive[j] = False
        drop_bucket(i)
        drop_bucket(j)
        paths.append(np.vstack([first, second]))
        widths.append(_stroke_width(paths[-1], coverage, dpi))
        merged = paths[-1]
        ends = np.vstack([ends, (merged[0, 0], merged[0, 1], merged[-1, 0], merged[-1, 1])])
        alive.append(True)
        index_bucket(len(paths) - 1)
        for other in neighbours(len(paths) - 1):
            if alive[other]:
                extended = merge_result(len(paths) - 1, other)
                if extended is not None:
                    heappush(heap, (extended[0], len(paths) - 1, other))
    return [(paths[i], widths[i]) for i in range(len(paths)) if alive[i]]


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(*(a - b)))


def _merge_pair(a: np.ndarray, b: np.ndarray, dpi: float, cap: float) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Orient a and b so a's end meets b's start, or None if they are not a dash gap.

    ``cap`` is the stroke width the skeleton throws away at the two facing
    ends, so the gap between skeleton endpoints is that much wider than the
    ink gap.
    """
    gap_limit = _scaled(_DASH_GAP_PX, dpi) + cap
    angle_limit = np.radians(_DASH_ANGLE_DEG)
    cosine, tangent = np.cos(angle_limit), np.tan(angle_limit)
    collinear_limit = _scaled(_DASH_COLLINEAR_PX, dpi)
    window = max(2, round(_scaled(15.0, dpi)))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for reverse_a in (False, True):
        for reverse_b in (False, True):
            tail = a[0] if reverse_a else a[-1]
            head = b[-1] if reverse_b else b[0]
            gap = _gap(head, tail)
            if gap > gap_limit:
                continue
            outward = _end_direction(a, at_start=reverse_a, window=window)
            inward = _end_direction(b, at_start=not reverse_b, window=window)
            # Both point away from the join, so a true dash gap looks opposite.
            if outward is None or inward is None or float(np.dot(outward, inward)) > -cosine:
                continue
            # A gap on a curve tilts the joining vector; allow the lateral
            # offset that still fits the angle the pairing allows.
            if abs(_cross(outward, head - tail)) > collinear_limit + gap * tangent:
                continue
            if best is None or gap < best[0]:
                best = (gap, a[::-1] if reverse_a else a, b[::-1] if reverse_b else b)
    return best


def _end_direction(points: np.ndarray, at_start: bool, window: int = 15) -> np.ndarray | None:
    """Unit vector pointing outward from one end of a path.

    Means of the window's thirds, not its endpoints: a kink at the tip of a
    short arc stub must not swing the heading the gap test compares.
    """
    if len(points) < 2:
        return None
    span = points[: min(len(points), window)] if at_start else points[-min(len(points), window) :]
    third = max(1, len(span) // 3)
    inner, outer = span[:third].mean(axis=0), span[-third:].mean(axis=0)
    vector = (inner - outer) if at_start else (outer - inner)
    norm = float(np.hypot(*vector))
    return vector / norm if norm else None


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    """Scalar 2-D cross product; np.cross raises on 2-D under NumPy 2.x."""
    return float(a[0] * b[1] - a[1] * b[0])


def _split_at_corners(path: np.ndarray, dpi: float) -> list[np.ndarray]:
    """Index spans between corner vertices of the simplified trace.

    A closed parcel traces as one cycle; straight runs only exist after
    splitting where the trace turns sharply. A smooth small arc still splits
    into spans here; separating corner from curvature is the curve fitting
    that lands with convert() (issue #22).
    """
    simplified = _simplify(path, dpi)
    if len(simplified) < 2:
        return [np.arange(len(path))]
    closed = len(path) > 3 and bool(np.allclose(path[0], path[-1]))
    # Simplification may drop the repeated closing vertex; keep it cyclically.
    closed_polygon = closed and len(simplified) > 3 and bool(np.allclose(simplified[0], simplified[-1]))
    polygon = simplified[:-1] if closed_polygon else simplified
    corners = _corner_indices(polygon, closed)
    # Polygon vertices are a subsequence of the path, so a monotone cursor maps
    # them without rescanning the whole path per corner.
    markers = []
    cursor = 0
    for corner in corners:
        vertex = polygon[corner]
        index = cursor + int(np.argmin(np.hypot(path[cursor:, 0] - vertex[0], path[cursor:, 1] - vertex[1])))
        cursor = index
        markers.append(index)
    markers = sorted(set(markers))
    if not markers or (len(markers) == 1 and closed):
        # No corner, or one corner that closes the cycle, which is still one span.
        return [np.arange(len(path))]
    if not closed:
        cuts = [0, *markers, len(path) - 1]
        return [np.arange(cuts[k], cuts[k + 1] + 1) for k in range(len(cuts) - 1) if cuts[k + 1] > cuts[k]]
    points = path[:-1] if np.allclose(path[0], path[-1]) else path
    spans = []
    for k, start in enumerate(markers):
        end = markers[(k + 1) % len(markers)]
        span = (
            np.arange(start, end + 1)
            if end >= start
            else np.concatenate([np.arange(start, len(points)), np.arange(0, end + 1)])
        )
        if len(span) >= 2:
            spans.append(span)
    return spans


def _corner_indices(polygon: np.ndarray, closed: bool) -> list[int]:
    """Vertices where the trace turns more than ``_CORNER_DEG``.

    A smooth small arc still splits here; which of its spans is a corner and
    which is curvature needs the curve fitting that lands with convert()
    (issue #22).
    """
    limit = np.radians(_CORNER_DEG)
    corners = []
    for k in range(len(polygon)):
        if not closed and k in (0, len(polygon) - 1):
            continue
        incoming = polygon[k] - polygon[k - 1]
        outgoing = polygon[(k + 1) % len(polygon)] - polygon[k]
        if np.hypot(*incoming) == 0 or np.hypot(*outgoing) == 0:
            continue
        turn = np.arctan2(_cross(incoming, outgoing), float(np.dot(incoming, outgoing)))
        if abs(turn) > limit:
            corners.append(k)
    return corners


def _drop_duplicates(entries: list[tuple[np.ndarray, float]], dpi: float) -> list[tuple[np.ndarray, float]]:
    """Drop runs that run alongside a substantially longer one.

    Thinning leaves staircase rungs next to the centreline it just walked; a
    run whose points mostly sit within a stroke width of a much longer run is
    that leftover (or an outline traced beside its twin), not a second line.
    Dashes survive: only their ends come near the neighbouring dash. Kept
    bounding boxes are bucketed so a page of varied-length runs is not an
    all-pairs comparison.
    """
    tolerance = _scaled(_DUPLICATE_PX, dpi)
    cell = max(tolerance, _scaled(32.0, dpi))
    kept: list[tuple[np.ndarray, float]] = []
    lengths: list[float] = []
    index: dict[tuple[int, int], list[int]] = {}
    long_kept: list[int] = []  # bboxes too large to bucket, checked by every query
    cells_of: list[set[tuple[int, int]]] = []  # the kept path's cells, reused across queries

    def cells_in(box: tuple[float, float, float, float]) -> list[tuple[int, int]]:
        return [
            (col, row)
            for col in range(int(box[0] // cell), int(box[2] // cell) + 1)
            for row in range(int(box[1] // cell), int(box[3] // cell) + 1)
        ]

    for path, width in sorted(entries, key=lambda entry: _arc_length(entry[0]), reverse=True):
        length = _arc_length(path)
        query_cells = cells_in(_bbox(path, 2.0 * tolerance))
        candidates = set(long_kept)
        if len(query_cells) <= _MAX_INDEX_CELLS:
            for key in query_cells:
                candidates.update(index.get(key, ()))
        else:
            candidates.update(range(len(kept)))  # a very long run: linear is cheaper
        duplicate = False
        for k in candidates:
            if lengths[k] < 2.0 * length:
                continue
            if _boxes_overlap(_bbox(path, tolerance), _bbox(kept[k][0], tolerance)) and _runs_alongside(
                path, cells_of[k], tolerance
            ):
                duplicate = True
                break
        if duplicate:
            continue
        kept.append((path, width))
        lengths.append(length)
        cells_of.append({(int(x // tolerance), int(y // tolerance)) for x, y in path})
        raw_cells = cells_in(_bbox(path, 0.0))
        if len(raw_cells) <= _MAX_INDEX_CELLS:
            for key in raw_cells:
                index.setdefault(key, []).append(len(kept) - 1)
        else:
            long_kept.append(len(kept) - 1)
    return kept


def _bbox(path: np.ndarray, margin: float) -> tuple[float, float, float, float]:
    return (
        float(path[:, 0].min()) - margin,
        float(path[:, 1].min()) - margin,
        float(path[:, 0].max()) + margin,
        float(path[:, 1].max()) + margin,
    )


def _boxes_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]


def _runs_alongside(short: np.ndarray, long_cells: set[tuple[int, int]], tolerance: float) -> bool:
    """True when most of ``short`` lies within ``tolerance`` of a long run's cells."""
    sample = short[:: max(1, len(short) // 100)]
    nearby = 0
    for x, y in sample:
        col, row = int(x // tolerance), int(y // tolerance)
        if any((col + dx, row + dy) in long_cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            nearby += 1
    return nearby / len(sample) >= 0.8


def _stroke_width(points: np.ndarray, coverage: np.ndarray, dpi: float, *, sections: int = 20) -> float:
    """Ink width across the run, in px, as the lower quartile of cross-sections.

    The lower quartile keeps a section that clips a crossing stroke or a
    neighbouring glyph from inflating the estimate for the whole run. Window
    and step scale with dpi, or a stroke reads narrower than it is above the
    reference resolution.
    """
    if len(points) < 2:
        return 0.0
    half, step = _scaled(4.0, dpi), _scaled(_WIDTH_STEP_PX, dpi)
    offsets = np.arange(-half, half + step / 2, step)
    widths = []
    for index in np.linspace(0, len(points) - 1, min(sections, len(points))).astype(int):
        back, ahead = max(index - 2, 0), min(index + 2, len(points) - 1)
        tangent = points[ahead] - points[back]
        norm = float(np.hypot(*tangent))
        if norm == 0:
            continue
        normal = np.array([-tangent[1], tangent[0]]) / norm
        cols = np.rint(points[index, 0] + normal[0] * offsets).astype(int)
        rows = np.rint(points[index, 1] + normal[1] * offsets).astype(int)
        # Samples off the raster are dropped, never clipped: clipping maps every
        # out-of-frame offset onto the edge pixel, so a 1 px line along an edge
        # counts its one row once per sample and reads wider than the boundary
        # gate. A partly visible section then measures the ink the raster can
        # attest to, which is the honest reading of a cropped stroke.
        inside = (cols >= 0) & (cols < coverage.shape[1]) & (rows >= 0) & (rows < coverage.shape[0])
        if not inside.any():
            continue
        widths.append(float(coverage[rows[inside], cols[inside]].sum()) * step)
    return float(np.percentile(widths, 25)) if widths else 0.0


def _is_distractor(points: np.ndarray, width: float, dpi: float) -> bool:
    """Thin or tiny linework: right-of-way lines, table borders, glyphs, monuments."""
    if width < _scaled(_MIN_STROKE_PX, dpi):
        return True
    # The skeleton stops short of a stroke's free ends and rounds it into
    # corners, so allow roughly that ink back before calling a short span a
    # monument mark. Straight spans stay within this allowance; the curved
    # fragments of a small arc are what issue #22's curve fitting must sort out.
    allowance = min(1.5 * width, _scaled(_MIN_EXTENT_PX / 2.0, dpi))
    extent = float(np.hypot(np.ptp(points[:, 0]), np.ptp(points[:, 1])))
    return extent < _scaled(_MIN_EXTENT_PX, dpi) - allowance


def _arc_length(points: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(points, axis=0).T)))


def _simplify(points: np.ndarray, dpi: float) -> np.ndarray:
    """Douglas-Peucker to a DPI-scaled tolerance."""
    if len(points) <= 2:
        return points.copy()
    epsilon = _scaled(_SIMPLIFY_PX, dpi)
    simplified = cv2.approxPolyDP(points.reshape(-1, 1, 2).astype(np.float32), epsilon, False)
    return simplified.reshape(-1, 2).astype(float)


def _scaled(px_at_reference_dpi: float, dpi: float) -> float:
    return px_at_reference_dpi * dpi / _REFERENCE_DPI
