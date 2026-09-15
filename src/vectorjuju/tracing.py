"""Trace boundary linework into straight runs.

Raster -> Otsu threshold -> optional text exclusion -> thin to a 1 px centreline
-> contract the skeleton into chains between junctions and pair those chains by
straightest continuation -> merge collinear dashes -> split at corners -> drop
linework that is not a boundary stroke.

Runs are simplified polylines in raster pixels (top-left origin, y down);
``convert()`` maps them to CAD units. Rejection is geometric, never
content-based: strokes thinner than a boundary stroke are right-of-way/setback
lines and curve-table borders, and runs too small to be a boundary stroke are
glyphs and monument marks.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from skimage.morphology import skeletonize

_REFERENCE_DPI = 200.0

# Thresholds are px at the reference DPI and scale with the run's dpi, so the
# same figure traces the same way at 100 and 400 dpi. 3.0 px separates the
# synthetic sheet's 1.4 pt boundary strokes (~3.9 px) from its 0.6-1.0 pt
# distractors (~1.7-2.8 px); the fixture test pins both sides of it.
_MIN_STROKE_PX = 3.0
_MIN_LENGTH_PX = 12.0
_MIN_EXTENT_PX = 22.0
_CHAIN_PRUNE_PX = 4.0
_DASH_GAP_PX = 10.0
_DASH_ANGLE_DEG = 12.0
_DASH_COLLINEAR_PX = 1.5
_SIMPLIFY_PX = 1.5
_CORNER_DEG = 15.0
_JOIN_DEG = 75.0
_DUPLICATE_PX = 2.0

_OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass(frozen=True)
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

    ``exclude_mask`` is a boolean array the size of the raster; true pixels are
    removed before tracing (text boxes go here once OCR lands).
    """
    gray = np.asarray(image.convert("L") if isinstance(image, Image.Image) else image)
    binary = _binarize(gray)
    if exclude_mask is not None:
        binary &= ~exclude_mask.astype(bool)
    # Ink coverage, not a binary width: anti-aliasing quantises px counts, but
    # the integrated coverage of a cross-section is the true stroke width.
    coverage = (255.0 - gray.astype(float)) / 255.0

    min_stroke = _scaled(_MIN_STROKE_PX, dpi)
    paths = [
        path
        for path in _trace_paths(skeletonize(binary), _scaled(_CHAIN_PRUNE_PX, dpi))
        if _stroke_width(path, coverage) >= min_stroke
    ]
    paths = _drop_duplicates(paths, dpi)
    spans = [path[span] for path in _merge_dashes(paths, dpi) for span in _split_at_corners(path, dpi)]
    # Merge again after splitting: a corner the trace crossed leaves a span
    # pair whose join is interior to the merged path.
    runs: list[Run] = []
    for span in _merge_dashes(_drop_duplicates(spans, dpi), dpi):
        if _is_distractor(span, coverage, dpi):
            continue
        runs.append(Run(points_px=_simplify(span, dpi)))
    return runs


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


def _merge_dashes(paths: list[np.ndarray], dpi: float) -> list[np.ndarray]:
    """Join collinear paths separated by a small gap into one run.

    Only paths whose endpoints land in the same or adjacent cells are compared,
    so the search stays linear in the fragment count. Greedy closest-pair-first:
    merging the tightest collinear gap each round keeps a dash chain from
    absorbing a merely nearby stroke.
    """
    remaining = [np.asarray(path, dtype=float) for path in paths]
    gap_limit = _scaled(_DASH_GAP_PX, dpi)
    while True:
        best: tuple[float, int, int, np.ndarray, np.ndarray] | None = None
        for i, j in _nearby_pairs(remaining, gap_limit):
            merge = _merge_pair(remaining[i], remaining[j], dpi)
            if merge is not None and (best is None or merge[0] < best[0]):
                best = (merge[0], i, j, merge[1], merge[2])
        if best is None:
            return remaining
        _, i, j, first, second = best
        remaining = [path for k, path in enumerate(remaining) if k not in (i, j)]
        remaining.append(np.vstack([first, second]))


def _nearby_pairs(paths: list[np.ndarray], limit: float) -> set[tuple[int, int]]:
    """Path index pairs with any endpoints in the same or an adjacent cell."""
    buckets: dict[tuple[int, int], list[int]] = {}
    for index, path in enumerate(paths):
        for point in (path[0], path[-1]):
            buckets.setdefault((int(point[0] // limit), int(point[1] // limit)), []).append(index)
    pairs: set[tuple[int, int]] = set()
    for (col, row), indices in buckets.items():
        for dc in (-1, 0, 1):
            for dr in (-1, 0, 1):
                for i in indices:
                    for j in buckets.get((col + dc, row + dr), []):
                        if i != j:
                            pairs.add((i, j) if i < j else (j, i))
    return pairs


def _merge_pair(a: np.ndarray, b: np.ndarray, dpi: float) -> tuple[float, np.ndarray, np.ndarray] | None:
    """Orient a and b so a's end meets b's start, or None if they are not a dash gap."""
    gap_limit = _scaled(_DASH_GAP_PX, dpi)
    collinear_limit = _scaled(_DASH_COLLINEAR_PX, dpi)
    cosine = np.cos(np.radians(_DASH_ANGLE_DEG))
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for reverse_a in (False, True):
        for reverse_b in (False, True):
            first = a[::-1] if reverse_a else a
            second = b[::-1] if reverse_b else b
            tail, head = first[-1], second[0]
            gap = float(np.hypot(*(head - tail)))
            if gap > gap_limit:
                continue
            outward = _end_direction(first, at_start=False)
            inward = _end_direction(second, at_start=True)
            # Both point away from the join, so a true dash gap looks opposite.
            if outward is None or inward is None or float(np.dot(outward, inward)) > -cosine:
                continue
            if abs(_cross(outward, head - tail)) > collinear_limit:
                continue
            if best is None or gap < best[0]:
                best = (gap, first, second)
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
    splitting where the trace turns sharply. Arcs spread their turn evenly and
    survive as a single span.
    """
    simplified = _simplify(path, dpi)
    if len(simplified) < 2:
        return [np.arange(len(path))]
    closed = len(path) > 3 and bool(np.allclose(path[0], path[-1]))
    # Simplification may drop the repeated closing vertex; keep it cyclically.
    closed_polygon = closed and len(simplified) > 3 and bool(np.allclose(simplified[0], simplified[-1]))
    polygon = simplified[:-1] if closed_polygon else simplified
    corners = _corner_indices(polygon, closed)
    if not corners:
        return [np.arange(len(path))]
    markers = sorted({int(np.argmin(np.hypot(*(path - polygon[corner]).T))) for corner in corners})
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
    """Vertices where the trace turns more than ``_CORNER_DEG``."""
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


def _drop_duplicates(paths: list[np.ndarray], dpi: float) -> list[np.ndarray]:
    """Drop runs that run alongside a substantially longer one.

    Thinning leaves staircase rungs next to the centreline it just walked; a
    run whose points mostly sit within a stroke width of a much longer run is
    that leftover (or an outline traced beside its twin), not a second line.
    Dashes survive: only their ends come near the neighbouring dash.
    """
    tolerance = _scaled(_DUPLICATE_PX, dpi)
    kept: list[np.ndarray] = []
    lengths: list[float] = []
    for path in sorted(paths, key=_arc_length, reverse=True):
        length = _arc_length(path)
        box = _bbox(path, tolerance)
        duplicate = False
        for longer, longer_length in zip(kept, lengths):
            if longer_length < 2.0 * length:
                break  # kept is longest-first
            if _boxes_overlap(box, _bbox(longer, tolerance)) and _runs_alongside(path, longer, tolerance):
                duplicate = True
                break
        if not duplicate:
            kept.append(path)
            lengths.append(length)
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


def _runs_alongside(short: np.ndarray, long: np.ndarray, tolerance: float) -> bool:
    """True when most of ``short`` lies within ``tolerance`` of ``long``."""
    cells = {(int(x // tolerance), int(y // tolerance)) for x, y in long}
    sample = short[:: max(1, len(short) // 100)]
    nearby = 0
    for x, y in sample:
        col, row = int(x // tolerance), int(y // tolerance)
        if any((col + dx, row + dy) in cells for dx in (-1, 0, 1) for dy in (-1, 0, 1)):
            nearby += 1
    return nearby / len(sample) >= 0.8


def _stroke_width(points: np.ndarray, coverage: np.ndarray, *, sections: int = 20, half: float = 4.0) -> float:
    """Ink width across the run, in px, as the lower quartile of cross-sections.

    The lower quartile keeps a section that clips a crossing stroke or a
    neighbouring glyph from inflating the estimate for the whole run.
    """
    if len(points) < 2:
        return 0.0
    step = 0.25
    offsets = np.arange(-half, half + step / 2, step)
    widths = []
    for index in np.linspace(0, len(points) - 1, min(sections, len(points))).astype(int):
        back, ahead = max(index - 2, 0), min(index + 2, len(points) - 1)
        tangent = points[ahead] - points[back]
        norm = float(np.hypot(*tangent))
        if norm == 0:
            continue
        normal = np.array([-tangent[1], tangent[0]]) / norm
        cols = np.clip(np.rint(points[index, 0] + normal[0] * offsets).astype(int), 0, coverage.shape[1] - 1)
        rows = np.clip(np.rint(points[index, 1] + normal[1] * offsets).astype(int), 0, coverage.shape[0] - 1)
        widths.append(float(coverage[rows, cols].sum()) * step)
    return float(np.percentile(widths, 25)) if widths else 0.0


def _is_distractor(points: np.ndarray, coverage: np.ndarray, dpi: float) -> bool:
    """Thin or tiny linework: right-of-way lines, table borders, glyphs, monuments."""
    if _stroke_width(points, coverage) < _scaled(_MIN_STROKE_PX, dpi):
        return True
    if float(np.hypot(np.ptp(points[:, 0]), np.ptp(points[:, 1]))) < _scaled(_MIN_EXTENT_PX, dpi):
        return True
    return _arc_length(points) < _scaled(_MIN_LENGTH_PX, dpi)


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
