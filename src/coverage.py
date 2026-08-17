"""Travel times between block group centroids, and the binary coverage matrix.

Travel time is great-circle distance at a constant assumed speed. That is a
deliberate simplification with a known direction of error: Pittsburgh sits at
the confluence of three rivers, so straight-line distance understates real
driving distance wherever a bridge or a hillside is in the way, and every
response time reported here is therefore optimistic. The speed constant is the
one lever that absorbs this, and ``README.md`` reports how the results move when
it is varied.

Swapping in a road network is a drop-in change: any callable with the signature
``(lat1, lon1, lat2, lon2) -> minutes`` can be passed to
``build_time_matrix``.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

EARTH_RADIUS_KM = 6371.0
DEFAULT_SPEED_KMH = 40.0

TimeFunction = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distance in kilometres. Broadcasts over arrays."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=float)) for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    inner = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2.0 * np.arcsin(np.sqrt(np.clip(inner, 0.0, 1.0)))


def haversine_minutes(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    speed_kmh: float = DEFAULT_SPEED_KMH,
) -> np.ndarray:
    """Great-circle travel time in minutes at a constant speed."""
    return haversine_km(lat1, lon1, lat2, lon2) / speed_kmh * 60.0


def euclidean_minutes(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
    speed_kmh: float = DEFAULT_SPEED_KMH,
    reference_latitude: float = 40.44,
) -> np.ndarray:
    """Flat-earth travel time, using a fixed degree-to-kilometre scale.

    Included as a cross-check on ``haversine_minutes``. Over an area the size of
    Pittsburgh the two agree to well under a second, so the choice between them
    is not what drives any result here.
    """
    km_per_degree_lat = 111.0
    km_per_degree_lon = 111.0 * np.cos(np.radians(reference_latitude))
    dy = (np.asarray(lat2, dtype=float) - np.asarray(lat1, dtype=float)) * km_per_degree_lat
    dx = (np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float)) * km_per_degree_lon
    return np.sqrt(dx**2 + dy**2) / speed_kmh * 60.0


def build_time_matrix(
    latitudes: Sequence[float],
    longitudes: Sequence[float],
    time_function: TimeFunction = haversine_minutes,
    **kwargs,
) -> np.ndarray:
    """Full ``n x n`` travel-time matrix, vectorised by broadcasting.

    ``time_matrix[i, j]`` is the time from demand point ``i`` to a station at
    candidate ``j``. The matrix is symmetric under the distance functions here.
    """
    lat = np.asarray(latitudes, dtype=float)
    lon = np.asarray(longitudes, dtype=float)
    return time_function(lat[:, None], lon[:, None], lat[None, :], lon[None, :], **kwargs)


def build_coverage_matrix(time_matrix: np.ndarray, max_minutes: float) -> np.ndarray:
    """Binary coverage: ``1`` where candidate ``j`` reaches demand point ``i`` in time."""
    return (time_matrix <= max_minutes).astype(np.int8)


def coverage_summary(coverage: np.ndarray) -> dict[str, float]:
    """Shape of the coverage matrix, as a sanity check before solving.

    A demand point that no candidate can reach makes plain set cover infeasible,
    so ``uncoverable_demand_points`` is the number to look at first.
    """
    per_candidate = coverage.sum(axis=0)
    per_demand = coverage.sum(axis=1)
    return {
        "demand_points": int(coverage.shape[0]),
        "candidates": int(coverage.shape[1]),
        "density": float(coverage.mean()),
        "mean_covered_per_candidate": float(per_candidate.mean()),
        "min_covered_per_candidate": int(per_candidate.min()),
        "max_covered_per_candidate": int(per_candidate.max()),
        "uncoverable_demand_points": int((per_demand == 0).sum()),
    }
