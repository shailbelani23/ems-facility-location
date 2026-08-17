"""Invariants for the coverage geometry and the two siting models.

These are property checks rather than golden-value checks. A golden value only
tells you the code changed; these tell you which guarantee broke. The one
golden-value test is the Pittsburgh instance itself, which pins the headline
result so that a refactor cannot quietly move it.

Run with ``pytest tests/`` from the repository root.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.coverage import (
    build_coverage_matrix,
    build_time_matrix,
    coverage_summary,
    euclidean_minutes,
    haversine_km,
    haversine_minutes,
)
from src.models import (
    assign_to_nearest,
    solve_max_cover,
    solve_set_cover,
    solve_set_cover_relaxation,
)

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

DOWNTOWN = (40.4406, -79.9959)
OAKLAND = (40.4418, -79.9530)


@pytest.fixture(scope="module")
def pittsburgh():
    """The real instance: 389 demand points and their 10-minute coverage matrix."""
    points = pd.read_csv(
        PROCESSED / "demand_points.csv", dtype={"geoid": str, "geoid_2020": str}
    )
    time_matrix = build_time_matrix(points["lat"], points["lon"], haversine_minutes)
    return points, time_matrix, build_coverage_matrix(time_matrix, 10.0)


# Geometry

def test_haversine_matches_known_distance() -> None:
    """Downtown to Oakland is about 3.6 km along the Boulevard of the Allies."""
    distance = float(haversine_km(*DOWNTOWN, *OAKLAND))
    assert 3.4 < distance < 3.8


def test_distance_to_self_is_zero() -> None:
    assert float(haversine_km(*DOWNTOWN, *DOWNTOWN)) == pytest.approx(0.0, abs=1e-9)


def test_time_matrix_is_symmetric_with_zero_diagonal(pittsburgh) -> None:
    _, time_matrix, _ = pittsburgh
    assert np.allclose(time_matrix, time_matrix.T)
    assert np.allclose(np.diag(time_matrix), 0.0)


def test_haversine_and_euclidean_agree_over_a_city(pittsburgh) -> None:
    """At this scale the projection choice cannot be what drives a result."""
    points, time_matrix, _ = pittsburgh
    flat = build_time_matrix(points["lat"], points["lon"], euclidean_minutes)
    assert np.abs(time_matrix - flat).max() < 0.1


def test_speed_scales_time_inversely() -> None:
    fast = float(haversine_minutes(*DOWNTOWN, *OAKLAND, speed_kmh=80.0))
    slow = float(haversine_minutes(*DOWNTOWN, *OAKLAND, speed_kmh=40.0))
    assert slow == pytest.approx(2.0 * fast, rel=1e-9)


def test_coverage_is_monotone_in_the_threshold(pittsburgh) -> None:
    """Relaxing the standard can only ever add coverage, never remove it."""
    _, time_matrix, _ = pittsburgh
    tight = build_coverage_matrix(time_matrix, 7.0)
    loose = build_coverage_matrix(time_matrix, 10.0)
    assert np.all(loose >= tight)


def test_every_candidate_covers_itself(pittsburgh) -> None:
    _, _, coverage = pittsburgh
    assert np.all(np.diag(coverage) == 1)


# Set cover

def test_set_cover_actually_covers_everything(pittsburgh) -> None:
    _, _, coverage = pittsburgh
    solution = solve_set_cover(coverage)
    assert np.all(coverage[:, solution.selected].sum(axis=1) >= 1)


def test_set_cover_is_not_beaten_by_dropping_a_station(pittsburgh) -> None:
    """Minimality: no proper subset of the solution is still feasible."""
    _, _, coverage = pittsburgh
    solution = solve_set_cover(coverage)
    for dropped in solution.selected:
        remaining = [j for j in solution.selected if j != dropped]
        assert not np.all(coverage[:, remaining].sum(axis=1) >= 1)


def test_lp_bound_lies_below_the_integer_optimum(pittsburgh) -> None:
    _, _, coverage = pittsburgh
    solution = solve_set_cover(coverage)
    assert solve_set_cover_relaxation(coverage) <= solution.objective + 1e-6


def test_set_cover_rejects_an_unreachable_demand_point() -> None:
    """An isolated demand point must raise rather than be silently skipped."""
    coverage = np.eye(3, dtype=np.int8)
    coverage[2, 2] = 0
    with pytest.raises(ValueError, match="unreachable"):
        solve_set_cover(coverage)


def test_set_cover_needs_one_station_when_one_reaches_everything() -> None:
    coverage = np.ones((6, 6), dtype=np.int8)
    assert solve_set_cover(coverage).n_facilities == 1


def test_set_cover_needs_n_stations_when_nothing_overlaps() -> None:
    coverage = np.eye(5, dtype=np.int8)
    assert solve_set_cover(coverage).n_facilities == 5


# Maximal covering

def test_max_cover_opens_exactly_p(pittsburgh) -> None:
    _, _, coverage = pittsburgh
    for p in (1, 2, 3):
        assert solve_max_cover(coverage, p).n_facilities == p


def test_max_cover_is_monotone_in_p(pittsburgh) -> None:
    """More budget cannot cover less."""
    _, _, coverage = pittsburgh
    values = [solve_max_cover(coverage, p).objective for p in (1, 2, 3, 4)]
    assert all(a <= b + 1e-9 for a, b in zip(values, values[1:]))


def test_max_cover_marks_only_genuinely_covered_points(pittsburgh) -> None:
    _, _, coverage = pittsburgh
    solution = solve_max_cover(coverage, 3)
    assert np.all(coverage[solution.covered][:, solution.selected].sum(axis=1) >= 1)


def test_max_cover_at_full_budget_matches_set_cover(pittsburgh) -> None:
    """With p equal to the set cover optimum, MCLP must reach full coverage."""
    points, _, coverage = pittsburgh
    p = solve_set_cover(coverage).n_facilities
    assert len(solve_max_cover(coverage, p).covered) == len(points)


def test_weighted_objective_never_covers_less_weight(pittsburgh) -> None:
    """Optimising for call volume must not lose to optimising for block count."""
    points, _, coverage = pittsburgh
    calls = points["call_count"].to_numpy(dtype=float)
    by_block = solve_max_cover(coverage, 1, weights=None)
    by_call = solve_max_cover(coverage, 1, weights=calls)
    assert by_call.objective >= calls[by_block.covered].sum() - 1e-6


# Assignment

def test_assignment_picks_the_nearest_station(pittsburgh) -> None:
    _, time_matrix, coverage = pittsburgh
    selected = solve_set_cover(coverage).selected
    response, station = assign_to_nearest(time_matrix, selected)
    assert np.allclose(response, time_matrix[:, selected].min(axis=1))
    assert np.all(np.isin(station, selected))


def test_full_coverage_means_no_one_waits_past_the_threshold(pittsburgh) -> None:
    _, time_matrix, coverage = pittsburgh
    response, _ = assign_to_nearest(time_matrix, solve_set_cover(coverage).selected)
    assert response.max() <= 10.0 + 1e-9


def test_assignment_requires_a_station() -> None:
    with pytest.raises(ValueError):
        assign_to_nearest(np.zeros((3, 3)), [])


# The headline result

def test_pittsburgh_headline_result_is_stable(pittsburgh) -> None:
    """Pins the numbers README.md quotes, so a refactor cannot move them quietly."""
    points, time_matrix, coverage = pittsburgh
    assert len(points) == 389
    assert int(points["call_count"].sum()) == 635_235

    stats = coverage_summary(coverage)
    assert stats["uncoverable_demand_points"] == 0
    assert stats["density"] == pytest.approx(0.530, abs=0.005)

    cover = solve_set_cover(coverage)
    assert cover.n_facilities == 4

    mclp = solve_max_cover(coverage, 3)
    assert len(mclp.covered) == 388

    # The single block group MCLP gives up carries one call in ten years, which
    # is the whole argument for preferring three stations to four.
    missed = sorted(set(range(len(points))) - set(mclp.covered))
    assert points.iloc[missed]["call_count"].sum() == 1
