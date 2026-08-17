"""Solve every model, write every table, figure and map.

Usage (from the repository root, after ``python -m src.data_prep``):

    python -m src.run_analysis

Outputs land in ``results/``. Every number quoted in README.md comes from here.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.coverage import (  # noqa: E402
    DEFAULT_SPEED_KMH,
    build_coverage_matrix,
    build_time_matrix,
    coverage_summary,
    euclidean_minutes,
    haversine_minutes,
)
from src.equity import attach_demographics, disparity, response_by_group  # noqa: E402
from src.mapping import solution_map  # noqa: E402
from src.models import (  # noqa: E402
    assign_to_nearest,
    solve_max_cover,
    solve_set_cover,
    solve_set_cover_relaxation,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MAPS = RESULTS / "maps"

THRESHOLD_MINUTES = 10.0
BENCHMARK_MINUTES = 5.0
SENSITIVITY_THRESHOLDS = [5, 6, 7, 8, 9, 10, 12, 15, 20]
SPEED_SENSITIVITY_KMH = [30.0, 35.0, 40.0, 45.0, 50.0]
P_VALUES = [1, 2, 3, 4, 5]
MCLP_P = 3

SERIOUS = ["E0", "E1", "E2"]
ROUTINE = ["E3", "E4", "E5"]

PRIMARY = "#B03A2E"
SECONDARY = "#2E5C8A"
ACCENT = "#3D8B5F"


# Loading

def load_inputs():
    points = pd.read_csv(PROCESSED / "demand_points.csv", dtype={"geoid": str, "geoid_2020": str})
    demographics = pd.read_csv(PROCESSED / "block_group_demographics.csv", dtype={"geoid": str})
    dispatch = pd.read_csv(PROCESSED / "grouped_dispatch.csv", dtype={"geoid": str})
    geometry = gpd.read_file(PROCESSED / "block_group_geometry.geojson")
    geometry["GEOID"] = geometry["GEOID"].astype(str)
    return points, demographics, dispatch, geometry


# Models

def run_threshold_sensitivity(points: pd.DataFrame, time_matrix: np.ndarray) -> pd.DataFrame:
    """How many stations plain set cover needs, as the standard tightens."""
    rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        coverage = build_coverage_matrix(time_matrix, threshold)
        stats = coverage_summary(coverage)
        if stats["uncoverable_demand_points"] > 0:
            rows.append(
                {
                    "Threshold (min)": threshold,
                    "Stations": np.nan,
                    "LP bound": np.nan,
                    "Integrality gap": np.nan,
                    "Solve seconds": np.nan,
                    "Note": f"{stats['uncoverable_demand_points']} demand points unreachable",
                }
            )
            continue

        solution = solve_set_cover(coverage)
        lp_bound = solve_set_cover_relaxation(coverage)
        rows.append(
            {
                "Threshold (min)": threshold,
                "Stations": int(solution.objective),
                "LP bound": round(lp_bound, 3),
                "Integrality gap": round(solution.objective - lp_bound, 3),
                "Solve seconds": round(solution.runtime, 3),
                "Note": "",
            }
        )
        print(
            f"  {threshold:>2} min -> {int(solution.objective)} stations "
            f"(LP bound {lp_bound:.2f}, {solution.runtime:.2f}s)"
        )
    return pd.DataFrame(rows)


def run_speed_sensitivity(points: pd.DataFrame) -> pd.DataFrame:
    """The assumed speed is the softest input, so vary it and report the swing."""
    rows = []
    for speed in SPEED_SENSITIVITY_KMH:
        matrix = build_time_matrix(
            points["lat"], points["lon"], haversine_minutes, speed_kmh=speed
        )
        coverage = build_coverage_matrix(matrix, THRESHOLD_MINUTES)
        if coverage_summary(coverage)["uncoverable_demand_points"] > 0:
            continue
        solution = solve_set_cover(coverage)
        response, _ = assign_to_nearest(matrix, solution.selected)
        rows.append(
            {
                "Speed (km/h)": speed,
                "Reach at 10 min (km)": round(speed / 60.0 * THRESHOLD_MINUTES, 2),
                "Stations": int(solution.objective),
                "Mean response (min)": round(float(response.mean()), 2),
                "Max response (min)": round(float(response.max()), 2),
            }
        )
        print(f"  {speed:g} km/h -> {int(solution.objective)} stations")
    return pd.DataFrame(rows)


def run_mclp_curve(points: pd.DataFrame, coverage: np.ndarray) -> pd.DataFrame:
    """Coverage against budget, under both MCLP objectives.

    The two objectives answer different questions. Maximising covered block
    groups treats every block group as one unit regardless of how much demand it
    generates; maximising covered calls weights each by its call volume. The
    columns are reported side by side so the difference is visible rather than
    assumed away.
    """
    calls = points["call_count"].to_numpy(dtype=float)
    total_calls = calls.sum()
    n_points = len(points)

    rows = [
        {
            "p": 0,
            "Blocks covered (block objective)": 0,
            "% blocks (block objective)": 0.0,
            "% calls (block objective)": 0.0,
            "% calls (call objective)": 0.0,
            "Same stations": True,
        }
    ]

    for p in P_VALUES:
        by_block = solve_max_cover(coverage, p, weights=None)
        by_call = solve_max_cover(coverage, p, weights=calls)

        rows.append(
            {
                "p": p,
                "Blocks covered (block objective)": len(by_block.covered),
                "% blocks (block objective)": 100.0 * len(by_block.covered) / n_points,
                "% calls (block objective)": 100.0 * calls[by_block.covered].sum() / total_calls,
                "% calls (call objective)": 100.0 * by_call.objective / total_calls,
                "Same stations": set(by_block.selected) == set(by_call.selected),
            }
        )
        print(
            f"  p={p}: block objective covers {len(by_block.covered)}/{n_points} blocks and "
            f"{100 * calls[by_block.covered].sum() / total_calls:5.1f}% of calls; "
            f"call objective covers {100 * by_call.objective / total_calls:5.1f}% of calls; "
            f"same stations: {set(by_block.selected) == set(by_call.selected)}"
        )

    return pd.DataFrame(rows)


# Figures

def figure_threshold_curve(sensitivity: pd.DataFrame) -> None:
    valid = sensitivity.dropna(subset=["Stations"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        valid["Threshold (min)"], valid["Stations"],
        marker="o", markersize=9, linewidth=2, color=PRIMARY, label="Integer optimum",
    )
    ax.plot(
        valid["Threshold (min)"], valid["LP bound"],
        marker="", linewidth=1.5, linestyle="--", color=SECONDARY, label="LP relaxation bound",
    )
    for _, row in valid.iterrows():
        ax.annotate(
            f"{int(row['Stations'])}",
            (row["Threshold (min)"], row["Stations"]),
            textcoords="offset points", xytext=(0, 11), ha="center", fontsize=10,
        )
    ax.axvline(THRESHOLD_MINUTES, color="grey", linestyle=":", linewidth=1)
    ax.set_xlabel("Response-time standard (minutes)")
    ax.set_ylabel("Stations required for full coverage")
    ax.set_title("Cost of the standard: flat from 8 to 10 minutes, then steep below 8")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "threshold_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_mclp_curve(curve: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.38
    x = np.arange(len(curve))

    ax.bar(
        x - width / 2, curve["% blocks (block objective)"], width,
        label="% of block groups covered", color=SECONDARY, edgecolor="black", linewidth=0.6,
    )
    ax.bar(
        x + width / 2, curve["% calls (call objective)"], width,
        label="% of call volume covered", color=PRIMARY, edgecolor="black", linewidth=0.6,
    )
    for index, row in curve.iterrows():
        ax.annotate(
            f"{row['% blocks (block objective)']:.0f}%",
            (index - width / 2, row["% blocks (block objective)"]),
            ha="center", va="bottom", fontsize=9,
        )
        ax.annotate(
            f"{row['% calls (call objective)']:.0f}%",
            (index + width / 2, row["% calls (call objective)"]),
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(curve["p"])
    ax.set_xlabel("Stations opened")
    ax.set_ylabel("% covered within 10 minutes")
    ax.set_ylim(0, 108)
    ax.set_title("Maximal covering: almost all of the benefit arrives in the first two stations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "mclp_coverage_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_equity(set_cover_stats: pd.DataFrame, mclp_stats: pd.DataFrame) -> None:
    groups = [g for g in set_cover_stats["Group"] if g != "No population"]
    counts = set_cover_stats.set_index("Group")["Block groups"]
    tick_labels = [f"{group}\n(n={counts[group]})" for group in groups]
    x = np.arange(len(groups))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, column, title in [
        (axes[0], "Call-weighted mean minutes", "Call-weighted mean response time"),
        (
            axes[1],
            f"% calls within {BENCHMARK_MINUTES:g} min",
            f"% of calls reached within {BENCHMARK_MINUTES:g} minutes",
        ),
    ]:
        for index, (stats, label, color) in enumerate(
            [(set_cover_stats, "Set cover (4 stations)", PRIMARY), (mclp_stats, "MCLP (3 stations)", SECONDARY)]
        ):
            values = stats.set_index("Group").reindex(groups)[column]
            ax.bar(
                x + width * (index - 0.5), values, width,
                label=label, color=color, edgecolor="black", linewidth=0.6,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Minutes")
    axes[1].set_ylabel("% of calls")
    axes[1].set_ylim(0, 105)
    axes[0].legend()
    fig.suptitle(
        "Response time by block group demographic majority. "
        "The two models trade the disparity in opposite directions.",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "equity_by_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure_demand_profile(points: pd.DataFrame, dispatch: pd.DataFrame, demographics: pd.DataFrame) -> None:
    """Where and when the calls are, and how concentrated they are."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # Call volume over time, split by acuity.
    panel = dispatch.copy()
    panel["band"] = np.where(
        panel["priority"].isin(SERIOUS), "Serious (E0-E2)",
        np.where(panel["priority"].isin(ROUTINE), "Routine (E3-E5)", "Other"),
    )
    yearly = panel.groupby(["call_year", "band"])["record_count"].sum().reset_index()
    for band, color in [("Serious (E0-E2)", PRIMARY), ("Routine (E3-E5)", SECONDARY)]:
        subset = yearly[yearly["band"] == band]
        axes[0].plot(subset["call_year"], subset["record_count"], marker="o", label=band, color=color)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Calls")
    axes[0].set_title("Call volume by acuity")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Concentration: the Lorenz curve of demand across block groups.
    calls = np.sort(points["call_count"].to_numpy(dtype=float))
    n = len(calls)
    cumulative = np.concatenate([[0], np.cumsum(calls) / calls.sum()])
    share = np.linspace(0, 1, len(cumulative))
    gini = 2 * np.sum(np.arange(1, n + 1) * calls) / (n * calls.sum()) - (n + 1) / n
    top_decile = calls[-max(1, n // 10) :].sum() / calls.sum()

    axes[1].plot(share, cumulative, color=PRIMARY, linewidth=2)
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    axes[1].fill_between(share, cumulative, share, alpha=0.15, color=PRIMARY)
    axes[1].annotate(
        f"Gini {gini:.2f}\nBusiest 10% of block groups\ncarry {100 * top_decile:.0f}% of calls",
        xy=(0.9, cumulative[int(0.9 * len(cumulative))]), xytext=(0.12, 0.70),
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1), fontsize=10,
    )
    axes[1].set_xlabel("Cumulative share of block groups, least busy first")
    axes[1].set_ylabel("Cumulative share of calls")
    axes[1].set_title("Demand is uneven but not dominated by a few hotspots")
    axes[1].grid(alpha=0.3)

    # Calls per capita by demographic majority. Bars are annotated with the
    # number of block groups behind them, because one of these groups is a
    # single block group and its rate should not be read as a group-level fact.
    joined = points.merge(
        demographics[["geoid", "demographic_group", "population"]].rename(columns={"geoid": "geoid_2020"}),
        on="geoid_2020", how="left",
    )
    joined["demographic_group"] = joined["demographic_group"].fillna("Unmatched")
    populated = joined[joined["population"] > 0]
    per_capita = (
        populated.groupby("demographic_group")
        .apply(lambda g: 1000 * g["call_count"].sum() / g["population"].sum(), include_groups=False)
        .sort_values(ascending=False)
    )
    counts = populated.groupby("demographic_group")["geoid_2020"].nunique()

    axes[2].bar(
        range(len(per_capita)), per_capita.to_numpy(),
        color=ACCENT, edgecolor="black", linewidth=0.6,
    )
    for index, group in enumerate(per_capita.index):
        axes[2].annotate(
            f"n={counts[group]}",
            (index, per_capita[group]), ha="center", va="bottom", fontsize=9,
        )
    axes[2].set_xticks(range(len(per_capita)))
    axes[2].set_xticklabels(per_capita.index, rotation=20, ha="right")
    axes[2].set_ylabel(f"Calls per 1,000 residents, {int(yearly['call_year'].min())}-{int(yearly['call_year'].max())}")
    axes[2].set_title("Call rate per capita, n = block groups behind each bar")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "demand_profile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Maps

def build_map_frame(
    geometry: gpd.GeoDataFrame,
    points: pd.DataFrame,
    demographics: pd.DataFrame,
    response: np.ndarray,
) -> gpd.GeoDataFrame:
    """Aggregate demand-point results up to the 2020 polygons used for drawing."""
    table = points.copy()
    table["response_minutes"] = response

    aggregated = (
        table.groupby("geoid_2020")
        .agg(calls_total=("call_count", "sum"), response_minutes=("response_minutes", "mean"))
        .reset_index()
        .rename(columns={"geoid_2020": "GEOID"})
    )

    frame = geometry.merge(aggregated, on="GEOID", how="left")
    frame = frame.merge(
        demographics.rename(columns={"geoid": "GEOID"})[
            ["GEOID", "population", "pct_white", "pct_black", "pct_asian", "avg_household_size"]
        ],
        on="GEOID", how="left",
    )
    frame["calls_total"] = frame["calls_total"].fillna(0)
    frame["calls_per_1k"] = np.where(
        frame["population"] > 0, 1000 * frame["calls_total"] / frame["population"], np.nan
    )
    return frame


def station_table(points: pd.DataFrame, selected: list[int]) -> pd.DataFrame:
    stations = points.iloc[selected].copy().reset_index(drop=True)
    stations["station_id"] = range(1, len(stations) + 1)
    return stations[["station_id", "geoid", "geoid_2020", "lat", "lon", "call_count"]]


# Main

def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)

    points, demographics, dispatch, geometry = load_inputs()
    print(f"Demand points: {len(points)}   Total calls: {points['call_count'].sum():,}")

    time_matrix = build_time_matrix(points["lat"], points["lon"], haversine_minutes)
    euclidean = build_time_matrix(points["lat"], points["lon"], euclidean_minutes)
    print(
        "Haversine against flat-earth travel time: max disagreement "
        f"{np.abs(time_matrix - euclidean).max():.4f} minutes"
    )

    coverage = build_coverage_matrix(time_matrix, THRESHOLD_MINUTES)
    stats = coverage_summary(coverage)
    print(
        f"Coverage matrix at {THRESHOLD_MINUTES:g} min: {stats['demand_points']} x "
        f"{stats['candidates']}, density {stats['density']:.3f}, "
        f"a candidate reaches {stats['min_covered_per_candidate']} to "
        f"{stats['max_covered_per_candidate']} demand points"
    )

    # --- Set cover -------------------------------------------------------
    print("\nSet cover:")
    cover_solution = solve_set_cover(coverage)
    lp_bound = solve_set_cover_relaxation(coverage)
    print(
        f"  {int(cover_solution.objective)} stations, LP bound {lp_bound:.3f}, "
        f"gap {cover_solution.objective - lp_bound:.3f}, solved in {cover_solution.runtime:.2f}s "
        f"({cover_solution.n_variables} vars, {cover_solution.n_constraints} constraints)"
    )

    cover_response, _ = assign_to_nearest(time_matrix, cover_solution.selected)
    cover_stations = station_table(points, cover_solution.selected)
    cover_stations.to_csv(RESULTS / "stations_set_cover.csv", index=False)

    weighted_mean = float(np.average(cover_response, weights=points["call_count"]))
    print(
        f"  Response time: mean {cover_response.mean():.2f} min, "
        f"call-weighted {weighted_mean:.2f} min, max {cover_response.max():.2f} min"
    )

    # --- MCLP ------------------------------------------------------------
    print(f"\nMaximal covering with p={MCLP_P}:")
    calls = points["call_count"].to_numpy(dtype=float)
    mclp_solution = solve_max_cover(coverage, MCLP_P, weights=None)
    mclp_response, _ = assign_to_nearest(time_matrix, mclp_solution.selected)
    mclp_stations = station_table(points, mclp_solution.selected)
    mclp_stations.to_csv(RESULTS / "stations_mclp.csv", index=False)

    uncovered_index = sorted(set(range(len(points))) - set(mclp_solution.covered))
    uncovered = points.iloc[uncovered_index].copy()
    uncovered["response_minutes"] = mclp_response[uncovered_index]
    print(
        f"  {len(mclp_solution.covered)}/{len(points)} block groups covered, "
        f"{100 * calls[mclp_solution.covered].sum() / calls.sum():.2f}% of calls, "
        f"{len(uncovered)} block group(s) left out carrying {int(uncovered['call_count'].sum()):,} calls"
    )

    # --- Sensitivity -----------------------------------------------------
    print("\nThreshold sensitivity:")
    threshold_table = run_threshold_sensitivity(points, time_matrix)
    threshold_table.to_csv(RESULTS / "threshold_sensitivity.csv", index=False)

    print("\nSpeed sensitivity at a 10-minute standard:")
    speed_table = run_speed_sensitivity(points)
    speed_table.to_csv(RESULTS / "speed_sensitivity.csv", index=False)

    print("\nMaximal covering, block objective against call objective:")
    mclp_curve = run_mclp_curve(points, coverage)
    mclp_curve.to_csv(RESULTS / "mclp_coverage_curve.csv", index=False)

    # --- Equity ----------------------------------------------------------
    print("\nEquity:")
    cover_equity = response_by_group(
        attach_demographics(points, demographics, cover_response), THRESHOLD_MINUTES, BENCHMARK_MINUTES
    )
    mclp_equity = response_by_group(
        attach_demographics(points, demographics, mclp_response), THRESHOLD_MINUTES, BENCHMARK_MINUTES
    )
    cover_equity.to_csv(RESULTS / "equity_set_cover.csv", index=False)
    mclp_equity.to_csv(RESULTS / "equity_mclp.csv", index=False)
    print("  Set cover:")
    print(cover_equity.to_string(index=False, float_format="%.2f"))
    print("  MCLP:")
    print(mclp_equity.to_string(index=False, float_format="%.2f"))
    print(
        f"  Call-weighted spread between best and worst served group: "
        f"set cover {disparity(cover_equity):.2f} min, MCLP {disparity(mclp_equity):.2f} min"
    )

    # --- Figures and maps ------------------------------------------------
    figure_threshold_curve(threshold_table)
    figure_mclp_curve(mclp_curve)
    figure_equity(cover_equity, mclp_equity)
    figure_demand_profile(points, dispatch, demographics)

    cover_frame = build_map_frame(geometry, points, demographics, cover_response)
    mclp_frame = build_map_frame(geometry, points, demographics, mclp_response)

    solution_map(
        cover_frame, cover_stations, THRESHOLD_MINUTES, DEFAULT_SPEED_KMH,
        station_label="Ambulance stations (set cover)",
    ).save(MAPS / "set_cover_solution.html")
    solution_map(
        mclp_frame, mclp_stations, THRESHOLD_MINUTES, DEFAULT_SPEED_KMH,
        uncovered=uncovered, station_label="Ambulance stations (MCLP)",
    ).save(MAPS / "mclp_solution.html")

    for path in sorted(MAPS.glob("*.html")):
        print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB")

    print(f"\nTables in {RESULTS}, figures in {FIGURES}, maps in {MAPS}")


if __name__ == "__main__":
    main()
