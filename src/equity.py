"""Who waits longer, and by how much.

A siting model that only reports mean response time can hide a great deal. The
objective in ``models.py`` is blind to demographics by construction, so any
disparity that shows up here is a property of where demand and geography put
the stations, not of anything the optimiser was told to care about. These
helpers make that disparity visible.

Response times are reported both unweighted, one row per block group, and
weighted by call volume, which is what an individual caller actually
experiences. The two can disagree, and when they do the weighted number is the
one that matters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROUP_ORDER = [
    "Majority White",
    "Majority Black",
    "Majority Asian",
    "Other or mixed",
    "No population",
]


def attach_demographics(
    demand_points: pd.DataFrame,
    demographics: pd.DataFrame,
    response_minutes: np.ndarray,
) -> pd.DataFrame:
    """Join response times onto block group demographics via the 2020 crosswalk."""
    table = demand_points.copy()
    table["response_minutes"] = np.asarray(response_minutes, dtype=float)

    columns = ["geoid", "demographic_group", "population", "pct_white", "pct_black", "pct_asian"]
    merged = table.merge(
        demographics[columns].rename(columns={"geoid": "geoid_2020"}),
        on="geoid_2020",
        how="left",
    )
    merged["demographic_group"] = merged["demographic_group"].fillna("Unmatched")
    return merged


def response_by_group(
    table: pd.DataFrame,
    threshold_minutes: float,
    benchmark_minutes: float = 5.0,
) -> pd.DataFrame:
    """Response-time distribution per demographic group.

    ``Call-weighted mean minutes`` weights each block group by its call volume,
    so it answers "how long does the average call wait" rather than "how long
    does the average block group wait".

    Two coverage columns are reported. The threshold column is the constraint
    the model was actually given, and once a solution satisfies it that column
    reads 100% for everyone and stops distinguishing anything. The benchmark
    column uses a tighter time, and it is the one that shows which groups are
    comfortably inside the standard and which are only just inside it.
    """
    rows = []
    for group, subset in table.groupby("demographic_group"):
        calls = subset["call_count"].to_numpy(dtype=float)
        times = subset["response_minutes"].to_numpy(dtype=float)
        total = calls.sum()
        rows.append(
            {
                "Group": group,
                "Block groups": len(subset),
                "Calls": int(total),
                "Mean minutes": times.mean(),
                "Call-weighted mean minutes": float(np.average(times, weights=calls))
                if total > 0
                else np.nan,
                "Median minutes": float(np.median(times)),
                "P95 minutes": float(np.percentile(times, 95)),
                f"% calls within {threshold_minutes:g} min": 100.0
                * calls[times <= threshold_minutes].sum() / total
                if total > 0
                else np.nan,
                f"% calls within {benchmark_minutes:g} min": 100.0
                * calls[times <= benchmark_minutes].sum() / total
                if total > 0
                else np.nan,
            }
        )

    result = pd.DataFrame(rows)
    result["_order"] = result["Group"].apply(
        lambda g: GROUP_ORDER.index(g) if g in GROUP_ORDER else len(GROUP_ORDER)
    )
    return result.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def disparity(summary: pd.DataFrame, column: str = "Call-weighted mean minutes") -> float:
    """Spread between the best and worst served populated groups, in minutes."""
    populated = summary[summary["Group"].isin(GROUP_ORDER[:4])][column].dropna()
    if populated.empty:
        return float("nan")
    return float(populated.max() - populated.min())
