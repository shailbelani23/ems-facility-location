"""Turn the raw Allegheny County dispatch extract into the tables the models use.

Three raw inputs go in:

* ``data/raw/pittsburgh_dispatch_data.csv``, roughly 2.07 million 911 dispatch
  records for Allegheny County, 2015 to 2024, one row per call, already carrying
  a block group GEOID and that block group's centroid.
* ``data/raw/social_explorer_data.csv``, ACS population and housing by block
  group.
* ``data/geo/tl_2020_42_bg/``, the 2020 TIGER block group shapefile for
  Pennsylvania.

Four tables come out, all small enough to commit:

* ``grouped_dispatch.csv``: call counts by service, priority, quarter, year and
  block group. This is the demand panel.
* ``demand_points.csv``: one row per block group, its centroid, total calls, and
  the 2020 block group its centroid falls inside. This doubles as the demand
  points and the candidate facility sites.
* ``block_group_demographics.csv``: population, race shares, housing, and a
  majority-group label per block group.
* ``block_group_geometry.geojson``: block group polygons clipped to the service
  area, so the maps redraw without the 39 MB statewide shapefile.

A vintage mismatch has to be handled on the way through. The dispatch records
span 2015 to 2024 and carry GEOIDs geocoded against *2010* census block groups,
while the TIGER shapefile and the Social Explorer extract are both *2020*
vintage. Joining the two on the GEOID string, which is the obvious thing to do
and what the first version of this project did, silently drops 87 of the 389
block groups: they simply do not exist as 2020 identifiers. Those 87 carry
26.4% of all calls, so the equity analysis ends up run on three quarters of the
demand with no warning that the quarter it lost is not a random quarter.

``crosswalk_to_2020`` fixes this by joining on geography instead of on strings:
each 2010 centroid is located inside whichever 2020 block group actually
contains it. That resolves all 389 points, and demographic coverage goes from
73.6% of calls to 100%. The 389 demand points collapse onto 332 distinct 2020
block groups, since the 2020 redraw merged some neighbours.

Run with ``python -m src.data_prep`` from the repository root.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
GEO = ROOT / "data" / "geo"

ALLEGHENY_PREFIX = "42003"
CITY = "PITTSBURGH"

# Social Explorer variable codes, from the second header row of the extract.
SOCIAL_COLUMNS = {
    "Geo_FIPS": "geoid",
    "Geo_NAME": "name",
    "SE_A03001_001": "population",
    "SE_A03001_002": "white",
    "SE_A03001_003": "black",
    "SE_A03001_005": "asian",
    "SE_A10024_001": "occupied_housing_units",
}


def load_dispatch() -> pd.DataFrame:
    """Raw dispatch records restricted to the City of Pittsburgh, with clean GEOIDs.

    GEOIDs arrive as floats in the export, which silently drops the leading zero
    on some states and turns the identifier into something that will not join.
    They are coerced to numeric, rounded to integer, and only then stringified.
    """
    dispatch = pd.read_csv(RAW / "pittsburgh_dispatch_data.csv", low_memory=False)
    dispatch = dispatch[dispatch["city_name"] == CITY].copy()

    dispatch["geoid"] = pd.to_numeric(dispatch["geoid"], errors="coerce")
    dispatch = dispatch.dropna(subset=["geoid"])
    dispatch["geoid"] = dispatch["geoid"].astype("int64").astype(str)
    dispatch = dispatch[dispatch["geoid"].str.startswith(ALLEGHENY_PREFIX)]

    return dispatch


def build_grouped_dispatch(dispatch: pd.DataFrame) -> pd.DataFrame:
    """Call counts by service, priority, quarter, year and block group."""
    return (
        dispatch.groupby(["service", "priority", "call_quarter", "call_year", "geoid"])
        .size()
        .reset_index(name="record_count")
    )


def build_demand_points(dispatch: pd.DataFrame) -> pd.DataFrame:
    """One row per block group: centroid and total call volume.

    Every block group in the service area is simultaneously a demand point and a
    candidate station site. That is a modelling choice, not a data limitation:
    it means the optimiser picks from 389 real, populated locations rather than
    from an arbitrary grid, and every reported travel time is centroid to
    centroid.
    """
    points = (
        dispatch.groupby("geoid")
        .agg(
            lon=("census_block_group_center__x", "first"),
            lat=("census_block_group_center__y", "first"),
            call_count=("_id", "count"),
        )
        .reset_index()
        .dropna(subset=["lat", "lon"])
        .sort_values("geoid")
        .reset_index(drop=True)
    )
    return points


def build_demographics() -> pd.DataFrame:
    """Population, race shares, housing and a majority-group label per block group.

    The majority label is assigned when one group exceeds 50% of the block
    group's population, otherwise the block group is Other or mixed. Blocks with
    no population are kept and labelled separately rather than dropped, because
    a few of them still generate calls, for example along commercial corridors.
    """
    social = pd.read_csv(RAW / "social_explorer_data.csv", header=1, low_memory=False)
    social = social.rename(columns=SOCIAL_COLUMNS)[list(SOCIAL_COLUMNS.values())]
    social["geoid"] = social["geoid"].astype(str)

    for column in ("population", "white", "black", "asian", "occupied_housing_units"):
        social[column] = pd.to_numeric(social[column], errors="coerce").fillna(0).astype(int)

    social = (
        social[social["geoid"].str.startswith(ALLEGHENY_PREFIX)]
        .drop_duplicates(subset="geoid", keep="first")
        .reset_index(drop=True)
    )

    has_population = social["population"] > 0
    for group in ("white", "black", "asian"):
        social[f"pct_{group}"] = (
            (100 * social[group] / social["population"]).where(has_population)
        )
    social["avg_household_size"] = (
        social["population"] / social["occupied_housing_units"]
    ).where(social["occupied_housing_units"] > 0)

    social["demographic_group"] = "Other or mixed"
    social.loc[social["pct_white"] >= 50, "demographic_group"] = "Majority White"
    social.loc[social["pct_black"] >= 50, "demographic_group"] = "Majority Black"
    social.loc[social["pct_asian"] >= 50, "demographic_group"] = "Majority Asian"
    social.loc[~has_population, "demographic_group"] = "No population"

    return social


def load_block_groups_2020() -> gpd.GeoDataFrame:
    """2020 TIGER block groups for Allegheny County, in WGS 84."""
    shapes = gpd.read_file(GEO / "tl_2020_42_bg" / "tl_2020_42_bg.shp")
    shapes["GEOID"] = shapes["GEOID"].astype(str)
    shapes = shapes[shapes["GEOID"].str.startswith(ALLEGHENY_PREFIX)].copy()
    return shapes[["GEOID", "NAMELSAD", "geometry"]].to_crs("EPSG:4326").reset_index(drop=True)


def crosswalk_to_2020(points: pd.DataFrame, block_groups: gpd.GeoDataFrame) -> pd.DataFrame:
    """Attach the 2020 block group that contains each demand point's centroid.

    This is a point-in-polygon join, not an areal-weighted crosswalk. For
    attaching demographics to a centroid that is what is wanted anyway, and it
    avoids depending on an external crosswalk file. It does mean a 2010 block
    group that straddles a redrawn 2020 boundary is attributed wholly to
    whichever side its centroid landed on.

    The demand points themselves keep their original centroids, so the coverage
    matrix and every optimisation result are untouched by this step. Only the
    demographic join and the map polygons depend on it.
    """
    located = gpd.GeoDataFrame(
        points.copy(),
        geometry=gpd.points_from_xy(points["lon"], points["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(located, block_groups[["GEOID", "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset="geoid", keep="first")

    result = points.copy()
    result["geoid_2020"] = joined.set_index("geoid").loc[points["geoid"], "GEOID"].to_numpy()

    unmatched = int(result["geoid_2020"].isna().sum())
    if unmatched:
        print(f"  Warning: {unmatched} demand points fall outside every 2020 block group.")
    return result


# About 2 m at this latitude. TIGER polygons trace individual property lines,
# which is far more detail than a city-wide choropleth can render, and the
# vertices are duplicated once per map layer. Simplifying at this tolerance
# shrinks the exported maps roughly threefold and moves total area by 0.0015%.
GEOMETRY_TOLERANCE_DEGREES = 2e-5


def build_geometry(
    block_groups: gpd.GeoDataFrame, service_area_geoids: set[str]
) -> gpd.GeoDataFrame:
    """The 2020 polygons the service area maps onto, simplified for display.

    Only the maps consume this. Nothing in the optimisation touches polygon
    geometry, so the simplification cannot move any reported result.
    """
    frame = block_groups[block_groups["GEOID"].isin(service_area_geoids)].copy()
    frame["geometry"] = frame.geometry.simplify(
        GEOMETRY_TOLERANCE_DEGREES, preserve_topology=True
    )
    return frame.reset_index(drop=True)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    print("Loading raw dispatch records ...")
    dispatch = load_dispatch()
    print(f"  Pittsburgh dispatch records: {len(dispatch):,}")

    grouped = build_grouped_dispatch(dispatch)
    grouped.to_csv(PROCESSED / "grouped_dispatch.csv", index=False)
    print(f"  Demand panel rows: {len(grouped):,}")

    points = build_demand_points(dispatch)
    print(f"  Block groups in service area: {len(points)}")
    print(f"  Total calls: {points['call_count'].sum():,}")

    block_groups = load_block_groups_2020()
    points = crosswalk_to_2020(points, block_groups)
    points.to_csv(PROCESSED / "demand_points.csv", index=False)

    naive_match = points["geoid"].isin(set(block_groups["GEOID"]))
    print(
        f"  Joining on the GEOID string alone would match {int(naive_match.sum())} of "
        f"{len(points)} points, holding "
        f"{100 * points.loc[naive_match, 'call_count'].sum() / points['call_count'].sum():.1f}% "
        "of calls."
    )
    print(
        f"  Joining on geography matches {int(points['geoid_2020'].notna().sum())} of "
        f"{len(points)} points, onto {points['geoid_2020'].nunique()} distinct 2020 block groups."
    )

    demographics = build_demographics()
    demographics.to_csv(PROCESSED / "block_group_demographics.csv", index=False)
    covered = points["geoid_2020"].isin(set(demographics["geoid"]))
    print(
        f"  Demand points with demographics: {int(covered.sum())} of {len(points)}, holding "
        f"{100 * points.loc[covered, 'call_count'].sum() / points['call_count'].sum():.1f}% of calls."
    )

    geometry = build_geometry(block_groups, set(points["geoid_2020"].dropna()))
    geometry.to_file(PROCESSED / "block_group_geometry.geojson", driver="GeoJSON")
    print(f"  Block groups with geometry: {len(geometry)}")


if __name__ == "__main__":
    main()
