"""Interactive Folium maps of the demand data and the siting solutions.

One implementation note that matters for repository size. The obvious way to get
per-polygon tooltips is to add one ``folium.GeoJson`` per block group, and that
is what the first version of this project did. It works, but it re-serialises
the polygon geometry once per layer per block group, and the two solution maps
came out at 24 MB and 16 MB. Passing the whole layer to a single
``folium.GeoJson`` with a ``GeoJsonTooltip`` writes the geometry once and
produces the same interaction, at roughly a twentieth of the size.
"""

from __future__ import annotations

import branca.colormap as cm
import folium
import geopandas as gpd
import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd

PITTSBURGH_CENTER = [40.4406, -79.9959]
DEFAULT_ZOOM = 11
BASEMAP = "CartoDB positron"

# Layer key -> (display label, colormap, tooltip format, whether shown on load)
LAYER_SPECS: dict[str, tuple[str, str, str, bool]] = {
    "response_minutes": ("Response time (min)", "RdYlGn_r", ".2f", True),
    "calls_total": ("EMS calls", "YlOrRd", ",.0f", False),
    "calls_per_1k": ("Calls per 1,000 residents", "Purples", ".1f", False),
    "population": ("Population", "Blues", ",.0f", False),
    "pct_black": ("% Black alone", "Oranges", ".1f", False),
    "pct_white": ("% White alone", "Reds", ".1f", False),
    "pct_asian": ("% Asian alone", "YlGn", ".1f", False),
    "avg_household_size": ("People per household", "YlOrBr", ".2f", False),
}


def _colormap(palette: str, low: float, high: float, caption: str) -> cm.LinearColormap:
    """A branca colour scale sampled from a matplotlib colormap.

    Going through matplotlib rather than branca's own named ramps means the
    ``_r`` reversed suffix and the palette names used in the static figures both
    work unchanged, so map and figure colours stay consistent.
    """
    samples = matplotlib.colormaps[palette](np.linspace(0.0, 1.0, 9))
    colors = [matplotlib.colors.to_hex(color) for color in samples]
    scale = cm.LinearColormap(colors, vmin=low, vmax=high)
    scale.caption = caption
    return scale


def _add_choropleth(
    fmap: folium.Map,
    frame: gpd.GeoDataFrame,
    column: str,
    label: str,
    palette: str,
    fmt: str,
    show: bool,
) -> None:
    """One choropleth layer, geometry serialised once."""
    layer = frame[["GEOID", "NAMELSAD", column, "geometry"]].dropna(subset=[column]).copy()
    if layer.empty:
        return

    low, high = float(layer[column].min()), float(layer[column].max())
    if high <= low:
        high = low + 1.0
    scale = _colormap(palette, low, high, label)

    folium.GeoJson(
        layer,
        name=label,
        show=show,
        style_function=lambda feature, col=column, sc=scale: {
            "fillColor": sc(feature["properties"][col]),
            "color": "#666666",
            "weight": 0.4,
            "fillOpacity": 0.65,
        },
        highlight_function=lambda _: {"weight": 2.5, "color": "#000000"},
        tooltip=folium.GeoJsonTooltip(
            fields=["NAMELSAD", "GEOID", column],
            aliases=["Block group", "GEOID", label],
            localize=True,
        ),
    ).add_to(fmap)

    if show:
        scale.add_to(fmap)


def _add_stations(
    fmap: folium.Map,
    stations: pd.DataFrame,
    threshold_minutes: float,
    speed_kmh: float,
    label: str,
) -> None:
    """Station markers, each with the straight-line reach implied by the threshold."""
    group = folium.FeatureGroup(name=label, show=True)
    radius_m = threshold_minutes / 60.0 * speed_kmh * 1000.0

    for _, row in stations.iterrows():
        folium.Circle(
            location=[row["lat"], row["lon"]],
            radius=radius_m,
            color="#B03A2E",
            weight=1,
            fill=True,
            fill_opacity=0.06,
            tooltip=f"{threshold_minutes:g}-minute reach",
        ).add_to(group)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7,
            color="black",
            weight=2,
            fill=True,
            fill_color="#B03A2E",
            fill_opacity=0.95,
            tooltip=f"Station {row['station_id']}",
            popup=folium.Popup(
                f"<b>Station {row['station_id']}</b><br>"
                f"Block group {row['geoid']}<br>"
                f"Local call volume: {row['call_count']:,}",
                max_width=280,
            ),
        ).add_to(group)

    group.add_to(fmap)


def solution_map(
    frame: gpd.GeoDataFrame,
    stations: pd.DataFrame,
    threshold_minutes: float,
    speed_kmh: float,
    uncovered: pd.DataFrame | None = None,
    station_label: str = "Ambulance stations",
) -> folium.Map:
    """Choropleth stack plus station markers for one siting solution."""
    fmap = folium.Map(location=PITTSBURGH_CENTER, zoom_start=DEFAULT_ZOOM, tiles=BASEMAP)

    for column, (label, palette, fmt, show) in LAYER_SPECS.items():
        if column in frame.columns:
            _add_choropleth(fmap, frame, column, label, palette, fmt, show)

    _add_stations(fmap, stations, threshold_minutes, speed_kmh, station_label)

    if uncovered is not None and len(uncovered):
        group = folium.FeatureGroup(name="Demand points left uncovered", show=True)
        for _, row in uncovered.iterrows():
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=5,
                color="#333333",
                weight=1.5,
                fill=True,
                fill_color="#999999",
                fill_opacity=0.85,
                tooltip=(
                    f"Uncovered: {row['geoid']}<br>"
                    f"{row['call_count']:,} calls<br>"
                    f"nearest station {row['response_minutes']:.1f} min"
                ),
            ).add_to(group)
        group.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    return fmap
