"""
Visualization helpers.

Note: regarding Kepler GL:
It only populates the default config
when the map is displayed interactively.
Therefore we have to import the whole config
from a previous interactive session.

Also: attribute-based determination of line width
is a bit flaky. It works best to normalize the input
before!
"""

import json
import logging
from itertools import zip_longest

import geopandas as gp
import matplotlib
import matplotlib.colors
import matplotlib.figure
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import pandas as pd
import tol_colors as tc
from keplergl import KeplerGl
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import FixedLocator

from intro_matsim import behavior, gis
from intro_matsim.analysis import prep_od_counts_per_h3
from intro_matsim.const import COL_H3_DEST, COL_H3_ORIG, SUBTOUR_ID, TRIP_COUNT
from intro_matsim.util import bin_seconds_of_day, float_str, int_str

logger = logging.getLogger(__name__)

CSET_QUALITATIVE = tc.colorsets["bright"]
CMAP_QUALITATIVE = ListedColormap(tc.colorsets["bright"], "bright")
CMAP_DIVERGING = tc.colormaps["sunset"]
ANNOTATION_FONT_SIZE = 10
MAP_LABEL_FONT_SIZE = 9


def plot_gdfs(
    gdfs, names: list = [], cities: gp.GeoDataFrame | None = None, crs=None
) -> matplotlib.figure.Figure:
    """
    Plot one or more GeoDataFrames on the same plot,
    optionally reprojecting them to a common CRS first.

    The gdfs are plotted on top of each other in the order they are passed.

    Args:
        gdfs: a single GeoDataFrame or a list of GeoDataFrames
        names: a list of names for the GeoDataFrames, used for the legend
        cities: a GeoDataFrame of cities to plot on top

    Limitation: polygon layers don't show up in legend. https://github.com/geopandas/geopandas/issues/660
    """
    fig, ax = plt.subplots()

    if not isinstance(gdfs, list):
        gdfs = [gdfs]

    for i, gdf, label in zip_longest(range(len(gdfs)), gdfs, names):
        if crs is not None:
            gdf = gdf.to_crs(crs)
        gdf.plot(
            ax=ax,
            color=CMAP_QUALITATIVE(i % CMAP_QUALITATIVE.N),
            zorder=i * 10,
            label=label,
        )

    if cities is not None:
        cities.plot(ax=ax, color=CMAP_QUALITATIVE(1), zorder=11)
        _plot_map_labels(fig, ax, cities, "black", "name", zorder=10)

    # ticks with lat/lon are not helpful, hide them
    plt.xticks([])
    plt.yticks([])

    plt.legend()
    return fig


def plot_subtour_stats(trips: pd.DataFrame, scale_factor: float, title_appendix: str):
    stats = trips["subtourLen"].value_counts().reset_index()
    stats["count"] = (stats["count"] / stats["subtourLen"]) * scale_factor
    stats = stats.set_index("subtourLen").sort_index()

    subtour_count = trips.groupby(SUBTOUR_ID).ngroups
    trip_count = len(trips)
    logger.info(
        "%s: %s subtours consisting of %s trips",
        title_appendix,
        subtour_count,
        trip_count,
    )
    logger.debug("subtour length distribution %s:\n%s", title_appendix, stats)

    fig, ax = plt.subplots()
    ax.pie(
        stats["count"],
        labels=stats.index.to_list(),
        autopct=lambda p: float_str(p, 1) + "%",
        colors=CSET_QUALITATIVE,
    )
    ax.set_title(f"Wegekettenlänge\n{title_appendix}")
    ax.set_xlabel("Anzahl der Wege pro Wegekette")
    ax.add_artist(
        _annotation({"Wege": trip_count, "Wegeketten": subtour_count}, scale_factor)
    )
    return fig


def plot_trip_purpose_stats(
    trips: pd.DataFrame, scale_factor: float, title_appendix: str
):
    def extract_purpose(chain):
        """remove home part of trip chain but leave chains not involving home intact"""
        tokens = chain.split("-")
        tokens = [token for token in tokens if token != "home"]
        if len(tokens) == 1:
            return tokens[0]
        return chain

    fig, ax = plt.subplots()
    purpose = trips["activityChain"].apply(extract_purpose)
    counts = pd.Series(purpose.value_counts())

    min_percent = 1
    mask = counts < round(counts.sum() / 100 * min_percent)
    if mask.any():
        other_sum = counts[mask].sum()
        counts = counts[~mask]
        counts["other"] = other_sum

    translation_dict = {
        "education": "Bildung",
        "errand": "Erledigung",
        "home": "Zuhause",
        "leisure": "Freizeit",
        "other": "Sonstiges",
        "shopping": "Einkaufen",
        "work": "Arbeit",
    }

    def translate_label(label):
        for en, de in translation_dict.items():
            if en in label:
                label = label.replace(en, de)
        return label

    labels = [translate_label(str(lab)) for lab in counts.index.to_list()]

    ax.pie(
        counts,
        labels=labels,
        autopct=lambda p: float_str(p, 1) + "%",
        colors=CSET_QUALITATIVE,
    )

    ax.set_title(f"Wegezweck\n{title_appendix}")
    ax.add_artist(_annotation({"Wege": len(trips)}, scale_factor))
    return fig


def plot_trip_departure_stats(
    trips: pd.DataFrame, scale_factor: float, title_appendix: str
):
    fig, ax = plt.subplots()
    bin_edges = [h for h in range(25)]
    dep_hour = trips["departureSecondsOfDay"].apply(lambda secs: secs // 3600)
    ax.hist(
        dep_hour,
        bins=bin_edges,
        edgecolor="black",
        align="left",
        color=CMAP_QUALITATIVE(0),
    )
    ax.set_title(f"Abfahrtszeit\n{title_appendix}")

    bar_width = 1
    ax.set_xlim(bin_edges[0] - bar_width / 2, bin_edges[-2] + bar_width / 2)
    tick_positions = bin_edges[:-1]
    tick_labels = [f"{h}-{h + 1} Uhr" for h in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)

    # upscaling labels (a bit clumsy because that way we don't use nice 'round' numbers)
    ax.yaxis.set_major_locator(FixedLocator(ax.get_yticks().tolist()))
    y_labels = [int_str(label * scale_factor) for label in ax.get_yticks()]
    ax.set_yticklabels(y_labels)

    ax.add_artist(_annotation({"Wege": len(trips)}, scale_factor))

    return fig


def traffic_spider(
    network: gp.GeoDataFrame,
    pd_stations: gp.GeoDataFrame,
    cities: gp.GeoDataFrame,
    scale_factor: float,
    title: str,
    max_line_width: int = 15,
) -> matplotlib.figure.Figure:
    """
    Args:
        network: network (must have the attribute "trip_count")
        pd_stations: for better orientation in the map
        cities: for better orientation in the map
    """
    max = network[TRIP_COUNT].max()
    linewidth = network[TRIP_COUNT] / max * max_line_width
    fig, ax = plt.subplots()
    network.plot(ax=ax, linewidth=linewidth, color=CMAP_QUALITATIVE(0), zorder=0)
    _plot_pd_stations_and_cities(fig, ax, pd_stations, cities)

    ax.set_title(title)
    # ticks for coordinates are not helpful, hide them
    ax.set_xticks([])
    ax.set_yticks([])

    if network.attrs[TRIP_COUNT] is not None:
        ax.add_artist(_annotation({"Wege": network.attrs[TRIP_COUNT]}, scale_factor))
    return fig


def traffic_spider_comparison(
    networks: list[gp.GeoDataFrame],
    subtitles: list[str],
    pd_stations: gp.GeoDataFrame,
    cities: gp.GeoDataFrame,
    scale_factor: float,
    title: str,
    max_linewidth: int = 15,
) -> matplotlib.figure.Figure:
    """
    Args:
        networks: networks (must have the attribute "trip_count")
        subtitles: one subtitle for each network
        pd_stations: for better orientation in the map
        cities: for better orientation in the map
    """
    fig, axes = plt.subplots(ncols=len(networks))
    max_trip_count = max([network[TRIP_COUNT].max() for network in networks])

    # Calculate the combined bounds of all networks
    combined_bounds = networks[0].total_bounds
    for network in networks[1:]:
        combined_bounds = [
            min(combined_bounds[0], network.total_bounds[0]),
            min(combined_bounds[1], network.total_bounds[1]),
            max(combined_bounds[2], network.total_bounds[2]),
            max(combined_bounds[3], network.total_bounds[3]),
        ]

    # Only include cities within the combined bbox of all networks
    cities_in_bbox = cities.cx[
        combined_bounds[0] : combined_bounds[2], combined_bounds[1] : combined_bounds[3]
    ]

    for i in range(len(networks)):
        ax = axes[i]
        network = networks[i]
        linewidth = network[TRIP_COUNT] / max_trip_count * max_linewidth
        network.plot(
            ax=ax, linewidth=linewidth, color=CMAP_QUALITATIVE(0), zorder=0, legend=True
        )
        _plot_pd_stations_and_cities(fig, ax, pd_stations, cities_in_bbox)

        if i < len(subtitles):
            ax.set_title(subtitles[i])
        ax.set_xticks([])
        ax.set_yticks([])

        # Set the same extent for all axes
        ax.set_xlim(combined_bounds[0], combined_bounds[2])
        ax.set_ylim(combined_bounds[1], combined_bounds[3])

        annotate_dict = {}
        if network.attrs[TRIP_COUNT] is not None:
            annotate_dict["Wege"] = network.attrs[TRIP_COUNT]
        annotate_dict["Wege auf meistbefahrenem Link"] = network[TRIP_COUNT].max()
        ax.add_artist(_annotation(annotate_dict, scale_factor))

    fig.suptitle(title)
    return fig


def plot_count_per_h3(
    h3_ids: pd.Series,
    pd_stations: gp.GeoDataFrame,
    cities: gp.GeoDataFrame,
    scale_factor: float,
    data_name: str,
    title: str,
    log_scale: bool = False,
) -> matplotlib.figure.Figure:
    """
    Plot distribution of a series of h3 cell ids.
    Arguments:
        data_name: what data are we plotting? e.g. Wegeketten
    """
    counts = pd.Series(h3_ids, name="h3").value_counts() * scale_factor
    df = pd.Series(counts).reset_index()
    geom = gis.h3cell_to_polygon(df["h3"]).to_crs(gis.AUSTRIA_LAMBERT)
    gdf = gp.GeoDataFrame(df, geometry=geom)
    fig, ax = plt.subplots()
    ax.set_title(title)
    norm = matplotlib.colors.LogNorm(vmin=gdf["count"].min(), vmax=gdf["count"].max())

    gdf.plot(
        ax=ax,
        column="count",
        cmap=CMAP_DIVERGING,
        norm=norm if log_scale else None,
        legend=True,
    )
    _plot_pd_stations_and_cities(fig, ax, pd_stations, cities)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_artist(_annotation({data_name: len(h3_ids)}, scale_factor))

    return fig


def plot_od_per_h3(
    od: gp.GeoDataFrame,
    pd_stations: gp.GeoDataFrame,
    cities: gp.GeoDataFrame,
    origins: bool = True,
    title_appendix: str = "",
) -> matplotlib.figure.Figure:
    """
    Plot distribution of trip origins or destinations
    based on H3 cells
    """
    col = COL_H3_ORIG if origins else COL_H3_DEST

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.set_title("Logarithmisch")
    od.plot(
        ax=ax1,
        column=col,
        cmap=CMAP_DIVERGING,
        legend=True,
        norm=matplotlib.colors.LogNorm(
            vmin=od[col][od[col] > 0].min(), vmax=od[col].max()
        ),
    )

    ax2.set_title("Linear")
    od.plot(
        ax=ax2,
        column=col,
        cmap=CMAP_DIVERGING,
        legend=True,
    )

    for ax in [ax1, ax2]:
        _plot_pd_stations_and_cities(fig, ax, pd_stations, cities)
        # ticks for coordinates are not helpful, hide them
        ax.set_xticks([])
        ax.set_yticks([])
        ax.add_artist(_annotation({"Wege": od.attrs[TRIP_COUNT]}, scale_factor=1))

    place = "Abfahrtsorte" if origins else "Zielorte"
    appendix = f"\n{title_appendix}" if len(title_appendix) > 0 else ""
    fig.suptitle(f"{place} {appendix}")
    return fig


def _annotation(
    data: dict, scale_factor: float, loc: str = "upper right"
) -> AnchoredText:
    """
    Print one line for the raw numbers and one for the upscaled numbers.
    (Or only one line if scale_factor is 1)
    Args:
        data: key = name to print (e.g. Wege), value = unscaled number
        scale_factor: of the model
    """
    raw_parts = []
    for key, value in data.items():
        raw_parts.append(f"{int_str(value)} {key}")
    raw_str = f"{float_str((1 / scale_factor) * 100)}%: ".replace(".", ",") + ", ".join(
        raw_parts
    )

    upscaled_parts = []
    for key, value in data.items():
        upscaled_parts.append(f"{int_str(value * scale_factor)} {key}")
    full_str = "100%: " + ", ".join(upscaled_parts)

    return AnchoredText(
        f"{raw_str}\n{full_str}" if scale_factor != 1 else full_str,
        loc=loc,
        prop=dict(size=ANNOTATION_FONT_SIZE),
        frameon=False,
    )


def _plot_pd_stations_and_cities(
    fig, ax, pd_stations: gp.GeoDataFrame, cities: gp.GeoDataFrame
):
    cities_filtered = cities.loc[~cities.index.isin(pd_stations.index)]
    cities_filtered.plot(ax=ax, color="gray", markersize=30, zorder=10)
    cities_filtered.plot(ax=ax, color="white", markersize=20, zorder=11)
    cities_filtered.plot(ax=ax, color="gray", markersize=10, zorder=12)
    _plot_map_labels(fig, ax, cities_filtered, "gray", "name", zorder=13)
    pd_stations.plot(ax=ax, color="black", markersize=45, zorder=20)
    pd_stations.plot(ax=ax, color="white", markersize=30, zorder=21)
    pd_stations.plot(ax=ax, color="black", markersize=15, zorder=22)
    _plot_map_labels(fig, ax, pd_stations, "black", "name", zorder=23)


def _plot_map_labels(fig, ax, gdf, color, field, **kwargs):
    transform = mtransforms.offset_copy(ax.transData, fig=fig, x=5, y=5, units="points")
    gdf.reset_index().apply(
        lambda item: ax.text(
            x=item.geometry.centroid.coords[0][0],
            y=item.geometry.centroid.coords[0][1],
            s=item[field],
            transform=transform,
            # horizontalalignment="left",
            fontsize=MAP_LABEL_FONT_SIZE,
            color=color,
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 0.5, "edgecolor": "none"},
            **kwargs,
        ),
        axis=1,
    )


def plot_pd_filling_levels_by_model(
    gpkg_path,
    station_name: str,
    bin_size_seconds: int = 60 * 60,
):
    """
    Collect departure times for first and second trips of subtours into bins.
    Then we approximate arrival / departure of the rider('s car) at the P&D station:
    - departure of first trip = arrival
    - departure of second trip = departure

    Numbers are divided by two to get the number of cars *parked* at the P&D station.

    Data is gathered from the results geopackage (processing step #5 - matchedSubtours)

    Returns a tuple of figure and dataframe
    """
    # necessary to avoid an annoying flood of info messages
    # see https://discourse.matplotlib.org/t/why-am-i-getting-this-matplotlib-error-for-plotting-a-categorical-variable/21758
    plt_logger = logging.getLogger("matplotlib")
    old_level = plt_logger.level
    plt_logger.setLevel(logging.WARNING)

    all_bins = set()
    arrivals = {}
    departures = {}

    for model in behavior.MODELS.values():
        bm_id = behavior.model_id(model.key, behavior.MODE_COMBINED)
        layer_name = f"{station_name}-5_{bm_id}_matchedSubtours"
        gdf = gis.read_geopackage_layer(
            gpkg_path, layer_name=layer_name, target_crs=gis.AUSTRIA_LAMBERT
        )
        arrival_times = gdf.loc[gdf["tripNr"] == 1]["departureSecondsOfDay"]
        departure_times = gdf.loc[gdf["tripNr"] == 2]["departureSecondsOfDay"]
        arr = bin_seconds_of_day(arrival_times, bin_size=bin_size_seconds)
        dep = bin_seconds_of_day(departure_times, bin_size=bin_size_seconds)
        all_bins.update(arr.index)
        all_bins.update(dep.index)
        arrivals[model.key] = arr["count"]
        departures[model.key] = dep["count"]

    all_bins = sorted(all_bins)
    df = pd.DataFrame(index=all_bins)
    for model_key in arrivals.keys():
        df[f"arr_{model_key}"] = arrivals[model_key].reindex(df.index, fill_value=0) / 2
        df[f"dep_{model_key}"] = (
            departures[model_key].reindex(df.index, fill_value=0) / 2
        )
        df[f"fillState_{model_key}"] = (
            arrivals[model_key].reindex(df.index, fill_value=0).cumsum()
            - departures[model_key].reindex(df.index, fill_value=0).cumsum()
        ) / 2
    df.index.name = "time"

    fig, ax = plt.subplots()
    for model in behavior.MODELS.values():
        arr = df[f"arr_{model.key}"]
        dep = df[f"dep_{model.key}"]
        fill_state = df[f"fillState_{model.key}"]
        ax.bar(
            arr.index,
            arr.values,
            align="edge",
            width=-0.3,
            label=f"Ankünfte ({model.name})",
            color=model.color_light,
            hatch="///",
            edgecolor="white",
            hatch_linewidth=2,
        )
        ax.bar(
            dep.index,
            dep.values,
            align="edge",
            width=0.3,
            label=f"Abfahrten ({model.name})",
            color=model.color_light,
            hatch="\\\\\\",
            edgecolor="white",
            hatch_linewidth=2,
        )
        ax.plot(
            fill_state.index,
            fill_state.values,
            label=f"Belegte Parkplätze ({model.name})",
            linewidth=5,
            color=model.color_dark,
        )
    ax.legend()
    ax.set_ylabel("Anzahl Pkw")
    ax.set_title(f"{station_name}: Füllstand P&D-Station nach Bereitschaftsmodell")
    ax.tick_params("x", rotation=90)

    plt_logger.setLevel(old_level)

    return fig, df


def plot_gdfs_kepler(gdfs, names: list = []):
    map = None
    if len(names) == len(gdfs) and len(set(names)) == len(gdfs):
        data = dict(zip(reversed(names), reversed(gdfs)))
        map = KeplerGl(data=data)
    else:
        logger.warning(
            "Number of names and GeoDataFrames do not match or names are not unique. Ignoring names."
        )
        map = KeplerGl()
        for i, gdf in enumerate(reversed(gdfs)):
            map.add_data(data=gdf, name=f"Layer {i}")

    return map


def plot_traffic_spider_and_od_per_h3_kepler(
    network: gp.GeoDataFrame | None,
    trips: gp.GeoDataFrame | pd.DataFrame | None,
    station: gp.GeoDataFrame,
    scale_factor: float,
    max_linewidth: int = 10,
) -> KeplerGl:
    """
    Plot a traffic spider and the
    distribution of trip origins or destinations
    based on H3 cells

    Args:
        network: network (must have the attribute "trip_count")
        station: should be a single station
    """
    kepler_data = {"station": station.reset_index()}

    if network is not None and TRIP_COUNT in network.columns:
        network = network.reset_index()
        network[TRIP_COUNT] = network[TRIP_COUNT] * scale_factor
        spider = network[["id", "osm_way_highway", TRIP_COUNT, "geometry"]]
        spider = spider.copy()
        spider["size"] = spider[TRIP_COUNT] / spider[TRIP_COUNT].max() * max_linewidth
        kepler_data["traffic_spider"] = spider

    h3_fields = {COL_H3_ORIG: "origins", COL_H3_DEST: "destinations"}
    if trips is not None:
        for field in h3_fields.keys():
            if field in trips.columns:
                counts = trips[field].value_counts() * scale_factor
                # it's OK to pass a Series of h3 cell ids to KeplerGl
                kepler_data[h3_fields[field]] = pd.Series(counts).reset_index()  # type: ignore

    config = json.load(open("data/keplergl_spider+od_config.json"))
    return KeplerGl(
        height=600,
        data=kepler_data,
        config=config,
    )


def traffic_spider_comparison_kepler(
    networks: list[gp.GeoDataFrame],
    layer_names: list[str],
    layer_colors: list[str],
    pd_stations: gp.GeoDataFrame,
    scale_factor: float,
    max_linewidth: int = 10,
) -> KeplerGl:
    """
    Args:
        networks: networks (must have the attribute "trip_count")
        layer_names: one for each network / spider
        pd_stations: for better orientation in the map
    """
    if len(networks) > 10:
        raise ValueError(
            "Too many networks for comparison, config must be extended to allow more."
        )
    max_trip_count = (
        max([network[TRIP_COUNT].max() for network in networks]) * scale_factor
    )

    kepler_data = {}
    for i in range(len(networks)):
        network = networks[i]
        network = network.reset_index()
        network[TRIP_COUNT] = network[TRIP_COUNT] * scale_factor
        spider = network[["id", "osm_way_highway", TRIP_COUNT, "geometry"]]
        spider = spider.copy()
        spider["size"] = spider[TRIP_COUNT] / max_trip_count * max_linewidth
        kepler_data[f"traffic_spider_{i}"] = spider

    config = json.load(open("data/keplergl_spider_comparison_config.json"))
    for i in range(0, len(kepler_data)):
        key = list(kepler_data.keys())[i]
        cfg_idx = i + 1  # one-off because station is the first layer
        config["config"]["visState"]["layers"][cfg_idx]["config"]["label"] = (
            layer_names[i]
        )
        config["config"]["visState"]["layers"][cfg_idx]["config"]["visConfig"][
            "strokeColor"
        ] = _convert_color_hex_string_to_rgb_tuple(layer_colors[i])
        config["config"]["visState"]["layers"][cfg_idx]["config"]["visConfig"][
            "sizeRange"
        ] = [
            0,
            round(kepler_data[key]["size"].max(), 2),
        ]

    # order of items in dict must stay the same,
    # otherwise our kepler.gl config crumbles
    data = {"station": pd_stations.reset_index()}
    for k, v in kepler_data.items():
        data[k] = v
    return KeplerGl(
        height=600,
        data=data,
        config=config,
    )


def _convert_color_hex_string_to_rgb_tuple(color: str) -> tuple[int, int, int]:
    """
    Convert a hex color string to an RGB tuple.
    """
    if color.startswith("#"):
        color = color[1:]  # remove the leading '#'
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))  # convert hex to RGB
