"""
P&D analysis for project INTRO
"""

import argparse
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import geopandas as gp
import matplotlib.pyplot as plt
import pandas as pd

from intro_matsim import analysis, behavior, gis, io, util, viz
from intro_matsim.const import (
    COL_ATT,
    COL_H3_DEST,
    COL_H3_ORIG,
    COL_LINKS,
    COL_LINKS_SHARED,
    COL_MODEL,
    COL_STATION,
    SUBTOUR_ID,
    TRIP_COUNT,
)

random.seed(42)  # for reproducibility

#### plotting setup ####
plt.rcParams.update(
    {
        "figure.figsize": (19.2, 10.8),
        "figure.dpi": 100,  # Exported DPI (1920x1080 when figsize is 19.2x10.8)
        "figure.constrained_layout.use": True,
        "font.size": 12,
        "figure.titlesize": 25,
        "axes.titlesize": 25,
        "axes.labelsize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.title_fontsize": 20,
        "legend.fontsize": 15,
        # Savefig settings
        "savefig.dpi": 300,  # High-quality export DPI
        "savefig.bbox": "tight",  # Trim whitespace around figures
    }
)
viz.ANNOTATION_FONT_SIZE = 10
viz.MAP_LABEL_FONT_SIZE = 9
FIG_OPTIONS = {
    "bbox_inches": "tight",
    "transparent": True,
    "pad_inches": 0.1,
}

#### input data & fixed configs ####
GPKG_PATH = Path("data/matsim-intro.gpkg")
MATSIM_NETWORK_LAYER = "matsim_model_vienna_xl_2023.2_network"
PARK_DRIVE_LAYER = "intro_parkdrive"
CITIES_LAYER = "intro_cities_towns"
BARRIER_LAYER = "intro_barriers"
ALL_TRIPS_CSV_PATH = Path("data/matsim_vienna_xl_2023.2_iteration0_trips-annotated.csv")
OUTPUT_PATH = Path("output")
CACHED_CAR_TRIPS_GPKG_PATH = OUTPUT_PATH / "analysis-carTripsCache.gpkg"
RESULTS_GPKG_PATH = OUTPUT_PATH / "analysis-results.gpkg"
# warning: scale factor should be an int, otherwise upscaling won't be exact
MATSIM_MODEL_SCALE_FACTOR = 8
# also note the hardcoded paths in behavior.py

#### analysis options ####
STATIONS = [
    "Ebreichsdorf",
    "Fels am Wagram",
    "Gerasdorf",
    "Hainfeld",
    "Herzogenburg",
    "Hollabrunn",
    "Klosterneuburg",
    "Korneuburg",
    # "Langenlois", # at edge of model, no useful results
    "Leobersdorf",
    "Stockerau",
    "Wiener Neustadt",
]
PARK_DRIVE_BUFFER_M = 5_000
MIN_SHARED_TRIP_LENGTH_M = 20_000
MATCH_OD_H3_RESOLUTION = 7
MATCH_TIME_BIN_SECS = 15 * 60
BEHAVIOR_MODE = behavior.MODE_COMBINED


def _prepare_logger(file_name) -> logging.Logger:
    # set up console and file handlers for root logger
    # (similar to logging.basicConfig)
    log_formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s.%(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger("")

    # Remove all existing handlers (avoid printing to console multiple times)
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(level=logging.DEBUG)
    logging.getLogger("").addHandler(console_handler)

    if file_name is not None:
        file_handler = logging.FileHandler(filename=file_name)
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(level=logging.DEBUG)
        logging.getLogger("").addHandler(file_handler)

    # set root level logging to info
    logging.getLogger("").setLevel("INFO")

    # main logger for our project
    logger = logging.getLogger("intro_matsim")
    logger.setLevel("INFO")
    return logger


def prepare_car_trips_for_pd_analysis(gdf_car_network: gp.GeoDataFrame) -> None:
    """
    Prepare trips by creating geometries and
    annotating which park & drive stations each trip passes
    (adding one bool column for each station).

    Writes all car trips (with a valid geometry) to a geopackage
    """
    gdf_pd = io.read_park_drive_stations(GPKG_PATH, PARK_DRIVE_LAYER)
    gdf_cities = io.read_cities(GPKG_PATH, CITIES_LAYER)
    gdf_barriers = io.read_layer(GPKG_PATH, BARRIER_LAYER)
    gdf_pd_buffers = gis.buffer_gdf(gdf_pd, PARK_DRIVE_BUFFER_M, gdf_barriers)
    gis.write_gpkg(gdf_pd_buffers, OUTPUT_PATH / "analysis-buffers.gpkg")
    fig = viz.plot_gdfs(
        [gdf_pd_buffers, gdf_barriers], names=["P&D", "Barriers"], cities=gdf_cities
    )
    fig.savefig(
        OUTPUT_PATH / "fullModel-buffers-barriers.png",
        **FIG_OPTIONS,
    )
    plt.close(fig)

    df_trips = pd.read_csv(ALL_TRIPS_CSV_PATH)
    _write_all_plots(
        filename_head="fullModel",
        filename_tail="allTrips",
        scale_factor=MATSIM_MODEL_SCALE_FACTOR,
        subtitle="Alle Wege im MATSim Modell",
        pd_stations=gdf_pd,
        cities=gdf_cities,
        trips=df_trips,
    )

    gdf_car_trips = io.build_all_car_trips(df_trips, gdf_car_network)
    if gdf_car_trips.empty:
        logger.error("No valid car trips (with geometry in the given network) found.")
        return

    analysis.assign_origin_destination_h3(gdf_car_trips, MATCH_OD_H3_RESOLUTION)
    _write_all_plots(
        filename_head="fullModel",
        filename_tail="allCarTrips",
        scale_factor=MATSIM_MODEL_SCALE_FACTOR,
        subtitle="Alle Wege mit dem Auto",
        pd_stations=gdf_pd,
        cities=gdf_cities,
        trips=gdf_car_trips,
        traffic_spider=analysis.prepare_traffic_spider(
            gdf_car_trips, COL_LINKS, gdf_car_network
        ),
    )

    for station_name in gdf_pd_buffers.index:
        logger.info(f"Intersecting car trips with P&D station {station_name}")
        gdf_pd_station = gdf_pd_buffers.loc[[station_name]]
        gdf_trips_via_pd = gis.intersect_gdfs(gdf_car_trips, gdf_pd_station)
        gdf_car_trips[station_name] = gdf_car_trips.index.isin(gdf_trips_via_pd.index)

    logger.info(
        f"Writing {len(gdf_car_trips)} car trips to {CACHED_CAR_TRIPS_GPKG_PATH}"
    )
    gis.write_gpkg(gdf_car_trips, CACHED_CAR_TRIPS_GPKG_PATH)


def analyze_pd_potential(gdf_car_network: gp.GeoDataFrame, station_name: str):
    """
    Analyze the potential of a park & drive station.
    Results of the filtering & matching steps are given in three flavours:
    one without any behavior model, and one for each of our two behavior models.
    """
    logger.info(f"Analysis for P&D station {station_name}")
    trip_stats = {}
    scale_factor = MATSIM_MODEL_SCALE_FACTOR

    # load all car trips passing the P&D station
    gdf_pd = io.read_park_drive_stations(GPKG_PATH, PARK_DRIVE_LAYER).loc[
        [station_name]
    ]
    gdf_cities = io.read_cities(GPKG_PATH, CITIES_LAYER)
    gdf_trips = io.read_trips(CACHED_CAR_TRIPS_GPKG_PATH, station_name)
    logger.info(f"{len(gdf_trips)} car trips pass the station in total")

    gdf_traffic_spider_all = analysis.prepare_traffic_spider(
        gdf_trips, COL_LINKS, gdf_car_network
    )
    _write_all_plots(
        filename_head=station_name,
        filename_tail="1_allCarTrips",
        scale_factor=scale_factor,
        subtitle=f"{station_name}: Alle Wege mit dem Auto",
        pd_stations=gdf_pd,
        cities=gdf_cities,
        trips=gdf_trips,
        traffic_spider=gdf_traffic_spider_all,
    )
    _update_trip_stats(gdf_trips, scale_factor, trip_stats, "1_allCarTrips")
    gdf_traffic_spider_all_scaled = gdf_traffic_spider_all.copy()
    gdf_traffic_spider_all_scaled[TRIP_COUNT] *= MATSIM_MODEL_SCALE_FACTOR
    gis.write_gpkg(
        gdf_traffic_spider_all_scaled,
        RESULTS_GPKG_PATH,
        f"{station_name}-1_allCarTrips-spider",
    )
    if {COL_H3_ORIG, COL_H3_DEST}.issubset(gdf_trips.columns):
        gdf_od = analysis.prep_od_counts_per_h3(gdf_trips, scale_factor)
        gis.write_gpkg(
            gdf_od,
            RESULTS_GPKG_PATH,
            f"{station_name}-1_allCarTrips-tripCounts",
        )

    gdf_trips = analysis.filter_trips_with_hxh_subtour(gdf_trips)
    logger.info(f"{len(gdf_trips)} car trips are part of a car-subtour with length 2")
    _update_trip_stats(gdf_trips, scale_factor, trip_stats, "2_twoTripSubtours")

    gdf_trips = analysis.filter_trips_with_min_shared_trip(
        gdf_trips, gdf_pd, PARK_DRIVE_BUFFER_M, MIN_SHARED_TRIP_LENGTH_M
    )
    logger.info(f"{len(gdf_trips)} car trips have a sufficiently long shared part")
    _update_trip_stats(gdf_trips, scale_factor, trip_stats, "3_suitableSharedPart")

    logger.info(f"Finding link ids for {len(gdf_trips)} shared trip parts")
    gdf_trips[COL_LINKS_SHARED] = gdf_trips.apply(
        lambda row: gis.clip_links(row[COL_LINKS], row["geom_shared"], gdf_car_network),
        axis=1,
    )

    gdf_trips = analysis.upscale_trips(gdf_trips, scale_factor)
    logger.info(f"{len(gdf_trips)} trips after upscaling *{scale_factor}")
    # upscale & reset scale factor afterwards
    scale_factor = 1
    # note: step was removed, quite redundant:
    # _update_trip_stats(gdf_trips, scale_factor, trip_stats, "upscaled")

    results = {}
    for behavior_model in behavior.MODELS.keys():
        results[behavior_model] = _analyze_pd_potential_with_behavior_model(
            gdf_trips,
            scale_factor,
            gdf_pd,
            gdf_cities,
            station_name,
            trip_stats,
            behavior_model,
        )

    df_trip_stats = pd.DataFrame(trip_stats)
    df_trip_stats.index.name = "attribute"
    df_trip_stats.to_csv(OUTPUT_PATH / f"{station_name}-tripStats.csv")

    # now that we have all trips, compare the different behavior models
    gdf_traffic_spider_all = analysis.upscale_traffic_spider(
        gdf_traffic_spider_all, MATSIM_MODEL_SCALE_FACTOR
    )

    for spider_key, step_id in {
        "allSpider": "5_spiderComparisonVsAllTrips_matchedSubtours",
        "sharedSpider": "6_spiderComparisonVsAllTrips_sharedTripPart",
    }.items():
        for model in behavior.MODELS.keys():
            model_spider = getattr(results[model], spider_key)
            if not gdf_traffic_spider_all.empty and not model_spider.empty:
                viz.traffic_spider_comparison(
                    [gdf_traffic_spider_all, model_spider],
                    [
                        "Alle Wege mit dem Auto",
                        f"P&D Potenzial\n{_model_text(model)}",
                    ],
                    gdf_pd,
                    gdf_cities,
                    scale_factor,
                    f"Verkehrsspinnen: {station_name}",
                ).savefig(
                    OUTPUT_PATH / f"{station_name}-{step_id}-{model}.png",
                    **FIG_OPTIONS,
                )

    any_result_is_empty = any(
        [
            result.allSpider.empty or result.sharedSpider.empty
            for result in results.values()
        ]
    )
    if any_result_is_empty:
        logger.warning(
            f"{station_name}: empty spider for at least one behavior model. "
            "Skipping comparison plots."
        )
        return

    for spider_key, step_id in {
        "allSpider": "5_spiderComparisonBehaviorModels_matchedSubtours",
        "sharedSpider": "6_spiderComparisonBehaviorModels_sharedTripPart",
    }.items():
        viz.traffic_spider_comparison(
            [getattr(result, spider_key) for result in results.values()],
            [behavior.MODELS[model].name for model in results.keys()],
            gdf_pd,
            gdf_cities,
            scale_factor,
            f"Verkehrsspinnen: {station_name}\nP&D Potenzial nach Bereitschaftsmodell",
        ).savefig(
            OUTPUT_PATH / f"{station_name}-{step_id}.png",
            **FIG_OPTIONS,
        )

        models = [getattr(result, spider_key) for result in results.values()]
        models.insert(0, gdf_traffic_spider_all)
        labels = [f"P&D Potenzial {_model_text(model)}" for model in results.keys()]
        labels.insert(0, "Alle Wege mit dem Auto")
        colors = ["#FFFB98", "#9226BE", "#D20000", "#F26C00"]
        kepler_html = OUTPUT_PATH / f"{station_name}-{step_id}-keplergl.html"
        viz.traffic_spider_comparison_kepler(
            list(reversed(models)),
            list(reversed(labels)),
            list(reversed(colors)),
            gdf_pd,
            scale_factor,
            max_linewidth=25,
        ).save_to_html(
            file_name=str(kepler_html),
            # read_only=True,
        )
        util.replace_html_title(
            kepler_html,
            f"{station_name}: Verkehrsspinnen - P&D Potenzial nach Bereitschaftsmodell vs. Alle Wege mit dem Auto",
        )


@dataclass
class TripAnalysisResult:
    all: gp.GeoDataFrame
    """full trips"""
    allSpider: gp.GeoDataFrame
    """traffic spider for full trips"""
    sharedPart: gp.GeoDataFrame
    """only the shared part of the trips, i.e. the part between home and P&D station"""
    sharedSpider: gp.GeoDataFrame
    """traffic spider for the shared part of the trips"""


def _analyze_pd_potential_with_behavior_model(
    gdf_trips: gp.GeoDataFrame,
    scale_factor: int,
    gdf_pd: gp.GeoDataFrame,
    gdf_cities: gp.GeoDataFrame,
    station_name: str,
    trip_stats: dict,
    model: str,
) -> TripAnalysisResult:
    """
    Analysis part for different behavior models.
    """
    bm_id = behavior.model_id(model, BEHAVIOR_MODE)
    logger.info(f"Using behavior model {bm_id} with {len(gdf_trips)} trips")

    gdf_trips = analysis.filter_trips_with_behavior_model(
        gdf_trips, model, BEHAVIOR_MODE
    )
    _update_trip_stats(gdf_trips, scale_factor, trip_stats, f"4_{bm_id}_behaviorFilter")

    gdf_trips = analysis.match_subtours(gdf_trips, MATCH_TIME_BIN_SECS)
    logger.info(f"After subtour matching {len(gdf_trips)} trip(s) remain.")
    _update_trip_stats(
        gdf_trips, scale_factor, trip_stats, f"5_{bm_id}_matchedSubtours"
    )
    gdf_traffic_spider_pd_only = analysis.prepare_traffic_spider(
        gdf_trips, COL_LINKS, gdf_car_network
    )
    _write_all_plots(
        filename_head=station_name,
        filename_tail=f"5_{bm_id}_matchedSubtours",
        scale_factor=scale_factor,
        subtitle=f"{station_name}: P&D Potenzial\n{_model_text(model)}",
        pd_stations=gdf_pd,
        cities=gdf_cities,
        trips=gdf_trips,
        traffic_spider=gdf_traffic_spider_pd_only,
    )
    gis.write_gpkg(
        gdf_trips, RESULTS_GPKG_PATH, f"{station_name}-5_{bm_id}_matchedSubtours"
    )
    gis.write_gpkg(
        gdf_traffic_spider_pd_only,
        RESULTS_GPKG_PATH,
        f"{station_name}-5_{bm_id}_matchedSubtours-spider",
    )
    if {COL_H3_ORIG, COL_H3_DEST}.issubset(gdf_trips.columns):
        gdf_od = analysis.prep_od_counts_per_h3(gdf_trips, scale_factor)
        gis.write_gpkg(
            gdf_od,
            RESULTS_GPKG_PATH,
            f"{station_name}-5_{bm_id}_matchedSubtours-tripCounts",
        )

    gdf_trips_shared = (
        gdf_trips.set_geometry("geom_shared")
        if not gdf_trips.empty
        else gp.GeoDataFrame()
    )
    _update_trip_stats(
        gdf_trips_shared, scale_factor, trip_stats, f"6_{bm_id}_sharedTripPart"
    )
    gdf_traffic_spider_shared = analysis.prepare_traffic_spider(
        gdf_trips_shared, COL_LINKS_SHARED, gdf_car_network
    )
    _write_all_plots(
        filename_head=station_name,
        filename_tail=f"6_{bm_id}_sharedTripPart",
        scale_factor=scale_factor,
        subtitle=f"{station_name}: P&D Potenzial (gemeinsamer Weg)\n{_model_text(model)}",
        pd_stations=gdf_pd,
        cities=gdf_cities,
        trips=gdf_trips_shared,
        traffic_spider=gdf_traffic_spider_shared,
    )
    gis.write_gpkg(
        gdf_trips_shared,
        RESULTS_GPKG_PATH,
        f"{station_name}-6_{bm_id}_sharedTripPart",
    )
    gis.write_gpkg(
        gdf_traffic_spider_shared,
        RESULTS_GPKG_PATH,
        f"{station_name}-6_{bm_id}_sharedTripPart-spider",
    )
    # not writing _sharedTripPart-tripCounts, does not change
    # in compared to step 5

    return TripAnalysisResult(
        gdf_trips,
        gdf_traffic_spider_pd_only,
        gdf_trips_shared,
        gdf_traffic_spider_shared,
    )


def _model_text(model: str) -> str:
    if model == behavior.MODEL_NONE:
        return "ohne Bereitschaftsmodell"
    return f"mit {behavior.MODELS[model].name}-Bereitschaftsmodell"


def _update_trip_stats(
    trips: gp.GeoDataFrame, scale_factor: float, stats: dict, desc: str
) -> None:
    """
    Adds an entry to the trip stats dict for the given trips.
    """
    if trips.empty:
        raw_stats = dict(
            personCount=0,
            subtourCount=0,
            tripCount=0,
            lengthKm=0,
        )
    else:
        raw_stats = dict(
            personCount=len(trips.reset_index().personId.unique()),
            subtourCount=len(trips.reset_index().set_index(SUBTOUR_ID).index.unique()),
            tripCount=len(trips),
            lengthKm=round(trips.geometry.length.sum() / 1000),
        )

    praw = f"-{(1 / scale_factor) * 100:.1f}pct"
    p100 = "-100pct"

    full_stats = {}
    gen_tuples = [(praw, 1), (p100, scale_factor)] if scale_factor != 1 else [(p100, 1)]
    for suffix, scale_factor in gen_tuples:
        for key, value in raw_stats.items():
            full_stats[f"{key}{suffix}"] = round(value * scale_factor)

    stats[desc] = full_stats


def _write_all_plots(
    filename_head: str,
    filename_tail: str,
    scale_factor: float,
    subtitle: str,
    pd_stations: gp.GeoDataFrame,
    cities: gp.GeoDataFrame,
    trips: gp.GeoDataFrame | pd.DataFrame,
    traffic_spider: gp.GeoDataFrame | None = None,
) -> None:
    if trips.empty:
        logger.warning("Skip plotting. No trips to plot")
        return

    if isinstance(trips, gp.GeoDataFrame):
        bbox = trips.total_bounds
        cities = cities.cx[bbox[0] : bbox[2], bbox[1] : bbox[3]]

    viz.plot_subtour_stats(trips, scale_factor, subtitle).savefig(
        OUTPUT_PATH / f"{filename_head}-subtourLength-{filename_tail}.png",
        **FIG_OPTIONS,
    )
    viz.plot_trip_purpose_stats(trips, scale_factor, subtitle).savefig(
        OUTPUT_PATH / f"{filename_head}-tripPurpose-{filename_tail}.png",
        **FIG_OPTIONS,
    )
    viz.plot_trip_departure_stats(trips, scale_factor, subtitle).savefig(
        OUTPUT_PATH / f"{filename_head}-tripDeparture-{filename_tail}.png",
        **FIG_OPTIONS,
    )

    if (
        isinstance(trips, gp.GeoDataFrame)
        and COL_H3_ORIG in trips.columns
        and COL_H3_DEST in trips.columns
    ):
        od_gdf = analysis.prep_od_counts_per_h3(trips, scale_factor)
        viz.plot_od_per_h3(
            od_gdf,
            pd_stations,
            cities,
            origins=True,
            title_appendix=subtitle,
        ).savefig(
            OUTPUT_PATH / f"{filename_head}-tripOrigins-{filename_tail}.png",
            **FIG_OPTIONS,
        )
        viz.plot_od_per_h3(
            od_gdf,
            pd_stations,
            cities,
            origins=False,
            title_appendix=subtitle,
        ).savefig(
            OUTPUT_PATH / f"{filename_head}-tripDestinations-{filename_tail}.png",
            **FIG_OPTIONS,
        )

        start = analysis.extract_start_of_subtour(trips)
        title = f"Abfahrtsorte (der Wegeketten)\n{subtitle}"
        viz.plot_count_per_h3(
            start, pd_stations, cities, scale_factor, "Wegeketten", title
        ).savefig(
            OUTPUT_PATH / f"{filename_head}-subtourOrigins-{filename_tail}.png",
            **FIG_OPTIONS,
        )

        activities = analysis.extract_first_destination_of_subtour(trips)
        title = f"Zielorte (Erstes Ziel der Wegeketten)\n{subtitle}"
        viz.plot_count_per_h3(
            activities, pd_stations, cities, scale_factor, "Wegeketten", title
        ).savefig(
            OUTPUT_PATH
            / f"{filename_head}-subtourFirstDestinations-{filename_tail}.png",
            **FIG_OPTIONS,
        )

    if traffic_spider is not None:
        viz.traffic_spider(
            traffic_spider,
            pd_stations,
            cities,
            scale_factor,
            f"Verkehrsspinne\n{subtitle}",
        ).savefig(
            OUTPUT_PATH / f"{filename_head}-spider-{filename_tail}.png", **FIG_OPTIONS
        )

    kepler_html = OUTPUT_PATH / f"{filename_head}-{filename_tail}-keplergl.html"
    viz.plot_traffic_spider_and_od_per_h3_kepler(
        traffic_spider, trips, pd_stations, scale_factor
    ).save_to_html(
        file_name=str(kepler_html),
        # read_only=True,
    )
    util.replace_html_title(
        kepler_html,
        f"{subtitle} - Verkehrsspinne, Abfahrts- und Zielorte der Wege",
    )

    # close all previously opened plots at once (to avoid memory issues)
    plt.close("all")


def combine_results() -> None:
    _combine_pd_potential()
    _combine_pd_filling_levels()


def _combine_pd_potential():
    all_stats = analysis.combine_trip_stats(STATIONS, OUTPUT_PATH)
    all_stats.to_csv(OUTPUT_PATH / "analysis-results.csv")
    _plot_and_export_pd_potential(
        all_stats,
        "lengthKm",
        "Ersparnis Autokilometer",
    )
    _plot_and_export_pd_potential(all_stats, "tripCount", "Anzahl Mitfahrgelegenheiten")


def _plot_and_export_pd_potential(stats: pd.DataFrame, yfield: str, ylabel: str):
    """
    To summarize the P&D potential we visualize (depending on yfield):
    - half of the matched trip count = number of ride sharings
    - half of the *shared* distance = actually saved driving distance
    """
    shared_trip_part = "6"
    df = stats.xs(yfield, level=COL_ATT)
    pivot_df = df.reset_index().pivot(
        index=COL_STATION, columns=COL_MODEL, values=shared_trip_part
    )
    # fix sorting of models
    pivot_df = pivot_df[behavior.MODELS.keys()]
    pivot_df /= 2
    pivot_df.to_csv(OUTPUT_PATH / f"analysis-pdPotential-{yfield}.csv")
    fig, ax = plt.subplots()
    model_colors = [behavior.MODELS[model].color_dark for model in pivot_df.columns]
    pivot_df.plot(
        ax=ax,
        kind="bar",
        width=0.8,
        ylabel=ylabel,
        title="P&D Potenzial nach Bereitschaftsmodell",
        color=model_colors,
    )
    legend_labels = [behavior.MODELS[model].name for model in pivot_df.columns]
    ax.legend(title="Bereitschaftsmodell", labels=legend_labels)
    ax.set_xlabel("P&D Station")
    ax.set_xticklabels(pivot_df.index, rotation=45, ha="right")
    fig.savefig(
        OUTPUT_PATH / f"analysis-pdPotential-{yfield}.png",
        **FIG_OPTIONS,
    )
    plt.close("all")


def _combine_pd_filling_levels():
    max_fill_levels = []
    # one filling level plot per station
    for station in STATIONS:
        fig, df = viz.plot_pd_filling_levels_by_model(RESULTS_GPKG_PATH, station)
        fig.savefig(
            OUTPUT_PATH / f"{station}-pdFillingLevel.png",
            **FIG_OPTIONS,
        )
        df.to_csv(OUTPUT_PATH / f"{station}-pdFillingLevel.csv")

        for model in behavior.MODELS.keys():
            max_fill_levels.append(
                {
                    "station": station,
                    "model": model,
                    "max_fill_level": df[f"fillState_{model}"].max(),
                }
            )

    # max filling levels per station and model
    df_max_fill = pd.DataFrame(max_fill_levels)
    df_max_fill_pivot = df_max_fill.pivot(
        index="station", columns="model", values="max_fill_level"
    )
    # fix sorting of models
    df_max_fill_pivot = df_max_fill_pivot[behavior.MODELS.keys()]
    df_max_fill_pivot.to_csv(OUTPUT_PATH / "analysis-pdMaxFillingLevel.csv")
    fig, ax = plt.subplots()
    model_colors = [
        behavior.MODELS[model].color_dark for model in behavior.MODELS.keys()
    ]
    df_max_fill_pivot.plot(
        ax=ax,
        kind="bar",
        width=0.8,
        ylabel="Anzahl Pkw",
        title="Maximalfüllstand P&D Station nach Bereitschaftsmodell",
        color=model_colors,
    )
    legend_labels = [behavior.MODELS[model].name for model in df_max_fill_pivot.columns]
    ax.legend(title="Bereitschaftsmodell", labels=legend_labels)
    ax.set_xlabel("P&D Station")
    ax.set_xticklabels(df_max_fill_pivot.index, rotation=45, ha="right")
    fig.savefig(
        OUTPUT_PATH / "analysis-pdMaxFillingLevel.png",
        **FIG_OPTIONS,
    )
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Only run the combine_results step and skip all other processing.",
    )
    args = parser.parse_args()

    run_prep = (not Path(CACHED_CAR_TRIPS_GPKG_PATH).exists()) and (
        not args.combine_only
    )
    run_calc_potential = not args.combine_only
    run_combine = True

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    logger = _prepare_logger(OUTPUT_PATH / "intro-matsim.log")
    logger.info("Starting...")

    if run_prep or run_calc_potential:
        # loading the network takes a while, so we do it only once
        gdf_car_network = io.read_car_network(GPKG_PATH, MATSIM_NETWORK_LAYER)
        if gdf_car_network.empty:
            logger.error("No valid car network found.")
            sys.exit(1)

    if run_prep:
        logger.info("Preparing car trips")
        prepare_car_trips_for_pd_analysis(gdf_car_network)

    if run_calc_potential:
        if not Path(CACHED_CAR_TRIPS_GPKG_PATH).exists():
            logger.error("Preparation of car trips failed.")
            sys.exit(1)
        for station_name in STATIONS:
            logger.info(f"Running analysis for P&D station {station_name}")
            analyze_pd_potential(gdf_car_network, station_name)

    if run_combine:
        logger.info("Combining results")
        combine_results()

    logger.info("Finished! (ﾉ^_^)ﾉ")
