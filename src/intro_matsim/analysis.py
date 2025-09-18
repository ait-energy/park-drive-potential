import logging
import random
from collections import Counter, defaultdict
from collections.abc import Collection
from pathlib import Path

import geopandas as gp
import pandas as pd

from intro_matsim import behavior, gis
from intro_matsim.behavior import BehaviorModel
from intro_matsim.const import (
    COL_ATT,
    COL_H3_DEST,
    COL_H3_ORIG,
    COL_MODEL,
    COL_STATION,
    SUBTOUR_ID,
    TRIP_COUNT,
)

logger = logging.getLogger(__name__)


def assign_origin_destination_h3(
    gdf_car_trips: gp.GeoDataFrame, h3_resolution: int
) -> None:
    if gdf_car_trips.empty:
        return

    gdf_car_trips[COL_H3_ORIG] = gis.point_to_h3cell(
        gdf_car_trips.geometry.apply(gis.first_point), h3_resolution
    )
    gdf_car_trips[COL_H3_DEST] = gis.point_to_h3cell(
        gdf_car_trips.geometry.apply(gis.last_point), h3_resolution
    )


def prep_od_counts_per_h3(
    trips: gp.GeoDataFrame,
    scale_factor: float,
) -> gp.GeoDataFrame:
    """
    Prepares upscaled counts of trip origins / destinations per H3 cell.
    Returns a GeoDataFrame with the h3 id as index

    Also stores the upscaled trip count in DataFrame.attrs
    """
    orig_counts = trips[COL_H3_ORIG].value_counts() * scale_factor
    dest_counts = trips[COL_H3_DEST].value_counts() * scale_factor

    all_h3 = pd.Index(orig_counts.index).union(dest_counts.index)
    df = pd.DataFrame(index=all_h3)
    df[COL_H3_ORIG] = orig_counts
    df[COL_H3_DEST] = dest_counts
    df = df.reset_index().rename(columns={"index": "h3"})

    geom = gis.h3cell_to_polygon(df["h3"]).to_crs(gis.AUSTRIA_LAMBERT)
    gdf = gp.GeoDataFrame(df, geometry=geom).set_index("h3")
    gdf.attrs[TRIP_COUNT] = len(trips) * scale_factor
    return gdf


def filter_trips_with_hxh_subtour(trips: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """
    Filter complete home - x - home subtours
    (i.e. remove subtours where only one trip passes the station
    and subtours consisting of only one trip)

    Note: start / end in same h3 cell is not enforced,
    and not necessary! What matters is that both trips
    pass the station.
    """
    if trips.empty:
        return gp.GeoDataFrame()

    mask = ((trips["tripNr"] == 1) & trips["activityChain"].str.startswith("home-")) | (
        (trips["tripNr"] == 2) & trips["activityChain"].str.endswith("-home")
    )
    trips = trips[mask]

    desired_len = 2
    trips = trips[trips["subtourLen"] == desired_len]
    return trips.groupby(SUBTOUR_ID).filter(lambda s: len(s) == desired_len)


def upscale_trips(trips: gp.GeoDataFrame, scale_factor: float) -> gp.GeoDataFrame:
    """
    Duplicate trips with the given scale factor.
    Adds an appendix to the personId to avoid duplicates.
    """
    int_scale_factor = round(scale_factor)
    if int_scale_factor != scale_factor:
        logger.warning(
            f"scale factor {scale_factor} is a float, upscaling will not be exact.,"
        )
    scale_factor = int_scale_factor
    trip_duplicates = [trips]
    for i in range(1, scale_factor):
        trip_copy = trips.copy()
        trip_copy["personId"] = trip_copy["personId"].astype(str) + f"#upscale{i}"
        trip_duplicates.append(trip_copy)
    return pd.concat(trip_duplicates, ignore_index=True)  # type: ignore


def filter_trips_with_behavior_model(
    trips: gp.GeoDataFrame, model: str, mode: str
) -> gp.GeoDataFrame:
    """
    For each subtour of each person the behavior model is invoked,
    i.e. decided if the person is willing to use P&D or not.
    """
    if trips.empty:
        return gp.GeoDataFrame()

    if model == behavior.MODEL_NONE:
        return trips

    behavior_model = BehaviorModel()

    def _behavior_filter(df):
        """
        Returns True if the trip should be kept, False otherwise.
        """
        row = df.iloc[0]
        probability = behavior_model.get_choice_probability(
            model_name=model,
            mode=mode,
            employed=row["isEmployed"],
            full_time=row["fullTimeWork"],
            in_education=row["inEducation"],
            age=row["age"],
        )
        return random.uniform(0, 1) < probability

    return trips.groupby(SUBTOUR_ID).filter(_behavior_filter)


def match_subtours(trips: gp.GeoDataFrame, time_bin_secs: int) -> gp.GeoDataFrame:
    """
    Match complete subtours (not single trips) based on
     - arrivalTimeAtActivity
     - departureTimeBackHome
     - activityH3Cell
    The matching is trivial: we assume that any pair of subtours can match.
    Subtours without a match are removed.
    """
    if trips.empty:
        return gp.GeoDataFrame()

    subtours = _build_subtour2(trips)
    logger.info(f"got {len(trips)} trip(s) with {len(subtours)} subtour(s).")
    subtours_matched = _match_subtours(subtours, time_bin_secs)
    filtered_trips = _filter_trips_in_subtours(trips, subtours_matched)
    logger.info(
        f"{len(filtered_trips)} trip(s) with {len(subtours_matched)} subtour(s) remain."
    )
    return filtered_trips


def _match_subtours(subtours: pd.DataFrame, time_bin_secs: int) -> pd.DataFrame:
    if subtours.empty:
        return pd.DataFrame()

    subtours["t1"] = subtours["arrivalTimeAtActivity"].apply(
        lambda secs: secs // time_bin_secs
    )
    subtours["t2"] = subtours["departureTimeBackHome"].apply(
        lambda secs: secs // time_bin_secs
    )
    groups = subtours.reset_index().groupby(["t1", "activityH3Cell", "t2"])
    # make sure each group has an even number of trips
    return groups.apply(lambda x: x.iloc[0 : len(x) - len(x) % 2], include_groups=False)


def _build_subtour2(trips: gp.GeoDataFrame) -> pd.DataFrame:
    """
    Collect trips into subtours, assuming there are two trips per subtours
    """
    if trips.empty:
        return pd.DataFrame()

    return trips.groupby(SUBTOUR_ID).apply(_group_to_subtour, include_groups=False)


def _filter_trips_in_subtours(
    trips: gp.GeoDataFrame, subtours: pd.DataFrame
) -> gp.GeoDataFrame:
    """
    Return only those trips that are referenced in the subtours
    """
    if trips.empty or subtours.empty:
        return gp.GeoDataFrame()

    idx = pd.Index(subtours[SUBTOUR_ID])
    return trips.set_index(SUBTOUR_ID).loc[idx].reset_index()


def _group_to_subtour(group_df) -> pd.Series:
    return pd.Series(
        {
            "arrivalTimeAtActivity": group_df.iloc[0]["arrivalSecondsOfDay"],
            "activityH3Cell": group_df.iloc[0][COL_H3_DEST],
            "departureTimeBackHome": group_df.iloc[1]["departureSecondsOfDay"],
        }
    )


def extract_start_of_subtour(trips: gp.GeoDataFrame) -> pd.Series:
    if trips.empty:
        return pd.Series()

    def _first_start(trips):
        return trips.loc[trips.index[0], COL_H3_ORIG]

    return (
        trips.reset_index()
        .groupby(SUBTOUR_ID)
        .apply(_first_start, include_groups=False)
    )


def extract_first_destination_of_subtour(trips: gp.GeoDataFrame) -> pd.Series:
    if trips.empty:
        return pd.Series()

    def _first_dest(trips):
        return trips.loc[trips.index[0], COL_H3_DEST]

    return (
        trips.reset_index().groupby(SUBTOUR_ID).apply(_first_dest, include_groups=False)
    )


def filter_trips_with_min_shared_trip(
    gdf_trips: gp.GeoDataFrame,
    gdf_pd: gp.GeoDataFrame,
    pd_buffer_m: float,
    min_shared_trip_length_m: float,
) -> gp.GeoDataFrame:
    """
    Filter trips (actually complete subtours)
    based on a minimum distance from the P&D station
    to the (shared) activity.

    Rationale: it's not realistic that people switch from
    their own car to P&D when the remaining trip distance
    is too short
    """
    if gdf_trips.empty or gdf_pd.empty:
        return gp.GeoDataFrame()

    gdf_pd_buffers = gis.buffer_gdf(gdf_pd, pd_buffer_m)
    gdf_trips = gis.add_geom_for_poly_split(gdf_trips, gdf_pd_buffers)
    first_trips = gdf_trips.loc[gdf_trips["tripNr"] == 1]
    min_meters = min_shared_trip_length_m - pd_buffer_m
    # make sure this is slighly positive to avoid empty geometries matching
    min_meters = max(0.1, min_meters)
    short_trips = first_trips.loc[first_trips.geom_after.length > min_meters]
    index = short_trips.set_index(SUBTOUR_ID).index
    gdf_trips = gdf_trips.set_index(SUBTOUR_ID).loc[index].reset_index()
    return gdf_trips


def prepare_traffic_spider(
    trips: gp.GeoDataFrame, link_col: str, network: gp.GeoDataFrame
) -> gp.GeoDataFrame:
    """
    Returns a subset of the given network
    with the additional attribute 'trip_count'.

    Only one link between two nodes remains
    to avoid plotting links over each other.
    Note: this is not necessarily the link
    with the correct digitization direction.

    In case there is no trip between two nodes,
    none of the links between them is present.

    Also stores the total trip count in DataFrame.attrs
    """
    if trips.empty or network.empty:
        return gp.GeoDataFrame()

    mapping = _link_map(_collect_links_between_nodes(network).values())
    counter = _link_count(trips, link_col, mapping)
    filtered_network = network[network.index.isin(counter.keys())]
    result = filtered_network.join(pd.Series(counter, name=TRIP_COUNT))
    result.attrs[TRIP_COUNT] = len(trips)
    return result


def upscale_traffic_spider(
    traffic_spider: gp.GeoDataFrame, scale_factor: float
) -> gp.GeoDataFrame:
    traffic_spider[TRIP_COUNT] *= scale_factor
    traffic_spider.attrs[TRIP_COUNT] *= scale_factor
    return traffic_spider


def _collect_links_between_nodes(
    network: pd.DataFrame,
) -> dict[tuple, set]:
    """
    Collects sets of links (i.e. their index) between two nodes,
    i.e. where 'fromNode' and 'toNode' are the same or reversed.

    Typically the sets are tuples, i.e. one forward and one backward link.
    However, for one-way links, the set contains only one link,
    and in some cases two nodes are connected by multiple links
    in the same direction.

    Returns
    -------
    dict
        key: sorted tuple of nodes
        value: set of link indices
    """
    # Use a dict to collect sets of indices for each normalized node pair
    idx_name = network.index.name if network.index.name else "index"
    link_dict = defaultdict(set)
    df_reset = network.reset_index()
    for row in df_reset.itertuples(index=False):
        from_node = getattr(row, "fromNode")
        to_node = getattr(row, "toNode")
        idx_val = getattr(row, idx_name)
        key = tuple(sorted([from_node, to_node]))
        link_dict[key].add(idx_val)

    return link_dict


def _link_map(sets: Collection[set]) -> dict:
    """
    Create a mapping of all set items to the first item in the set,
    including the first item itself.
    """
    mapping = {}
    for s in sets:
        if len(s) == 0:
            continue
        items = sorted(s)
        first = items.pop(0)
        mapping[first] = first
        mapping.update({e: first for e in items})
    return mapping


def _link_count(trips: gp.GeoDataFrame, link_col: str, mapping: dict) -> Counter:
    counter = Counter()

    def _split_and_map(links_str):
        return [mapping[link] for link in links_str.split(",")]

    trips[link_col].apply(lambda l: counter.update(_split_and_map(l)))
    return counter


def combine_trip_stats(stations: list[str], output_path) -> pd.DataFrame:
    """
    Read raw trip stats CSVs from the output directory
    and combine them into a single DataFrame.
    """
    all_stats = []
    for station in stations:
        stats = prep_raw_trip_stats(output_path / f"{station}-tripStats.csv")
        if stats.empty:
            continue
        all_stats.append(stats)
        stats[COL_STATION] = station
    return (
        pd.concat(all_stats).reset_index().set_index([COL_STATION, COL_MODEL, COL_ATT])
    )


def prep_raw_trip_stats(path: Path) -> pd.DataFrame:
    """
    Read raw trip stats CSV and transform it into a nicer format
    """
    if not path.exists():
        logging.warning(f"Trip stats file {path} does not exist.")
        return pd.DataFrame()
    df = pd.read_csv(path, index_col=0)
    df = df.loc[df.index.str.endswith("-100pct")]
    df.index = df.index.str.replace("-100pct", "", regex=False)

    results = []
    for model in behavior.MODELS.keys():
        model_id = behavior.model_id(model, behavior.MODE_COMBINED)
        cols = df.columns[
            (df.columns.str.get(0).astype(int) < 4)
            | (df.columns.str.contains(model_id))
        ]
        model_results = df[cols].copy()
        model_results.columns = model_results.columns.str[:1]
        model_results[COL_MODEL] = model
        results.append(model_results)

    combined_res = pd.concat(results)
    return combined_res.reset_index().set_index([COL_MODEL, COL_ATT])
